//! The fused sampler draws the same token as the three stages run separately,
//! and its result does not depend on the order the interpreter visits lanes.
//!
//! WHY: `TokenSampler::program` fuses a parallel writer (`logit_adjust`, one
//! invocation per vocabulary entry) with two invocation-gated readers
//! (`softmax_top_k` and `nucleus_select`, both reading the whole adjusted row)
//! into one dispatch. That is correct only if the fuser places a grid-wide
//! dependency between the arms. Nothing in the recorded registry fixture proves
//! it: a fixture is one evaluation, and an evaluator that happens to run the
//! writer first records a passing row for a program that races on a device.
//!
//! Two independent observations pin it. Executing the fused program must agree
//! with executing the three stage programs in sequence, which is the definition
//! of the fusion being meaning-preserving. And the fused program must give the
//! same token under reversed and rotated lane order, because a gated reader
//! that could observe a partly written row would read different values once the
//! lanes that fill that row run in a different order.
//!
//! What it does not catch: a hazard that only appears at a grid wider than the
//! vocabulary used here, and a backend that ignores the synchronisation the IR
//! carries. The second is the backend's own parity suite.
#![cfg(feature = "llm")]

use vyre_foundation::ir::{BufferAccess, BufferDecl, Program};
use vyre_libs::llm::sampling::{logit_adjust, nucleus_select, TokenSampler};
use vyre_libs::nn::moe::softmax_top_k;
use vyre_reference::value::Value;
use vyre_reference::{is_reference_input, output_index};

const VOCABULARY: u32 = 12;
const CANDIDATES: u32 = 5;
const TEMPERATURE: f32 = 0.8;
const PENALTY: f32 = 1.3;
const TOP_P: f32 = 0.9;
const UNIFORM: f32 = 0.42;

fn logits() -> Vec<f32> {
    (0..VOCABULARY)
        .map(|i| ((f32::from(i as u16) * 0.37).sin() * 3.0) - 0.4)
        .collect()
}

fn counts() -> Vec<u32> {
    (0..VOCABULARY).map(|i| i % 3).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Bytes one zero-filled buffer needs, derived from its own declaration.
fn byte_len(decl: &BufferDecl) -> usize {
    decl.static_byte_len()
        .expect("Fix: a sampler buffer declared an unmeasurable size")
        .expect("Fix: a sampler buffer has no static element count")
}

/// Inputs in the interpreter's ABI order, zero-filled except for the named
/// buffers this test supplies.
fn inputs(program: &Program, supplied: &[(&str, Vec<u8>)]) -> Vec<Value> {
    program
        .buffers()
        .iter()
        .filter(|decl| is_reference_input(decl))
        .map(|decl| {
            supplied
                .iter()
                .find(|(name, _)| *name == decl.name())
                .map(|(_, bytes)| Value::from(bytes.clone()))
                .unwrap_or_else(|| Value::from(vec![0u8; byte_len(decl)]))
        })
        .collect()
}

fn read_u32(values: &[Value], program: &Program, name: &str) -> u32 {
    let index = output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: {name} must be a returned output"));
    let bytes = values[index].to_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_bytes(values: &[Value], program: &Program, name: &str) -> Vec<u8> {
    let index = output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: {name} must be a returned output"));
    values[index].to_bytes().to_vec()
}

fn sampler() -> TokenSampler<'static> {
    TokenSampler {
        logits: "logits",
        counts: "counts",
        uniform: "uniform",
        token: "token",
        vocabulary: VOCABULARY,
        candidates: CANDIDATES,
        temperature: TEMPERATURE,
        repetition_penalty: PENALTY,
        top_p: TOP_P,
    }
}

fn fused_inputs(program: &Program) -> Vec<Value> {
    inputs(
        program,
        &[
            ("logits", f32_bytes(&logits())),
            ("counts", u32_bytes(&counts())),
            ("uniform", f32_bytes(&[UNIFORM])),
        ],
    )
}

/// Run the three stages as separate programs, threading each stage's output
/// into the next, and return the drawn token.
fn unfused_token() -> u32 {
    let adjust = logit_adjust(
        "logits",
        "counts",
        "adjusted",
        VOCABULARY,
        TEMPERATURE,
        PENALTY,
    );
    let adjusted = read_bytes(
        &vyre_reference::reference_eval(
            &adjust,
            &inputs(
                &adjust,
                &[
                    ("logits", f32_bytes(&logits())),
                    ("counts", u32_bytes(&counts())),
                ],
            ),
        )
        .expect("Fix: the adjust stage must evaluate"),
        &adjust,
        "adjusted",
    );

    let select = softmax_top_k("adjusted", "selected", "weights", VOCABULARY, CANDIDATES);
    let selection =
        vyre_reference::reference_eval(&select, &inputs(&select, &[("adjusted", adjusted)]))
            .expect("Fix: the selection stage must evaluate");
    let selected = read_bytes(&selection, &select, "selected");
    let weights = read_bytes(&selection, &select, "weights");

    let draw = nucleus_select("selected", "weights", "uniform", "token", CANDIDATES, TOP_P);
    let drawn = vyre_reference::reference_eval(
        &draw,
        &inputs(
            &draw,
            &[
                ("selected", selected),
                ("weights", weights),
                ("uniform", f32_bytes(&[UNIFORM])),
            ],
        ),
    )
    .expect("Fix: the draw stage must evaluate");
    read_u32(&drawn, &draw, "token")
}

#[test]
fn fusing_the_sampler_draws_the_token_the_separate_stages_draw() {
    let program = sampler().program().expect("Fix: the sampler must build");
    let fused = read_u32(
        &vyre_reference::reference_eval(&program, &fused_inputs(&program))
            .expect("Fix: the fused sampler must evaluate"),
        &program,
        "token",
    );
    assert_eq!(
        fused,
        unfused_token(),
        "the fused dispatch must draw the token the unfused pipeline draws"
    );
}

#[test]
fn the_gated_reader_never_sees_a_partly_written_row() {
    let program = sampler().program().expect("Fix: the sampler must build");
    let baseline = read_u32(
        &vyre_reference::reference_eval(&program, &fused_inputs(&program))
            .expect("Fix: the fused sampler must evaluate"),
        &program,
        "token",
    );

    let reversed = read_u32(
        &vyre_reference::reference_eval_lane_reversed(&program, &fused_inputs(&program))
            .expect("Fix: the fused sampler must evaluate in reverse lane order"),
        &program,
        "token",
    );
    assert_eq!(
        reversed, baseline,
        "reversing lane order changed the draw, so a reader observed the writer's partial row"
    );

    for by in 1..VOCABULARY {
        let rotated = read_u32(
            &vyre_reference::reference_eval_lane_rotated(&program, &fused_inputs(&program), by)
                .expect("Fix: the fused sampler must evaluate in rotated lane order"),
            &program,
            "token",
        );
        assert_eq!(
            rotated, baseline,
            "rotating lane order by {by} changed the draw, so the arms are not ordered"
        );
    }
}

/// Fusing concatenates buffer declarations, so each stage's output stays an
/// output of the fused program unless the composition demotes it. A demoted
/// buffer is still allocated by the backend; what changes is that the host no
/// longer reads it back, which is what `is_output` selects.
#[test]
fn only_the_token_is_read_back_from_the_fused_program() {
    let program = sampler().program().expect("Fix: the sampler must build");
    let read_back: Vec<&str> = program
        .buffers()
        .iter()
        .filter(|decl| decl.is_output())
        .map(BufferDecl::name)
        .collect();
    assert_eq!(
        read_back,
        vec!["token"],
        "the fused sampler reads back the token and nothing else"
    );

    let staged: Vec<&str> = program
        .buffers()
        .iter()
        .filter(|decl| decl.access() != BufferAccess::Workgroup)
        .filter(|decl| !decl.is_output() && decl.is_backend_allocated_output())
        .map(BufferDecl::name)
        .collect();
    assert!(
        !staged.is_empty(),
        "the stages hand each other a row, so the fused program must still allocate one"
    );
    for name in staged {
        assert!(
            program
                .buffers()
                .iter()
                .find(|decl| decl.name() == name)
                .is_some_and(BufferDecl::is_pipeline_live_out),
            "{name} is allocated but not marked live out, so deadness analysis may drop its writes"
        );
    }
}
