//! Parity suite for the cpu-ref backend.
//!
//! Verifies that `CpuRefBackend::dispatch` produces correct results for
//! every major IR shape: arithmetic, bitwise, control flow, memory access,
//! and multi-buffer programs. Each test constructs a `Program`, dispatches
//! through the `VyreBackend` trait surface, and asserts byte-exact output.

use vyre_driver::DispatchConfig;
use vyre_driver_reference::CpuRefEvaluator;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::dispatch_fixtures;
use dispatch_fixtures::{binary_program, dispatch_no_input, dispatch_with_inputs, u32_out_buffer};

// ---------------------------------------------------------------
// Scalar expression shapes: store, arithmetic, bitwise
// ---------------------------------------------------------------

/// One scalar shape: the expression the program stores into `out[0]`, and the
/// word the reference backend must return for it.
struct ScalarCase {
    name: &'static str,
    value: fn() -> Expr,
    expected: u32,
}

const SCALAR_CASES: &[ScalarCase] = &[
    ScalarCase {
        name: "store literal",
        value: || Expr::u32(42),
        expected: 42,
    },
    ScalarCase {
        name: "store zero",
        value: || Expr::u32(0),
        expected: 0,
    },
    ScalarCase {
        name: "add",
        value: || Expr::add(Expr::u32(10), Expr::u32(32)),
        expected: 42,
    },
    ScalarCase {
        name: "sub",
        value: || Expr::sub(Expr::u32(50), Expr::u32(8)),
        expected: 42,
    },
    ScalarCase {
        name: "mul",
        value: || Expr::mul(Expr::u32(6), Expr::u32(7)),
        expected: 42,
    },
    ScalarCase {
        name: "bitxor",
        value: || Expr::bitxor(Expr::u32(0xFF), Expr::u32(0x55)),
        expected: 0xAA,
    },
    ScalarCase {
        name: "bitand",
        value: || Expr::bitand(Expr::u32(0xFF), Expr::u32(0x0F)),
        expected: 0x0F,
    },
    ScalarCase {
        name: "bitor",
        value: || Expr::bitor(Expr::u32(0xF0), Expr::u32(0x0F)),
        expected: 0xFF,
    },
];

#[test]
fn scalar_expression_shapes_dispatch_to_their_pinned_word() {
    for case in SCALAR_CASES {
        let program = Program::wrapped(
            vec![u32_out_buffer("out", 0)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), (case.value)())],
        );
        assert_eq!(
            dispatch_no_input(&program),
            vec![case.expected.to_le_bytes().to_vec()],
            "Fix: cpu-ref must evaluate the {} shape to {}.",
            case.name,
            case.expected
        );
    }
}

// ---------------------------------------------------------------
// Input buffer passthrough (read → write)
// ---------------------------------------------------------------

#[test]
fn input_buffer_passthrough() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("val", Expr::load("input", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("val")),
        ],
    );
    let input = 99u32.to_le_bytes().to_vec();
    let outputs = dispatch_with_inputs(&program, &[input]);
    assert_eq!(outputs, vec![99u32.to_le_bytes().to_vec()]);
}

/// WHY: a reference input the caller never supplied is an ABI failure. Zero
/// synthesis would answer it with fabricated data and hide the caller's defect
/// behind a plausible-looking result.
#[test]
fn missing_input_buffer_is_rejected_rather_than_synthesized() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("val", Expr::load("input", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("val")),
        ],
    );
    let error = CpuRefEvaluator
        .evaluate(&program, &[], &DispatchConfig::default())
        .expect_err("a missing reference input must be rejected");
    assert!(
        error
            .to_string()
            .contains("missing an input buffer for `input`"),
        "reference backend must name the buffer it never received: {error}"
    );
}

/// WHY: backend-allocated outputs are not host inputs. Accepting an initializer
/// creates a second submission geometry that can diverge from target backends.
#[test]
fn backend_allocated_output_initializer_is_rejected() {
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );

    let error = CpuRefEvaluator
        .evaluate(
            &program,
            &[&0_u32.to_le_bytes()[..]],
            &DispatchConfig::default(),
        )
        .expect_err("backend-allocated output initializers must be rejected");
    assert!(
        error.to_string().contains("extra input buffer"),
        "reference backend must report the non-canonical input: {error}"
    );
}

// ---------------------------------------------------------------
// Two-buffer XOR (the README example)
// ---------------------------------------------------------------

#[test]
fn two_buffer_xor() {
    let program = binary_program(Expr::bitxor);
    let a = 0xAAu32.to_le_bytes().to_vec();
    let b = 0x55u32.to_le_bytes().to_vec();
    let outputs = dispatch_with_inputs(&program, &[a, b]);
    // 0xAA ^ 0x55 = 0xFF = 255
    assert_eq!(outputs, vec![255u32.to_le_bytes().to_vec()]);
}

// ---------------------------------------------------------------
// Conditional: if-then store
// ---------------------------------------------------------------

#[test]
fn conditional_if_true() {
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![
            // Store 0 first, then conditionally overwrite with 42
            Node::store("out", Expr::u32(0), Expr::u32(0)),
            Node::if_then(
                Expr::bool(true),
                vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
            ),
        ],
    );
    let outputs = dispatch_no_input(&program);
    assert_eq!(outputs, vec![42u32.to_le_bytes().to_vec()]);
}

#[test]
fn conditional_if_false() {
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![
            // Store 99, then conditionally overwrite  -  but condition is false
            Node::store("out", Expr::u32(0), Expr::u32(99)),
            Node::if_then(
                Expr::bool(false),
                vec![Node::store("out", Expr::u32(0), Expr::u32(0))],
            ),
        ],
    );
    let outputs = dispatch_no_input(&program);
    // if-false branch not taken → 99 survives
    assert_eq!(outputs, vec![99u32.to_le_bytes().to_vec()]);
}

// ---------------------------------------------------------------
// Evaluator determinism: same program twice = same bytes
// ---------------------------------------------------------------

#[test]
fn evaluators_produce_identical_bytes_on_same_inputs() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("val", Expr::load("a", Expr::u32(0))),
            Node::store("out", Expr::u32(0), Expr::var("val")),
        ],
    );
    let input_bytes = 77u32.to_le_bytes();

    let out1 = evaluator
        .evaluate(&program, &[&input_bytes[..]], &DispatchConfig::default())
        .expect("eval 1");
    let out2 = evaluator
        .evaluate(&program, &[&input_bytes[..]], &DispatchConfig::default())
        .expect("eval 2");

    assert_eq!(
        out1, out2,
        "Fix: evaluate must produce deterministic bytes."
    );
}

#[test]
fn extra_input_buffers_rejected() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let result = evaluator.evaluate(&program, &[&[0; 4], &[0; 4]], &DispatchConfig::default());
    assert!(
        result.is_err(),
        "Fix: extra input buffers must be rejected."
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("extra input buffer"),
        "Fix: error must name extra input buffer, got: {err_msg}"
    );
}

#[test]
fn determinism_guarantee() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("v", Expr::add(Expr::load("a", Expr::u32(0)), Expr::u32(1))),
            Node::store("out", Expr::u32(0), Expr::var("v")),
        ],
    );
    let input = 100u32.to_le_bytes();
    let config = DispatchConfig::default();

    let out1 = evaluator
        .evaluate(&program, &[&input[..]], &config)
        .unwrap();
    let out2 = evaluator
        .evaluate(&program, &[&input[..]], &config)
        .unwrap();
    assert_eq!(
        out1, out2,
        "Fix: cpu-ref must be deterministic  -  identical inputs must produce identical outputs."
    );
}
#[test]
fn caller_supplied_physical_grid_cannot_change_the_semantic_answer() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32),
            BufferDecl::read("b", 1, DataType::U32),
            u32_out_buffer("out", 2),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("a", Expr::u32(0)), Expr::load("b", Expr::u32(0))),
        )],
    );
    let a = 17u32.to_le_bytes();
    let b = 25u32.to_le_bytes();
    let inputs = [a.as_slice(), b.as_slice()];

    let default_config = DispatchConfig::default();
    let default_output = evaluator
        .evaluate(&program, &inputs, &default_config)
        .expect("default config evaluate");

    let mut modified_config = DispatchConfig::default();
    modified_config.dispatch_elements = Some(100_000);
    modified_config.dispatch_grid = Some([64, 4, 2]);
    modified_config.grid_override = Some([128, 1, 1]);

    let modified_output = evaluator
        .evaluate(&program, &inputs, &modified_config)
        .expect("modified config evaluate");

    assert_eq!(
        default_output, modified_output,
        "Fix: physical launch policy (grid_override, dispatch_grid, dispatch_elements) must not change the semantic answer of reference evaluation"
    );
    assert_eq!(
        default_output,
        vec![42u32.to_le_bytes().to_vec()],
        "Fix: output must match 17 + 25 = 42"
    );
}
