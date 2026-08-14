//! Byte-stability golden for what the reference backend computes.
//!
//! The reference interpreter is the oracle every backend is diffed against, so
//! a refactor inside it is only safe if the bytes it returns do not move. This
//! pins two surfaces:
//!
//! - every registered dual-reference facet, over a fixed hostile seed set,
//!   enumerated from the registry so a new facet turns the corpus stale;
//! - the shared neutral program corpus, run through
//!   [`vyre_reference::reference_eval`].
//!
//! The program corpus is `vyre_lower::program_stability_corpus`, shared with the
//! emitted-artifact goldens: one `Program` pins both what a backend emits for it
//! and what the oracle computes for it. The section format, the comparison, and
//! the hex rendering live in `vyre_lower::artifact_golden`.

#![forbid(unsafe_code)]

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;

use vyre_lower::artifact_golden::{
    assert_matches_golden, contains_case, hex_words, render_sections, write_golden,
};
use vyre_lower::program_stability_corpus;
use vyre_reference::value::Value;
use vyre_reference::{dual_op_ids, reference_eval, resolve_dual};

/// Fixed hostile seeds every dual facet is evaluated over.
const FACET_SEEDS: [u32; 8] = [
    0,
    1,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0x0000_ffff,
    0xdead_beef,
    0x0f0f_0f0f,
];

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/reference_outputs.txt")
}

/// Widen one seed into the byte input a dual facet consumes.
///
/// Facets read a fixed-width prefix, so the widest supported operand set is
/// supplied once and narrower facets ignore the tail.
fn facet_input(seed: u32) -> Vec<u8> {
    let left = seed
        .wrapping_mul(0x85eb_ca6b)
        .rotate_left((seed ^ 0x13) & 31);
    let right = seed
        .wrapping_mul(0xc2b2_ae35)
        .rotate_right((seed ^ 0x29) & 31);
    let mut input = Vec::with_capacity(48);
    for word in [left, right, left ^ right, left.wrapping_add(right)] {
        input.extend_from_slice(&word.to_le_bytes());
    }
    for word in [
        (u64::from(left) << 32) | u64::from(right),
        (u64::from(right) << 32) | u64::from(left),
    ] {
        input.extend_from_slice(&word.to_le_bytes());
    }
    input.extend_from_slice(&f32::from_bits(left).to_le_bytes());
    input.extend_from_slice(&f32::from_bits(right).to_le_bytes());
    input
}

/// Render one dual facet's outputs across the seed set, asserting the two
/// independent references agree before either is pinned.
fn render_facet(op_id: &str) -> String {
    let (reference_a, reference_b) =
        resolve_dual(op_id).expect("Fix: a registered dual facet must resolve");
    let mut text = String::new();
    for seed in FACET_SEEDS {
        let input = facet_input(seed);
        let output_a = reference_a(&input);
        let output_b = reference_b(&input);
        assert_eq!(
            output_a, output_b,
            "Fix: dual references for {op_id} diverged at seed {seed:#010x}"
        );
        writeln!(text, "seed {seed:#010x}").expect("string write");
        text.push_str(&hex_words(&output_a));
    }
    text
}

/// Render one neutral program case's oracle outputs.
fn render_program(case: &program_stability_corpus::StabilityCase) -> String {
    let values = case
        .inputs
        .iter()
        .map(|bytes| Value::Bytes(Arc::from(bytes.clone().into_boxed_slice())))
        .collect::<Vec<_>>();
    let outputs = reference_eval(&case.program, &values).unwrap_or_else(|error| {
        panic!(
            "Fix: shared stability case `{}` must evaluate: {error}",
            case.id
        )
    });
    let mut text = String::new();
    for (index, output) in outputs.iter().enumerate() {
        writeln!(text, "output {index}").expect("string write");
        text.push_str(&hex_words(&output.to_bytes()));
    }
    text
}

/// Section id under which one dual facet is pinned.
fn facet_section(op_id: &str) -> String {
    format!("dual::{op_id}")
}

/// Section id under which one neutral program case is pinned.
fn program_section(case_id: &str) -> String {
    format!("program::{case_id}")
}

/// Render every registered dual facet plus the shared neutral program corpus.
fn render_corpus() -> String {
    let facets = dual_op_ids()
        .iter()
        .map(|op_id| (facet_section(op_id), render_facet(op_id)));
    let programs = program_stability_corpus::cases()
        .into_iter()
        .map(|case| (program_section(case.id), render_program(&case)));
    render_sections(facets.chain(programs))
}

/// WHY: the reference interpreter is the conformance oracle. A change in the
/// bytes it computes is a change in what every backend is graded against, so it
/// must never happen as a side effect of a refactor.
#[test]
fn reference_outputs_match_the_pinned_corpus() {
    assert_matches_golden(&golden_path(), &render_corpus());
}

/// WHY: reference evaluation must be a pure function of program and input. A
/// renderer that depended on iteration order or an address would pass the
/// golden once and fail the next run.
#[test]
fn reference_outputs_are_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

/// WHY: a pinned corpus that no longer names every registered dual facet, or
/// every shared program case, would silently stop covering it.
#[test]
fn pinned_corpus_covers_every_facet_and_shared_case() {
    let golden =
        std::fs::read_to_string(golden_path()).expect("pinned reference corpus must exist");
    for op_id in dual_op_ids() {
        assert!(
            contains_case(&golden, &facet_section(op_id)),
            "Fix: pinned reference corpus is missing dual facet `{op_id}`; re-bless it."
        );
    }
    for case in program_stability_corpus::cases() {
        assert!(
            contains_case(&golden, &program_section(case.id)),
            "Fix: pinned reference corpus is missing program case `{}`; re-bless it.",
            case.id
        );
    }
}

#[test]
#[ignore = "bless: rewrites the pinned reference-output corpus"]
fn bless_pinned_reference_corpus() {
    write_golden(&golden_path(), &render_corpus());
}
