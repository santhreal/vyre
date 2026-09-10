//! Byte-stability golden for what the reference oracle computes.
//!
//! The reference interpreter is the oracle every backend is diffed against, so
//! a refactor inside it is only safe if the bytes it returns do not move. This
//! pins the shared neutral program corpus run through
//! [`vyre_reference::reference_eval`].
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
use vyre_reference::reference_eval;
use vyre_reference::value::Value;

fn golden_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-reference/tests/golden/reference_outputs.txt")
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

/// Section id under which one neutral program case is pinned.
fn program_section(case_id: &str) -> String {
    format!("program::{case_id}")
}

/// Render the shared neutral program corpus.
fn render_corpus() -> String {
    render_sections(
        program_stability_corpus::cases()
            .into_iter()
            .map(|case| (program_section(case.id), render_program(&case))),
    )
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

/// WHY: a pinned corpus that no longer names every shared program case would
/// silently stop covering it.
#[test]
fn pinned_corpus_covers_every_shared_case() {
    let golden =
        std::fs::read_to_string(golden_path()).expect("pinned reference corpus must exist");
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
