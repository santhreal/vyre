//! Emitted PTX must stay byte-identical across refactors of this crate.
//!
//! The shared success corpus is rendered to one pinned text file. Any change to
//! register allocation, instruction selection, directive order, or whitespace
//! moves those bytes and fails here. That is the contract every emitter
//! deduplication turns on: shared code may move, emitted text may not.
//!
//! The corpus, the section format, and the comparison live in
//! `vyre_lower::artifact_golden`. This file supplies only the PTX rendering.

use std::path::PathBuf;

use vyre_lower::artifact_golden;

fn golden_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-emit-ptx/tests/golden/success_corpus.ptx")
}

fn render_corpus() -> String {
    artifact_golden::render_success_corpus(|descriptor| {
        vyre_emit_ptx::emit(descriptor).unwrap_or_else(|error| {
            panic!("Fix: success-corpus descriptor must emit PTX: {error:?}")
        })
    })
}

#[test]
fn emitted_ptx_matches_the_pinned_corpus() {
    artifact_golden::assert_matches_golden(&golden_path(), &render_corpus());
}

/// A pinned corpus that no longer names every case would silently stop
/// checking the cases it dropped.
#[test]
fn pinned_corpus_covers_every_shared_success_case() {
    let golden = std::fs::read_to_string(golden_path()).expect("pinned PTX corpus must exist");
    for case in vyre_lower::emit_adversarial_corpus::success_cases() {
        assert!(
            golden.contains(&format!("===== {}\n", case.id)),
            "Fix: pinned PTX corpus is missing case `{}`; re-bless it.",
            case.id
        );
    }
}

/// Emission must be a pure function of the descriptor. A renderer that
/// depended on iteration order or an address would pass the golden once and
/// fail the next run.
#[test]
fn emitted_ptx_is_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

#[test]
#[ignore = "writes the pinned corpus; run deliberately and review the diff"]
fn bless_pinned_ptx_corpus() {
    artifact_golden::write_golden(&golden_path(), &render_corpus());
}
