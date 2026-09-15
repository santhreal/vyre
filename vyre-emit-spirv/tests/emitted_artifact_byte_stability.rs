//! Emitted SPIR-V must stay byte-identical across refactors of this crate.
//!
//! The shared success corpus is rendered to one pinned text file as hex words.
//! Any change to the module this crate hands the SPIR-V writer, or to the
//! writer options it chooses, moves those bytes and fails here. That is the
//! contract every emitter deduplication turns on: shared code may move,
//! emitted bytes may not.
//!
//! The corpus, the section format, and the comparison live in
//! `vyre_lower::artifact_golden`. This file supplies only the SPIR-V rendering.

use std::path::PathBuf;

use vyre_lower::artifact_golden;

fn golden_path() -> PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
        .join("vyre-emit-spirv/tests/golden/success_corpus.spv.hex")
}

fn render_corpus() -> String {
    artifact_golden::render_success_corpus(|descriptor| {
        let bytes = vyre_emit_spirv::emit_bytes(descriptor).unwrap_or_else(|error| {
            panic!("Fix: success-corpus descriptor must emit SPIR-V: {error:?}")
        });
        artifact_golden::hex_words(&bytes)
    })
}

#[test]
fn emitted_spirv_matches_the_pinned_corpus() {
    artifact_golden::assert_matches_golden(&golden_path(), &render_corpus());
}

/// A pinned corpus that no longer names every case would silently stop
/// checking the cases it dropped.
#[test]
fn pinned_corpus_covers_every_shared_success_case() {
    let golden = std::fs::read_to_string(golden_path()).expect("pinned SPIR-V corpus must exist");
    for case in vyre_lower::emit_adversarial_corpus::success_cases() {
        assert!(
            golden.contains(&format!("===== {}\n", case.id)),
            "Fix: pinned SPIR-V corpus is missing case `{}`; re-bless it.",
            case.id
        );
    }
}

/// Emission must be a pure function of the descriptor.
#[test]
fn emitted_spirv_is_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

#[test]
#[ignore = "writes the pinned corpus; run deliberately and review the diff"]
fn bless_pinned_spirv_corpus() {
    artifact_golden::write_golden(&golden_path(), &render_corpus());
}
