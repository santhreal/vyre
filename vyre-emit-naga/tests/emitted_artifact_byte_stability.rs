//! The module this crate emits must stay byte-identical across its refactors.
//!
//! The module itself is not bytes, so the pin is its shader-text serialization:
//! two modules serialize identically exactly when they carry the same types,
//! globals, expressions, and statements in the same order. Any change to what
//! this emitter builds moves that text and fails here. That is the contract
//! every emitter deduplication turns on: shared code may move, emitted output
//! may not.
//!
//! Pinning the text also pins the two crates that re-serialize this module into
//! their own artifacts, since both start from exactly what is pinned here.
//!
//! The corpus, the section format, and the comparison live in
//! `vyre_lower::artifact_golden`. This file supplies only the serialization.

use std::path::PathBuf;

use naga::back::wgsl::{write_string, WriterFlags};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use vyre_lower::artifact_golden;

fn golden_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/success_corpus.wgsl")
}

fn render_corpus() -> String {
    artifact_golden::render_success_corpus(|descriptor| {
        let module = vyre_emit_naga::emit(descriptor).unwrap_or_else(|error| {
            panic!("Fix: success-corpus descriptor must emit a module: {error:?}")
        });
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|error| {
                panic!("Fix: emitted module must validate before serialization: {error:?}")
            });
        write_string(&module, &info, WriterFlags::empty()).unwrap_or_else(|error| {
            panic!("Fix: validated module must serialize to shader text: {error:?}")
        })
    })
}

#[test]
fn emitted_module_matches_the_pinned_corpus() {
    artifact_golden::assert_matches_golden(&golden_path(), &render_corpus());
}

/// A pinned corpus that no longer names every case would silently stop
/// checking the cases it dropped.
#[test]
fn pinned_corpus_covers_every_shared_success_case() {
    let golden = std::fs::read_to_string(golden_path()).expect("pinned module corpus must exist");
    for case in vyre_lower::emit_adversarial_corpus::success_cases() {
        assert!(
            golden.contains(&format!("===== {}\n", case.id)),
            "Fix: pinned module corpus is missing case `{}`; re-bless it.",
            case.id
        );
    }
}

/// Emission must be a pure function of the descriptor. `emit` allocates handles
/// as it walks the descriptor, so an iteration order that depended on a hash
/// seed would pass the golden once and fail the next run.
#[test]
fn emitted_module_is_deterministic_across_runs() {
    assert_eq!(render_corpus(), render_corpus());
}

#[test]
#[ignore = "writes the pinned corpus; run deliberately and review the diff"]
fn bless_pinned_module_corpus() {
    artifact_golden::write_golden(&golden_path(), &render_corpus());
}
