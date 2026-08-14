//! Wire-format fuzz infrastructure contracts.
//!
//! `Program::from_wire` is an untrusted parser surface. These tests keep the
//! libFuzzer target, corpus layout, and nightly release hook from drifting.

use std::fs;
use std::path::PathBuf;
use vyre_test_support::monorepo::vyre_workspace_root;

fn workspace_root() -> PathBuf {
    vyre_workspace_root()
}

#[test]
fn program_wire_fuzz_corpus_is_nontrivial_and_contains_named_valid_programs() {
    let corpus_dir = workspace_root().join("vyre-foundation/fuzz/corpus/program_wire");
    let entries = fs::read_dir(&corpus_dir)
        .unwrap_or_else(|error| panic!("{} must be readable: {error}", corpus_dir.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("fuzz corpus entry must be readable: {error}"))
                .path()
        })
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();

    assert!(
        entries.len() >= 64,
        "Fix: program_wire fuzz corpus has only {} entries; keep regression seeds checked in.",
        entries.len()
    );
    for named_seed in [
        "empty_program.vir0",
        "literal_u32_store.vir0",
        "bin_op_add.vir0",
        "if_then_else.vir0",
        "barrier_only.vir0",
    ] {
        assert!(
            corpus_dir.join(named_seed).is_file(),
            "Fix: program_wire fuzz corpus must keep named valid-program seed `{named_seed}`."
        );
    }
}
