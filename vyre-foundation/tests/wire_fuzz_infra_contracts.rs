//! Wire-format fuzz infrastructure contracts.
//!
//! `Program::from_wire` is an untrusted parser surface. These tests keep the
//! libFuzzer target, corpus layout, and nightly release hook from drifting.
//!
//! WHY the corpus is replayed here and not only under libFuzzer: a fuzz corpus
//! that nothing reads on an ordinary test run is a directory, not a regression
//! suite. The three invariants below are the fuzz target's own invariants, so a
//! wire-format change that breaks a seed the fuzzer once found is red on any
//! machine, without a nightly run and without libFuzzer installed.
//!
//! What these do NOT catch: an input the corpus does not contain. Only the
//! fuzzer finds those, and a crash it finds belongs here as a new seed.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use vyre_foundation::ir::Program;
use vyre_test_support::monorepo::vyre_workspace_root;

/// Seeds whose names claim they decode to a valid program.
///
/// These are hand-authored rather than fuzzer-found, so they are named here: a
/// rename or deletion is a decision, not a corpus trim.
const NAMED_VALID_SEEDS: [&str; 5] = [
    "empty_program.vir0",
    "literal_u32_store.vir0",
    "bin_op_add.vir0",
    "if_then_else.vir0",
    "barrier_only.vir0",
];

fn corpus_directory() -> PathBuf {
    vyre_workspace_root().join("vyre-foundation/fuzz/corpus/program_wire")
}

/// Every corpus entry, read at run time so a seed added tomorrow is covered.
fn corpus_entries(directory: &Path) -> Vec<(String, Vec<u8>)> {
    let mut entries: Vec<(String, Vec<u8>)> = fs::read_dir(directory)
        .unwrap_or_else(|error| {
            panic!(
                "Fix: {} must be readable and checked in; the program_wire fuzz corpus is a \
                 tracked regression suite, not a local scratch directory: {error}",
                directory.display()
            )
        })
        .map(|entry| {
            entry.unwrap_or_else(|error| panic!("Fix: fuzz corpus entry must be readable: {error}"))
                .path()
        })
        .filter(|path| path.is_file())
        .map(|path| {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_else(|| panic!("Fix: corpus entry {} needs a UTF-8 name", path.display()))
                .to_string();
            let bytes = fs::read(&path)
                .unwrap_or_else(|error| panic!("Fix: read {}: {error}", path.display()));
            (name, bytes)
        })
        .collect();
    entries.sort_by(|left, right| left.0.cmp(&right.0));
    assert!(
        !entries.is_empty(),
        "Fix: the program_wire fuzz corpus is empty, so every contract below guards nothing."
    );
    entries
}

#[test]
fn program_wire_fuzz_corpus_keeps_its_named_seeds_and_no_byte_identical_pair() {
    let directory = corpus_directory();
    let entries = corpus_entries(&directory);

    for named_seed in NAMED_VALID_SEEDS {
        assert!(
            entries.iter().any(|(name, _)| name == named_seed),
            "Fix: program_wire fuzz corpus must keep named valid-program seed `{named_seed}`."
        );
    }

    let mut by_content: BTreeMap<&[u8], Vec<&str>> = BTreeMap::new();
    for (name, bytes) in &entries {
        by_content
            .entry(bytes.as_slice())
            .or_default()
            .push(name.as_str());
    }
    let duplicates: Vec<String> = by_content
        .values()
        .filter(|names| names.len() > 1)
        .map(|names| names.join(" == "))
        .collect();
    assert!(
        duplicates.is_empty(),
        "Fix: {} program_wire corpus entries are byte-identical to another entry, so the fuzzer \
         spends its budget re-deciding the same input: {}. Delete the copy.",
        duplicates.len(),
        duplicates.join(", ")
    );
}

#[test]
fn every_program_wire_corpus_seed_replays_the_fuzz_target_invariants() {
    let directory = corpus_directory();
    let mut decoded = 0usize;
    for (name, bytes) in corpus_entries(&directory) {
        // Invariant 1: never panic on arbitrary input, and Invariant 2: every
        // rejection carries a Fix: hint.
        let program = match Program::from_wire(&bytes) {
            Ok(program) => program,
            Err(error) => {
                let message = error.to_string();
                assert!(
                    message.contains("Fix:"),
                    "Fix: `from_wire` rejected corpus seed `{name}` without a Fix: hint, so the \
                     caller is told what happened and not what to do: {message}"
                );
                continue;
            }
        };
        decoded += 1;

        // Invariant 3: from_wire . to_wire . from_wire == from_wire.
        let round = program.to_wire().unwrap_or_else(|error| {
            panic!(
                "Fix: `to_wire` failed for corpus seed `{name}`, which had just decoded; a program \
                 that decodes must re-encode: {error}"
            )
        });
        let reparsed = Program::from_wire(&round).unwrap_or_else(|error| {
            panic!(
                "Fix: re-decoding canonical `to_wire` bytes for corpus seed `{name}` failed: \
                 {error}"
            )
        });
        assert!(
            program.structural_eq(&reparsed),
            "Fix: corpus seed `{name}` is not structurally equal to itself after a wire \
             round-trip, so the encoder and decoder disagree about it."
        );
    }

    for named_seed in NAMED_VALID_SEEDS {
        let bytes = fs::read(directory.join(named_seed))
            .unwrap_or_else(|error| panic!("Fix: read named seed `{named_seed}`: {error}"));
        Program::from_wire(&bytes).unwrap_or_else(|error| {
            panic!(
                "Fix: named valid-program seed `{named_seed}` no longer decodes, so either the \
                 wire format changed without migrating the seed or the seed is misnamed: {error}"
            )
        });
    }

    assert!(
        decoded >= NAMED_VALID_SEEDS.len(),
        "Fix: only {decoded} corpus seeds decoded to a program, fewer than the {} named valid \
         seeds; a corpus of nothing but rejections never exercises the round-trip.",
        NAMED_VALID_SEEDS.len()
    );
}
