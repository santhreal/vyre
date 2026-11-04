//! `read_source_bounded` is the one reader every gate uses for a tracked Rust
//! source or manifest, so its cap and its failures hold for all of them.
//!
//! WHY: gates kept their own readers with caps from 2 MiB to 16 MiB, and two
//! of them skipped a file they could not read, which reported the tree clean
//! past a source nobody judged. These tests pin the shared reader's contract:
//! a file at the cap reads whole, one byte over is refused rather than
//! truncated, and every failure names the path.
//!
//! What this does not catch: a gate that bypasses the reader with its own
//! `read_to_string`, or one that maps the error to a skip.

use std::fs;

use structure_gate::{read_source_bounded, MAX_SOURCE_BYTES};

fn cap() -> usize {
    usize::try_from(MAX_SOURCE_BYTES).expect("Fix: the source cap must fit in usize")
}

#[test]
fn a_source_at_the_cap_reads_whole() {
    let dir = tempfile::tempdir().expect("Fix: create a temp directory");
    let path = dir.path().join("at_cap.rs");
    fs::write(&path, "a".repeat(cap())).expect("Fix: write the fixture");

    let text = read_source_bounded(&path).expect("Fix: a source at the cap must read");
    assert_eq!(
        text.len(),
        cap(),
        "Fix: a source at the cap must read whole"
    );
}

#[test]
fn a_source_one_byte_over_the_cap_is_refused_not_truncated() {
    let dir = tempfile::tempdir().expect("Fix: create a temp directory");
    let path = dir.path().join("over_cap.rs");
    fs::write(&path, "a".repeat(cap() + 1)).expect("Fix: write the fixture");

    let error = read_source_bounded(&path).expect_err("Fix: a source over the cap must be refused");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    let message = error.to_string();
    assert!(
        message.contains("over_cap.rs") && message.contains(&MAX_SOURCE_BYTES.to_string()),
        "Fix: the refusal must name the file and the cap, got: {message}"
    );
}

#[test]
fn a_missing_source_is_an_error_naming_the_path() {
    let dir = tempfile::tempdir().expect("Fix: create a temp directory");
    let path = dir.path().join("missing.rs");

    let error = read_source_bounded(&path).expect_err("Fix: a missing source must be an error");
    assert_eq!(error.kind(), std::io::ErrorKind::NotFound);
    assert!(
        error.to_string().contains("missing.rs"),
        "Fix: the error must name the path, got: {error}"
    );
}

#[test]
fn a_source_that_is_not_utf8_is_an_error_naming_the_path() {
    let dir = tempfile::tempdir().expect("Fix: create a temp directory");
    let path = dir.path().join("binary.rs");
    fs::write(&path, [0x66, 0x6e, 0xff, 0xfe]).expect("Fix: write the fixture");

    let error = read_source_bounded(&path).expect_err("Fix: a non-UTF-8 source must be an error");
    assert!(
        error.to_string().contains("binary.rs"),
        "Fix: the error must name the path, got: {error}"
    );
}
