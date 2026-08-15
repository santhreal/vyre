//! The two source-fingerprint producers must spell the same tree the same way.
//!
//! `xtask` links no vyre crate, so the recorder that stamps a gate artifact and
//! the probe that stamps a measured benchmark report cannot share an
//! implementation. This crate links both, and it is the only place that can
//! hold them to the same string. Without this test the two would drift and the
//! first thing to notice would be a release gate comparing a gate artifact's
//! fingerprint against a benchmark artifact's and finding two trees where there
//! is one.

use std::path::Path;

#[test]
fn both_producers_fingerprint_a_clean_checkout_identically() {
    let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
    xtask::fixture_checkout::seeded(dir.path());

    assert_eq!(
        probe_fingerprint(dir.path()),
        xtask::source_provenance::capture(dir.path())
            .expect("Fix: a clean checkout names its commit."),
        "Fix: the benchmark probe and the artifact recorder must name one tree one way."
    );
}

#[test]
fn both_producers_fingerprint_a_dirty_checkout_identically() {
    let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
    xtask::fixture_checkout::seeded(dir.path());
    std::fs::write(dir.path().join("tracked.txt"), "changed\n")
        .expect("Fix: dirty the tracked file.");
    std::fs::write(dir.path().join("untracked.txt"), "new\n")
        .expect("Fix: add an untracked file.");

    let probe = probe_fingerprint(dir.path());
    let recorder = xtask::source_provenance::capture(dir.path())
        .expect("Fix: a dirty checkout still names a commit.");

    assert!(
        probe.contains(":dirty=true:worktree="),
        "Fix: the probe must mark a dirty tree dirty; probe={probe}"
    );
    assert_eq!(
        probe, recorder,
        "Fix: two producers of one worktree digest must agree byte for byte."
    );
}

#[test]
fn both_producers_ignore_the_evidence_the_run_is_writing() {
    let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
    xtask::fixture_checkout::seeded(dir.path());
    std::fs::create_dir_all(dir.path().join("release/evidence/metadata"))
        .expect("Fix: create the evidence directory.");
    std::fs::write(
        dir.path().join("release/evidence/metadata/matrix.json"),
        "{}\n",
    )
    .expect("Fix: write an evidence artifact.");

    assert_eq!(
        probe_fingerprint(dir.path()),
        xtask::source_provenance::capture(dir.path())
            .expect("Fix: the checkout still names a commit."),
        "Fix: both producers must exclude release/evidence from the dirty scan."
    );
}

fn probe_fingerprint(root: &Path) -> String {
    vyre_bench::probes::source_fingerprint(&vyre_bench::probes::capture_git_info_at(root))
}

