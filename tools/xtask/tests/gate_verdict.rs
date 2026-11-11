//! A failing gate run still emits its report.
//!
//! This closes the class where a verdict short-circuits emission. A gate
//! computes its findings, then the tree is compared against a pre-run snapshot
//! for mutation. When the mutation verdict is judged before the report is
//! rendered, every finding the run computed is discarded and only a second run
//! recovers them. An edit to any workspace file while the gate reads the tree
//! produces that mutation, so under parallel work the discarded report is the
//! common case rather than a corner.
//!
//! Not covered here: whether a mutation is attributed to the gate that ran or
//! to a concurrent writer. A content digest carries no author, so the guard
//! cannot draw that distinction and this suite does not claim it does.

use xtask::gate::{finish_run, render, Finding, Report};

const MUTATION: &str = "gate `g` modified workspace file `a.rs` without --write";

fn one_finding() -> Report {
    Report {
        findings: vec![Finding::new("message", "fix")],
        notes: vec!["note".to_string()],
        ..Report::default()
    }
}

#[test]
fn a_mutated_run_still_renders_every_finding_it_computed() {
    let report = one_finding();
    let verdict = finish_run("g", &report, &[MUTATION.to_string()], false);
    assert_eq!(verdict.stdout, render("g", &report));
    assert_eq!(verdict.stderr, vec![format!("Fix: {MUTATION}")]);
    assert!(verdict.failed);
}

#[test]
fn a_mutated_run_renders_its_report_even_when_silence_was_requested() {
    let report = Report::clean();
    let verdict = finish_run("g", &report, &[MUTATION.to_string()], true);
    assert_eq!(verdict.stdout, render("g", &report));
    assert!(verdict.failed);
}

#[test]
fn a_clean_run_that_requested_silence_prints_nothing_and_passes() {
    let verdict = finish_run("g", &Report::clean(), &[], true);
    assert!(verdict.stdout.is_empty());
    assert!(verdict.stderr.is_empty());
    assert!(!verdict.failed);
}

#[test]
fn a_clean_run_without_silence_prints_its_zero_count_and_passes() {
    let report = Report::clean();
    let verdict = finish_run("g", &report, &[], false);
    assert_eq!(verdict.stdout, render("g", &report));
    assert!(verdict.stderr.is_empty());
    assert!(!verdict.failed);
}

#[test]
fn a_finding_alone_fails_and_emits_no_mutation_line() {
    let report = one_finding();
    let verdict = finish_run("g", &report, &[], false);
    assert_eq!(verdict.stdout, render("g", &report));
    assert!(verdict.stderr.is_empty());
    assert!(verdict.failed);
}

#[test]
fn every_mutation_is_emitted_not_only_the_first() {
    let mutations = vec!["one".to_string(), "two".to_string(), "three".to_string()];
    let verdict = finish_run("g", &Report::clean(), &mutations, false);
    assert_eq!(
        verdict.stderr,
        vec![
            "Fix: one".to_string(),
            "Fix: two".to_string(),
            "Fix: three".to_string(),
        ]
    );
    assert!(verdict.failed);
}
