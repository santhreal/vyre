//! Contracts for `xtask feature-isolation`.
//!
//! WHY: a crate can compile under its default features and under
//! `--all-features` and still be uncompilable under one feature on its own,
//! because `--all-features` is a union that supplies whatever a feature forgot
//! to require. `vyre-libs --features matching-regex` was one instance of that
//! class. Fixing the instance closes nothing, so the gate judges every
//! (member, feature) pair, and these contracts hold the gate to the two
//! properties that make it a class closure rather than a second instance fix:
//! the axis is derived from the manifests at run time, and a pair with no
//! recorded decision is a failure.
//!
//! Not covered here: whether a pair actually compiles. That is the sweep, it
//! costs a full workspace build per pair, and CI owns it.

use std::fs;
use std::path::Path;

use xtask::gates::feature_isolation::{
    agreement_failures, derive_pairs, first_error, load_rows, Pair, Row, BASELINE,
};

use super::common::workspace_root;

fn row(member: &str, feature: &str, outcome: &str, reason: Option<&str>) -> Row {
    Row {
        member: member.to_string(),
        feature: feature.to_string(),
        outcome: outcome.to_string(),
        reason: reason.map(str::to_string),
    }
}

fn pair(member: &str, feature: &str) -> Pair {
    Pair {
        member: member.to_string(),
        feature: feature.to_string(),
    }
}

fn fixture(root: &Path, directory: &str, package: &str, features: &str) {
    let path = root.join(directory);
    fs::create_dir_all(&path).expect("Fix: fixture crate directory must be creatable");
    fs::write(
        path.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n{features}"
        ),
    )
    .expect("Fix: fixture crate manifest must be writable");
}

fn fixture_workspace(members: &[&str]) -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    let quoted = members
        .iter()
        .map(|member| format!("\"{member}\""))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        directory.path().join("Cargo.toml"),
        format!("[workspace]\nresolver = \"2\"\nmembers = [{quoted}]\n"),
    )
    .expect("Fix: fixture workspace manifest must be writable");
    directory
}

/// WHY: the axis is the whole gate. Read from the manifests it is complete by
/// construction; written down it is complete until someone adds a feature. The
/// per-member `--no-default-features` probe is part of the axis because a
/// member with no features at all still has to build without them, and because
/// it is what makes a NEW MEMBER red rather than merely unjudged.
#[test]
fn the_axis_is_every_declared_feature_plus_a_baseline_per_member() {
    let workspace = fixture_workspace(&["quiet", "loud"]);
    let root = workspace.path();
    fixture(root, "quiet", "quiet-crate", "");
    fixture(
        root,
        "loud",
        "loud-crate",
        "[features]\ndefault = [\"alpha\"]\nalpha = []\nbeta = [\"alpha\"]\n",
    );

    let derived = derive_pairs(root).expect("Fix: the fixture workspace must derive an axis");

    assert_eq!(
        derived,
        vec![
            pair("loud-crate", BASELINE),
            pair("loud-crate", "alpha"),
            pair("loud-crate", "beta"),
            pair("quiet-crate", BASELINE),
        ],
        "the axis must be every non-default feature plus one baseline per member"
    );
}

/// WHY: `[features]` is not the whole feature surface. An optional dependency
/// that no feature names with the `dep:` prefix publishes a feature of its own,
/// which a consumer can enable and which reading only the declared table would
/// leave unjudged. `vyre-runtime`'s `remote-cache = ["ureq"]` was exactly that
/// shape. The suppressed twin is the control: adding `dep:` must remove the
/// pair rather than leave a row that can never be selected.
#[test]
fn an_optional_dependency_with_no_dep_prefix_joins_the_axis() {
    let bare = fixture_workspace(&["bare"]);
    fixture(
        bare.path(),
        "bare",
        "bare-crate",
        "[features]\nremote = [\"ureq\"]\n\n[dependencies]\nureq = { version = \"3\", optional = true }\nserde = \"1\"\n",
    );
    assert_eq!(
        derive_pairs(bare.path()).expect("Fix: the fixture workspace must derive an axis"),
        vec![
            pair("bare-crate", BASELINE),
            pair("bare-crate", "remote"),
            pair("bare-crate", "ureq"),
        ],
        "a bare optional dependency publishes a feature and must be judged"
    );

    let prefixed = fixture_workspace(&["prefixed"]);
    fixture(
        prefixed.path(),
        "prefixed",
        "prefixed-crate",
        "[features]\nremote = [\"dep:ureq\"]\n\n[dependencies]\nureq = { version = \"3\", optional = true }\n",
    );
    assert_eq!(
        derive_pairs(prefixed.path()).expect("Fix: the fixture workspace must derive an axis"),
        vec![
            pair("prefixed-crate", BASELINE),
            pair("prefixed-crate", "remote"),
        ],
        "`dep:` suppresses the implicit feature, so it must not be on the axis"
    );
}

/// WHY: a member that declares a feature after the file was written must turn
/// the gate red. This is the fail-by-default property; without it the gate
/// judges the set someone had in mind at the time, which is what a hardcoded
/// list already did.
#[test]
fn a_pair_with_no_recorded_decision_names_itself_and_fails() {
    let pairs = vec![pair("crate-a", BASELINE), pair("crate-a", "fresh")];
    let rows = vec![row("crate-a", BASELINE, "compiles", None)];

    let failures = agreement_failures(&pairs, &rows);

    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].contains("crate-a --no-default-features --features fresh")
            && failures[0].contains("no row in xtask/feature-isolation.toml"),
        "the failure must name the unjudged pair: {}",
        failures[0]
    );
}

/// WHY: a row for a feature that was renamed or deleted keeps reporting a
/// decision about nothing, and it reads as coverage of a pair that no longer
/// exists.
#[test]
fn a_row_for_a_pair_no_manifest_declares_fails_as_stale() {
    let pairs = vec![pair("crate-a", BASELINE)];
    let rows = vec![
        row("crate-a", BASELINE, "compiles", None),
        row("crate-a", "renamed-away", "compiles", None),
    ];

    let failures = agreement_failures(&pairs, &rows);

    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].contains("renamed-away") && failures[0].contains("stale row"),
        "the failure must name the stale row: {}",
        failures[0]
    );
}

/// WHY: rows are matched by (member, feature), so a second row for the same
/// pair is never consulted. Two rows recording opposite outcomes would let the
/// sweep agree with whichever one the lookup reached first.
#[test]
fn a_duplicated_pair_fails_rather_than_shadowing_its_second_row() {
    let pairs = vec![pair("crate-a", BASELINE)];
    let rows = vec![
        row("crate-a", BASELINE, "compiles", None),
        row("crate-a", BASELINE, "blocked", Some("something")),
    ];

    let failures = agreement_failures(&pairs, &rows);

    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("recorded more than once")),
        "{failures:?}"
    );
}

/// WHY: `blocked` is the escape hatch, and an escape hatch justified by a
/// schedule never closes. Each of these reasons is a promise, not a constraint,
/// and `UNREVIEWED` is what `--sweep --write` itself emits, so regenerating the
/// file cannot launder an unfixed break into an accepted one.
#[test]
fn a_blocked_row_needs_a_constraint_and_not_a_schedule() {
    let pairs = vec![pair("crate-a", "gpu")];
    for excuse in [
        None,
        Some(""),
        Some("todo"),
        Some("not fixed yet, tracked elsewhere in the backlog rows"),
        Some("UNREVIEWED: E0432 at crate-a/src/lib.rs:12"),
        Some("temporarily broken while the driver split lands upstream"),
        Some("too short"),
    ] {
        let failures = agreement_failures(&pairs, &[row("crate-a", "gpu", "blocked", excuse)]);
        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("with no real reason")),
            "`{excuse:?}` must be rejected as a reason: {failures:?}"
        );
    }

    let real = "the feature selects a vendor driver whose headers are not in this workspace";
    assert_eq!(
        agreement_failures(&pairs, &[row("crate-a", "gpu", "blocked", Some(real))]),
        Vec::<String>::new(),
        "a stated technical constraint must be accepted"
    );
}

/// WHY: an outcome outside the two words is a row nothing can compare against,
/// and a reason left behind on a pair that now compiles is a stale explanation
/// that outlives the problem it described.
#[test]
fn an_unknown_outcome_and_a_reason_on_a_passing_row_both_fail() {
    let pairs = vec![pair("crate-a", "gpu")];

    let unknown = agreement_failures(&pairs, &[row("crate-a", "gpu", "probably", None)]);
    assert!(
        unknown
            .iter()
            .any(|failure| failure.contains("records outcome `probably`")),
        "{unknown:?}"
    );

    let leftover = agreement_failures(
        &pairs,
        &[row(
            "crate-a",
            "gpu",
            "compiles",
            Some("was broken before the manifest edge landed"),
        )],
    );
    assert!(
        leftover
            .iter()
            .any(|failure| failure.contains("still carries a reason")),
        "{leftover:?}"
    );
}

/// WHY: "vyre-libs fails" sends a reader to a crate. The sweep's whole value as
/// a report is the code and the line, and the diagnostic stream carries
/// warnings, build-script noise and non-error messages ahead of the first error.
#[test]
fn the_first_error_is_reported_with_its_code_and_line() {
    let stream = concat!(
        r#"{"reason":"compiler-artifact","package_id":"x"}"#,
        "\n",
        r#"{"reason":"compiler-message","message":{"level":"warning","code":null,"message":"unused","spans":[{"file_name":"a.rs","line_start":1}]}}"#,
        "\n",
        r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"E0432"},"message":"unresolved import","spans":[{"file_name":"vyre-libs/src/matching/regex.rs","line_start":12}]}}"#,
        "\n",
        r#"{"reason":"compiler-message","message":{"level":"error","code":{"code":"E0599"},"message":"no method","spans":[]}}"#,
        "\n",
    );

    assert_eq!(
        first_error(stream).as_deref(),
        Some("E0432 at vyre-libs/src/matching/regex.rs:12")
    );
    assert_eq!(first_error("not json at all\n"), None);
    assert_eq!(
        first_error(
            r#"{"reason":"compiler-message","message":{"level":"error","code":null,"message":"m","spans":[]}}"#
        )
        .as_deref(),
        Some("error at no span"),
        "an uncoded error still has to be reported"
    );
}

/// WHY: the two properties above are worth nothing if the checked-in file does
/// not currently satisfy them. This is the gate's fast half, run against the
/// real tree in process, so a feature added on this commit fails here as well
/// as in CI.
#[test]
fn the_checked_in_declaration_agrees_with_the_tracked_manifests() {
    let root = workspace_root();
    let pairs = derive_pairs(&root).expect("Fix: the workspace manifests must derive an axis");
    let rows = load_rows(&root).expect("Fix: xtask/feature-isolation.toml must be readable");

    assert!(
        pairs.len() > 100,
        "the axis collapsed to {} pair(s); the derivation is reading the wrong manifests",
        pairs.len()
    );
    assert_eq!(
        agreement_failures(&pairs, &rows),
        Vec::<String>::new(),
        "run `cargo run -p xtask -- feature-isolation --sweep --write` and record a decision for each new pair"
    );
}
