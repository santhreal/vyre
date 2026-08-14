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
    agreement_failures, derive_pairs, first_error, load_rows, render, Observation, Pair, Row,
    BASELINE, DEFAULTS,
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

fn fixture(root: &Path, directory: &str, package: &str, tables: &str) {
    let path = root.join(directory);
    fs::create_dir_all(&path).expect("Fix: fixture crate directory must be creatable");
    fs::write(
        path.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n{tables}"
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
/// it is what makes a NEW MEMBER red rather than merely unjudged. The plain
/// default build is part of it because that is what `cargo check -p <member>`
/// resolves, and neither the baseline probe nor any single-feature probe is that
/// selection: `vyre-aot` and `vyre-pass-engine` both stopped compiling under it
/// while every recorded row stayed green.
#[test]
fn the_axis_is_every_declared_feature_plus_a_baseline_and_default_build_per_member() {
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
            pair("loud-crate", DEFAULTS),
            pair("loud-crate", BASELINE),
            pair("loud-crate", "alpha"),
            pair("loud-crate", "beta"),
            pair("quiet-crate", DEFAULTS),
            pair("quiet-crate", BASELINE),
        ],
        "the axis must be every non-default feature plus a baseline and a default build per member"
    );
}

/// WHY: the one-feature-at-a-time axis judges what a consumer could write, not
/// what this workspace does write. `vyre-libs` asks `vyre-primitives` for
/// `graph`, `inventory-registry` and `text` together, and no single-feature
/// probe covers that combination: cargo unifies features across a build, so a
/// whole-workspace check hides a break inside it behind whichever unrelated
/// member enables the missing piece. The spelling is canonical so two edges
/// asking for the same set in a different order are one judged point.
#[test]
fn every_selection_a_workspace_edge_asks_for_joins_the_axis() {
    let workspace = fixture_workspace(&["host", "dep", "tool"]);
    let root = workspace.path();
    fixture(
        root,
        "dep",
        "dep-crate",
        "[features]\ndefault = [\"alpha\"]\nalpha = []\nbeta = []\ngamma = []\n",
    );
    fixture(
        root,
        "host",
        "host-crate",
        "[dependencies]\n\
         dep-crate = { path = \"../dep\", default-features = false, features = [\"gamma\", \"beta\"] }\n\
         [dev-dependencies]\n\
         tool-crate = { path = \"../tool\", features = [\"delta\"] }\n",
    );
    fixture(root, "tool", "tool-crate", "[features]\ndelta = []\n");

    let derived = derive_pairs(root).expect("Fix: the fixture workspace must derive an axis");

    assert!(
        derived.contains(&pair("dep-crate", "beta,gamma")),
        "the edge's own combination must be judged, spelled in sorted order: {derived:?}"
    );
    assert!(
        !derived.contains(&pair("dep-crate", "gamma,beta")),
        "one selection must have one spelling, or the same combination is judged twice: {derived:?}"
    );
    assert!(
        derived.contains(&pair("tool-crate", "(default),delta")),
        "a dev-dependency edge keeping defaults must be judged, and keeping defaults is part of the selection: {derived:?}"
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
            pair("bare-crate", DEFAULTS),
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
            pair("prefixed-crate", DEFAULTS),
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

/// WHY: one column spells four different cargo selections, so the column is
/// only readable if the mapping is exact. Getting it wrong does not fail loudly:
/// a `(default)` row that silently swept `--no-default-features` would record an
/// outcome for a selection nobody asked about, and every row would still look
/// judged.
#[test]
fn the_selection_column_spells_the_cargo_flags_it_stands_for() {
    let flags = |feature: &str| pair("crate-a", feature).cargo_flags();

    assert_eq!(flags(BASELINE), vec!["--no-default-features"]);
    assert!(
        flags(DEFAULTS).is_empty(),
        "the default build is the selection with no feature flags at all"
    );
    assert_eq!(
        flags("gpu"),
        vec!["--no-default-features", "--features", "gpu"]
    );
    assert_eq!(
        flags("alpha,beta"),
        vec!["--no-default-features", "--features", "alpha,beta"]
    );
    assert_eq!(
        flags("(default),alpha,beta"),
        vec!["--features", "alpha,beta"],
        "an edge that keeps defaults must not be swept with them off"
    );
}

/// WHY: the fail-by-default message is what a reader acts on, and the four kinds
/// need different actions. A missing default build means the crate does not
/// build the way a consumer builds it; a missing edge selection means this
/// workspace asks for a combination nothing judges. Reporting both as "a new
/// feature" sends the reader to the feature table, which is neither.
#[test]
fn an_unjudged_selection_names_which_kind_it_is() {
    let pairs = vec![
        pair("crate-a", BASELINE),
        pair("crate-a", DEFAULTS),
        pair("crate-a", "alpha,beta"),
    ];

    let failures = agreement_failures(&pairs, &[]);

    assert_eq!(failures.len(), 3, "{failures:?}");
    assert!(
        failures.iter().any(|failure| failure.contains("a new member is unjudged")),
        "{failures:?}"
    );
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("a new default build is unjudged")),
        "{failures:?}"
    );
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("a new edge selection is unjudged")),
        "{failures:?}"
    );
}

/// WHY: recording one new selection used to mean re-observing every other, so
/// the data went stale rather than pay a multi-hour sweep, which is how a row
/// recorded `compiles` outlived the code that made it true. A write that merges
/// must keep a reviewed decision it did not re-observe, must drop a row for a
/// selection the manifests no longer declare, and must never invent a `compiles`
/// for a selection nobody observed.
#[test]
fn a_write_merges_observations_over_recorded_rows_and_drops_stale_ones() {
    let axis = vec![
        pair("crate-a", DEFAULTS),
        pair("crate-a", "gpu"),
        pair("crate-a", "fresh"),
    ];
    let observed = vec![(
        pair("crate-a", DEFAULTS),
        Observation {
            compiles: true,
            first_error: None,
        },
    )];
    let previous = vec![
        row("crate-a", DEFAULTS, "blocked", Some("stale explanation")),
        row(
            "crate-a",
            "gpu",
            "blocked",
            Some("the CUDA driver API is not linkable on this runner"),
        ),
        row("crate-a", "renamed-away", "compiles", None),
    ];

    let rendered = render(&axis, &observed, &previous);

    assert!(
        rendered.contains("feature = \"gpu\"\noutcome = \"blocked\"\nreason = \"the CUDA driver API is not linkable on this runner\""),
        "an unobserved row keeps its reviewed decision verbatim: {rendered}"
    );
    assert!(
        !rendered.contains("stale explanation"),
        "an observation that now compiles must replace the recorded reason: {rendered}"
    );
    assert!(
        !rendered.contains("renamed-away"),
        "a row for a selection off the axis must not survive a write: {rendered}"
    );
    assert!(
        rendered.contains("feature = \"fresh\"\noutcome = \"blocked\"\nreason = \"UNREVIEWED: never observed\""),
        "a selection with neither an observation nor a row must be written as unreviewed, not as passing: {rendered}"
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
        "run `cargo run -p xtask --bin xtask -- feature-isolation --sweep --write --only-unrecorded` and record a decision for each new selection"
    );
}
