//! The architecture manifest joined to cargo, and the two documents rendered
//! from the join.
//!
//! Every fixture here is a whole checkout: a workspace manifest, member
//! manifests cargo can read, and a `docs/CRATE_OWNERSHIP.toml` describing them.
//! The mutations each change one of those three and assert the exact finding,
//! because the join is the contract and a fixture that only exercises the
//! manifest reader would pass while the workspace disagreed with it.

use std::fs;
use std::path::Path;

use xtask::gate::Report;
use xtask::gates::crate_registry::CrateOwnership;

use super::workspace_sources::{run_gate, track_fixture, workspace_root};

/// Run the gate over a fixture checkout.
fn run(root: &Path) -> Report {
    run_gate("crate-ownership", &CrateOwnership, root, false)
}

/// Every message the gate reported, joined for a failure diagnostic.
fn messages(report: &Report) -> String {
    report.finding_messages()
}

/// One member manifest, with the publication class every member declares.
fn write_member(root: &Path, path: &str, package: &str, tables: &str) {
    let directory = root.join(path);
    fs::create_dir_all(&directory).expect("Fix: fixture crate directory must be creatable");
    fs::write(
        directory.join("Cargo.toml"),
        format!(
            "[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\
             [package.metadata.vyre]\npublication_class = \"internal-engine\"\n{tables}"
        ),
    )
    .expect("Fix: fixture crate manifest must be writable");
}

/// The workspace manifest, with an optional `[workspace.dependencies]` body.
fn write_workspace(root: &Path, members: &[&str], workspace_dependencies: &str) {
    fs::create_dir_all(root.join("docs")).expect("Fix: fixture docs directory must be creatable");
    let members = members
        .iter()
        .map(|member| format!("\"{member}\""))
        .collect::<Vec<_>>()
        .join(", ");
    fs::write(
        root.join("Cargo.toml"),
        format!(
            "[workspace]\nresolver = \"2\"\nmembers = [{members}]\n{workspace_dependencies}"
        ),
    )
    .expect("Fix: fixture workspace manifest must be writable");
}

/// One `[[layer]]` row.
fn layer_row(name: &str, rank: i64, extra: &str) -> String {
    format!("[[layer]]\nname = \"{name}\"\nrank = {rank}\npurpose = \"the {name} layer\"\n{extra}\n")
}

/// One `[[crate]]` row. The seam is the package name, which is unique per row.
fn crate_row(package: &str, path: &str, layer: &str, extra: &str) -> String {
    format!(
        "[[crate]]\npackage = \"{package}\"\npath = \"{path}\"\nlayer = \"{layer}\"\n\
         seam = \"{package}-seam\"\ninterface = \"the {package} interface\"\n\
         responsibility = \"Prove the fixture contract.\"\n{extra}\n"
    )
}

/// A manifest from layer rows and crate rows.
fn manifest(layers: &[String], crates: &[String]) -> String {
    format!(
        "schema_version = 4\n\n{}\n{}",
        layers.concat(),
        crates.concat()
    )
}

/// Write the manifest and turn the fixture into a checkout the gate can read.
fn seal(root: &Path, registry: String) {
    fs::write(root.join("docs/CRATE_OWNERSHIP.toml"), registry)
        .expect("Fix: fixture manifest must be writable");
    track_fixture(root);
}

/// A two-member checkout with one legal downward edge: `high` reaches `low`.
///
/// Every mutation below starts from this and changes exactly one thing, so the
/// finding it asserts is the only difference between red and green.
fn legal_pair(root: &Path, workspace_dependencies: &str, high_tables: &str) {
    write_workspace(root, &["high", "low"], workspace_dependencies);
    write_member(root, "high", "high", high_tables);
    write_member(root, "low", "low", "");
    seal(
        root,
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );
}

/// The plain form of the legal pair: one required downward normal dependency.
fn legal_fixture(root: &Path) {
    legal_pair(
        root,
        "",
        "[dependencies]\nlow = { version = \"0.1.0\", path = \"../low\" }\n",
    );
}

/// The checked-in manifest and both generated documents must agree with the
/// workspace cargo resolves.
///
/// Derived from the checkout at run time: the member roster, every dependency
/// table, and every feature table come from cargo, so adding a workspace member
/// or a backwards edge turns this red with no edit here.
#[test]
fn workspace_manifest_and_generated_documents_are_current() {
    let report = run(&workspace_root());
    assert!(
        report.findings.is_empty(),
        "Fix: regenerate or repair the architecture manifest evidence:\n{}",
        messages(&report)
    );
}

/// The fixture the mutations are measured against must be green, or every one
/// of them proves nothing about the thing it changed.
#[test]
fn the_unmutated_fixture_reports_nothing() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    legal_fixture(temp.path());

    let report = run(temp.path());
    assert!(
        report.findings.is_empty(),
        "Fix: the baseline fixture must be clean or no mutation below is measured:\n{}",
        messages(&report)
    );
}

/// Adding a workspace member must turn the gate red until a row describes it.
///
/// A member with no row has no layer and no seam, so every direction rule skips
/// it silently. Fourteen crates once disappeared from the ownership document
/// this way while cargo still built and linked them.
#[test]
fn an_added_member_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low", "newcomer"], "");
    write_member(temp.path(), "high", "high", "");
    write_member(temp.path(), "low", "low", "");
    write_member(temp.path(), "newcomer", "newcomer", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("workspace member `newcomer` has no manifest row")
            && messages.contains("workspace package `newcomer` has no manifest row"),
        "Fix: an added member must be named by path and by package; got\n{messages}"
    );
}

/// A declaration the workspace stopped needing must fail as loudly as a missing
/// one.
///
/// This is the stale-declaration class that survives the per-edge roster's
/// deletion, in all three of its remaining forms: a row describing no member, a
/// `consumed_by` entry no edge crosses, and a `facade_exported` flag no
/// exporting layer reaches. The two records row 79 names were per-edge rows of
/// the first kind, and per-edge staleness is now unrepresentable.
#[test]
fn a_stale_declaration_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    write_member(
        temp.path(),
        "high",
        "high",
        "[dependencies]\nlow = { version = \"0.1.0\", path = \"../low\" }\n",
    );
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[
                layer_row("upper", 4, ""),
                layer_row("lower", 1, "consumed_by = [\"upper\", \"retired\"]"),
                layer_row("retired", 2, ""),
            ],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", "facade_exported = true"),
                crate_row("ghost", "ghost", "retired", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("manifest row `ghost` is not a workspace member"),
        "Fix: a row describing no member must be named; got\n{messages}"
    );
    assert!(
        messages.contains(
            "layer `lower` admits consumer layer `retired` and no production edge crosses it"
        ),
        "Fix: an admission no edge crosses must be named; got\n{messages}"
    );
    assert!(
        messages
            .contains("`low` declares `facade_exported` and no exporting layer depends on it"),
        "Fix: an export flag no facade reaches must be named; got\n{messages}"
    );
}

/// An edge only a workspace-inherited feature resolves is still an edge, and is
/// held to the layer DAG under the feature that turns it on.
///
/// `[workspace.dependencies]` unifies features across every member that writes
/// `dep = { workspace = true }`, so a feature activated there is invisible in
/// the member manifest that carries the edge. Four edges in this workspace are
/// resolvable only this way.
#[test]
fn a_feature_unified_hidden_edge_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(
        temp.path(),
        &["high", "low"],
        "[workspace.dependencies]\nhigh = { version = \"0.1.0\", path = \"high\", features = [\"reaches-up\"] }\n",
    );
    write_member(
        temp.path(),
        "high",
        "high",
        "[dependencies]\nlow = { version = \"0.1.0\", path = \"../low\" }\n",
    );
    write_member(
        temp.path(),
        "low",
        "low",
        "[dependencies]\nhigh = { workspace = true, optional = true }\n\
         [features]\nreaches-up = [\"dep:high\"]\n",
    );
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains(
            "`low` in layer `lower` (rank 1) depends under feature `reaches-up` on `high` in layer `upper` (rank 4)"
        ),
        "Fix: a feature-activated upward edge must name the feature that turns it on; got\n{messages}"
    );
}

/// Moving an edge into a production table must fail when the destination layer
/// admits no consumer.
///
/// `consumed_by = []` is how a layer states that nothing may reach it in
/// production. The shared test-fixture crate is crossed by forty-four members
/// and by `[dev-dependencies]` only, which cargo resolves in a separate graph;
/// promoting one of those to a normal dependency links the fixtures into a
/// shipped artifact.
#[test]
fn a_production_edge_where_only_a_development_edge_is_admitted_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "fixtures"], "");
    write_member(
        temp.path(),
        "high",
        "high",
        "[dependencies]\nfixtures = { version = \"0.1.0\", path = \"../fixtures\" }\n",
    );
    write_member(temp.path(), "fixtures", "fixtures", "");
    seal(
        temp.path(),
        manifest(
            &[
                layer_row("upper", 4, ""),
                layer_row("test-fixtures", 1, "consumed_by = []"),
            ],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("fixtures", "fixtures", "test-fixtures", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let production = messages(&report);
    assert!(
        production.contains(
            "`high` in layer `upper` depends always on `fixtures` over the `fixtures-seam` seam, and layer `test-fixtures` admits no consumer in `upper`"
        ),
        "Fix: a production edge into a layer that admits none must name the seam it crosses; got\n{production}"
    );

    let development = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(development.path(), &["high", "fixtures"], "");
    write_member(
        development.path(),
        "high",
        "high",
        "[dev-dependencies]\nfixtures = { version = \"0.1.0\", path = \"../fixtures\" }\n",
    );
    write_member(development.path(), "fixtures", "fixtures", "");
    seal(
        development.path(),
        manifest(
            &[
                layer_row("upper", 4, ""),
                layer_row("test-fixtures", 1, "consumed_by = []"),
            ],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("fixtures", "fixtures", "test-fixtures", ""),
            ],
        ),
    );

    let report = run(development.path());
    assert!(
        report.findings.is_empty(),
        "Fix: a development edge is how a crate tests against its fixtures and must stay legal:\n{}",
        messages(&report)
    );
}

/// A production edge to a layer the manifest ranks at or above the consumer's
/// own must fail closed.
#[test]
fn a_reverse_layer_edge_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    write_member(
        temp.path(),
        "low",
        "low",
        "[dependencies]\nhigh = { version = \"0.1.0\", path = \"../high\" }\n",
    );
    write_member(temp.path(), "high", "high", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains(
            "`low` in layer `lower` (rank 1) depends always on `high` in layer `upper` (rank 4)"
        ),
        "Fix: a reverse edge must name both layers and both ranks; got\n{messages}"
    );
}

/// A cycle in the resolved production graph must fail closed naming the exact
/// cycle path, including one inside a single layer.
///
/// An intra-layer cycle is the case the rank rule cannot see: a layer is a set
/// of crates that may reach each other, so every edge in the cycle is legal on
/// its own.
#[test]
fn a_dependency_cycle_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["a", "b"], "");
    write_member(
        temp.path(),
        "a",
        "a",
        "[dependencies]\nb = { version = \"0.1.0\", path = \"../b\" }\n",
    );
    write_member(
        temp.path(),
        "b",
        "b",
        "[dependencies]\na = { version = \"0.1.0\", path = \"../a\" }\n",
    );
    seal(
        temp.path(),
        manifest(
            &[layer_row("one", 1, "")],
            &[
                crate_row("a", "a", "one", ""),
                crate_row("b", "b", "one", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("dependency cycle detected: a -> b -> a"),
        "Fix: a cycle must name its path; got\n{messages}"
    );
}

/// A member of a layer that exports declared seams may only reach a seam whose
/// row declares itself exported.
///
/// Rank permits the facade to reach every crate in the workspace, because it
/// outranks all of them. The export flag on the destination is the whole of what
/// keeps the curated surface from re-exporting an emitter or a pass engine, and
/// it names no layer, so it survives a layer rename.
#[test]
fn a_facade_import_of_an_unexported_seam_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["facade", "internal"], "");
    write_member(
        temp.path(),
        "facade",
        "facade",
        "[dependencies]\ninternal = { version = \"0.1.0\", path = \"../internal\" }\n",
    );
    write_member(temp.path(), "internal", "internal", "");
    seal(
        temp.path(),
        manifest(
            &[
                layer_row("curated", 9, "exports_declared_seams = true"),
                layer_row("engine", 1, ""),
            ],
            &[
                crate_row("facade", "facade", "curated", ""),
                crate_row("internal", "internal", "engine", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains(
            "`facade` exports declared seams and depends always on `internal`, whose row does not set `facade_exported`"
        ),
        "Fix: a facade import of an unexported seam must be named; got\n{messages}"
    );
}

/// Two rows claiming one seam name leave every edge into that seam crossing an
/// interface with two owners.
///
/// This is the shape the twenty-two package split left behind: twenty-three
/// packages all owned one concern, so nothing named which of them a consumer
/// was depending on.
#[test]
fn two_rows_owning_one_seam_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    write_member(temp.path(), "high", "high", "");
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", "").replace("low-seam", "high-seam"),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`high` and `low` both own the `high-seam` seam"),
        "Fix: a shared seam name must name both packages; got\n{messages}"
    );
}

/// A member that declares no publication class fails against its own manifest.
///
/// The class has one home, beside the `publish` key it qualifies. A row that
/// states one instead is a retired key, and it is rejected by name so the stale
/// copy cannot stay readable beside the live one.
#[test]
fn a_missing_publication_class_and_a_retired_row_key_both_fail_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    fs::create_dir_all(temp.path().join("high"))
        .expect("Fix: fixture crate directory must be creatable");
    fs::write(
        temp.path().join("high/Cargo.toml"),
        "[package]\nname = \"high\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .expect("Fix: fixture crate manifest must be writable");
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row(
                    "low",
                    "low",
                    "lower",
                    "publication_class = \"internal-engine\"",
                ),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`high` declares no `package.metadata.vyre.publication_class`"),
        "Fix: a member with no class must be named against its own manifest; got\n{messages}"
    );
    assert!(
        messages.contains("uses the retired `publication_class` key"),
        "Fix: a second home for the class must be rejected by name; got\n{messages}"
    );
}

/// A `[[crate.dependency]]` row must be rejected by name.
///
/// The per-edge roster was the second statement of the actual dependency graph,
/// and it drifted: two of its edges outlived the manifests that declared them.
/// Cargo owns those edges now, so a row that states one again is describing a
/// graph nothing reads.
#[test]
fn a_retired_per_edge_roster_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    write_member(
        temp.path(),
        "high",
        "high",
        "[dependencies]\nlow = { version = \"0.1.0\", path = \"../low\" }\n",
    );
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row(
                    "high",
                    "high",
                    "upper",
                    "[[crate.dependency]]\npackage = \"low\"\npurpose = \"reach low\"\n",
                ),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("uses the retired `dependency` key"),
        "Fix: a per-edge roster row must be rejected by name; got\n{messages}"
    );
}

/// An optional edge no feature of its consumer can turn on is an edge no build
/// resolves.
///
/// It reads as a live dependency to anyone counting rebuild fan-out, and it is
/// the shape a half-finished feature rename leaves.
#[test]
fn an_unactivatable_optional_edge_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    write_member(
        temp.path(),
        "high",
        "high",
        "[dependencies]\nlow = { version = \"0.1.0\", path = \"../low\", optional = true }\n",
    );
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`high` declares `low` optional and no feature activates it"),
        "Fix: an unactivatable optional edge must be named; got\n{messages}"
    );
}

/// A layer no `[[layer]]` row declares, and a layer no member occupies, are both
/// findings.
///
/// The layer roster and the member roster are one file, so neither half may
/// carry an entry the other does not. An unranked layer would otherwise be
/// skipped by every direction rule.
#[test]
fn an_unranked_layer_and_an_empty_layer_both_fail_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["high", "low"], "");
    write_member(temp.path(), "high", "high", "");
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("vacant", 2, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "undeclared", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("`low` sits in layer `undeclared` and no [[layer]] row declares it"),
        "Fix: an unranked layer must be named; got\n{messages}"
    );
    assert!(
        messages.contains("layer `vacant` holds no workspace member"),
        "Fix: a layer no member occupies must be named; got\n{messages}"
    );
}

/// A row whose declared path is not where the package lives must fail closed.
///
/// The path is the join key between a row and a member manifest, and it is what
/// the publication-class read follows, so a wrong path silently reads another
/// crate's class.
#[test]
fn a_row_path_that_does_not_match_the_member_fails_closed() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    write_workspace(temp.path(), &["crates/high", "low"], "");
    write_member(temp.path(), "crates/high", "high", "");
    write_member(temp.path(), "low", "low", "");
    seal(
        temp.path(),
        manifest(
            &[layer_row("upper", 4, ""), layer_row("lower", 1, "")],
            &[
                crate_row("high", "high", "upper", ""),
                crate_row("low", "low", "lower", ""),
            ],
        ),
    );

    let report = run(temp.path());
    let messages = messages(&report);
    assert!(
        messages.contains("package `high` is declared at `high` and lives at `crates/high`"),
        "Fix: a path mismatch must name both sides; got\n{messages}"
    );
}
