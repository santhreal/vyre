//! Tree contracts for the typed configuration space model.
//!
//! The rule under test is that one capability name has one owning package and
//! that the declared constraints leave every schedulable cell with an
//! assignment. Both halves are injected red here: the fixtures build a
//! workspace that violates one clause at a time, so a predicate that stopped
//! looking would show up as a green case rather than as a lower count.

use std::collections::BTreeSet;
use std::path::Path;

use tempfile::TempDir;
use xtask::checkout::checkout_root;
use xtask::config_space::*;

/// A workspace of `(directory, manifest, lib source)` plus an ownership record.
///
/// Every fixture needs the record, because a model built without the roster
/// would report a facade that matches a roster it never loaded.
fn workspace(members: &[(&str, &str, &str)], record: &str) -> TempDir {
    let temp = TempDir::new().expect("create temp dir");
    let root = temp.path();
    let names: Vec<String> = members
        .iter()
        .map(|(directory, ..)| format!("\"{directory}\""))
        .collect();
    std::fs::write(
        root.join("Cargo.toml"),
        format!("[workspace]\nmembers = [{}]\n", names.join(", ")),
    )
    .expect("write workspace manifest");
    for (directory, manifest, source) in members {
        let member = root.join(directory);
        std::fs::create_dir_all(member.join("src")).expect("create member src");
        std::fs::write(member.join("Cargo.toml"), manifest).expect("write member manifest");
        std::fs::write(member.join("src/lib.rs"), source).expect("write member source");
    }
    std::fs::create_dir_all(root.join("docs")).expect("create docs");
    std::fs::write(root.join(OWNERSHIP_RECORD_PATH), record).expect("write ownership record");
    temp
}

/// A one-package facade roster over `package`, publishing `features`.
fn roster(package: &str, features: &[(&str, &str)]) -> String {
    let mut text = format!("[facade]\npackage = \"{package}\"\ndefault = []\nlocal = []\n");
    for (name, domain) in features {
        text.push_str(&format!(
            "\n[[facade.domain]]\npackage = \"{domain}\"\noptional = false\n\n[[facade.feature]]\nname = \"{name}\"\ndomain = \"{domain}\"\n"
        ));
    }
    text
}

fn model_of(root: &Path) -> ConfigurationModel {
    ConfigurationModel::inspect_workspace(root).expect("inspect workspace")
}

fn mentions(findings: &[String], needle: &str) -> bool {
    findings.iter().any(|finding| finding.contains(needle))
}

#[test]
fn the_workspace_configuration_space_has_an_assignment() {
    let root = checkout_root();
    let model = model_of(&root);

    assert_eq!(model.schema_version, CONFIG_SPACE_SCHEMA_VERSION);
    match &model.satisfiability {
        Satisfiability::Satisfiable(assignment) => {
            assert!(
                !assignment.enabled.is_empty() && !assignment.activated.is_empty(),
                "a witness names the features and packages the cell turns on"
            );
        }
        Satisfiability::Unsatisfiable(conflicts) => {
            panic!("the workspace configuration space must have an assignment: {conflicts:?}")
        }
    }
    assert!(
        model.scheduled_cells > model.crates.len(),
        "every package contributes at least a no-features cell and the model schedules more"
    );
    for audit in &model.build_script_audits {
        assert!(
            audit.is_deterministic && !audit.accesses_undeclared_host_state,
            "build script `{}` must be deterministic and bounded",
            audit.path
        );
    }
}

/// WHY: the covering set is what a scheduler runs, so a package whose feature
/// never gets a cell of its own is a feature only unification ever enables.
/// Derived from the manifests at run time, so a new feature is covered or the
/// case fails.
#[test]
fn every_declared_feature_has_an_isolated_cell() {
    let root = checkout_root();
    let model = model_of(&root);

    let mut checked = 0usize;
    for entry in &model.crates {
        let cells: BTreeSet<&str> = model
            .covering_sets
            .iter()
            .filter(|cell| cell.package == entry.name)
            .filter(|cell| cell.enabled_features.len() == 1)
            .map(|cell| cell.enabled_features[0].as_str())
            .collect();
        for feature in &entry.features {
            if feature == "default" {
                continue;
            }
            assert!(
                cells.contains(feature.as_str()),
                "`{}` feature `{feature}` has no isolated cell",
                entry.name
            );
            checked += 1;
        }
    }
    assert!(
        checked > 100,
        "the workspace declares more than 100 features; a loop over {checked} of them proves nothing"
    );
}

/// WHY: the four provenance cases are the whole answer to "which package
/// decides what this name means". The match has no catch-all arm, so a fifth
/// case cannot be added without a decision recorded here.
#[test]
fn every_body_shape_classifies_into_one_provenance() {
    let temp = workspace(
        &[
            (
                "owner",
                "[package]\nname = \"owner\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nbase = []\ncapability = [\"base\"]\n",
                "",
            ),
            (
                "facade",
                "[package]\nname = \"facade\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nlocal = []\ncapability = [\"owner/capability\"]\nbundle = [\"capability\"]\ndivided = [\"owner/divided\", \"local\"]\n\n[dependencies]\nowner = { path = \"../owner\" }\n",
                "",
            ),
        ],
        &roster("facade", &[("capability", "owner")]),
    );
    let root = temp.path();
    let model = model_of(root);

    // The classification is read back through the findings the model reports,
    // because those are what a caller acts on.
    let cases = [
        ("capability", Provenance::Forwarded),
        ("bundle", Provenance::Aggregated),
        ("local", Provenance::Declared),
        ("divided", Provenance::Divided),
    ];
    for (feature, expected) in cases {
        let divided_reported = mentions(
            &model.feature_ownership_findings,
            &format!("Feature `{feature}` of `facade`"),
        );
        match expected {
            Provenance::Divided => assert!(
                divided_reported,
                "a divided body must be a finding, got: {:?}",
                model.feature_ownership_findings
            ),
            Provenance::Forwarded | Provenance::Aggregated | Provenance::Declared => assert!(
                !divided_reported,
                "`{feature}` is not divided but was reported as one: {:?}",
                model.feature_ownership_findings
            ),
        }
    }
    assert!(
        mentions(
            &model.forward_findings,
            "forwards to `owner/divided`, which `owner` does not declare"
        ),
        "a forward into a name the dependency does not declare must be a finding, got: {:?}",
        model.forward_findings
    );
}

/// WHY: two packages declaring one name with different bodies is the shape that
/// makes one `--features` flag mean two things. A facade that forwards the name
/// is the shape that does not, and the predicate has to separate them.
#[test]
fn two_declarations_of_one_name_are_a_finding_and_a_forward_is_not() {
    let temp = workspace(
        &[
            (
                "alpha",
                "[package]\nname = \"alpha\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nsplit = []\n",
                "#[cfg(feature = \"split\")]\npub fn alpha() {}\n",
            ),
            (
                "beta",
                "[package]\nname = \"beta\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nsplit = []\nuse_product = []\n",
                "#[cfg(feature = \"split\")]\npub fn beta() {}\n",
            ),
            (
                "facade",
                "[package]\nname = \"facade\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nsplit = [\"alpha/split\"]\n\n[dependencies]\nalpha = { path = \"../alpha\" }\n",
                "",
            ),
        ],
        &roster("facade", &[("split", "alpha")]),
    );
    let model = model_of(temp.path());

    assert!(
        mentions(
            &model.feature_ownership_findings,
            "Feature `split` is declared by 2 packages"
        ),
        "one name declared by two packages must be a finding, got: {:?}",
        model.feature_ownership_findings
    );
    assert!(
        mentions(
            &model.feature_ownership_findings,
            "`alpha` declares the name here"
        ) && mentions(
            &model.feature_ownership_findings,
            "`beta` declares the name here"
        ),
        "the finding must name both declaring packages, got: {:?}",
        model.feature_ownership_findings
    );
    assert!(
        !mentions(&model.feature_ownership_findings, "`facade`"),
        "a forwarding facade is not a declaring package, got: {:?}",
        model.feature_ownership_findings
    );
    assert!(
        mentions(&model.capability_naming_findings, "use_product"),
        "a verb-prefixed feature name must be a finding, got: {:?}",
        model.capability_naming_findings
    );
}

/// WHY: an exemption that no longer covers two declarations reads as a live
/// waiver while waiving nothing, which is how an allowlist rots into a list of
/// names nobody can account for.
#[test]
fn a_shared_capability_exemption_that_covers_nothing_is_a_finding() {
    let mut record = roster("solo", &[("only", "solo")]);
    record.push_str(
        "\n[[configuration.shared_capability]]\nname = \"only\"\nreason = \"stale exemption\"\n",
    );
    let temp = workspace(
        &[(
            "solo",
            "[package]\nname = \"solo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nonly = []\n",
            "#[cfg(feature = \"only\")]\npub fn only() {}\n",
        )],
        &record,
    );
    let model = model_of(temp.path());

    assert!(
        mentions(
            &model.feature_ownership_findings,
            "recorded as a shared capability but 1 package(s) declare it"
        ),
        "an exemption covering one declaration must be a finding, got: {:?}",
        model.feature_ownership_findings
    );
}

/// WHY: the previous verdict was "no other finding was reported", which could
/// not fail on its own claim. A conflicting pair both reachable from one cell
/// has to come back unsatisfiable, and the conflict has to name the cell.
#[test]
fn a_conflicting_pair_reachable_from_one_cell_is_unsatisfiable() {
    let mut record = roster("solo", &[("fast", "solo")]);
    record.push_str(
        "\n[[configuration.exclusive]]\npackage = \"solo\"\nfeatures = [\"fast\", \"exact\"]\nreason = \"the two select different rounding\"\n",
    );
    let temp = workspace(
        &[(
            "solo",
            "[package]\nname = \"solo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\ndefault = [\"both\"]\nboth = [\"fast\", \"exact\"]\nfast = []\nexact = []\n",
            "",
        )],
        &record,
    );
    let model = model_of(temp.path());

    let conflicts = model.satisfiability.conflicts();
    assert!(
        !conflicts.is_empty(),
        "a cell enabling both halves of an exclusive pair has no assignment"
    );
    assert!(
        conflicts
            .iter()
            .any(|conflict| conflict.violated.contains("different rounding")
                && conflict.cell.contains("solo")),
        "the conflict must name the cell and the recorded reason, got: {conflicts:?}"
    );
}

/// WHY: a constraint over a feature no package declares can never bind, so it
/// reads as a live constraint while constraining nothing. Both of the entries
/// this record shipped with were of that shape.
#[test]
fn a_constraint_naming_an_undeclared_feature_is_a_finding() {
    let mut record = roster("solo", &[("only", "solo")]);
    record.push_str(
        "\n[[configuration.exclusive]]\npackage = \"solo\"\nfeatures = [\"only\", \"absent\"]\nreason = \"unreachable\"\n",
    );
    let temp = workspace(
        &[(
            "solo",
            "[package]\nname = \"solo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nonly = []\n",
            "#[cfg(feature = \"only\")]\npub fn only() {}\n",
        )],
        &record,
    );
    let model = model_of(temp.path());

    assert!(
        mentions(
            &model.facade_findings,
            "names `absent` which the package does not declare"
        ),
        "a constraint over an absent feature must be a finding, got: {:?}",
        model.facade_findings
    );
    assert!(
        model.satisfiability.is_satisfiable(),
        "dropping an unbindable constraint leaves the space satisfiable"
    );
}

/// WHY: a `cfg` reading a name the manifest never declares is a branch no
/// configuration can select, so the code inside it is unreachable by
/// construction.
#[test]
fn a_cfg_reading_an_undeclared_feature_is_a_finding() {
    let temp = workspace(
        &[(
            "solo",
            "[package]\nname = \"solo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[features]\nonly = []\n",
            "#[cfg(feature = \"ghost\")]\npub fn ghost() {}\n\n#[cfg(feature = \"only\")]\npub fn only() {}\n",
        )],
        &roster("solo", &[("only", "solo")]),
    );
    let model = model_of(temp.path());

    assert!(
        mentions(&model.unreachable_cfg_findings, "ghost"),
        "a cfg over an undeclared feature must be a finding, got: {:?}",
        model.unreachable_cfg_findings
    );
    assert!(
        !mentions(&model.unreachable_cfg_findings, "only"),
        "a declared feature is reachable, got: {:?}",
        model.unreachable_cfg_findings
    );
}

/// WHY: the roster is the one copy of the facade surface, so a roster row the
/// facade does not publish, or a facade feature the roster does not record,
/// puts the fact back in two places. Derived from both files at run time, so
/// adding a domain to one of them turns this red.
#[test]
fn the_roster_and_the_facade_manifest_publish_the_same_features() {
    let root = checkout_root();
    let model = model_of(&root);
    assert!(
        model.facade_findings.is_empty(),
        "the roster and the facade manifest must agree, got: {:?}",
        model.facade_findings
    );

    let (path, generated) =
        ConfigurationModel::facade_manifest(&root).expect("render the facade manifest");
    let committed = std::fs::read_to_string(root.join(&path)).expect("read the facade manifest");
    assert_eq!(
        committed.replace("\r\n", "\n"),
        generated,
        "`{path}` differs from what the roster renders; run `configuration-model --write`"
    );
}

/// WHY: a model read back at the wrong schema version would be interpreted
/// against fields it does not have, so the read fails instead.
#[test]
fn a_stale_configuration_space_schema_fails_closed() {
    let root = checkout_root();
    let mut model = model_of(&root);
    model.schema_version = CONFIG_SPACE_SCHEMA_VERSION + 41;
    let text = model.to_toml().expect("serialize");
    assert_eq!(
        ConfigurationModel::from_toml(&text).unwrap_err(),
        ConfigSpaceError::StaleSchemaVersion {
            expected: CONFIG_SPACE_SCHEMA_VERSION,
            found: CONFIG_SPACE_SCHEMA_VERSION + 41,
        }
    );
}
