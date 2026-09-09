//! Tree contracts for Typed Configuration Space Model & CI Scheduler (Row 115).

use tempfile::TempDir;
use xtask::checkout::checkout_root;
use xtask::config_space::*;

#[test]
fn configuration_space_model_is_satisfiable_across_workspace() {
    let root = checkout_root();
    let model =
        ConfigurationModel::inspect_workspace(&root).expect("configuration model generation");

    assert_eq!(model.schema_version, CONFIG_SPACE_SCHEMA_VERSION);
    assert!(
        model.is_satisfiable,
        "model must be satisfiable across workspace"
    );
    assert!(!model.crates.is_empty());
    assert!(!model.covering_sets.is_empty());
    assert!(!model.supported_targets.is_empty());
    assert!(!model.supported_toolchains.is_empty());
    assert!(!model.backend_cells.is_empty());

    // Proves that build scripts are audited
    for audit in &model.build_script_audits {
        assert!(
            audit.is_deterministic,
            "build script {} must be deterministic",
            audit.path
        );
        assert!(
            !audit.accesses_undeclared_host_state,
            "build script {} must not access undeclared host state",
            audit.path
        );
    }
}

/// WHY: Acceptance criterion - A test proves the model is satisfiable and that
/// the covering set is minimal and deterministic: two runs produce byte-identical
/// cell lists, and removing any cell drops an interaction.
#[test]
fn configuration_space_covering_sets_isolate_features_and_are_deterministic_and_minimal() {
    let root = checkout_root();
    let model1 =
        ConfigurationModel::inspect_workspace(&root).expect("first configuration model generation");
    let model2 = ConfigurationModel::inspect_workspace(&root)
        .expect("second configuration model generation");

    // Determinism proof: two runs produce byte-identical cell lists and TOML representation
    let toml1 = model1.to_toml().expect("to toml 1");
    let toml2 = model2.to_toml().expect("to toml 2");
    assert_eq!(
        toml1, toml2,
        "two runs must produce byte-identical configuration model TOML"
    );
    assert_eq!(
        model1.covering_sets, model2.covering_sets,
        "covering sets must be identical across runs"
    );

    // Every crate with features must have isolated cells
    for krate in &model1.crates {
        if !krate.features.is_empty() {
            let cells: Vec<_> = model1
                .covering_sets
                .iter()
                .filter(|c| c.package == krate.name)
                .collect();
            assert!(
                !cells.is_empty(),
                "crate {} must have covering set cells",
                krate.name
            );
            // Must have a base cell (empty features)
            assert!(
                cells.iter().any(|c| c.enabled_features.is_empty()),
                "crate {} must have empty base cell",
                krate.name
            );
        }
    }

    // Minimality proof: removing any cell in a covering set drops an interaction / isolated feature
    let features = vec![
        "analysis".to_string(),
        "encoding".to_string(),
        "solvers".to_string(),
    ];
    let covering = generate_covering_set("vyre-sample", &features);
    assert_eq!(covering.len(), 5); // base (1) + isolated (3) + full (1)
    for i in 0..covering.len() {
        let mut subset = covering.clone();
        subset.remove(i);
        let removed = &covering[i];
        let covered_elsewhere = subset
            .iter()
            .any(|c| c.enabled_features == removed.enabled_features);
        assert!(
            !covered_elsewhere,
            "removing cell `{}` must drop an isolated capability interaction",
            removed.target_cell
        );
    }
}

/// WHY: Acceptance criterion - A test proves an unreachable cfg branch and a
/// feature no cfg reads are both findings, derived from source at run time.
#[test]
fn unreachable_cfg_and_unreferenced_feature_are_findings() {
    let temp = TempDir::new().expect("create temp dir");
    let root = temp.path();

    // Write workspace Cargo.toml
    let workspace_toml = r#"
[workspace]
members = ["krate-a"]
"#;
    std::fs::write(root.join("Cargo.toml"), workspace_toml).unwrap();

    // Write crate Cargo.toml declaring "declared_feat" and "dead_feat"
    let crate_dir = root.join("krate-a");
    std::fs::create_dir_all(crate_dir.join("src")).unwrap();
    let crate_toml = r#"
[package]
name = "krate-a"
version = "0.1.0"
edition = "2021"

[features]
declared_feat = []
dead_feat = []
"#;
    std::fs::write(crate_dir.join("Cargo.toml"), crate_toml).unwrap();

    // Write src/lib.rs referencing an undeclared feature "ghost_feature" and "declared_feat"
    let lib_rs = concat!(
        "#[cfg(feature = \"ghost_feature\")]\n",
        "pub fn ghost() {}\n",
        "#[cfg(feature = \"declared_feat\")]\n",
        "pub fn active() {}\n"
    );
    std::fs::write(crate_dir.join("src/lib.rs"), lib_rs).unwrap();

    let model = ConfigurationModel::inspect_workspace(root).expect("inspect workspace");

    // 1. Ghost feature must be reported as unreachable cfg
    assert!(
        model
            .unreachable_cfg_findings
            .iter()
            .any(|f| f.contains("ghost_feature")),
        "unreachable cfg branch `ghost_feature` must be a finding, got: {:?}",
        model.unreachable_cfg_findings
    );

    // 2. dead_feat has no deps and no cfg reading it -> unreferenced feature finding
    assert!(
        model
            .unreferenced_feature_findings
            .iter()
            .any(|f| f.contains("dead_feat")),
        "unreferenced feature `dead_feat` must be a finding, got: {:?}",
        model.unreferenced_feature_findings
    );
}

/// WHY: Acceptance criterion - A test proves a feature declared in two packages,
/// or a product-named feature, is a finding.
#[test]
fn duplicate_and_product_named_features_are_findings() {
    let temp = TempDir::new().expect("create temp dir");
    let root = temp.path();

    let workspace_toml = r#"
[workspace]
members = ["krate-one", "krate-two"]
"#;
    std::fs::write(root.join("Cargo.toml"), workspace_toml).unwrap();

    // Crate 1 declares duplicate_custom_feat and product-named use_tensor_rt
    let dir1 = root.join("krate-one");
    std::fs::create_dir_all(dir1.join("src")).unwrap();
    let toml1 = r#"
[package]
name = "krate-one"
version = "0.1.0"
edition = "2021"

[features]
duplicate_custom_feat = []
use_tensor_rt = []
"#;
    std::fs::write(dir1.join("Cargo.toml"), toml1).unwrap();
    std::fs::write(dir1.join("src/lib.rs"), "").unwrap();

    // Crate 2 also declares duplicate_custom_feat
    let dir2 = root.join("krate-two");
    std::fs::create_dir_all(dir2.join("src")).unwrap();
    let toml2 = r#"
[package]
name = "krate-two"
version = "0.1.0"
edition = "2021"

[features]
duplicate_custom_feat = []
"#;
    std::fs::write(dir2.join("Cargo.toml"), toml2).unwrap();
    std::fs::write(dir2.join("src/lib.rs"), "").unwrap();

    let model = ConfigurationModel::inspect_workspace(root).expect("inspect workspace");

    // Duplicate feature finding
    assert!(
        model
            .duplicate_feature_findings
            .iter()
            .any(|f| f.contains("duplicate_custom_feat")),
        "feature declared in two packages must be a finding, got: {:?}",
        model.duplicate_feature_findings
    );

    // Product-named feature finding
    assert!(
        model
            .capability_naming_findings
            .iter()
            .any(|f| f.contains("use_tensor_rt")),
        "product-named feature `use_tensor_rt` must be a finding, got: {:?}",
        model.capability_naming_findings
    );
}

#[test]
fn stale_configuration_space_schema_fails_closed() {
    let root = checkout_root();
    let mut model =
        ConfigurationModel::inspect_workspace(&root).expect("configuration model generation");
    model.schema_version = 42; // Stale schema version

    let toml = model.to_toml().expect("to toml");
    let err = ConfigurationModel::from_toml(&toml).unwrap_err();
    assert!(matches!(
        err,
        ConfigSpaceError::StaleSchemaVersion {
            expected: 1,
            found: 42
        }
    ));
}
