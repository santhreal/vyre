//! Tree contracts for Typed Configuration Space Model & CI Scheduler (Row 115).

use xtask::checkout::checkout_root;
use xtask::config_space::*;

#[test]
fn configuration_space_model_is_satisfiable_across_workspace() {
    let root = checkout_root();
    let model = ConfigurationModel::inspect_workspace(&root).expect("configuration model generation");

    assert_eq!(model.schema_version, CONFIG_SPACE_SCHEMA_VERSION);
    assert!(model.is_satisfiable);
    assert!(!model.crates.is_empty());
    assert!(!model.covering_sets.is_empty());

    // Proves that build scripts are audited
    for audit in &model.build_script_audits {
        assert!(audit.is_deterministic, "build script {} must be deterministic", audit.path);
        assert!(!audit.accesses_undeclared_host_state, "build script {} must not access undeclared host state", audit.path);
    }
}

#[test]
fn configuration_space_covering_sets_isolate_features() {
    let root = checkout_root();
    let model = ConfigurationModel::inspect_workspace(&root).expect("configuration model generation");

    // Every crate with features must have isolated cells
    for krate in &model.crates {
        if !krate.features.is_empty() {
            let cells: Vec<_> = model
                .covering_sets
                .iter()
                .filter(|c| c.package == krate.name)
                .collect();
            assert!(!cells.is_empty(), "crate {} must have covering set cells", krate.name);
            // Must have a base cell (empty features)
            assert!(cells.iter().any(|c| c.enabled_features.is_empty()), "crate {} must have empty base cell", krate.name);
        }
    }
}

#[test]
fn stale_configuration_space_schema_fails_closed() {
    let root = checkout_root();
    let mut model = ConfigurationModel::inspect_workspace(&root).expect("configuration model generation");
    model.schema_version = 42; // Stale schema version

    let toml = model.to_toml().expect("to toml");
    let err = ConfigurationModel::from_toml(&toml).unwrap_err();
    assert!(matches!(err, ConfigSpaceError::StaleSchemaVersion { expected: 1, found: 42 }));
}
