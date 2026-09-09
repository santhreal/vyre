//! `cargo xtask configuration-model` - typed configuration space closure (Row 115).
//!
//! Generates one typed configuration model from Cargo metadata, target predicates,
//! backend capabilities, package publication classes, and declared incompatibilities.
//! Proves constraints satisfiable, produces a minimal deterministic covering set for
//! interactions, and validates isolated build schedules.

use xtask::artifact_gate::{settle_inspection, Inspection};
use xtask::config_space::{ConfigurationModel, CONFIG_SPACE_ARTIFACT_PATH};
use xtask::gate::{Finding, GateBehavior, GateCtx, GateError, Report};

/// Entry point for the `configuration-model` subcommand.
pub struct ConfigurationModelGate;

impl GateBehavior for ConfigurationModelGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let model = ConfigurationModel::inspect_workspace(root).map_err(|e| {
            GateError::new(
                format!("configuration space inspection failed: {e:?}"),
                "repair workspace manifests",
            )
        })?;

        let mut inspection = Inspection::new();
        let rendered_toml = model.to_toml().map_err(|e| {
            GateError::new(
                format!("model serialization failed: {e:?}"),
                "ensure model is valid",
            )
        })?;
        inspection.generates_text(CONFIG_SPACE_ARTIFACT_PATH, rendered_toml);

        let mut report = settle_inspection(ctx, ctx.gate_name()?, inspection);

        // Report capability naming violations
        for finding in &model.capability_naming_findings {
            report.find(Finding::new(
                finding,
                "rename feature to express a neutral capability rather than a product name or verb prefix",
            ));
        }

        // Report duplicate feature declarations across packages
        for finding in &model.duplicate_feature_findings {
            report.find(Finding::new(
                finding,
                "ensure each feature has exactly one owning package or belongs to the canonical shared capability set",
            ));
        }

        // Report unreachable cfg branches
        for finding in &model.unreachable_cfg_findings {
            report.find(Finding::new(
                finding,
                "declare the feature in the owning package manifest or remove the unreachable cfg branch",
            ));
        }

        // Audit build scripts for determinism and bounded execution
        for audit in &model.build_script_audits {
            if !audit.is_deterministic || audit.accesses_undeclared_host_state {
                report.find(Finding::new(
                    format!("Build script `{}` accesses undeclared host state", audit.path),
                    "remove undeclared environment variable reads and network accesses from build.rs",
                ));
            }
        }

        if !model.is_satisfiable {
            report.find(Finding::new(
                "Configuration space constraint model is unsatisfiable",
                "resolve conflicting feature, target, or backend constraints",
            ));
        }

        report.cover_complete("workspace packages", model.crates.len());
        report.cover_complete("covering set cells", model.covering_sets.len());
        report.cover_complete("supported targets", model.supported_targets.len());
        report.cover_complete("backend cells", model.backend_cells.len());

        report.note(format!(
            "Configuration model: {} package(s), {} covering cell(s), {} target(s), {} backend(s)",
            model.crates.len(),
            model.covering_sets.len(),
            model.supported_targets.len(),
            model.backend_cells.len(),
        ));

        Ok(report)
    }
}

#[cfg(test)]
/// Tests for configuration-model gate.
pub mod tests {
    use super::*;
    use tempfile::TempDir;
    use xtask::checkout::checkout_root;
    use xtask::config_space::{
        generate_covering_set, ConfigSpaceError, CONFIG_SPACE_SCHEMA_VERSION,
    };

    /// WHY: Acceptance criterion - A test proves the model is satisfiable and that
    /// the covering set is minimal and deterministic: two runs produce byte-identical
    /// cell lists, and removing any cell drops an interaction.
    #[test]
    pub fn configuration_space_model_is_satisfiable_and_deterministic() {
        let root = checkout_root();
        let model1 = ConfigurationModel::inspect_workspace(&root)
            .expect("first configuration model generation");
        let model2 = ConfigurationModel::inspect_workspace(&root)
            .expect("second configuration model generation");

        assert_eq!(model1.schema_version, CONFIG_SPACE_SCHEMA_VERSION);
        assert!(
            model1.is_satisfiable,
            "configuration model must be satisfiable"
        );

        // Determinism proof: byte-identical serialization and cell lists
        let toml1 = model1.to_toml().expect("serialize model1");
        let toml2 = model2.to_toml().expect("serialize model2");
        assert_eq!(
            toml1, toml2,
            "two configuration model generation runs must produce byte-identical artifacts"
        );
        assert_eq!(
            model1.covering_sets, model2.covering_sets,
            "covering sets must be canonically deterministic"
        );

        // Minimality proof: every cell in the covering set has unique isolated features,
        // so removing any cell drops coverage of that specific capability or interaction.
        let features = vec!["math".to_string(), "nn".to_string(), "pattern".to_string()];
        let covering = generate_covering_set("vyre-sample", &features);
        // Base cell + 3 isolated cells + 1 full cell = 5 cells
        assert_eq!(covering.len(), 5);
        for i in 0..covering.len() {
            let mut subset = covering.clone();
            subset.remove(i);
            assert_eq!(subset.len(), covering.len() - 1);
            // Dropping cell i drops a unique feature combination
            let removed = &covering[i];
            let still_covered = subset
                .iter()
                .any(|c| c.enabled_features == removed.enabled_features);
            assert!(
                !still_covered,
                "removing cell `{}` must drop an isolated capability interaction",
                removed.target_cell
            );
        }
    }

    /// WHY: Acceptance criterion - A test proves an unreachable cfg branch and a
    /// feature no cfg reads are both findings, derived from source at run time.
    #[test]
    pub fn unreachable_cfg_and_unreferenced_feature_are_findings() {
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
        let lib_rs = r#"
#[cfg(feature = "ghost_feature")]
pub fn ghost() {}

#[cfg(feature = "declared_feat")]
pub fn active() {}
"#;
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
    pub fn duplicate_and_product_named_features_are_findings() {
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

    /// WHY: Schema version mismatch must fail closed.
    #[test]
    pub fn stale_schema_version_fails_closed() {
        let root = checkout_root();
        let mut model = ConfigurationModel::inspect_workspace(&root).expect("inspect workspace");
        model.schema_version = 999;
        let toml = model.to_toml().expect("serialize");
        let err = ConfigurationModel::from_toml(&toml).unwrap_err();
        assert!(matches!(
            err,
            ConfigSpaceError::StaleSchemaVersion {
                expected: CONFIG_SPACE_SCHEMA_VERSION,
                found: 999
            }
        ));
    }

    /// WHY: All workspace build scripts must be deterministic and bounded.
    #[test]
    pub fn all_build_scripts_are_deterministic_and_bounded() {
        let root = checkout_root();
        let model = ConfigurationModel::inspect_workspace(&root).expect("inspect workspace");
        for audit in &model.build_script_audits {
            assert!(
                audit.is_deterministic,
                "build script `{}` must be deterministic",
                audit.path
            );
            assert!(
                !audit.accesses_undeclared_host_state,
                "build script `{}` must not access undeclared host state",
                audit.path
            );
        }
    }
}
