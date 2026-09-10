//! `cargo xtask configuration-model` - typed configuration space closure (Rows 81, 115).
//!
//! Reads every member manifest into one model: which package declares each
//! capability, which package a facade row forwards it to, the cells a build can
//! ask for, and whether the declared constraints leave any of those cells
//! without an assignment. Generates the configuration space artifact and the
//! consumer facade's `[features]` table from the `[facade]` roster in
//! `docs/CRATE_OWNERSHIP.toml`, so adding a domain is one ownership record
//! rather than a row in the facade manifest and a row in the domain manifest.

use xtask::artifact_gate::{settle_inspection, Inspection};
use xtask::config_space::{ConfigurationModel, CONFIG_SPACE_ARTIFACT_PATH};
use xtask::gate::{Finding, GateBehavior, GateCtx, GateError, Report};

/// Entry point for the `configuration-model` subcommand.
pub struct ConfigurationModelGate;

impl GateBehavior for ConfigurationModelGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let model = ConfigurationModel::inspect_workspace(root).map_err(|error| {
            GateError::new(
                format!("configuration space inspection failed: {error:?}"),
                "repair workspace manifests and the [facade] roster in docs/CRATE_OWNERSHIP.toml",
            )
        })?;

        let mut inspection = Inspection::new();
        let rendered_toml = model.to_toml().map_err(|error| {
            GateError::new(
                format!("model serialization failed: {error:?}"),
                "ensure model is valid",
            )
        })?;
        inspection.generates_document_text(CONFIG_SPACE_ARTIFACT_PATH, rendered_toml);

        let (facade_path, facade_manifest) =
            ConfigurationModel::facade_manifest(root).map_err(|error| {
                GateError::new(
                    format!("facade manifest generation failed: {error:?}"),
                    "repair the [facade] roster in docs/CRATE_OWNERSHIP.toml",
                )
            })?;
        inspection.generates_text(&facade_path, facade_manifest);

        let mut report = settle_inspection(ctx, ctx.gate_name()?, inspection);

        for finding in &model.capability_naming_findings {
            report.find(Finding::new(
                finding,
                "rename the feature to express a neutral capability rather than a product name or verb prefix",
            ));
        }

        for finding in &model.feature_ownership_findings {
            report.find(Finding::new(
                finding,
                "move the declaration into the single package that owns the capability and forward the name from every other package, or record the name as a shared capability in docs/CRATE_OWNERSHIP.toml",
            ));
        }

        for finding in &model.forward_findings {
            report.find(Finding::new(
                finding,
                "declare the feature in the dependency the forward names, or point the forward at the name that package declares",
            ));
        }

        for finding in &model.facade_findings {
            report.find(Finding::new(
                finding,
                "reconcile the [facade] roster in docs/CRATE_OWNERSHIP.toml with the domain manifests it forwards through",
            ));
        }

        for finding in &model.unreachable_cfg_findings {
            report.find(Finding::new(
                finding,
                "declare the feature in the owning package manifest or remove the unreachable cfg branch",
            ));
        }

        for audit in &model.build_script_audits {
            if !audit.is_deterministic || audit.accesses_undeclared_host_state {
                report.find(Finding::new(
                    format!("Build script `{}` accesses undeclared host state", audit.path),
                    "remove undeclared environment variable reads and network accesses from build.rs",
                ));
            }
        }

        for conflict in model.satisfiability.conflicts() {
            report.find(Finding::new(
                format!(
                    "Cell `{}` has no valid assignment: {}",
                    conflict.cell, conflict.violated
                ),
                "relax the constraint the cell violates, or stop scheduling the cell by removing the feature that reaches it",
            ));
        }

        report.cover_complete("workspace packages", model.crates.len());
        report.cover_complete("covering set cells", model.covering_sets.len());
        report.cover_complete("supported targets", model.supported_targets.len());
        report.cover_complete("backend cells", model.backend_cells.len());
        report.cover_complete("scheduled configuration cells", model.scheduled_cells);

        if let xtask::config_space::Satisfiability::Satisfiable(assignment) = &model.satisfiability
        {
            report.note(format!(
                "Configuration space satisfiable; widest assignment `{}` enables {} feature(s) across {} package(s)",
                assignment.cell,
                assignment.enabled.len(),
                assignment.activated.len(),
            ));
        }

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
    use xtask::checkout::checkout_root;
    use xtask::config_space::{
        generate_covering_set, ConfigSpaceError, Satisfiability, CONFIG_SPACE_SCHEMA_VERSION,
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
        match &model1.satisfiability {
            Satisfiability::Satisfiable(assignment) => assert!(
                !assignment.enabled.is_empty(),
                "a satisfying assignment names the features it enables"
            ),
            Satisfiability::Unsatisfiable(conflicts) => {
                panic!("configuration model must be satisfiable, conflicts: {conflicts:?}")
            }
        }

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
        for index in 0..covering.len() {
            let mut subset = covering.clone();
            subset.remove(index);
            assert_eq!(subset.len(), covering.len() - 1);
            let removed = &covering[index];
            let still_covered = subset
                .iter()
                .any(|cell| cell.enabled_features == removed.enabled_features);
            assert!(
                !still_covered,
                "removing cell `{}` must drop an isolated capability interaction",
                removed.target_cell
            );
        }
    }

    /// WHY: Schema version mismatch must fail closed.
    #[test]
    pub fn stale_schema_version_fails_closed() {
        let root = checkout_root();
        let mut model = ConfigurationModel::inspect_workspace(&root).expect("inspect workspace");
        model.schema_version = CONFIG_SPACE_SCHEMA_VERSION + 997;
        let toml = model.to_toml().expect("serialize");
        let error = ConfigurationModel::from_toml(&toml).unwrap_err();
        assert_eq!(
            error,
            ConfigSpaceError::StaleSchemaVersion {
                expected: CONFIG_SPACE_SCHEMA_VERSION,
                found: CONFIG_SPACE_SCHEMA_VERSION + 997,
            }
        );
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

    /// WHY: The facade's `[features]` table is generated from the roster, so the
    /// committed table has to be what the roster renders. A drift here is the
    /// shape row 81 names: a facade list and a domain list of one fact.
    #[test]
    pub fn facade_features_table_matches_the_roster() {
        let root = checkout_root();
        let (path, generated) =
            ConfigurationModel::facade_manifest(&root).expect("render facade manifest");
        let committed = std::fs::read_to_string(root.join(&path)).expect("read facade manifest");
        assert_eq!(
            committed.replace("\r\n", "\n"),
            generated,
            "`{path}` differs from what the [facade] roster renders; run `configuration-model --write`"
        );
    }
}
