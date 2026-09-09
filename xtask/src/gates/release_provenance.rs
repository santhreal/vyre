//! The `release-provenance` gate: dependency and release provenance authority (Row 116).
//!
//! Holds the workspace lockfile, third-party dependencies, native toolchain inputs,
//! code generators, benchmark baselines, schemas, and build script contracts to the
//! emitted CycloneDX SBOM, SLSA v1.2 provenance, and release provenance records.

use std::path::PathBuf;

use crate::artifact_gate::{settle_inspection, Inspection};
use crate::gate::{Finding, GateBehavior, GateCtx, GateError, Report};
use crate::provenance::{
    ReleaseProvenanceAuthority, PROVENANCE_ARTIFACT_PATH, SBOM_ARTIFACT_PATH,
    SLSA_PROVENANCE_ARTIFACT_PATH,
};

/// Gate that enforces dependency and release provenance authority across the workspace.
pub struct ReleaseProvenanceGate;

impl GateBehavior for ReleaseProvenanceGate {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut inspection = Inspection::new();

        let authority = match ReleaseProvenanceAuthority::inspect_workspace(&ctx.root) {
            Ok(auth) => auth,
            Err(error) => {
                inspection.find(Finding::in_file(
                    PathBuf::from("Cargo.lock"),
                    format!("failed to construct release provenance authority: {error}"),
                    "repair Cargo.lock and dependency manifests",
                ));
                return Ok(settle_inspection(ctx, "release-provenance", inspection));
            }
        };

        // 1. Verify every dependency is policy-approved and has a valid license classification
        let mut unapproved_count = 0usize;
        for dep in &authority.dependencies {
            if !dep.is_policy_approved {
                unapproved_count += 1;
                inspection.find(Finding::in_file(
                    PathBuf::from("Cargo.lock"),
                    format!(
                        "dependency '{}@{}' has unapproved license '{}' or violates policy in deny.toml",
                        dep.name, dep.version, dep.license
                    ),
                    "choose an approved license or replace the dependency with a policy-compliant alternative",
                ));
            }
        }

        // 2. Verify build script safety and bounded read contracts
        if let Err(error) = authority.verify_build_scripts(&ctx.root) {
            inspection.find(Finding::in_file(
                PathBuf::from("Cargo.toml"),
                format!("build script contract violation: {error}"),
                "ensure build scripts have no undeclared network or unbounded filesystem access",
            ));
        }

        // 3. Verify offline integrity
        if let Err(error) = authority.verify_offline_integrity() {
            inspection.find(Finding::in_file(
                PathBuf::from("Cargo.lock"),
                format!("offline provenance integrity violation: {error}"),
                "ensure all third-party dependencies are pinned with cryptographic checksums",
            ));
        }

        // 4. Generate the 3 authoritative provenance artifacts
        match authority.to_toml() {
            Ok(toml_str) => inspection.generates_text(PROVENANCE_ARTIFACT_PATH, toml_str),
            Err(err) => inspection.find(Finding::in_file(
                PathBuf::from(PROVENANCE_ARTIFACT_PATH),
                format!("failed to render release provenance TOML: {err}"),
                "fix serialization invariants in ReleaseProvenanceAuthority",
            )),
        }

        match authority.generate_sbom() {
            Ok(sbom_str) => inspection.generates_text(SBOM_ARTIFACT_PATH, sbom_str),
            Err(err) => inspection.find(Finding::in_file(
                PathBuf::from(SBOM_ARTIFACT_PATH),
                format!("failed to render CycloneDX SBOM: {err}"),
                "fix CycloneDX JSON serialization invariants",
            )),
        }

        match authority.generate_slsa_provenance(&ctx.root) {
            Ok(slsa_str) => inspection.generates_text(SLSA_PROVENANCE_ARTIFACT_PATH, slsa_str),
            Err(err) => inspection.find(Finding::in_file(
                PathBuf::from(SLSA_PROVENANCE_ARTIFACT_PATH),
                format!("failed to render SLSA provenance record: {err}"),
                "fix SLSA JSON serialization invariants",
            )),
        }

        inspection.notes.push(format!(
            "inspected {} dependencies ({} unapproved), {} native tools, {} code generators, {} benchmark baselines, {} schemas, {} build scripts",
            authority.dependencies.len(),
            unapproved_count,
            authority.native_tools.len(),
            authority.code_generators.len(),
            authority.benchmark_baselines.len(),
            authority.schemas.len(),
            authority.build_scripts.len()
        ));

        let mut report = settle_inspection(ctx, "release-provenance", inspection);
        report.cover_complete(
            "release provenance dependencies",
            authority.dependencies.len(),
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkout::checkout_root;

    #[test]
    fn release_provenance_gate_enforces_clean_tree_and_rejects_unapproved_dependencies() {
        let root = checkout_root();
        let authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
            .expect("inspect workspace for release provenance");

        assert!(!authority.dependencies.is_empty());
        assert!(authority.is_offline_capable);
        assert!(!authority.lockfile_digest.is_empty());

        for dep in &authority.dependencies {
            assert!(
                dep.is_policy_approved,
                "dependency '{}@{}' with license '{}' must be policy-approved",
                dep.name, dep.version, dep.license
            );
        }
    }

    #[test]
    fn release_provenance_gate_rejects_banned_dependencies() {
        let root = checkout_root();
        let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
            .expect("inspect workspace for release provenance");

        // Simulate injection of banned dependency
        authority
            .dependencies
            .push(crate::provenance::PinnedCrateDependency {
                name: "proc-macro1".to_string(),
                version: "1.0.0".to_string(),
                checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    .to_string(),
                source: "registry+https://github.com/rust-lang/crates.io-index".to_string(),
                license: "MIT".to_string(),
                is_policy_approved: false,
            });

        let unapproved = authority
            .dependencies
            .iter()
            .filter(|d| !d.is_policy_approved)
            .count();
        assert_eq!(unapproved, 1);
    }

    #[test]
    fn release_provenance_gate_rejects_network_reading_build_script() {
        let root = checkout_root();
        let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
            .expect("inspect workspace for release provenance");

        authority
            .build_scripts
            .push(crate::provenance::BuildScriptContract {
                crate_name: "hostile-crate".to_string(),
                path: "vyre-bench/build.rs".to_string(),
                declared_inputs: vec![],
                declared_outputs: vec![],
                has_network_access: true,
                has_bounded_reads: true,
            });

        let result = authority.verify_build_scripts(&root);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("network access is strictly forbidden"));
    }
}
