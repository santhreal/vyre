//! The `release-provenance` gate: dependency and release provenance authority.
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

        let unapproved = unapproved_license_findings(&authority);
        let unapproved_count = unapproved.len();
        for finding in unapproved {
            inspection.find(finding);
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
            Ok(toml_str) => inspection.generates_document_text(PROVENANCE_ARTIFACT_PATH, toml_str),
            Err(err) => inspection.find(Finding::in_file(
                PathBuf::from(PROVENANCE_ARTIFACT_PATH),
                format!("failed to render release provenance TOML: {err}"),
                "fix serialization invariants in ReleaseProvenanceAuthority",
            )),
        }

        match authority.generate_sbom() {
            Ok(sbom_str) => inspection.generates_evidence_text(
                SBOM_ARTIFACT_PATH,
                crate::evidence_record::MeasurementRecord::HostOnly,
                sbom_str,
            ),
            Err(err) => inspection.find(Finding::in_file(
                PathBuf::from(SBOM_ARTIFACT_PATH),
                format!("failed to render CycloneDX SBOM: {err}"),
                "fix CycloneDX JSON serialization invariants",
            )),
        }

        match authority.generate_slsa_provenance(&ctx.root) {
            Ok(slsa_str) => inspection.generates_evidence_text(
                SLSA_PROVENANCE_ARTIFACT_PATH,
                crate::evidence_record::MeasurementRecord::HostOnly,
                slsa_str,
            ),
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

/// One finding per dependency whose license the policy does not admit.
///
/// The verdict is separated from the run so it can be proven without a
/// checkout: asking the live tree whether its own dependencies are approved
/// answers what the tree happens to hold, not what the gate does with an
/// unapproved one, and the tree is green precisely when that path never runs.
fn unapproved_license_findings(authority: &ReleaseProvenanceAuthority) -> Vec<Finding> {
    authority
        .dependencies
        .iter()
        .filter(|dependency| !dependency.is_policy_approved)
        .map(|dependency| {
            Finding::in_file(
                PathBuf::from("Cargo.lock"),
                format!(
                    "dependency '{}@{}' has unapproved license '{}' or violates policy in deny.toml",
                    dependency.name, dependency.version, dependency.license
                ),
                "choose an approved license or replace the dependency with a policy-compliant alternative",
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    use crate::checkout::checkout_root;
    use crate::provenance::PinnedCrateDependency;

    fn dependency(name: &str, license: &str, approved: bool) -> PinnedCrateDependency {
        PinnedCrateDependency {
            name: name.to_string(),
            version: "1.0.0".to_string(),
            checksum: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                .to_string(),
            source: "registry+https://github.com/rust-lang/crates.io-index".to_string(),
            license: license.to_string(),
            is_policy_approved: approved,
        }
    }

    /// WHY: this is the one verdict the gate reaches on its own, and a green
    /// checkout never exercises it. A test that asked the live tree whether its
    /// dependencies are approved passed on the answer being yes and would go on
    /// passing if the mapping to a finding were deleted. The finding has to name
    /// the dependency and the license, because a reader of `Cargo.lock` has no
    /// other way to tell which of several hundred rows the gate refused.
    #[test]
    fn an_unapproved_license_becomes_a_finding_naming_the_dependency() {
        let root = checkout_root();
        let mut authority = ReleaseProvenanceAuthority::inspect_workspace(&root)
            .expect("inspect workspace for release provenance");
        authority.dependencies = vec![
            dependency("approved-crate", "MIT", true),
            dependency("agpl-crate", "AGPL-3.0-only", false),
        ];

        let findings = unapproved_license_findings(&authority);
        assert_eq!(findings.len(), 1, "only the unapproved dependency is named");
        assert!(
            findings[0].message.contains("agpl-crate@1.0.0")
                && findings[0].message.contains("AGPL-3.0-only"),
            "the finding must name the dependency and its license: {}",
            findings[0].message
        );
        assert_eq!(findings[0].file.as_deref(), Some(Path::new("Cargo.lock")));

        authority.dependencies = vec![dependency("approved-crate", "MIT", true)];
        assert!(
            unapproved_license_findings(&authority).is_empty(),
            "an approved dependency states nothing"
        );
    }

    /// WHY: a build script that reaches the network makes the build
    /// irreproducible and unattributable, and the release provenance record
    /// this gate writes would state otherwise.
    #[test]
    fn a_build_script_reaching_the_network_is_refused() {
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

        let error = authority
            .verify_build_scripts(&root)
            .expect_err("a network-reading build script is refused");
        assert!(
            error.to_string().contains("network access is strictly forbidden"),
            "{error}"
        );
    }
}
