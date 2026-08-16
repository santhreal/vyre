//! Authoritative metadata for every registered gate.
//!
//! The implementation registry owns executable code. This table owns the
//! orthogonal facts the runner must validate before execution: area membership,
//! authoritative subject class, generated paths, and proof method. Registry and
//! metadata agreement is checked in both directions. Named subsets are derived
//! from `areas`; no second list of gate names exists.

use crate::gate::GateMetadata;

/// One metadata row keyed by the command-line gate name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GateMetadataRow {
    /// Registered gate name.
    pub name: &'static str,
    /// Static enforcement contract.
    pub metadata: GateMetadata,
}

/// Every gate metadata row, sorted by gate name.
pub static GATE_METADATA: &[GateMetadataRow] = &[
    GateMetadataRow { name: "abstraction-gate", metadata: GateMetadata { areas: &["prepublish"], subject: "registered operations", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "architecture-contract", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "backend-extension", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "backend-matrix", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "bench-baselines", metadata: GateMetadata { areas: &["structure"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "bench-coverage", metadata: GateMetadata { areas: &["structure", "benchmarks"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "bench-crossback", metadata: GateMetadata { areas: &["ir"], subject: "registered IR corpus", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "bench-release", metadata: GateMetadata { areas: &["prepublish", "benchmarks"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "bench-smoke-runtime", metadata: GateMetadata { areas: &["structure", "benchmarks"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "catalog", metadata: GateMetadata { areas: &["prepublish"], subject: "registered operations", artifacts: &["docs/generated/catalog.toml"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "check-tier-deps", metadata: GateMetadata { areas: &["structure"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "ci-matrix", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "ci-required", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "cli-docs", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "compile", metadata: GateMetadata { areas: &["ir"], subject: "registered IR corpus", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "conformance-matrix", metadata: GateMetadata { areas: &["contract-rules", "release-evidence"], subject: "workspace contract subjects", artifacts: &["release/evidence/conformance/conformance-matrix.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "contract-in-source", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "cross-target", metadata: GateMetadata { areas: &["prepublish"], subject: "registered operations", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "dep-drift", metadata: GateMetadata { areas: &["prepublish"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "doc-claims", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "docs-check", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "docs-coupling", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "docs-references", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "docs-register", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "dup-scan", metadata: GateMetadata { areas: &["structure"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "evidence-paths", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "example-capability", metadata: GateMetadata { areas: &["structure"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "feature-isolation", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "feature-matrix", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &["release/evidence/metadata/feature-matrix.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "file-size", metadata: GateMetadata { areas: &["source-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "frozen-contracts", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &["docs/frozen-traits/AlgebraicLaw.txt", "docs/frozen-traits/EnforceGate.txt", "docs/frozen-traits/ExprVisitor.txt", "docs/frozen-traits/Lowerable.txt", "docs/frozen-traits/MutationClass.txt", "docs/frozen-traits/PassBoundaryClass.txt", "docs/frozen-traits/VyreBackend.txt"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "gate1", metadata: GateMetadata { areas: &["prepublish"], subject: "registered operations", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "gpu-loudness", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "heuristic-audit", metadata: GateMetadata { areas: &["structure"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-blocking-wait", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-inventory", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-nested-rows", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-owned-dispatch", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-reserve", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-scan", metadata: GateMetadata { areas: &["structure"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-unbounded-cache", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hot-path-unbounded-read", metadata: GateMetadata { areas: &["hot-path-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "hygiene-matrix", metadata: GateMetadata { areas: &["structure"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "internal-dep-versions", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "invariant-paths", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "launch-state", metadata: GateMetadata { areas: &["contract-rules", "release-evidence"], subject: "workspace contract subjects", artifacts: &["release/evidence/final/public-launch-state.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "layering", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lego-audit", metadata: GateMetadata { areas: &["composition"], subject: "registered operations", artifacts: &["audits/lego-composition.tsv"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lego-quick", metadata: GateMetadata { areas: &["composition"], subject: "registered operations", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lint-expect-fix", metadata: GateMetadata { areas: &["lint-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lint-one-policy", metadata: GateMetadata { areas: &["lint-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lint-unsafe-budget", metadata: GateMetadata { areas: &["lint-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lint-unsafe-justification", metadata: GateMetadata { areas: &["lint-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "list-ops", metadata: GateMetadata { areas: &["prepublish"], subject: "registered operations", artifacts: &["docs/generated/op-inventory.toml"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "lockfile-clean", metadata: GateMetadata { areas: &["prepublish"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "metadata-matrix", metadata: GateMetadata { areas: &["contract-rules", "release-evidence"], subject: "workspace contract subjects", artifacts: &["release/evidence/metadata/metadata-matrix.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "neutral-crates", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "op-matrix", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &["docs/optimization/OP_MATRIX.toml"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "op-names", metadata: GateMetadata { areas: &["cat-a"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "operation-schema", metadata: GateMetadata { areas: &["prepublish"], subject: "registered operations", artifacts: &["docs/generated/OP_SCHEMA.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "optimization-corpus", metadata: GateMetadata { areas: &["ir", "release-evidence"], subject: "registered IR corpus", artifacts: &["release/evidence/optimization/optimization-case-manifest.json", "release/evidence/optimization/optimization-corpus-contracts.json", "release/evidence/optimization/optimization-corpus.json", "release/evidence/optimization/optimization-family-manifest.json", "release/evidence/optimization/optimizer-pass-manifest.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "optimization-docs", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &["docs/generated/optimizer-passes.toml"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "optimization-matrix", metadata: GateMetadata { areas: &["ir", "release-evidence"], subject: "registered IR corpus", artifacts: &["release/evidence/optimization/optimization-matrix.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "package-readiness", metadata: GateMetadata { areas: &["prepublish", "release-evidence"], subject: "release contract subjects", artifacts: &["release/evidence/metadata/package-readiness.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "parity-testing-isolated", metadata: GateMetadata { areas: &["cat-a"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "path-deps-resolve", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "platform-boundary", metadata: GateMetadata { areas: &["cat-a", "prepublish"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "platform-consumer-docs", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "primitive-admission-gate", metadata: GateMetadata { areas: &["composition"], subject: "registered operations", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "print-composition", metadata: GateMetadata { areas: &["ir"], subject: "registered IR corpus", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "program-wire-fields", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "proptest-coverage", metadata: GateMetadata { areas: &["lint-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "public-api-paths", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &["xtask/public-api-paths.toml"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "public-api-snapshot", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "readback-ring", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "release-benchmarks", metadata: GateMetadata { areas: &["prepublish", "benchmarks"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "release-conformance", metadata: GateMetadata { areas: &["prepublish", "release-evidence"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "release-docs", metadata: GateMetadata { areas: &["docs"], subject: "owned documentation artifacts", artifacts: &["CHANGELOG.md", "release/evidence/docs/release-notes-body.md"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "release-evidence", metadata: GateMetadata { areas: &["prepublish", "release-evidence"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "release-workload-matrix", metadata: GateMetadata { areas: &["prepublish", "benchmarks"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "repo-hygiene", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "shader-source", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "shrink", metadata: GateMetadata { areas: &["ir"], subject: "registered IR corpus", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "single-backlog", metadata: GateMetadata { areas: &["repo-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "source-include-module", metadata: GateMetadata { areas: &["source-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "source-parses", metadata: GateMetadata { areas: &["source-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "source-reachability", metadata: GateMetadata { areas: &["source-rules"], subject: "tracked source files", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "trace-f32", metadata: GateMetadata { areas: &["ir"], subject: "registered IR corpus", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "unification", metadata: GateMetadata { areas: &["contract-rules"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "verify-rewrite-proofs", metadata: GateMetadata { areas: &["ir"], subject: "registered IR corpus", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "version-matrix", metadata: GateMetadata { areas: &["prepublish", "release-evidence"], subject: "release contract subjects", artifacts: &["release/evidence/metadata/version-matrix.json"], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "vyre-release-gate", metadata: GateMetadata { areas: &["prepublish"], subject: "release contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "whats-similar", metadata: GateMetadata { areas: &["composition"], subject: "registered operations", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "workspace-check", metadata: GateMetadata { areas: &["cat-a"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "workspace-clippy", metadata: GateMetadata { areas: &["cat-a"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "workspace-docs", metadata: GateMetadata { areas: &["cat-a"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "workspace-membership", metadata: GateMetadata { areas: &["manifest-rules"], subject: "workspace manifests", artifacts: &[], proof: "definition-site mutation tests" } },
    GateMetadataRow { name: "workspace-tests", metadata: GateMetadata { areas: &["cat-a"], subject: "workspace contract subjects", artifacts: &[], proof: "definition-site mutation tests" } },
];

/// Resolve one gate's enforcement contract.
#[must_use]
pub fn metadata(name: &str) -> Option<GateMetadata> {
    GATE_METADATA
        .binary_search_by_key(&name, |row| row.name)
        .ok()
        .map(|index| GATE_METADATA[index].metadata)
}

/// Stable area names derived from the flat metadata rows.
#[must_use]
pub fn areas() -> Vec<&'static str> {
    let mut areas = GATE_METADATA
        .iter()
        .flat_map(|row| row.metadata.areas.iter().copied())
        .collect::<Vec<_>>();
    areas.sort_unstable();
    areas.dedup();
    areas
}

/// Registered gate names in one area.
#[must_use]
pub fn gates_in_area(area: &str) -> Vec<&'static str> {
    GATE_METADATA
        .iter()
        .filter(|row| row.metadata.areas.contains(&area))
        .map(|row| row.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: adding a metadata row out of order breaks binary lookup and can make
    /// a live gate look unclassified. Exact sorted uniqueness closes both the
    /// shadowing and unreachable-row variants.
    #[test]
    fn metadata_names_are_sorted_and_unique() {
        assert!(GATE_METADATA.windows(2).all(|rows| rows[0].name < rows[1].name));
    }

    #[test]
    fn every_metadata_row_has_a_complete_static_contract() {
        for row in GATE_METADATA {
            assert!(
                row.metadata.failures().is_empty(),
                "{}: {:?}",
                row.name,
                row.metadata.failures()
            );
        }
    }
}
