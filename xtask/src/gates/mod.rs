//! The gates this crate implements, and the sweep that runs the registry.
//!
//! A gate here reads the repository and reports findings. There is no other
//! category: what used to be a composite that re-ran other subcommands is now a
//! named subset of the registry, and the cargo invocations it drove are gates of
//! their own, so the sweep holds them to a baseline like everything else.
//!
//! `gates::sweep` is the runner and the wiring meta-check that keeps every
//! registered gate connected to a pinned baseline and a workflow.

pub mod architecture_contract;
pub mod backend_parity;
pub mod bench;
pub mod check_tier_deps;
pub mod ci_contract;
pub mod ci_registry;
pub mod ci_steps;
pub mod codeowners;
pub mod crate_pages;
pub mod crate_readmes;
pub mod crate_registry;
pub mod dedup_report;
pub mod dep_drift;
pub mod device_test_gating;
pub mod dispatch_surface;
pub mod doc_contract;
pub mod docs_references;
pub mod dup_scan;
pub mod evidence_paths;
pub mod evidence_provenance;
pub mod example_capability;
pub mod feature_isolation;
pub mod feature_msrv;
pub mod file_size;
pub mod finding_capability;
#[cfg(test)]
pub mod fixture_checkout;
pub mod frozen_contract;
pub mod gate_canon;
pub mod gpu_loudness;
mod host_oracle_closure;
pub mod host_oracle_elimination;
mod host_oracle_elimination_ast;
mod host_oracle_elimination_classify;
mod host_oracle_elimination_eval;
mod host_oracle_elimination_extract;
mod host_oracle_elimination_records;
mod host_oracle_elimination_scanners;
#[cfg(test)]
mod host_oracle_elimination_test_fixtures;
#[cfg(test)]
mod host_oracle_elimination_tests_part1;
#[cfg(test)]
mod host_oracle_elimination_tests_part2;
#[cfg(test)]
mod host_oracle_elimination_tests_part3;
mod host_oracle_elimination_visitor;
pub mod hot_path;
pub mod hot_path_scan;
pub mod hygiene_matrix;
pub mod implementation_family;
pub mod inventory_walk;
pub mod layering;
pub mod lego_quick;
pub mod lint_hygiene;
pub mod lockfile;
pub mod manifest_contract;
pub mod metal_parity;
pub mod module_layout;
pub mod op_names;
pub mod oracle_sweeps;
pub mod ownership;
pub mod parity_testing;
pub mod placement_predicate;
pub mod platform_boundary;
pub mod platform_docs;
pub mod proptest_coverage;
pub mod public_api;
pub mod public_api_paths;
pub mod repo_hygiene;
pub mod scan;
pub mod schedule_ownership;
pub mod script_ledger;
pub mod shader_source;
pub mod source_reachability;
pub mod sweep;
pub mod test_material;
pub mod test_only_capability;
pub mod testing_guides;
pub mod unification;
pub mod use_paths;
pub mod wire_determinism;
pub mod workspace_build;
pub mod worktree_lifetime;

use crate::gate::GateBehavior;

/// Every gate behavior this module implements, keyed only to pair it with its
/// authoritative descriptor at registry construction.
pub static GATES: &[(&str, &dyn GateBehavior)] = &[
    (
        "architecture-contract",
        &architecture_contract::ArchitectureContract,
    ),
    ("cuda-parity", &backend_parity::CudaParity),
    ("spirv-parity", &backend_parity::SpirvParity),
    ("bench-baselines", &bench::BenchBaselines),
    ("bench-coverage", &bench::BenchCoverage),
    ("bench-smoke-runtime", &bench::BenchSmokeRuntime),
    ("check-tier-deps", &check_tier_deps::CheckTierDeps),
    ("ci-concurrency", &ci_contract::CiConcurrency),
    ("ci-matrix", &ci_contract::CiMatrix),
    ("ci-required", &ci_contract::CiRequired),
    ("ci-registry", &ci_registry::CiRegistry),
    ("ci-steps", &ci_steps::CiSteps),
    ("ci-shell", &ci_contract::CiShell),
    ("codeowners", &codeowners::Codeowners),
    ("crate-pages", &crate_pages::CratePages),
    ("crate-readmes", &crate_readmes::CrateReadmes),
    ("crate-ownership", &crate_registry::CrateOwnership),
    ("dep-drift", &dep_drift::DepDrift),
    ("device-test-gating", &device_test_gating::DeviceTestGating),
    ("hot-path-nested-rows", &dispatch_surface::NestedRows),
    ("hot-path-owned-dispatch", &dispatch_surface::OwnedDispatch),
    ("contract-in-source", &doc_contract::ContractInSource),
    ("doc-claims", &doc_contract::DocClaims),
    ("docs-references", &docs_references::DocsReferences),
    ("dup-scan", &dup_scan::DupScan),
    ("evidence-paths", &evidence_paths::EvidencePaths),
    ("invariant-paths", &evidence_paths::InvariantPaths),
    (
        "evidence-provenance",
        &evidence_provenance::EvidenceProvenance,
    ),
    ("example-capability", &example_capability::ExampleCapability),
    ("feature-isolation", &feature_isolation::FeatureIsolation),
    ("feature-msrv", &feature_msrv::FeatureMsrv),
    ("file-size", &file_size::FileSize),
    ("backend-extension", &frozen_contract::BackendExtension),
    ("frozen-contracts", &frozen_contract::FrozenContracts),
    ("program-wire-fields", &frozen_contract::ProgramWireFields),
    ("readback-ring", &frozen_contract::ReadbackRing),
    ("gate-canon", &gate_canon::GateCanon),
    ("gpu-loudness", &gpu_loudness::GpuLoudness),
    (
        "host-oracle-elimination",
        &host_oracle_elimination::HostOracleElimination,
    ),
    ("hot-path-blocking-wait", &hot_path::BlockingWait),
    ("hot-path-reserve", &hot_path::ReserveArgument),
    ("hot-path-unbounded-cache", &hot_path::UnboundedCache),
    ("hot-path-unbounded-read", &hot_path::UnboundedRead),
    ("hot-path-scan", &hot_path_scan::HotPathScan),
    ("hygiene-matrix", &hygiene_matrix::HygieneMatrix),
    ("hot-path-inventory", &inventory_walk::InventoryWalk),
    ("layering", &layering::Layering),
    ("neutral-crates", &layering::NeutralCrates),
    ("lego-quick", &lego_quick::LegoQuick),
    ("lint-expect-fix", &lint_hygiene::ExpectHasFix),
    ("lint-one-policy", &lint_hygiene::OneLintPolicy),
    ("lint-unsafe-budget", &lint_hygiene::UnsafeBudget),
    (
        "lint-unsafe-justification",
        &lint_hygiene::UnsafeJustification,
    ),
    ("lockfile-clean", &lockfile::LockfileClean),
    (
        "internal-dep-versions",
        &manifest_contract::InternalDepVersions,
    ),
    ("path-deps-resolve", &manifest_contract::PathDepsResolve),
    (
        "workspace-membership",
        &manifest_contract::WorkspaceMembership,
    ),
    ("metal-parity", &metal_parity::MetalParity),
    ("module-layout", &module_layout::ModuleLayout),
    ("op-names", &op_names::OpNames),
    ("oracle-sweeps", &oracle_sweeps::OracleSweeps),
    (
        "parity-testing-isolated",
        &parity_testing::ParityTestingIsolated,
    ),
    (
        "placement-predicates",
        &placement_predicate::PlacementPredicates,
    ),
    ("platform-boundary", &platform_boundary::PlatformBoundary),
    (
        "platform-consumer-docs",
        &platform_docs::PlatformConsumerDocs,
    ),
    ("proptest-coverage", &proptest_coverage::ProptestCoverage),
    ("public-api-snapshot", &public_api::PublicApiSnapshot),
    ("public-api-paths", &public_api_paths::PublicApiPaths),
    ("repo-hygiene", &repo_hygiene::RepoHygiene),
    ("single-backlog", &repo_hygiene::SingleBacklog),
    ("schedule-ownership", &schedule_ownership::ScheduleOwnership),
    ("script-ledger", &script_ledger::ScriptLedger),
    ("shader-source", &shader_source::ShaderSource),
    (
        "source-include-module",
        &source_reachability::IncludeIsNotAModule,
    ),
    ("source-parses", &source_reachability::SourceParses),
    (
        "source-reachability",
        &source_reachability::SourceReachability,
    ),
    (
        "test-material-placement",
        &test_material::TestMaterialPlacement,
    ),
    (
        "test-only-capability",
        &test_only_capability::TestOnlyCapability,
    ),
    ("testing-guides", &testing_guides::TestingGuides),
    ("unification", &unification::Unification),
    ("wire-determinism", &wire_determinism::WireDeterminism),
    ("workspace-check", &workspace_build::WorkspaceCheck),
    ("workspace-clippy", &workspace_build::WorkspaceClippy),
    ("workspace-docs", &workspace_build::WorkspaceDocs),
    ("workspace-tests", &workspace_build::WorkspaceTests),
];
