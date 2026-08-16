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
pub mod crate_pages;
pub mod crate_readmes;
pub mod crate_registry;
pub mod dedup_report;
pub mod dep_drift;
pub mod dispatch_surface;
pub mod doc_contract;
pub mod docs_references;
pub mod dup_scan;
pub mod evidence_paths;
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
pub mod script_ledger;
pub mod shader_source;
pub mod source_reachability;
pub mod sweep;
pub mod test_material;
pub mod testing_guides;
pub mod unification;
pub mod use_paths;
pub mod wire_determinism;
pub mod workspace_build;

use crate::gate::Gate;

/// Every gate this module owns.
///
/// The registry is assembled from one slice per area at run time, so adding a
/// gate is adding it here and nowhere else. The sweep enumerates what this
/// yields, which is why a gate cannot be registered and left unswept.
pub static GATES: &[&dyn Gate] = &[
    &architecture_contract::ArchitectureContract,
    &backend_parity::CudaParity,
    &backend_parity::SpirvParity,
    &bench::BenchBaselines,
    &bench::BenchCoverage,
    &bench::BenchSmokeRuntime,
    &check_tier_deps::CheckTierDeps,
    &ci_contract::CiMatrix,
    &ci_contract::CiRequired,
    &ci_registry::CiRegistry,
    &ci_steps::CiSteps,
    &crate_pages::CratePages,
    &crate_readmes::CrateReadmes,
    &crate_registry::CrateOwnership,
    &dep_drift::DepDrift,
    &dispatch_surface::NestedRows,
    &dispatch_surface::OwnedDispatch,
    &doc_contract::ContractInSource,
    &doc_contract::DocClaims,
    &docs_references::DocsReferences,
    &dup_scan::DupScan,
    &evidence_paths::EvidencePaths,
    &evidence_paths::InvariantPaths,
    &example_capability::ExampleCapability,
    &feature_isolation::FeatureIsolation,
    &feature_msrv::FeatureMsrv,
    &file_size::FileSize,
    &frozen_contract::BackendExtension,
    &frozen_contract::FrozenContracts,
    &frozen_contract::ProgramWireFields,
    &frozen_contract::ReadbackRing,
    &gate_canon::GateCanon,
    &gpu_loudness::GpuLoudness,
    &hot_path::BlockingWait,
    &hot_path::ReserveArgument,
    &hot_path::UnboundedCache,
    &hot_path::UnboundedRead,
    &hot_path_scan::HotPathScan,
    &hygiene_matrix::HygieneMatrix,
    &inventory_walk::InventoryWalk,
    &layering::Layering,
    &layering::NeutralCrates,
    &lego_quick::LegoQuick,
    &lint_hygiene::ExpectHasFix,
    &lint_hygiene::OneLintPolicy,
    &lint_hygiene::UnsafeBudget,
    &lint_hygiene::UnsafeJustification,
    &lockfile::LockfileClean,
    &manifest_contract::InternalDepVersions,
    &manifest_contract::PathDepsResolve,
    &manifest_contract::WorkspaceMembership,
    &metal_parity::MetalParity,
    &op_names::OpNames,
    &oracle_sweeps::OracleSweeps,
    &parity_testing::ParityTestingIsolated,
    &placement_predicate::PlacementPredicates,
    &platform_boundary::PlatformBoundary,
    &platform_docs::PlatformConsumerDocs,
    &proptest_coverage::ProptestCoverage,
    &public_api::PublicApiSnapshot,
    &public_api_paths::PublicApiPaths,
    &repo_hygiene::RepoHygiene,
    &repo_hygiene::SingleBacklog,
    &script_ledger::ScriptLedger,
    &shader_source::ShaderSource,
    &source_reachability::IncludeIsNotAModule,
    &source_reachability::SourceParses,
    &source_reachability::SourceReachability,
    &test_material::TestMaterialPlacement,
    &testing_guides::TestingGuides,
    &unification::Unification,
    &wire_determinism::WireDeterminism,
    &workspace_build::WorkspaceCheck,
    &workspace_build::WorkspaceClippy,
    &workspace_build::WorkspaceDocs,
    &workspace_build::WorkspaceTests,
];
