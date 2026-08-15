//! The subcommands that produce release evidence and close the release.
//!
//! Everything here either writes an artifact under `release/evidence/` or
//! reads one back and decides whether the release may proceed, together with
//! the release manifests (`release/release-train.toml`,
//! `release/repo-boundary.toml`) those decisions are made against.

pub mod conformance_evidence_semantics;
pub mod conformance_op_matrix;
pub mod conformance_workflows;
pub mod feature_matrix;
pub mod launch_contract;
pub mod launch_state;
pub mod metadata_matrix;
pub mod package_readiness;
pub mod release_conformance;
pub mod release_gate;
pub mod release_train;
pub mod repo_boundary;
pub mod version_matrix;

/// Every release gate implemented in this crate, plus the ones it delegates.
///
/// A delegated gate is not a category. It answers the same contract as a local
/// one; the only difference is that its owning package links a vyre crate, so
/// the runner builds that package and reads the report back off its stdout.
pub static GATES: &[&dyn crate::gate::Gate] = &[
    &feature_matrix::FeatureMatrixGate,
    &launch_state::LaunchStateGate,
    &metadata_matrix::MetadataMatrixGate,
    &package_readiness::PackageReadinessGate,
    &release_conformance::ReleaseConformanceGate,
    &version_matrix::VersionMatrixGate,
    &crate::gate::Delegated {
        name: "backend-matrix",
        help: "Hold the CUDA-first, WGPU-fallback backend policy to the tree and the recorded probe.",
        package: "xtask-evidence",
        generates: true,
    },
    &crate::gate::Delegated {
        name: "bench-release",
        help: "Hold the canonical release benchmark axes to the recorded CUDA evidence.",
        package: "xtask-evidence",
        generates: false,
    },
    &crate::gate::Delegated {
        name: "op-matrix",
        help: "Hold docs/optimization/OP_MATRIX.toml to the live operation registry.",
        package: "xtask-registry",
        generates: true,
    },
    &crate::gate::Delegated {
        name: "optimization-corpus",
        help: "Hold the semantic Program optimizer corpus evidence to the corpus generator.",
        package: "xtask-registry",
        generates: true,
    },
    &crate::gate::Delegated {
        name: "optimization-matrix",
        help: "Hold the optimizer pass matrix evidence to the registered passes.",
        package: "xtask-registry",
        generates: true,
    },
    &crate::gate::Delegated {
        name: "release-workload-matrix",
        help: "Hold the release workload matrix to the benchmark registry.",
        package: "xtask-evidence",
        generates: true,
    },
];
