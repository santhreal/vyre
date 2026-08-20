//! xtask subcommands that read benchmark and release evidence provenance.
//!
//! A recorded measurement only describes the tree it was taken on. Deciding
//! whether it still does means fingerprinting the source and reading the host
//! devices, which is what `vyre-bench` owns and why this crate links it. The
//! release checks that only read manifests stay in `xtask`.

pub mod bench;
pub mod gpu_release_floor;
pub mod release;
#[cfg(test)]
pub(crate) mod report_fixture;

/// Every gate this crate implements, in the shape the dispatcher runs them.
///
/// `xtask` links no vyre crate, so it cannot call these directly. It builds
/// this crate's binary and runs it, and the binary resolves the name against
/// this table, runs the gate and prints one `Report` on stdout.
pub static GATES: &[(&str, &'static dyn xtask::gate::GateBehavior)] = &[
    (
        "backend-matrix",
        &release::backend_matrix::BackendMatrixGate,
    ),
    (
        "bench-crossback",
        &bench::bench_crossback::BenchCrossbackGate,
    ),
    ("bench-release", &bench::bench_release::BenchReleaseGate),
    (
        "release-benchmarks",
        &bench::release_benchmarks::ReleaseBenchmarksGate,
    ),
    (
        "release-evidence",
        &release::release_evidence::ReleaseEvidenceGate,
    ),
    (
        "release-workload-matrix",
        &release::release_workload_matrix::ReleaseWorkloadMatrixGate,
    ),
    (
        "vyre-release-gate",
        &release::vyre_release_gate::VyreReleaseGate,
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delegated_contracts() {
        xtask::delegate::assert_delegated_crate_contracts("xtask-evidence", GATES);
    }
}
