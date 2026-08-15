//! xtask subcommands that read benchmark and release evidence provenance.
//!
//! A recorded measurement only describes the tree it was taken on. Deciding
//! whether it still does means fingerprinting the source and reading the host
//! devices, which is what `vyre-bench` owns and why this crate links it. The
//! release checks that only read manifests stay in `xtask`.

pub mod bench;
pub mod release;
#[cfg(test)]
pub(crate) mod report_fixture;

/// Every gate this crate implements, in the shape the dispatcher runs them.
///
/// `xtask` links no vyre crate, so it cannot call these directly. It builds
/// this crate's binary and runs it, and the binary resolves the name against
/// this table, runs the gate and prints one `Report` on stdout.
pub static GATES: &[&dyn xtask::gate::Gate] = &[
    &release::backend_matrix::BackendMatrixGate,
    &bench::bench_crossback::BenchCrossbackGate,
    &bench::bench_release::BenchReleaseGate,
    &bench::release_benchmarks::ReleaseBenchmarksGate,
    &release::release_evidence::ReleaseEvidenceGate,
    &release::release_workload_matrix::ReleaseWorkloadMatrixGate,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: this crate is built and run only when `xtask` decides a gate
    /// belongs here, so its table and the assignment in `xtask` are two
    /// declarations that must agree exactly, once each, and answer to nothing
    /// else. The checker derives both sides at call time.
    #[test]
    fn the_gate_table_agrees_with_the_assignment_in_xtask() {
        assert_eq!(
            xtask::subcommands::delegate_table_problems("xtask-evidence", GATES),
            Vec::<String>::new()
        );
    }
}
