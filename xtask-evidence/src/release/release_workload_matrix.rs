//! Hold the release workload matrix to the benchmark case registry.
//!
//! This gate used to shell into `cargo run -p vyre-bench -- release-matrix`,
//! which meant checking a derived artifact rebuilt the benchmark harness. The
//! matrix is a pure function of the case registry and the bench target manifest,
//! both of which `vyre-bench` exposes as a library, so the derivation happens in
//! process here and no gate spawns a compiler.

use vyre_bench::release_matrix::{build_release_matrix, ReleaseWorkloadMatrix};
use xtask::artifact_gate::Inspection;

/// The artifact this gate owns, relative to the workspace root.
const ARTIFACT: &str = "release/evidence/benchmarks/release-workload-matrix.json";

xtask::artifact_gate! {
    /// Holds the workload matrix to the benchmark cases the registry declares.
    ReleaseWorkloadMatrixGate,
    name: "release-workload-matrix",
    help: "Rebuild release/evidence/benchmarks/release-workload-matrix.json from the benchmark case \
       registry and the bench target manifest, and report each line the committed artifact \
       disagrees on. Proves every required release workload family matches at least one \
       registered case, that each family naming a CPU state-of-the-art baseline has one, and \
       that the matrix carries no blockers. Proves nothing about any measurement: no benchmark \
       runs here and no artifact any family names is read.",
    inspect: |ctx| inspect(),
}

/// What the case registry declares, and the artifact recording it.
fn inspect() -> Inspection {
    let mut inspection = Inspection::new();
    let registry = vyre_bench::registry::collect_all();
    let matrix = build_release_matrix(&registry);
    report_blockers(&matrix, &mut inspection);
    inspection.generates(ARTIFACT, &matrix);
    inspection
}

/// Every judgement the matrix makes about the case registry.
///
/// `blockers` was a field in a file nobody opened. The release workflow read it
/// with `jq` after the fact, so a blocked matrix was still a zero exit from the
/// command that wrote it.
fn report_blockers(matrix: &ReleaseWorkloadMatrix, inspection: &mut Inspection) {
    for blocker in &matrix.blockers {
        inspection.blocked(
            ARTIFACT,
            blocker.clone(),
            "Register the benchmark case or bench target the sentence names, or drop the release \
             workload family that requires it.",
        );
    }
    if matrix.matched_required_families < matrix.required_closed_families {
        inspection.blocked(
            ARTIFACT,
            format!(
                "{} of {} required release workload families matched a benchmark case",
                matrix.matched_required_families, matrix.required_closed_families
            ),
            "Register a case for each unmatched family. A release workload with no case is a \
             workload nothing measures.",
        );
    }
    for family in &matrix.missing_required_cpu_sota_100x_families {
        inspection.blocked(
            ARTIFACT,
            format!("release workload family `{family}` declares no CPU state-of-the-art 100x case"),
            "Register a cpu-sota contract case for the family, or remove it from the required \
             100x set.",
        );
    }
}
