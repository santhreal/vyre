//! Hold the release workload matrix to the benchmark case registry.
//!
//! This gate used to shell into `cargo run -p vyre-bench -- release-matrix`,
//! which meant checking a derived artifact rebuilt the benchmark harness. The
//! matrix is a pure function of the case registry and the bench target manifest,
//! both of which `vyre-bench` exposes as a library, so the derivation happens in
//! process here and no gate spawns a compiler.
//!
//! This gate is also the only writer of the document. The benchmark generator
//! used to write it a second way, by spawning `vyre-bench release-matrix
//! --output`, which serializes the same struct with no provenance head: the
//! committed file then held whichever of the two forms ran last, and a
//! comparison against either one failed on the other.

use std::path::Path;

use vyre_bench::release_matrix::{build_release_matrix, ReleaseWorkloadMatrix};
use xtask::artifact_gate::Inspection;
use xtask::gate::Finding;

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

/// Write the matrix from the case registry, returning everything that found.
///
/// The benchmark generator regenerates the matrix before it measures against it,
/// and it is the same derivation as the gate: one producer, reached from both
/// callers, so the document cannot depend on which one ran.
pub(crate) fn regenerate(root: &Path) -> Vec<Finding> {
    let inspection = inspect();
    let mut findings = inspection.findings;
    findings.extend(xtask::artifact_gate::settle(
        root,
        "release-workload-matrix",
        &inspection.artifacts,
        true,
    ));
    findings
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
            format!(
                "release workload family `{family}` declares no CPU state-of-the-art 100x case"
            ),
            "Register a cpu-sota contract case for the family, or remove it from the required \
             100x set.",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The committed matrix is what this gate would write today.
    ///
    /// A second producer wrote the same document without a provenance head, so a
    /// comparison against the committed file could only pass on the tree that
    /// happened to write it last, and the test that made that comparison was
    /// deleted rather than fixed. The head is lifted off before comparing,
    /// because it names the tree the body was recorded from and not the tree
    /// running this test.
    #[test]
    fn the_committed_matrix_body_is_what_the_registry_derives() {
        let root = xtask::checkout::checkout_root();
        let committed = std::fs::read_to_string(root.join(ARTIFACT))
            .expect("Fix: the committed release workload matrix must be readable.");
        let (fingerprint, body) = xtask::artifact_gate::split_provenance(&committed);
        assert!(
            fingerprint.is_some(),
            "Fix: the recorded matrix must name the tree it was recorded from."
        );

        let registry = vyre_bench::registry::collect_all();
        let derived = xtask::output_arg::render_evidence_json(&build_release_matrix(&registry))
            .expect("Fix: the derived matrix must serialize.");

        assert_eq!(
            body.trim_end(),
            derived.trim_end(),
            "Fix: regenerate the matrix with `release-workload-matrix --write`; the committed body \
             disagrees with the case registry."
        );
    }

    /// WHY: The authoritative descriptor and workload-matrix producer must agree on
    /// the exact output path so comparison is immutable and write mutations
    /// are never undeclared.
    #[test]
    fn authoritative_descriptor_declares_exact_release_workload_matrix_artifact() {
        let descriptor = xtask::gate_metadata::descriptor_by_name("release-workload-matrix");
        let mut expected: Vec<&str> = vec![ARTIFACT];
        expected.sort_unstable();
        let mut actual: Vec<&str> = descriptor.artifacts.to_vec();
        actual.sort_unstable();
        assert_eq!(
            actual, expected,
            "Fix: release-workload-matrix gate descriptor must declare exactly the canonical workload evidence artifact (`{ARTIFACT}`)"
        );
    }
}
