//! What the op matrix claims about a backend, against what the recorded run of
//! that backend observed.
//!
//! `docs/optimization/OP_MATRIX.toml` is generated from the live registry, so
//! every cell is reproducible. Reproducible is not the same as true: the
//! generator has only host-side facts, and a backend's answer comes from a
//! device. A cell reading `supported` for all 349 operations is a claim nobody
//! checked, and a cell reading `not_applicable` is a refusal nobody checked
//! either. One generator briefly wrote seven of the latter by mistaking what a
//! program needs for what a backend refuses, and every other gate stayed green
//! because they all judge the matrix against itself.
//!
//! The conformance run is the observation. `release/evidence/conformance/`
//! carries one artifact per backend, each with a `pairs` array recording
//! whether that operation executed against `vyre-reference` on that backend.
//! This compares the two, in both directions: a claim with no observation
//! behind it, and an observation the matrix denies. Either one means the
//! release evidence and the parity suite disagree, which is the state this
//! exists to make impossible to ship.

use std::collections::BTreeMap;
use std::path::Path;

use xtask::release::conformance_op_matrix::OpMatrixReleaseBackendSpec;

/// Backend as the matrix spells it, paired with its recorded artifact and the
/// `backend_id` that artifact uses.
///
/// The three spellings differ for the reference backend: the matrix column is
/// `reference`, the file is `reference-conformance.json` and the runner writes
/// `cpu-ref` inside it. Keeping all three together is what stops a rule from
/// silently matching nothing.
const RECORDED_BACKENDS: &[(&str, &str, &str)] = &[
    (
        "reference",
        "release/evidence/conformance/reference-conformance.json",
        "cpu-ref",
    ),
    (
        "cuda",
        "release/evidence/conformance/cuda-conformance.json",
        "cuda",
    ),
    (
        "wgpu",
        "release/evidence/conformance/wgpu-conformance.json",
        "wgpu",
    ),
];

/// Statuses that assert the backend runs the operation.
const CLAIMS_SUPPORT: &str = "supported";

/// Blockers for every disagreement between the matrix and the recorded runs.
pub(super) fn disagreements(root: &Path, specs: &[OpMatrixReleaseBackendSpec]) -> Vec<String> {
    let mut blockers = Vec::new();
    for (backend, artifact, recorded_id) in RECORDED_BACKENDS {
        let observed = match read_pairs(&root.join(artifact), recorded_id) {
            Ok(observed) => observed,
            Err(problem) => {
                blockers.push(format!(
                    "cannot judge OP_MATRIX `{backend}` cells against a recorded run: {problem}"
                ));
                continue;
            }
        };
        for spec in specs.iter().filter(|spec| spec.backend == *backend) {
            match (spec.status.as_str(), observed.get(spec.op_id.as_str())) {
                (CLAIMS_SUPPORT, Some(true)) => {}
                (CLAIMS_SUPPORT, Some(false)) => blockers.push(format!(
                    "OP_MATRIX claims `{}:{backend}` is supported, and the recorded {backend} \
                     conformance run reports it failing",
                    spec.op_id
                )),
                (CLAIMS_SUPPORT, None) => blockers.push(format!(
                    "OP_MATRIX claims `{}:{backend}` is supported, and the recorded {backend} \
                     conformance run does not cover it",
                    spec.op_id
                )),
                (status, Some(true)) => blockers.push(format!(
                    "OP_MATRIX declares `{}:{backend}` as `{status}`, and the recorded {backend} \
                     conformance run observes it passing",
                    spec.op_id
                )),
                (_, Some(false) | None) => {}
            }
        }
    }
    blockers
}

/// Every operation the recorded run reports on, and whether it passed.
///
/// A pair naming a different backend is a mixed-up artifact rather than a
/// coverage gap, so it is reported instead of ignored.
fn read_pairs(path: &Path, recorded_id: &str) -> Result<BTreeMap<String, bool>, String> {
    let text = super::read_text_bounded(path)
        .map_err(|error| format!("{} is unreadable: {error}", path.display()))?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("{} is not JSON: {error}", path.display()))?;
    let pairs = document
        .get("pairs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} records no `pairs` array", path.display()))?;
    let mut observed = BTreeMap::new();
    for pair in pairs {
        let op_id = pair
            .get("op_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| format!("{} has a pair with no `op_id`", path.display()))?;
        let backend_id = pair
            .get("backend_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        if backend_id != recorded_id {
            return Err(format!(
                "{} records `{op_id}` under backend `{backend_id}`, not `{recorded_id}`",
                path.display()
            ));
        }
        let passed = pair
            .get("passed")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                format!(
                    "{} has a pair for `{op_id}` with no `passed`",
                    path.display()
                )
            })?;
        observed.insert(op_id.to_string(), passed);
    }
    if observed.is_empty() {
        return Err(format!("{} records zero pairs", path.display()));
    }
    Ok(observed)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn spec(op_id: &str, backend: &str, status: &str) -> OpMatrixReleaseBackendSpec {
        OpMatrixReleaseBackendSpec {
            op_id: op_id.to_string(),
            backend: backend.to_string(),
            status: status.to_string(),
            test_paths: Vec::new(),
            unreadable_test_paths: Vec::new(),
            test_case_classes: BTreeSet::new(),
        }
    }

    /// Write the three artifacts, giving `wgpu` the pairs supplied and the
    /// other two a passing pair for every operation named.
    fn recorded(pairs: &[(&str, bool)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("a temp directory");
        std::fs::create_dir_all(root.path().join("release/evidence/conformance"))
            .expect("the evidence directory");
        for (_backend, artifact, recorded_id) in RECORDED_BACKENDS {
            let rows = pairs
                .iter()
                .map(|(op_id, passed)| {
                    serde_json::json!({
                        "op_id": op_id,
                        "backend_id": recorded_id,
                        "passed": if *recorded_id == "wgpu" { *passed } else { true },
                    })
                })
                .collect::<Vec<_>>();
            let document = serde_json::json!({ "pairs": rows });
            std::fs::write(
                root.path().join(artifact),
                serde_json::to_string(&document).expect("the document serializes"),
            )
            .expect("the artifact is written");
        }
        root
    }

    /// WHY: this is the state the whole rule exists for. Every other gate reads
    /// the matrix, so a cell claiming a backend runs an operation was checked
    /// only against itself and stayed green whatever the device did.
    #[test]
    fn a_claim_the_recorded_run_does_not_cover_is_a_blocker() {
        let root = recorded(&[("op::covered", true)]);
        let blockers = disagreements(root.path(), &[spec("op::absent", "wgpu", "supported")]);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].contains("does not cover it"), "{blockers:?}");
    }

    /// WHY: a failing pair and an absent pair are different defects with
    /// different corrective actions, and collapsing them sends a reader to
    /// look for missing coverage when the run said the operation is broken.
    #[test]
    fn a_claim_the_recorded_run_disproves_is_a_blocker() {
        let root = recorded(&[("op::broken", false)]);
        let blockers = disagreements(root.path(), &[spec("op::broken", "wgpu", "supported")]);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].contains("reports it failing"), "{blockers:?}");
    }

    /// WHY: the direction that actually shipped. A generator wrote seven
    /// `not_applicable` cells for operations the recorded wgpu run observes
    /// passing, and nothing failed, because no rule read the observation.
    #[test]
    fn a_refusal_the_recorded_run_contradicts_is_a_blocker() {
        let root = recorded(&[("op::runs", true)]);
        let blockers = disagreements(root.path(), &[spec("op::runs", "wgpu", "not_applicable")]);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].contains("observes it passing"), "{blockers:?}");
    }

    /// WHY: agreement in both directions must be silent, or the rule reports on
    /// every release and gets switched off. A refusal the run also refuses is
    /// agreement, not a finding.
    #[test]
    fn agreement_in_either_direction_is_silent() {
        let root = recorded(&[("op::runs", true), ("op::refused", false)]);
        let blockers = disagreements(
            root.path(),
            &[
                spec("op::runs", "wgpu", "supported"),
                spec("op::refused", "wgpu", "not_applicable"),
                spec("op::runs", "cuda", "supported"),
                spec("op::runs", "reference", "supported"),
            ],
        );
        assert_eq!(blockers, Vec::<String>::new());
    }

    /// WHY: a missing artifact must fail closed. Treating it as "nothing to
    /// compare" makes deleting the evidence the cheapest way to pass.
    #[test]
    fn a_missing_recorded_run_is_a_blocker_rather_than_a_pass() {
        let root = tempfile::tempdir().expect("a temp directory");
        let blockers = disagreements(root.path(), &[spec("op::runs", "wgpu", "supported")]);
        assert_eq!(blockers.len(), RECORDED_BACKENDS.len(), "{blockers:?}");
        assert!(
            blockers
                .iter()
                .all(|blocker| blocker.contains("cannot judge OP_MATRIX")),
            "{blockers:?}"
        );
    }

    /// WHY: `reference` in the matrix is `cpu-ref` in the artifact. Matching on
    /// the matrix spelling would compare against zero pairs and report every
    /// reference cell as uncovered, which is the failure mode that gets a rule
    /// reverted rather than fixed.
    #[test]
    fn the_reference_column_is_matched_to_the_cpu_ref_artifact() {
        let root = recorded(&[("op::runs", true)]);
        let blockers = disagreements(root.path(), &[spec("op::runs", "reference", "supported")]);
        assert_eq!(blockers, Vec::<String>::new());
    }

    /// WHY: an artifact whose pairs name another backend has been copied or
    /// renamed, and reading it as coverage would certify one device with
    /// another device's results.
    #[test]
    fn an_artifact_recording_another_backend_is_rejected() {
        let root = recorded(&[("op::runs", true)]);
        let document = serde_json::json!({
            "pairs": [{ "op_id": "op::runs", "backend_id": "metal", "passed": true }]
        });
        std::fs::write(
            root.path()
                .join("release/evidence/conformance/wgpu-conformance.json"),
            serde_json::to_string(&document).expect("the document serializes"),
        )
        .expect("the artifact is written");
        let blockers = disagreements(root.path(), &[spec("op::runs", "wgpu", "supported")]);
        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].contains("not `wgpu`"), "{blockers:?}");
    }
}
