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
use xtask::source_provenance;

/// Where the per-backend conformance runs are recorded, as one pathspec for the
/// carrier walk that dates them.
const CONFORMANCE_EVIDENCE_DIR: &str = "release/evidence/conformance";

/// Backend as the matrix spells it, paired with its recorded artifact and the
/// `backend_id` that artifact uses.
///
/// The three spellings differ for the reference backend: the matrix column is
/// `reference`, the file is `reference-conformance.json` and the runner writes
/// `cpu-ref` inside it. Keeping all three together is what stops a rule from
/// silently matching nothing.
pub const RECORDED_BACKENDS: &[(&str, &str, &str)] = &[
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
///
/// A record is dated before it is read. A device run measures the source it
/// was taken against, so a record pinned to a tree the carrier commit does not
/// hold answers a question about some other source, and every cell judged
/// against it is judged on a guess. Reading one anyway is how a passing
/// operation was reported as failing on wgpu: the record predated the emitter
/// change that fixed it. An undatable record is reported as unusable, once per
/// backend, and its cells are left unjudged rather than judged wrongly.
pub(super) fn disagreements(root: &Path, specs: &[OpMatrixReleaseBackendSpec]) -> Vec<String> {
    let mut blockers = Vec::new();
    let carriers = match source_provenance::carrier_commits(root, CONFORMANCE_EVIDENCE_DIR) {
        Ok(carriers) => carriers,
        Err(issue) => {
            blockers.push(format!(
                "cannot date the recorded conformance runs, so none of them can be read as \
                 evidence about this tree: {issue}"
            ));
            return blockers;
        }
    };
    for (backend, artifact, recorded_id) in RECORDED_BACKENDS {
        if let Err(verdict) = record_is_about_this_tree(root, artifact, carriers.get(*artifact)) {
            blockers.push(format!(
                "the recorded {backend} conformance run in `{artifact}` is not evidence about \
                 this tree, so no OP_MATRIX `{backend}` cell was judged against it: {verdict}"
            ));
            continue;
        }
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

/// Whether a recorded run measured the source the commit carrying it holds.
///
/// A record that is not committed, and one whose worktree copy differs from
/// the committed copy, has nothing here to be dated against: it was just
/// regenerated, and the artifact gate judges its body against the tree. Every
/// other record either names a source the carrier reproduces or names one
/// nothing here can rebuild, and the second is not evidence about this tree.
fn record_is_about_this_tree(
    root: &Path,
    artifact: &str,
    carrier: Option<&String>,
) -> Result<(), String> {
    let Some(carrier) = carrier else {
        return Ok(());
    };
    let Ok(worktree) = super::read_text_bounded(&root.join(artifact)) else {
        // Unreadable here means `read_pairs` reports it against its own path.
        return Ok(());
    };
    let object = format!("{carrier}:{artifact}");
    let committed = source_provenance::committed_texts(root, std::slice::from_ref(&object))
        .into_iter()
        .next()
        .flatten()
        .ok_or_else(|| format!("`{object}` could not be read out of the object store"))?;
    if worktree != committed {
        return Ok(());
    }
    let (record, _) = xtask::artifact_gate::split_provenance(&committed);
    let record = record.map_err(|issue| issue.predicate())?;
    let fingerprint = record.tree.source_fingerprint().ok_or_else(|| {
        "it records no source fingerprint, so nothing names the source it measured".to_string()
    })?;
    source_provenance::resolves_against(root, fingerprint, carrier)
}

/// Every operation the recorded run reports on, and whether it passed.
///
/// A pair naming a different backend is a mixed-up artifact rather than a
/// coverage gap, so it is reported instead of ignored.
fn read_pairs(path: &Path, recorded_id: &str) -> Result<BTreeMap<String, bool>, String> {
    #[derive(serde::Deserialize)]
    struct ConformanceEvidenceDoc {
        pairs: Option<Vec<EvidencePairRow>>,
    }

    #[derive(serde::Deserialize)]
    struct EvidencePairRow {
        op_id: Option<String>,
        #[serde(default)]
        backend_id: String,
        passed: Option<bool>,
    }

    let text = super::read_text_bounded(path)
        .map_err(|error| format!("{} is unreadable: {error}", path.display()))?;
    let document: ConformanceEvidenceDoc = serde_json::from_str(&text)
        .map_err(|error| format!("{} is not JSON: {error}", path.display()))?;
    let pairs = document
        .pairs
        .ok_or_else(|| format!("{} records no `pairs` array", path.display()))?;
    let mut observed = BTreeMap::new();
    for pair in pairs {
        let op_id = pair
            .op_id
            .ok_or_else(|| format!("{} has a pair with no `op_id`", path.display()))?;
        if pair.backend_id != recorded_id {
            return Err(format!(
                "{} records `{op_id}` under backend `{}`, not `{recorded_id}`",
                path.display(),
                pair.backend_id
            ));
        }
        let passed = pair.passed.ok_or_else(|| {
            format!(
                "{} has a pair for `{op_id}` with no `passed`",
                path.display()
            )
        })?;
        observed.insert(op_id, passed);
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
    ///
    /// Bodies only. A caller decides whether they get a stamp and a commit,
    /// because whether a record can be dated is the thing under test.
    fn write_records(root: &Path, pairs: &[(&str, bool)]) {
        std::fs::create_dir_all(root.join("release/evidence/conformance"))
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
            let body = format!(
                "{}\n",
                serde_json::to_string_pretty(&document).expect("the document serializes")
            );
            std::fs::write(root.join(artifact), body).expect("the artifact is written");
        }
    }

    /// Stamp every written record with the provenance of the tree as it stands.
    fn stamp_records(root: &Path) {
        let provenance = xtask::evidence_record::EvidenceProvenance::capture(
            root,
            xtask::evidence_record::MeasurementRecord::HostOnly,
        )
        .expect("the fixture checkout names a tree");
        for (_backend, artifact, _recorded_id) in RECORDED_BACKENDS {
            let path = root.join(artifact);
            let body = std::fs::read_to_string(&path).expect("the artifact is readable");
            let stamped =
                xtask::evidence_record::stamp(&body, &provenance).expect("the artifact stamps");
            std::fs::write(&path, stamped).expect("the artifact is written");
        }
    }

    /// A checkout carrying three records that each name the source their own
    /// commit holds, which is the only state in which a cell may be judged.
    fn recorded(pairs: &[(&str, bool)]) -> tempfile::TempDir {
        let root = tempfile::tempdir().expect("a temp directory");
        xtask::fixture_checkout::seeded(root.path());
        write_records(root.path(), pairs);
        stamp_records(root.path());
        xtask::fixture_checkout::commit_worktree(root.path(), "record conformance runs");
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
        xtask::fixture_checkout::seeded(root.path());
        let blockers = disagreements(root.path(), &[spec("op::runs", "wgpu", "supported")]);
        assert_eq!(blockers.len(), RECORDED_BACKENDS.len(), "{blockers:?}");
        assert!(
            blockers
                .iter()
                .all(|blocker| blocker.contains("cannot judge OP_MATRIX")),
            "{blockers:?}"
        );
    }

    /// WHY: the defect this dating rule closes. A wgpu record captured before
    /// an emitter change was read as current, so an operation that passes on
    /// the device was reported as failing and the corrective action a reader
    /// was handed was to fix a lowering that is already correct. A record the
    /// carrier commit does not reproduce is refused, and the wording has to
    /// send a reader to the recapture rather than to the operation.
    #[test]
    fn a_record_the_carrier_does_not_reproduce_is_refused_rather_than_read() {
        let root = tempfile::tempdir().expect("a temp directory");
        xtask::fixture_checkout::seeded(root.path());
        write_records(root.path(), &[("op::runs", false)]);
        stamp_records(root.path());
        // A source change the same commit carries, which the stamp predates.
        std::fs::write(root.path().join("tracked.txt"), "changed after the run\n")
            .expect("the tracked source changes");
        xtask::fixture_checkout::commit_worktree(root.path(), "record conformance runs");

        let blockers = disagreements(root.path(), &[spec("op::runs", "wgpu", "supported")]);

        assert_eq!(blockers.len(), RECORDED_BACKENDS.len(), "{blockers:?}");
        for blocker in &blockers {
            assert!(
                blocker.contains("is not evidence about this tree"),
                "Fix: a record the carrier does not reproduce must be refused as a record; \
                 {blocker}"
            );
            assert!(
                blocker.contains("does not name the source"),
                "Fix: the refusal must name why the record cannot be dated; {blocker}"
            );
        }
    }

    /// WHY: the two verdicts must stay distinguishable. A stale record and a
    /// device that really failed are different defects with different
    /// corrective actions, and the whole cost of the original defect was that
    /// a reader could not tell them apart. Both halves are asserted against
    /// the same claim so that collapsing either wording turns this red.
    #[test]
    fn a_stale_record_and_a_real_failure_read_differently() {
        let stale = tempfile::tempdir().expect("a temp directory");
        xtask::fixture_checkout::seeded(stale.path());
        write_records(stale.path(), &[("op::runs", false)]);
        stamp_records(stale.path());
        std::fs::write(stale.path().join("tracked.txt"), "changed after the run\n")
            .expect("the tracked source changes");
        xtask::fixture_checkout::commit_worktree(stale.path(), "record conformance runs");
        let claim = [spec("op::runs", "wgpu", "supported")];

        let refused = disagreements(stale.path(), &claim);
        let failing = disagreements(recorded(&[("op::runs", false)]).path(), &claim);

        assert!(
            refused.iter().all(|blocker| !blocker.contains("failing")),
            "Fix: a record nothing can date says nothing about whether the op failed; {refused:?}"
        );
        assert_eq!(failing.len(), 1, "{failing:?}");
        assert!(
            failing[0].contains("reports it failing")
                && !failing[0].contains("is not evidence about this tree"),
            "Fix: a record the carrier reproduces is read, and a failing pair stays a failing \
             pair; {failing:?}"
        );
    }

    /// WHY: the adversarial boundary. Dating a record against the commit
    /// carrying it must not refuse one that was just regenerated and is not
    /// committed yet, or every recapture would have to be committed before the
    /// gate that judges it could run, and a release would be gated on a commit
    /// nobody could make green first. A worktree copy that differs from the
    /// committed copy is a fresh capture, and its cells are judged.
    #[test]
    fn a_regenerated_record_is_judged_rather_than_refused() {
        let root = recorded(&[("op::runs", true)]);
        write_records(root.path(), &[("op::runs", false)]);
        stamp_records(root.path());

        let blockers = disagreements(root.path(), &[spec("op::runs", "wgpu", "supported")]);

        assert_eq!(blockers.len(), 1, "{blockers:?}");
        assert!(blockers[0].contains("reports it failing"), "{blockers:?}");
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
