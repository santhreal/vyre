//! Every committed evidence artifact names the source the commit carrying it
//! holds.
//!
//! A recorded `source_fingerprint` used to be judged on shape alone: it named a
//! commit, and nothing checked that commit against the tree the artifact was
//! committed into. Two artifacts were recorded against a commit hundreds of
//! non-evidence files behind the one that carries them, and 23 more carried a
//! worktree digest over `git status` output, which no reader can rebuild once
//! the changes it described are committed.
//!
//! This gate reads the committed copy of each artifact, not the worktree copy: a
//! regenerated artifact that is not committed yet has no commit to be checked
//! against, and the artifact gate already compares its body against the tree.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::source_provenance;

/// Where release evidence lives.
const EVIDENCE_DIR: &str = "release/evidence";

/// The command that re-records an artifact against the current tree.
const FIX: &str = "regenerate the artifact with its owning gate's `--write` on a tree whose \
                   remaining changes the next commit captures, and commit both together";

/// A committed fingerprint resolves against the commit that carries it.
pub struct CommittedEvidenceProvenance;

impl crate::gate::GateBehavior for CommittedEvidenceProvenance {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let tracked = git(ctx, &["ls-files", "-z", "--", EVIDENCE_DIR])?;
        let carriers = carrier_commits(ctx)?;
        let mut uncommitted = 0_usize;
        let mut unattributable = 0_usize;
        let mut judged = 0_usize;
        let mut committed: Vec<(String, String)> = Vec::new();
        for path in tracked
            .split(|byte| *byte == 0)
            .filter(|path| !path.is_empty())
        {
            let path = String::from_utf8_lossy(path).into_owned();
            match carriers.get(&path) {
                Some(carrier) => committed.push((path, carrier.clone())),
                None => uncommitted += 1,
            }
        }
        let objects: Vec<String> = committed
            .iter()
            .map(|(path, carrier)| format!("{carrier}:{path}"))
            .collect();
        let contents = source_provenance::committed_texts(&ctx.root, &objects);
        for ((path, carrier), content) in committed.iter().zip(contents) {
            let Some(content) = content else {
                return Err(GateError::new(
                    format!("`{carrier}:{path}` could not be read out of the object store"),
                    "fetch the history that carries the committed evidence",
                ));
            };
            if !path.ends_with(".json") {
                continue;
            }
            let (record, _) = crate::artifact_gate::split_provenance(&content);
            let record = match record {
                Ok(record) => record,
                Err(issue) => {
                    report.find(Finding::in_file(
                        PathBuf::from(path),
                        format!("`{path}` {}", issue.predicate()),
                        FIX,
                    ));
                    continue;
                }
            };
            let Some(fingerprint) = record.tree.source_fingerprint() else {
                unattributable += 1;
                continue;
            };
            judged += 1;
            if let Err(verdict) =
                source_provenance::resolves_against(&ctx.root, fingerprint, carrier)
            {
                report.find(Finding::in_file(PathBuf::from(path), verdict, FIX));
            }
        }
        report.cover_complete("committed evidence fingerprints", judged);
        report.note(format!(
            "{judged} committed fingerprint(s) judged, {unattributable} artifact(s) record that \
             nothing can attribute them, {uncommitted} not committed yet"
        ));
        Ok(report)
    }
}

/// The newest commit touching each committed artifact under [`EVIDENCE_DIR`],
/// as this gate's error type.
fn carrier_commits(ctx: &GateCtx) -> Result<BTreeMap<String, String>, GateError> {
    source_provenance::carrier_commits(&ctx.root, EVIDENCE_DIR).map_err(|issue| {
        GateError::new(
            issue,
            "judge a checkout with its history present; a shallow clone cannot resolve the \
             commit an artifact was recorded against",
        )
    })
}

/// Run one git command in the judged tree, or name what could not be read.
fn git(ctx: &GateCtx, arguments: &[&str]) -> Result<Vec<u8>, GateError> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(&ctx.root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("git {arguments:?} could not run: {error}"),
                "install git; evidence provenance is a claim about a checkout",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "git {arguments:?} failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "judge a checkout with its history present; a shallow clone cannot resolve the \
             commit an artifact was recorded against",
        ));
    }
    Ok(output.stdout)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::fixture_checkout;
    use crate::gate::GateBehavior;

    /// Write an artifact stamped the way the recorder stamps one.
    ///
    /// The stamp goes on through `evidence_record::stamp`, so a change to
    /// where the block sits or what it holds reaches these tests instead of
    /// leaving them agreeing with a shape nothing writes any more.
    fn write_artifact(root: &Path, name: &str, fingerprint: &str) {
        let dir = root.join(EVIDENCE_DIR).join("metadata");
        std::fs::create_dir_all(&dir).expect("Fix: create the evidence directory.");
        let mut provenance = crate::evidence_record::EvidenceProvenance::capture(
            root,
            crate::evidence_record::MeasurementRecord::Unattributable {
                reason: "the fixture records no measurement".to_string(),
                recapture: "run the owning gate with --write".to_string(),
            },
        )
        .expect("Fix: the fixture checkout must name a tree.");
        provenance.tree = recorded_against(fingerprint);
        let body = "{\n  \"schema_version\": 1\n}\n";
        let stamped = crate::evidence_record::stamp(body, &provenance)
            .expect("Fix: stamp the fixture artifact.");
        std::fs::write(dir.join(name), stamped).expect("Fix: write the evidence artifact.");
    }

    /// A tree record naming exactly the fingerprint a case wants judged.
    fn recorded_against(fingerprint: &str) -> crate::evidence_record::TreeRecord {
        let commit = source_provenance::recorded_commit(fingerprint)
            .expect("Fix: a fixture fingerprint names a commit.")
            .to_string();
        crate::evidence_record::TreeRecord::Attributed {
            branch: "fixture".to_string(),
            commit,
            commit_timestamp: "0".to_string(),
            parent_commit: String::new(),
            dirty: fingerprint.contains(":dirty=true"),
            source_fingerprint: fingerprint.to_string(),
        }
    }

    fn findings(root: &Path) -> String {
        CommittedEvidenceProvenance
            .run(&GateCtx::new(root.to_path_buf(), Vec::new()))
            .expect("Fix: the fixture checkout must be judgeable.")
            .finding_messages()
    }

    #[test]
    fn an_artifact_recorded_against_the_source_its_commit_carries_is_admitted() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        fixture_checkout::seeded(dir.path());
        let base = fixture_checkout::head(dir.path());
        write_artifact(
            dir.path(),
            "matrix.json",
            &format!("git:{base}:dirty=false"),
        );
        fixture_checkout::commit_worktree(dir.path(), "record evidence");

        assert!(
            findings(dir.path()).is_empty(),
            "Fix: an artifact whose commit changes nothing but evidence records a clean tree."
        );
    }

    #[test]
    fn an_artifact_recorded_before_a_source_change_its_own_commit_carries_is_a_finding() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        fixture_checkout::seeded(dir.path());
        let base = fixture_checkout::head(dir.path());
        std::fs::write(dir.path().join("tracked.txt"), "changed\n")
            .expect("Fix: change the tracked source.");
        write_artifact(
            dir.path(),
            "matrix.json",
            &format!("git:{base}:dirty=false"),
        );
        fixture_checkout::commit_worktree(dir.path(), "record evidence");

        let found = findings(dir.path());

        assert!(
            found.contains("does not name the source"),
            "Fix: a fingerprint that omits a source change its own commit carries names another \
             tree, and the verdict must say so; found={found}"
        );
    }

    #[test]
    fn a_dirty_recording_the_commit_captures_is_admitted() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        fixture_checkout::seeded(dir.path());
        let base = fixture_checkout::head(dir.path());
        std::fs::write(dir.path().join("tracked.txt"), "changed\n")
            .expect("Fix: change the tracked source.");
        let fingerprint = source_provenance::capture(dir.path())
            .expect("Fix: a dirty checkout still names a commit.");
        assert!(
            fingerprint.starts_with(&format!("git:{base}:dirty=true:worktree=")),
            "Fix: the recorder must record the uncommitted change; fingerprint={fingerprint}"
        );
        write_artifact(dir.path(), "matrix.json", &fingerprint);
        fixture_checkout::commit_worktree(dir.path(), "record evidence");

        assert!(
            findings(dir.path()).is_empty(),
            "Fix: a worktree digest a reader can rebuild from the commit must resolve."
        );
    }

    /// WHY: this gate's claim is that every committed evidence artifact names
    /// the source the commit carrying it holds. An artifact with no block at
    /// all names nothing, so it is a finding here rather than a silent row in
    /// a count. The stamped artifact beside it proves the finding is about the
    /// unstamped file and not about the run.
    ///
    /// What it does not catch: an artifact whose block is present and whose
    /// contents are a lie about a device. That is the corpus gate's judgement.
    #[test]
    fn an_artifact_carrying_no_provenance_block_is_a_finding() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        fixture_checkout::seeded(dir.path());
        let base = fixture_checkout::head(dir.path());
        write_artifact(
            dir.path(),
            "matrix.json",
            &format!("git:{base}:dirty=false"),
        );
        std::fs::write(
            dir.path().join(EVIDENCE_DIR).join("metadata/plain.json"),
            "{\n  \"schema_version\": 1\n}\n",
        )
        .expect("Fix: write an unstamped artifact.");
        fixture_checkout::commit_worktree(dir.path(), "record evidence");

        let report = CommittedEvidenceProvenance
            .run(&GateCtx::new(dir.path().to_path_buf(), Vec::new()))
            .expect("Fix: the fixture checkout must be judgeable.");
        let found = report.finding_messages();

        assert!(
            found.contains("plain.json") && found.contains("carries no provenance block"),
            "Fix: an unstamped committed artifact attributes nothing and the verdict must name \
             it; found={found}"
        );
        assert!(
            !found.contains("matrix.json"),
            "Fix: the stamped artifact beside it resolves and must stay silent; found={found}"
        );
    }

    #[test]
    fn each_artifact_is_judged_against_the_commit_that_carries_it() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        fixture_checkout::seeded(dir.path());
        let first_base = fixture_checkout::head(dir.path());
        write_artifact(
            dir.path(),
            "first.json",
            &format!("git:{first_base}:dirty=false"),
        );
        fixture_checkout::commit_worktree(dir.path(), "record the first artifact");

        let second_base = fixture_checkout::head(dir.path());
        write_artifact(
            dir.path(),
            "second.json",
            &format!("git:{second_base}:dirty=false"),
        );
        fixture_checkout::commit_worktree(dir.path(), "record the second artifact");

        let report = CommittedEvidenceProvenance
            .run(&GateCtx::new(dir.path().to_path_buf(), Vec::new()))
            .expect("Fix: the fixture checkout must be judgeable.");

        assert!(
            report.findings.is_empty(),
            "Fix: an artifact recorded one commit ago is judged against that commit, not against \
             the newest one; {}",
            report.finding_messages()
        );
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("2 committed fingerprint(s) judged")),
            "Fix: both artifacts must be judged; notes={:?}",
            report.notes
        );
    }
}
