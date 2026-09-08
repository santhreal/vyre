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
pub struct EvidenceProvenance;

impl crate::gate::GateBehavior for EvidenceProvenance {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let mut report = Report::clean();
        let tracked = git(ctx, &["ls-files", "-z", "--", EVIDENCE_DIR])?;
        let carriers = carrier_commits(ctx)?;
        let mut uncommitted = 0_usize;
        let mut unstamped = 0_usize;
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
            let (Some(fingerprint), _) = crate::artifact_gate::split_provenance(&content) else {
                unstamped += 1;
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
            "{judged} committed fingerprint(s) judged, {unstamped} artifact(s) carry none, \
             {uncommitted} not committed yet"
        ));
        Ok(report)
    }
}

/// The newest commit touching each committed artifact under [`EVIDENCE_DIR`].
///
/// One history walk answers every path. `git log -1 -- <path>` per artifact
/// walks the whole history again for each one, which is what made this gate
/// cost tens of minutes on a network checkout: the commits arrive newest-first,
/// so the first mention of a path is the commit that carries it.
fn carrier_commits(ctx: &GateCtx) -> Result<BTreeMap<String, String>, GateError> {
    let log = git(
        ctx,
        &[
            "-c",
            "core.quotePath=false",
            "log",
            "--format=%H",
            "--name-only",
            "--no-renames",
            "--",
            EVIDENCE_DIR,
        ],
    )?;
    let mut carriers = BTreeMap::new();
    let mut commit = String::new();
    for line in String::from_utf8_lossy(&log).lines() {
        if line.is_empty() {
            continue;
        }
        if is_commit_id(line) {
            commit = line.to_string();
            continue;
        }
        if commit.is_empty() {
            return Err(GateError::new(
                format!("git log reported path `{line}` before any commit"),
                "read the log as commit-then-paths records",
            ));
        }
        carriers
            .entry(line.to_string())
            .or_insert_with(|| commit.clone());
    }
    Ok(carriers)
}

/// Whether a log line is a commit id rather than a path.
///
/// A path under `release/evidence` cannot be 40 hex characters, so the two
/// record kinds are distinguishable without a separator.
fn is_commit_id(line: &str) -> bool {
    line.len() == 40 && line.bytes().all(|byte| byte.is_ascii_hexdigit())
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
    fn write_artifact(root: &Path, name: &str, fingerprint: &str) {
        let dir = root.join(EVIDENCE_DIR).join("metadata");
        std::fs::create_dir_all(&dir).expect("Fix: create the evidence directory.");
        std::fs::write(
            dir.join(name),
            format!(
                "{{\n  \"source_fingerprint\": \"{fingerprint}\",\n  \"schema_version\": 1\n}}\n"
            ),
        )
        .expect("Fix: write the evidence artifact.");
    }

    fn findings(root: &Path) -> String {
        EvidenceProvenance
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

    #[test]
    fn an_artifact_carrying_no_fingerprint_is_left_to_the_artifact_gate() {
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

        let report = EvidenceProvenance
            .run(&GateCtx::new(dir.path().to_path_buf(), Vec::new()))
            .expect("Fix: the fixture checkout must be judgeable.");

        assert!(
            report.findings.is_empty(),
            "Fix: an artifact carrying no fingerprint is the artifact gate's to report; {}",
            report.finding_messages()
        );
        assert!(
            report
                .notes
                .iter()
                .any(|note| note.contains("1 artifact(s) carry none")),
            "Fix: the count must state what was left unjudged; notes={:?}",
            report.notes
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

        let report = EvidenceProvenance
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
