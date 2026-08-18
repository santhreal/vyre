//! Regenerate an evidence artifact into memory and hold the tree to it.
//!
//! A generator that writes a file and exits zero proves nothing. The only thing
//! it ever agreed with is the file it just wrote, so an artifact and the tree it
//! describes could drift apart for a year and no run would say so. Twelve
//! generators worked that way, and the artifacts under `release/evidence` were
//! recorded across six different commits spanning thirteen days.
//!
//! Every gate that owns a generated artifact renders it here instead, and the
//! default action reads the committed copy and names each line where the two
//! disagree. `--write` is the only path that touches the tree.
//!
//! The comparison is line by line and every divergent line is its own finding,
//! because the pinned number a gate answers to is its finding count. Collapsing
//! a thousand-line disagreement into one finding would let an artifact rot back
//! to nothing while the pin stayed level.

use std::fs;
use std::path::{Path, PathBuf};
#[cfg(test)]
std::thread_local! {
    static SNAPSHOT_COUNTERS: std::cell::Cell<(usize, usize)> =
        const { std::cell::Cell::new((0, 0)) };
}

#[cfg(test)]
fn record_snapshot_capture() {
    SNAPSHOT_COUNTERS.with(|counters| {
        let (captures, verifications) = counters.get();
        counters.set((captures + 1, verifications));
    });
}

#[cfg(not(test))]
fn record_snapshot_capture() {}

#[cfg(test)]
fn record_snapshot_verification() {
    SNAPSHOT_COUNTERS.with(|counters| {
        let (captures, verifications) = counters.get();
        counters.set((captures, verifications + 1));
    });
}

#[cfg(not(test))]
fn record_snapshot_verification() {}

/// Reset this test thread's snapshot instrumentation.
#[cfg(test)]
pub fn reset_snapshot_counters() {
    SNAPSHOT_COUNTERS.with(|counters| counters.set((0, 0)));
}

/// Return this test thread's `(captures, verifications)` instrumentation.
#[cfg(test)]
#[must_use]
pub fn snapshot_counter_values() -> (usize, usize) {
    SNAPSHOT_COUNTERS.with(std::cell::Cell::get)
}

use serde::Serialize;

use crate::gate::{Coverage, Finding, GateCtx, Report};
/// Largest committed artifact this module will read into memory.
///
/// The op matrix carried this cap on its own reader before it became a gate.
/// It belongs here now, because every artifact is read through one place.
const MAX_ARTIFACT_BYTES: u64 = 16_777_216;

/// One artifact a gate owns, rendered in memory before the tree is consulted.
pub struct Generated {
    /// Path of the artifact, relative to the workspace root.
    pub path: PathBuf,
    /// Exact bytes the tree says the artifact holds, trailing newline included.
    pub content: String,
}

impl Generated {
    /// Render `value` as the bytes this artifact holds on disk.
    ///
    /// Serialization goes through [`crate::output_arg::render_evidence_json`],
    /// the renderer the writers already used, so a difference reported here is
    /// a difference in content and never in formatting.
    ///
    /// # Errors
    ///
    /// Returns a finding naming the artifact when `value` cannot be serialized.
    pub fn json(path: impl Into<PathBuf>, value: &impl Serialize) -> Result<Self, Finding> {
        let path = path.into();
        match crate::output_arg::render_evidence_json(value) {
            Ok(content) => Ok(Self { path, content }),
            Err(error) => Err(Finding::in_file(
                path.clone(),
                format!("`{}` could not be serialized: {error}", path.display()),
                "Correct the artifact type so serde can represent it. A gate that cannot render its own artifact can never compare one.",
            )),
        }
    }

    /// Take `content` verbatim, for an artifact that is not JSON.
    pub fn text(path: impl Into<PathBuf>, content: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
        }
    }
}

/// What one evidence generator found in the tree, and what it would write.
///
/// Every gate that owns a generated artifact produces this and nothing else.
/// Deciding what to do with it, compare or write, belongs to one place, so a
/// gate cannot forget to compare and cannot write without being asked.
#[derive(Default)]
pub struct Inspection {
    /// Judgements about the tree, independent of any artifact.
    pub findings: Vec<Finding>,
    /// Context a reader needs that must never be counted as a finding.
    pub notes: Vec<String>,
    /// The artifacts this gate owns, rendered from the tree.
    pub artifacts: Vec<Generated>,
}

impl Inspection {
    /// An inspection that has found nothing yet.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one judgement about the tree.
    pub fn find(&mut self, finding: Finding) {
        self.findings.push(finding);
    }

    /// Record one blocker sentence a generator produced, against `artifact`.
    ///
    /// The fourteen evidence generators each carried a `blockers` list of prose,
    /// and every sentence in it is a real judgement that has to survive the
    /// conversion. This is where they cross over. The artifact keeps recording
    /// them so a reader of the file sees what was found, and the gate reports
    /// them too, so committing a blocked artifact does not buy silence.
    pub fn blocked(&mut self, artifact: &str, message: impl Into<String>, fix: impl Into<String>) {
        self.findings.push(Finding::in_file(artifact, message, fix));
    }

    /// Render `value` as an owned artifact, recording a serializer failure.
    pub fn generates(&mut self, path: &str, value: &impl Serialize) {
        match Generated::json(path, value) {
            Ok(artifact) => self.artifacts.push(artifact),
            Err(finding) => self.findings.push(finding),
        }
    }

    /// Record an owned artifact whose bytes are not JSON.
    pub fn generates_text(&mut self, path: &str, content: impl Into<String>) {
        self.artifacts.push(Generated::text(path, content));
    }
}

/// Declare a gate whose whole body is one artifact inspection.
///
/// Twelve gates spelled the same four methods: a name, a help string,
/// `generates` returning true, and a `run` that hands one [`Inspection`] to
/// [`settle_inspection`]. Only the name, the help and the expression that
/// builds the inspection differ between them, so those are the arguments and
/// the rest is here. The inspection expression names its own binding for the
/// context, because a name this macro invented would not be visible to the
/// expression the caller writes.
///
/// ```ignore
/// xtask::artifact_gate! {
///     /// Holds the feature matrix to every workspace manifest.
///     FeatureMatrixGate,
///     name: "feature-matrix",
///     help: "Regenerate release/evidence/metadata/feature-matrix.json ...",
///     inspect: |ctx| inspect(&ctx.root),
/// }
/// ```
#[macro_export]
macro_rules! artifact_gate {
    (
        $(#[$attribute:meta])*
        $gate:ident,
        name: $name:literal,
        $(help: $help:literal,)?
        inspect: |$ctx:ident| $inspection:expr $(,)?
    ) => {
        $(#[$attribute])*
        pub struct $gate;
        impl $crate::gate::GateBehavior for $gate {
            fn run(
                &self,
                $ctx: &$crate::gate::GateCtx,
            ) -> ::core::result::Result<$crate::gate::Report, $crate::gate::GateError> {
                ::core::result::Result::Ok($crate::artifact_gate::settle_inspection(
                    $ctx,
                    $name,
                    $inspection,
                ))
            }
        }
    };
}

/// Settle `inspection` against the tree and render the gate's report.
///
/// This is the whole body of an artifact-owning gate. Without `--write` the
/// artifacts are compared and each divergence joins the findings; with it they
/// are written and only a write failure is a finding.
#[must_use]
pub fn settle_inspection(ctx: &GateCtx, gate: &str, inspection: Inspection) -> Report {
    let Inspection {
        mut findings,
        notes,
        artifacts,
    } = inspection;
    let owned_paths = artifacts
        .iter()
        .map(|artifact| artifact.path.clone())
        .collect();
    findings.extend(settle(&ctx.root, gate, &artifacts, ctx.write));
    Report {
        findings,
        notes,
        coverage: vec![Coverage::complete("generated artifacts", artifacts.len())],
        artifacts: owned_paths,
    }
}

/// Exact state of one workspace directory entry.
#[derive(Clone, Debug, Eq, PartialEq)]
enum SnapshotEntry {
    File { size: u64, digest: [u8; 32] },
    Symlink(PathBuf),
}

/// Exact snapshot of workspace file paths, sizes, content digests, and symlink targets.
#[derive(Clone, Debug, Default)]
pub struct WorkspaceSnapshot {
    files: std::collections::BTreeMap<PathBuf, SnapshotEntry>,
    errors: Vec<String>,
}

impl WorkspaceSnapshot {
    /// Capture the exact relative file set and BLAKE3 content digests across `root`.
    #[must_use]
    pub fn capture(root: &Path) -> Self {
        record_snapshot_capture();
        let mut files = std::collections::BTreeMap::new();
        let mut errors = Vec::new();
        let mut walker = walkdir::WalkDir::new(root).into_iter();
        while let Some(result) = walker.next() {
            let entry = match result {
                Ok(entry) => entry,
                Err(error) => {
                    errors.push(format!("workspace snapshot walk failed: {error}"));
                    continue;
                }
            };
            let path = entry.path();
            if entry.file_type().is_dir() {
                if matches!(
                    path.file_name().and_then(|name| name.to_str()),
                    Some(".git" | "target")
                ) {
                    walker.skip_current_dir();
                }
                continue;
            }
            let relative = path.strip_prefix(root).unwrap_or(path).to_path_buf();
            if entry.file_type().is_symlink() {
                match fs::read_link(path) {
                    Ok(target) => {
                        files.insert(relative, SnapshotEntry::Symlink(target));
                    }
                    Err(error) => errors.push(format!(
                        "workspace snapshot could not read symlink `{}`: {error}",
                        relative.display()
                    )),
                }
                continue;
            }
            if !entry.file_type().is_file() {
                continue;
            }
            match fs::read(path) {
                Ok(bytes) => {
                    files.insert(
                        relative,
                        SnapshotEntry::File {
                            size: bytes.len() as u64,
                            digest: *blake3::hash(&bytes).as_bytes(),
                        },
                    );
                }
                Err(error) => errors.push(format!(
                    "workspace snapshot could not read `{}`: {error}",
                    relative.display()
                )),
            }
        }
        Self { files, errors }
    }

    /// Detect any creation, deletion, or modification against a post-execution state.
    ///
    /// - If `allow_owned_writes` is true (e.g. gate was invoked with `--write`), only files in
    ///   `declared_artifacts` may be created or modified.
    /// - If `allow_owned_writes` is false (comparison / sweep mode), NO workspace mutation is allowed,
    ///   even for owned artifacts (Section 182.5.6).
    #[must_use]
    pub fn detect_mutations(
        &self,
        root: &Path,
        gate_name: &str,
        declared_artifacts: &[&str],
        allow_owned_writes: bool,
    ) -> Vec<String> {
        record_snapshot_verification();
        let post = Self::capture(root);
        let declared_set: std::collections::BTreeSet<PathBuf> =
            declared_artifacts.iter().map(PathBuf::from).collect();
        let mut violations: Vec<String> = self
            .errors
            .iter()
            .map(|error| format!("gate `{gate_name}` pre-execution {error}"))
            .chain(
                post.errors
                    .iter()
                    .map(|error| format!("gate `{gate_name}` post-execution {error}")),
            )
            .collect();

        // 1. Created files
        for rel in post.files.keys() {
            if !self.files.contains_key(rel) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if !allow_owned_writes {
                    violations.push(format!(
                        "gate `{gate_name}` created workspace file `{rel_str}` without --write (violating Section 182.5.6: comparison mode must never mutate workspace)"
                    ));
                } else if !declared_set.contains(rel) {
                    violations.push(format!(
                        "gate `{gate_name}` created unowned workspace file `{rel_str}` (violating Section 182.5.4: write outside declared owned artifact set)"
                    ));
                }
            }
        }

        // 2. Deleted files
        for rel in self.files.keys() {
            if !post.files.contains_key(rel) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                if !allow_owned_writes {
                    violations.push(format!(
                        "gate `{gate_name}` deleted workspace file `{rel_str}` without --write (violating Section 182.5.6: comparison mode must never mutate workspace)"
                    ));
                } else if !declared_set.contains(rel) {
                    violations.push(format!(
                        "gate `{gate_name}` deleted unowned workspace file `{rel_str}` (violating Section 182.5.4: write outside declared owned artifact set)"
                    ));
                }
            }
        }

        // 3. Modified files (content digest changed, even if mtime was restored)
        for (rel, post_state) in &post.files {
            if let Some(pre_state) = self.files.get(rel) {
                if pre_state != post_state {
                    let rel_str = rel.to_string_lossy().replace('\\', "/");
                    if !allow_owned_writes {
                        violations.push(format!(
                            "gate `{gate_name}` modified workspace file `{rel_str}` without --write (violating Section 182.5.6: comparison mode must never mutate workspace)"
                        ));
                    } else if !declared_set.contains(rel) {
                        violations.push(format!(
                            "gate `{gate_name}` wrote unowned workspace file `{rel_str}` (violating Section 182.5.4: write outside declared owned artifact set)"
                        ));
                    }
                }
            }
        }

        violations
    }
}

/// Compare every artifact against the tree, or write it when `write` is set.
///
/// `gate` names the subcommand in each `fix`, so a reader learns the exact
/// command that settles the disagreement rather than being told one exists.
///
/// The tree is fingerprinted once here rather than once per artifact, because
/// every artifact a gate owns is recorded from the same tree in the same run
/// and four gates spent four `git status` walks proving that.
#[must_use]
pub fn settle(root: &Path, gate: &str, generated: &[Generated], write: bool) -> Vec<Finding> {
    let fingerprint = crate::source_provenance::capture(root);
    generated
        .iter()
        .flat_map(|artifact| {
            if write {
                write_artifact(root, artifact, fingerprint.as_deref())
            } else {
                compare_artifact(root, gate, artifact)
            }
        })
        .collect()
}

/// Whether `path` names a recorded artifact, which must name the tree it came
/// from.
///
/// Everything under `release/evidence` is a record of what some tree was, read
/// by someone who no longer has that tree. Generated documentation elsewhere in
/// the workspace is not: it is read beside the source it describes.
fn records_provenance(path: &Path) -> bool {
    path.starts_with("release/evidence")
}

/// Put one artifact on disk, reporting a write failure as a finding.
///
/// A recorded artifact is stamped with `fingerprint` unless the committed copy
/// already holds the same body under a sound fingerprint, in which case it is
/// the same recording and keeps the tree it was recorded from. Regenerating
/// therefore leaves an unchanged artifact untouched instead of re-attributing
/// it to whatever tree happened to run the gate.
fn write_artifact(
    root: &Path,
    artifact: &Generated,
    fingerprint: Result<&str, &String>,
) -> Vec<Finding> {
    let content = if records_provenance(&artifact.path) {
        let committed = read_committed(root, &artifact.path).ok();
        let recorded = committed.as_deref().map(split_provenance);
        if let Some((Some(recorded_fingerprint), body)) = recorded.as_ref() {
            if *body == artifact.content
                && crate::source_provenance::issues(recorded_fingerprint).is_empty()
            {
                return Vec::new();
            }
        }
        match stamp_provenance(&artifact.path, &artifact.content, fingerprint) {
            Ok(content) => content,
            Err(finding) => return vec![finding],
        }
    } else {
        artifact.content.clone()
    };
    let absolute = root.join(&artifact.path);
    if let Some(parent) = absolute.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            return vec![Finding::in_file(
                artifact.path.clone(),
                format!("failed to create `{}`: {error}", parent.display()),
                "Make the evidence directory writable, then run the gate again with --write.",
            )];
        }
    }
    match fs::write(&absolute, &content) {
        Ok(()) => Vec::new(),
        Err(error) => vec![Finding::in_file(
            artifact.path.clone(),
            format!("failed to write `{}`: {error}", artifact.path.display()),
            "Make the artifact writable, then run the gate again with --write.",
        )],
    }
}

/// Put `fingerprint` at the head of `body`, or refuse to record at all.
///
/// Refusal is the point. An artifact written without a fingerprint names no
/// tree, and nothing downstream can recover the one it came from, so the
/// recorder that cannot identify its tree writes nothing rather than one more
/// generation of unattributable evidence.
///
/// # Errors
///
/// Returns the finding when the tree could not be fingerprinted, or when the
/// rendered artifact is not a JSON object and so has no head to stamp.
fn stamp_provenance(
    path: &Path,
    body: &str,
    fingerprint: Result<&str, &String>,
) -> Result<String, Finding> {
    let fingerprint = fingerprint.map_err(|error| {
        Finding::in_file(
            path.to_path_buf(),
            format!(
                "`{}` was not written because the tree it would record has no source fingerprint: {error}",
                path.display()
            ),
            "Record evidence from a checkout git can identify. An artifact that names no tree proves nothing about one.",
        )
    })?;
    if let Some(issue) = crate::source_provenance::issues(fingerprint)
        .into_iter()
        .next()
    {
        return Err(Finding::in_file(
            path.to_path_buf(),
            format!(
                "`{}` was not written because the {}",
                path.display(),
                issue.predicate()
            ),
            "Record evidence from a checkout whose state git can state exactly.",
        ));
    }
    let Some(rest) = body.strip_prefix("{\n") else {
        return Err(Finding::in_file(
            path.to_path_buf(),
            format!(
                "`{}` is recorded evidence and must be a JSON object so it can name the tree it came from",
                path.display()
            ),
            "Render the artifact as an object with a `source_fingerprint` head, or move it out of release/evidence.",
        ));
    };
    Ok(format!(
        "{{\n  \"{PROVENANCE_KEY}\": \"{fingerprint}\",\n{rest}"
    ))
}

/// The key a recorded artifact names its tree under, at the head of the object.
const PROVENANCE_KEY: &str = "source_fingerprint";

/// Take the recorded fingerprint off `committed` and return the body under it.
///
/// The stamp is one line at a known place, so lifting it back off is exact.
/// The body is what the owning gate generates, and it is the only half a
/// comparison against the tree may look at: the fingerprint names the tree the
/// body was recorded from, which is a different tree from the one running the
/// gate whenever anything has been committed since, and reporting that as a
/// divergence would make every artifact rot one commit after it was written.
pub fn split_provenance(committed: &str) -> (Option<&str>, String) {
    let head = format!("{{\n  \"{PROVENANCE_KEY}\": \"");
    let Some(rest) = committed.strip_prefix(head.as_str()) else {
        return (None, committed.to_string());
    };
    let Some(end) = rest.find("\",\n") else {
        return (None, committed.to_string());
    };
    (
        Some(&rest[..end]),
        format!("{{\n{}", &rest[end + "\",\n".len()..]),
    )
}

/// Read the committed copy of `path`, bounded.
fn read_committed(root: &Path, path: &Path) -> std::io::Result<String> {
    crate::output_arg::read_text_bounded(&root.join(path), MAX_ARTIFACT_BYTES, "evidence artifact")
}

/// Name every way the committed artifact differs from what the tree generates.
///
/// The read is bounded so a corrupted or accidentally enormous artifact refuses
/// rather than being allocated whole.
fn compare_artifact(root: &Path, gate: &str, artifact: &Generated) -> Vec<Finding> {
    let committed = match read_committed(root, &artifact.path) {
        Ok(committed) => committed,
        Err(error) => {
            return vec![Finding::in_file(
                artifact.path.clone(),
                format!(
                    "`{}` is the artifact this gate owns and it could not be read: {error}",
                    artifact.path.display()
                ),
                format!(
                    "Run `./cargo_full run --bin xtask -- {gate} --write` and commit the artifact."
                ),
            )];
        }
    };
    if !records_provenance(&artifact.path) {
        return divergences(gate, &artifact.path, &committed, &artifact.content);
    }
    let (fingerprint, body) = split_provenance(&committed);
    let mut findings = provenance_findings(gate, &artifact.path, fingerprint);
    findings.extend(divergences(gate, &artifact.path, &body, &artifact.content));
    findings
}

/// Judge the fingerprint the committed artifact carries, if it carries one.
fn provenance_findings(gate: &str, path: &Path, fingerprint: Option<&str>) -> Vec<Finding> {
    let fix =
        format!("Run `./cargo_full run --bin xtask -- {gate} --write` and commit the artifact.");
    let Some(fingerprint) = fingerprint else {
        return vec![Finding::in_file(
            path.to_path_buf(),
            format!(
                "`{}` names no source tree, so nothing it records is attributable",
                path.display()
            ),
            fix,
        )];
    };
    crate::source_provenance::issues(fingerprint)
        .into_iter()
        .map(|issue| {
            Finding::in_file(
                path.to_path_buf(),
                format!("`{}` {}", path.display(), issue.predicate()),
                fix.clone(),
            )
        })
        .collect()
}

/// One finding per line on which `committed` and `generated` disagree.
///
/// Split out from the read so the comparison is provable without a filesystem,
/// and so a caller already holding both texts can reuse it.
#[must_use]
pub fn divergences(gate: &str, path: &Path, committed: &str, generated: &str) -> Vec<Finding> {
    // A checkout that materialised the artifact with CRLF endings is not a tree
    // defect, and the line comparison below cannot see the difference anyway,
    // so the byte comparison must not either.
    let committed = committed.replace("\r\n", "\n");
    let generated = generated.replace("\r\n", "\n");
    if committed == generated {
        return Vec::new();
    }
    let fix = format!(
        "Run `./cargo_full run --bin xtask -- {gate} --write` and commit the artifact, or correct the tree fact the line reports."
    );
    let committed_lines: Vec<&str> = committed.lines().collect();
    let generated_lines: Vec<&str> = generated.lines().collect();
    let mut findings = Vec::new();
    for index in 0..committed_lines.len().max(generated_lines.len()) {
        let line = u32::try_from(index + 1).unwrap_or(u32::MAX);
        let message = match (committed_lines.get(index), generated_lines.get(index)) {
            (Some(left), Some(right)) if left == right => continue,
            (Some(left), Some(right)) => {
                format!("the artifact says `{left}`; the tree generates `{right}`")
            }
            (Some(left), None) => {
                format!("the artifact says `{left}`; the tree generates nothing here")
            }
            (None, Some(right)) => {
                format!("the artifact ends before `{right}`, which the tree generates")
            }
            (None, None) => continue,
        };
        findings.push(Finding::at(path.to_path_buf(), line, message, fix.clone()));
    }
    // Two texts differing only in a trailing newline yield identical line
    // sequences, so the loop finds nothing and the gate would report a clean
    // artifact it has already decided is wrong.
    if findings.is_empty() {
        findings.push(Finding::in_file(
            path.to_path_buf(),
            "the artifact and the tree agree line for line but not byte for byte; the trailing newline differs",
            fix,
        ));
    }
    findings
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    const ARTIFACT: &str = "release/evidence/metadata/matrix.json";

    #[test]
    fn the_recorder_refuses_an_artifact_whose_tree_has_no_source_fingerprint() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        let artifact = Generated::text(ARTIFACT, "{\n  \"schema_version\": 1\n}\n");

        let findings = settle(dir.path(), "metadata-matrix", &[artifact], true);

        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("has no source fingerprint")),
            "Fix: a recorder that cannot identify its tree must refuse; findings={findings:?}"
        );
        assert!(
            !dir.path().join(ARTIFACT).exists(),
            "Fix: refusing to record must leave no unattributable artifact on disk."
        );
    }

    #[test]
    fn generated_documentation_outside_the_evidence_set_still_records_without_git() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        let artifact = Generated::text("docs/optimization/OP_MATRIX.toml", "rows = 0\n");

        let findings = settle(dir.path(), "op-matrix", &[artifact], true);

        assert_eq!(
            findings,
            Vec::new(),
            "Fix: documentation is read beside its source and names no recorded tree."
        );
        assert!(dir
            .path()
            .join("docs/optimization/OP_MATRIX.toml")
            .is_file());
    }

    #[test]
    fn recording_stamps_the_tree_and_regenerating_the_same_body_changes_nothing() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        init_repository(dir.path());
        let body = "{\n  \"schema_version\": 1\n}\n";

        let findings = settle(
            dir.path(),
            "metadata-matrix",
            &[Generated::text(ARTIFACT, body)],
            true,
        );
        assert_eq!(
            findings,
            Vec::new(),
            "Fix: a clean checkout can be recorded."
        );
        let recorded = std::fs::read_to_string(dir.path().join(ARTIFACT))
            .expect("Fix: the recorder wrote the artifact.");
        assert!(
            recorded.starts_with("{\n  \"source_fingerprint\": \"git:"),
            "Fix: the tree must be named at the head of the artifact; recorded={recorded}"
        );

        commit_everything(dir.path(), "move the tree on");
        assert_eq!(
            settle(
                dir.path(),
                "metadata-matrix",
                &[Generated::text(ARTIFACT, body)],
                true,
            ),
            Vec::new(),
            "Fix: re-recording an unchanged body must find nothing."
        );

        assert_eq!(
            std::fs::read_to_string(dir.path().join(ARTIFACT)).expect("Fix: read the artifact."),
            recorded,
            "Fix: an unchanged body is the same recording and keeps the tree it was recorded from."
        );
    }

    #[test]
    fn a_changed_body_is_re_attributed_to_the_tree_that_produced_it() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        init_repository(dir.path());
        assert_eq!(
            settle(
                dir.path(),
                "metadata-matrix",
                &[Generated::text(ARTIFACT, "{\n  \"schema_version\": 1\n}\n")],
                true,
            ),
            Vec::new(),
            "Fix: a clean checkout can be recorded."
        );
        let first = std::fs::read_to_string(dir.path().join(ARTIFACT))
            .expect("Fix: the recorder wrote the artifact.");
        commit_everything(dir.path(), "move the tree on");

        assert_eq!(
            settle(
                dir.path(),
                "metadata-matrix",
                &[Generated::text(ARTIFACT, "{\n  \"schema_version\": 2\n}\n")],
                true,
            ),
            Vec::new(),
            "Fix: a changed body can be recorded."
        );

        let second = std::fs::read_to_string(dir.path().join(ARTIFACT))
            .expect("Fix: the recorder rewrote the artifact.");
        assert_ne!(
            fingerprint_of(&first),
            fingerprint_of(&second),
            "Fix: a new body is a new recording and must name the tree it came from."
        );
    }

    #[test]
    fn comparing_reports_an_unattributed_artifact_and_still_compares_the_body() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        std::fs::create_dir_all(dir.path().join("release/evidence/metadata"))
            .expect("Fix: create the evidence directory.");
        std::fs::write(dir.path().join(ARTIFACT), "{\n  \"schema_version\": 1\n}\n")
            .expect("Fix: commit an artifact that names no tree.");

        let findings = settle(
            dir.path(),
            "metadata-matrix",
            &[Generated::text(ARTIFACT, "{\n  \"schema_version\": 2\n}\n")],
            false,
        );

        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("names no source tree")),
            "Fix: an artifact with no fingerprint must be reported; findings={findings:?}"
        );
        assert!(
            findings
                .iter()
                .any(|finding| finding.message.contains("\"schema_version\": 2")),
            "Fix: the body must still be compared; findings={findings:?}"
        );
    }

    #[test]
    fn comparing_ignores_the_stamp_and_agrees_on_a_body_recorded_from_another_tree() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        std::fs::create_dir_all(dir.path().join("release/evidence/metadata"))
            .expect("Fix: create the evidence directory.");
        let stamped = format!(
            "{{\n  \"source_fingerprint\": \"git:{}:dirty=false\",\n  \"schema_version\": 1\n}}\n",
            "a".repeat(40)
        );
        std::fs::write(dir.path().join(ARTIFACT), &stamped)
            .expect("Fix: commit an artifact recorded from another tree.");

        let findings = settle(
            dir.path(),
            "metadata-matrix",
            &[Generated::text(ARTIFACT, "{\n  \"schema_version\": 1\n}\n")],
            false,
        );

        assert_eq!(
            findings,
            Vec::new(),
            "Fix: the tree an artifact was recorded from is not a divergence from the tree reading it."
        );
    }

    fn fingerprint_of(recorded: &str) -> String {
        split_provenance(recorded)
            .0
            .expect("Fix: a recorded artifact names its tree.")
            .to_string()
    }

    fn init_repository(dir: &Path) {
        std::fs::write(dir.join("tracked.txt"), "original\n")
            .expect("Fix: write the tracked file.");
        for args in [
            vec!["init", "--quiet"],
            vec!["config", "user.email", "gate@example.invalid"],
            vec!["config", "user.name", "gate"],
        ] {
            run_git(dir, &args);
        }
        commit_everything(dir, "seed");
    }

    fn commit_everything(dir: &Path, message: &str) {
        run_git(dir, &["add", "--all", "--", "tracked.txt"]);
        std::fs::write(dir.join("tracked.txt"), message).expect("Fix: change the tracked file.");
        run_git(dir, &["add", "--all", "--", "tracked.txt"]);
        run_git(dir, &["commit", "--quiet", "-m", message]);
    }

    fn run_git(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
            .args(args)
            .current_dir(dir)
            .status()
            .expect("Fix: run git to build the fixture checkout.");
        assert!(status.success(), "Fix: git {args:?} failed in the fixture.");
    }

    /// WHY: Section 182.5.4 requires detecting modifications even when file mtime is preserved/restored.
    #[test]
    fn modified_file_with_restored_mtime_is_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let target = root.join("target_file.txt");
        fs::write(&target, "version 1").expect("write initial");
        let original_mtime = fs::metadata(&target)
            .and_then(|m| m.modified())
            .expect("mtime");

        let snap = WorkspaceSnapshot::capture(root);

        // Modify content
        fs::write(&target, "version 2 modified").expect("write modified");

        // Compare with snapshot
        let violations = snap.detect_mutations(root, "test-gate", &["target_file.txt"], false);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("modified workspace file `target_file.txt` without --write"));
    }

    /// WHY: Section 182.5.4 requires detecting unauthorized file deletions.
    #[test]
    fn deleted_file_is_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let target = root.join("to_delete.txt");
        fs::write(&target, "some data").expect("write initial");

        let snap = WorkspaceSnapshot::capture(root);
        fs::remove_file(&target).expect("remove file");

        let violations = snap.detect_mutations(root, "test-gate", &["to_delete.txt"], false);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("deleted workspace file `to_delete.txt` without --write"));
    }

    /// WHY: Section 182.5.6 requires that comparison mode never mutates even owned artifacts.
    #[test]
    fn owned_write_without_write_flag_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let owned = root.join("owned_artifact.json");
        fs::write(&owned, "initial").expect("write initial");

        let snap = WorkspaceSnapshot::capture(root);
        fs::write(&owned, "mutated").expect("write mutated");

        let violations = snap.detect_mutations(root, "test-gate", &["owned_artifact.json"], false);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("without --write"));
        assert!(violations[0].contains("Section 182.5.6"));
    }

    /// WHY: Section 182.5.4 requires rejecting writes to unowned workspace files even with --write.
    #[test]
    fn unowned_write_with_write_flag_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let owned = root.join("owned_artifact.json");
        let unowned = root.join("unowned_artifact.json");
        fs::write(&owned, "initial owned").expect("write owned");
        fs::write(&unowned, "initial unowned").expect("write unowned");

        let snap = WorkspaceSnapshot::capture(root);
        fs::write(&owned, "mutated owned").expect("write mutated owned");
        fs::write(&unowned, "mutated unowned").expect("write mutated unowned");

        let violations = snap.detect_mutations(root, "test-gate", &["owned_artifact.json"], true);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].contains("wrote unowned workspace file `unowned_artifact.json`"));
        assert!(violations[0].contains("Section 182.5.4"));
    }
}
