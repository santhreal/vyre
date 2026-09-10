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

use crate::evidence_record::{
    self, EvidenceArtifact, EvidenceProvenance, MeasurementRecord, ProvenanceIssue,
};
use crate::gate::{Coverage, Finding, GateCtx, Report};
/// Largest committed artifact this module will read into memory.
///
/// The op matrix carried this cap on its own reader before it became a gate.
/// It belongs here now, because every artifact is read through one place.
pub const MAX_ARTIFACT_BYTES: u64 = 16_777_216;

/// How one generated artifact is read, and therefore what it must carry.
///
/// There is no third state and no default. A generator picks the arm that
/// describes its reader, so an artifact under `release/evidence` cannot reach
/// the writer without the class of what produced it.
pub enum Attribution {
    /// Read beside the source it describes; the reader has the tree.
    BesideSource,
    /// Read by someone who no longer has the tree, so it carries provenance.
    ///
    /// The generator states what took part in the measurement, which is the
    /// one fact only it knows. The tree and the host are facts of the run and
    /// [`settle`] supplies them, so a generator cannot state them wrongly.
    Recorded(MeasurementRecord),
}

/// One artifact a gate owns, rendered in memory before the tree is consulted.
pub struct Generated {
    /// Path of the artifact, relative to the workspace root.
    pub path: PathBuf,
    /// Exact bytes the body holds on disk, trailing newline included.
    pub content: String,
    /// How the artifact is read, and therefore what it must carry.
    pub attribution: Attribution,
}

impl Generated {
    /// Render one recorded evidence artifact from its envelope.
    ///
    /// Serialization goes through [`crate::output_arg::render_evidence_json`],
    /// the renderer the writers already used, so a difference reported here is
    /// a difference in content and never in formatting.
    ///
    /// # Errors
    ///
    /// Returns a finding naming the artifact when the body cannot be
    /// serialized.
    pub fn evidence<T: Serialize + ?Sized>(
        path: impl Into<PathBuf>,
        artifact: &EvidenceArtifact<'_, T>,
    ) -> Result<Self, Finding> {
        let path = path.into();
        match artifact.render_body() {
            Ok(content) => Ok(Self {
                path,
                content,
                attribution: Attribution::Recorded(artifact.measurement().clone()),
            }),
            Err(error) => Err(unserializable(path, &error)),
        }
    }

    /// Take `content` verbatim for a recorded artifact rendered elsewhere.
    pub fn evidence_text(
        path: impl Into<PathBuf>,
        measurement: MeasurementRecord,
        content: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
            attribution: Attribution::Recorded(measurement),
        }
    }

    /// Render one generated document, read beside the source it describes.
    ///
    /// # Errors
    ///
    /// Returns a finding naming the artifact when `value` cannot be serialized.
    pub fn document(path: impl Into<PathBuf>, value: &impl Serialize) -> Result<Self, Finding> {
        let path = path.into();
        match crate::output_arg::render_evidence_json(value) {
            Ok(content) => Ok(Self {
                path,
                content,
                attribution: Attribution::BesideSource,
            }),
            Err(error) => Err(unserializable(path, &error)),
        }
    }

    /// Take `content` verbatim, for a document that is not JSON.
    pub fn document_text(path: impl Into<PathBuf>, content: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            content: content.into(),
            attribution: Attribution::BesideSource,
        }
    }
}

/// The finding a gate reports when it cannot render its own artifact.
fn unserializable(path: PathBuf, error: &str) -> Finding {
    Finding::in_file(
        path.clone(),
        format!("`{}` could not be serialized: {error}", path.display()),
        "Correct the artifact type so serde can represent it. A gate that cannot render its own artifact can never compare one.",
    )
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

    /// Render `body` as one recorded evidence artifact of class `measurement`.
    ///
    /// `measurement` is positional and has no default. A generator that has not
    /// decided whether a device took part cannot call this.
    pub fn generates_evidence(
        &mut self,
        path: &str,
        measurement: MeasurementRecord,
        body: &impl Serialize,
    ) {
        let artifact = EvidenceArtifact::new(measurement, body);
        match Generated::evidence(path, &artifact) {
            Ok(artifact) => self.artifacts.push(artifact),
            Err(finding) => self.findings.push(finding),
        }
    }

    /// Record one recorded evidence artifact whose bytes are already rendered.
    pub fn generates_evidence_text(
        &mut self,
        path: &str,
        measurement: MeasurementRecord,
        content: impl Into<String>,
    ) {
        self.artifacts
            .push(Generated::evidence_text(path, measurement, content));
    }

    /// Render `value` as one generated document read beside its source.
    pub fn generates_document(&mut self, path: &str, value: &impl Serialize) {
        match Generated::document(path, value) {
            Ok(artifact) => self.artifacts.push(artifact),
            Err(finding) => self.findings.push(finding),
        }
    }

    /// Record one generated document whose bytes are not JSON.
    pub fn generates_document_text(&mut self, path: &str, content: impl Into<String>) {
        self.artifacts.push(Generated::document_text(path, content));
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
    ///
    /// Every gate invocation captures this twice, before and after the gate
    /// runs, so the whole checkout is read twice per gate. The reads are
    /// latency-bound rather than core-bound on a network checkout, which made
    /// one gate spend minutes inside `fs::read` and nothing inside its own
    /// contract: the walk names the files serially and the reads run over
    /// `structure_gate::workspace_manifest::read_lanes` lanes, the same bound
    /// the source-corpus reads use, so the descriptor budget is shared under
    /// one rule.
    #[must_use]
    pub fn capture(root: &Path) -> Self {
        record_snapshot_capture();
        let mut files = std::collections::BTreeMap::new();
        let mut errors = Vec::new();
        let mut contents = Vec::new();
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
            contents.push((relative, path.to_path_buf()));
        }
        let (read, read_errors) = digest_files(&contents);
        files.extend(read);
        errors.extend(read_errors);
        Self { files, errors }
    }

    /// Detect any creation, deletion, or modification against a post-execution state.
    ///
    /// - If `allow_owned_writes` is true (e.g. gate was invoked with `--write`), only files in
    ///   `declared_artifacts` may be created or modified.
    /// - If `allow_owned_writes` is false (comparison / sweep mode), NO workspace mutation is allowed,
    ///   even for owned artifacts (Section 182.5.6).
    ///
    /// A path git ignores is not part of the checkout a gate certifies. Every
    /// artifact a reviewer reads is tracked, and an ignored path is where a
    /// concurrent process on the same checkout writes its scratch: a job log
    /// appearing mid-run reported another process's write as this gate's
    /// mutation and failed a gate that had touched nothing.
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
        let differing = self.differing_paths(&post);
        let ignored = ignored_paths(root, &differing);

        // 1. Created files
        for rel in post.files.keys() {
            if !self.files.contains_key(rel) && !ignored.contains(rel) {
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
            if !post.files.contains_key(rel) && !ignored.contains(rel) {
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
                if pre_state != post_state && !ignored.contains(rel) {
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

    /// Every path whose presence or content differs between two snapshots.
    ///
    /// The ignore query runs over this set rather than the whole checkout, so
    /// a run that mutated nothing asks git nothing.
    fn differing_paths(&self, post: &Self) -> std::collections::BTreeSet<PathBuf> {
        let mut differing = std::collections::BTreeSet::new();
        for (rel, post_state) in &post.files {
            match self.files.get(rel) {
                Some(pre_state) if pre_state == post_state => {}
                _ => {
                    differing.insert(rel.clone());
                }
            }
        }
        for rel in self.files.keys() {
            if !post.files.contains_key(rel) {
                differing.insert(rel.clone());
            }
        }
        differing
    }
}

/// The subset of `candidates` git ignores, asked in one invocation.
///
/// `git check-ignore` consults the index, so a tracked path is never reported
/// here however its name reads: a generated artifact that a gate writes stays
/// a mutation. A checkout git cannot answer for yields nothing, which keeps
/// every path a mutation and is the strict answer.
fn ignored_paths(
    root: &Path,
    candidates: &std::collections::BTreeSet<PathBuf>,
) -> std::collections::BTreeSet<PathBuf> {
    let mut ignored = std::collections::BTreeSet::new();
    if candidates.is_empty() {
        return ignored;
    }
    let mut child = match std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["check-ignore", "--stdin", "-z"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => child,
        Err(_) => return ignored,
    };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        for path in candidates {
            let written = stdin
                .write_all(path.to_string_lossy().as_bytes())
                .and_then(|()| stdin.write_all(&[0]));
            if written.is_err() {
                break;
            }
        }
    }
    let Ok(output) = child.wait_with_output() else {
        return ignored;
    };
    for name in output.stdout.split(|byte| *byte == 0) {
        if name.is_empty() {
            continue;
        }
        ignored.insert(PathBuf::from(String::from_utf8_lossy(name).into_owned()));
    }
    ignored
}

/// Read and digest every named file, over bounded parallel lanes.
///
/// Each entry is a relative path and the absolute path to read. A read that
/// fails contributes an error line and no entry, which is what a serial read
/// did: a file deleted between the walk and the read is reported, not guessed
/// at.
///
/// # Panics
///
/// Resumes the panic of a digest lane that failed. A lane turns a failed read
/// into an error line, so a panic is a defect here; a partial digest set would
/// report an artifact as unchanged because its lane never reported.
fn digest_files(contents: &[(PathBuf, PathBuf)]) -> (Vec<(PathBuf, SnapshotEntry)>, Vec<String>) {
    if contents.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let lanes = structure_gate::workspace_manifest::read_lanes(contents.len());
    let chunk = contents.len().div_ceil(lanes);
    std::thread::scope(|scope| {
        let handles: Vec<_> = contents
            .chunks(chunk)
            .map(|chunk| scope.spawn(move || digest_chunk(chunk)))
            .collect();
        let mut entries = Vec::with_capacity(contents.len());
        let mut errors = Vec::new();
        for handle in handles {
            let (lane_entries, lane_errors) = handle.join().expect(
                "a snapshot read lane must not panic. Fix: the panic is in digest_chunk above, so \
                 read its message and correct that read, not this join",
            );
            entries.extend(lane_entries);
            errors.extend(lane_errors);
        }
        (entries, errors)
    })
}

/// Read and digest one lane's files.
fn digest_chunk(contents: &[(PathBuf, PathBuf)]) -> (Vec<(PathBuf, SnapshotEntry)>, Vec<String>) {
    let mut entries = Vec::with_capacity(contents.len());
    let mut errors = Vec::new();
    for (relative, path) in contents {
        match fs::read(path) {
            Ok(bytes) => entries.push((
                relative.clone(),
                SnapshotEntry::File {
                    size: bytes.len() as u64,
                    digest: *blake3::hash(&bytes).as_bytes(),
                },
            )),
            Err(error) => errors.push(format!(
                "workspace snapshot could not read `{}`: {error}",
                relative.display()
            )),
        }
    }
    (entries, errors)
}

/// Compare every artifact against the tree, or write it when `write` is set.
///
/// `gate` names the subcommand in each `fix`, so a reader learns the exact
/// command that settles the disagreement rather than being told one exists.
///
/// The tree and the host are read once here rather than once per artifact,
/// because every artifact a gate owns is recorded from one tree on one host in
/// one run, and four gates spent four `git status` walks proving that.
#[must_use]
pub fn settle(root: &Path, gate: &str, generated: &[Generated], write: bool) -> Vec<Finding> {
    generated
        .iter()
        .flat_map(|artifact| {
            if write {
                write_artifact(root, artifact)
            } else {
                compare_artifact(root, gate, artifact)
            }
        })
        .collect()
}

/// Whether `path` names a recorded artifact, which must name the tree it came
/// from.
///
/// A JSON artifact under `release/evidence` is a record of what some tree was,
/// read by someone who no longer has that tree. Generated documentation
/// elsewhere in the workspace is not: it is read beside the source it
/// describes. Neither is the prose under `release/evidence`, which carries the
/// release notes rather than a measurement and has no object head to name a
/// tree in.
///
/// The pair of components is matched wherever it appears rather than only at
/// the front, because the benchmark writers name their artifacts absolutely
/// and a check that only recognised the relative form would let every one of
/// them through.
#[must_use]
pub fn records_provenance(path: &Path) -> bool {
    if path.extension().and_then(std::ffi::OsStr::to_str) != Some("json") {
        return false;
    }
    let mut components = path.components();
    while let Some(component) = components.next() {
        if component.as_os_str() == std::ffi::OsStr::new("release")
            && components.clone().next().map(|next| next.as_os_str())
                == Some(std::ffi::OsStr::new("evidence"))
        {
            return true;
        }
    }
    false
}

/// The complete provenance of `artifact`, when it is a recorded one.
///
/// The two halves meet here. The generator supplied the measurement class, the
/// run supplies the tree and the host, and neither half can be left out: a
/// recorded artifact outside `release/evidence` and a document inside it are
/// both reported rather than written, so the path and the class cannot
/// disagree about how the artifact will be read.
fn provenance_of(root: &Path, artifact: &Generated) -> Result<Option<EvidenceProvenance>, Finding> {
    let recorded = records_provenance(&artifact.path);
    match (&artifact.attribution, recorded) {
        (Attribution::BesideSource, false) => Ok(None),
        (Attribution::BesideSource, true) => Err(Finding::in_file(
            artifact.path.clone(),
            format!(
                "`{}` is under release/evidence and was generated as a document, so it would carry no provenance",
                artifact.path.display()
            ),
            "Record it with `generates_evidence`, naming what took part in producing it, or generate it outside release/evidence.",
        )),
        (Attribution::Recorded(_), false) => Err(Finding::in_file(
            artifact.path.clone(),
            format!(
                "`{}` is generated as a recorded measurement and is not under release/evidence, so nothing reads its provenance",
                artifact.path.display()
            ),
            "Generate it with `generates_document`, or move the artifact under release/evidence.",
        )),
        (Attribution::Recorded(measurement), true) => {
            EvidenceProvenance::capture(root, measurement.clone())
                .map(Some)
                .map_err(|error| {
                    Finding::in_file(
                        artifact.path.clone(),
                        format!(
                            "`{}` was not written because the tree it would record cannot be identified: {error}",
                            artifact.path.display()
                        ),
                        "Record evidence from a checkout git can identify. An artifact that names no tree proves nothing about one.",
                    )
                })
        }
    }
}

/// Write one recorded evidence artifact from a writer that owns no inspection.
///
/// The benchmark writers render their evidence as a `serde_json::Value` built
/// at run time and put it on disk themselves. They used to do that through the
/// plain document writer, which is how six evidence artifacts came to carry no
/// provenance at all. They go through the same stamp as every other recorded
/// artifact now, and [`crate::json_document::write`] refuses an evidence path
/// so a seventh cannot appear beside them.
///
/// # Errors
///
/// Returns the sentence the caller reports when the tree cannot be identified,
/// the body cannot be serialized, or the artifact cannot be written.
pub fn write_recorded(
    root: &Path,
    relative: &Path,
    measurement: MeasurementRecord,
    value: &impl Serialize,
) -> Result<(), String> {
    let recorded = EvidenceArtifact::new(measurement, value);
    let generated = Generated::evidence(relative, &recorded).map_err(|finding| finding.message)?;
    let findings = write_artifact(root, &generated);
    if findings.is_empty() {
        return Ok(());
    }
    Err(Finding::messages(&findings))
}

/// Put one artifact on disk, reporting a write failure as a finding.
///
/// A recorded artifact is stamped with the provenance of this run every time it
/// is written, including when the body is unchanged. The stamp states the tree,
/// the host and the device the record was taken from, and that tree is the one
/// the commit carrying it captures, so keeping an older stamp on an unchanged
/// body records a state the artifact is no longer committed alongside.
fn write_artifact(root: &Path, artifact: &Generated) -> Vec<Finding> {
    let provenance = match provenance_of(root, artifact) {
        Ok(provenance) => provenance,
        Err(finding) => return vec![finding],
    };
    let content = match provenance {
        None => artifact.content.clone(),
        Some(provenance) => match evidence_record::stamp(&artifact.content, &provenance) {
            Ok(content) => content,
            Err(error) => {
                return vec![Finding::in_file(
                    artifact.path.clone(),
                    format!("`{}` was not written: {error}", artifact.path.display()),
                    "Render the artifact as a JSON object so it can carry the provenance head, or generate it outside release/evidence.",
                )]
            }
        },
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

/// The fingerprint the committed artifact records its tree with, if it has one.
///
/// Readers that only need the tree half of the record ask for this rather than
/// re-deriving where the stamp sits.
#[must_use]
pub fn recorded_fingerprint(committed: &str) -> Option<String> {
    match evidence_record::split(committed).0 {
        Ok(provenance) => match provenance.tree {
            evidence_record::TreeRecord::Attributed {
                source_fingerprint, ..
            } => Some(source_fingerprint),
            evidence_record::TreeRecord::Unattributable { .. } => None,
        },
        Err(_) => None,
    }
}

/// Take the recorded provenance off `committed` and return the body under it.
///
/// The stamp is one line at a known place, so lifting it back off is exact.
/// The body is what the owning gate generates, and it is the only half a
/// comparison against the tree may look at: the provenance names the tree the
/// body was recorded from, which is a different tree from the one running the
/// gate whenever anything has been committed since, and reporting that as a
/// divergence would make every artifact rot one commit after it was written.
#[must_use]
pub fn split_provenance(committed: &str) -> (Result<EvidenceProvenance, ProvenanceIssue>, String) {
    evidence_record::split(committed)
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
    if let Err(finding) = provenance_of(root, artifact) {
        return vec![finding];
    }
    if !records_provenance(&artifact.path) {
        return divergences(gate, &artifact.path, &committed, &artifact.content);
    }
    let (provenance, body) = split_provenance(&committed);
    let mut findings = provenance_findings(gate, &artifact.path, provenance.as_ref());
    findings.extend(divergences(gate, &artifact.path, &body, &artifact.content));
    findings
}

/// Judge whether the committed artifact carries a provenance block at all.
///
/// The content of the block is the corpus gate's judgement, not an owning
/// gate's: whether a commit is an ancestor of the branch, and whether a device
/// result names a device, are facts about the corpus and are answered once for
/// every artifact in it rather than once per owning gate.
fn provenance_findings(
    gate: &str,
    path: &Path,
    provenance: Result<&EvidenceProvenance, &ProvenanceIssue>,
) -> Vec<Finding> {
    match provenance {
        Ok(_) => Vec::new(),
        Err(issue) => vec![Finding::in_file(
            path.to_path_buf(),
            format!("`{}` {}", path.display(), issue.predicate()),
            format!(
                "Run `./cargo_full run --bin xtask -- {gate} --write` and commit the artifact."
            ),
        )],
    }
}

/// One finding for an artifact that disagrees with what the tree generates.
///
/// Split out from the read so the comparison is provable without a filesystem,
/// and so a caller already holding both texts can reuse it.
///
/// A stale artifact is one defect: it was not regenerated. The comparison is
/// positional, so a single inserted line offsets every line after it and a
/// per-line finding count reports that one defect once per line of the file.
/// `configuration-model` reported 8384 findings for one unregenerated
/// artifact that way, which buries every other finding the gate produced and
/// makes the count useless as a pin. The finding names the first line that
/// disagrees, so the diagnosis survives, and counts the rest.
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
    let mut first: Option<(u32, String)> = None;
    let mut disagreeing = 0usize;
    for index in 0..committed_lines.len().max(generated_lines.len()) {
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
        disagreeing += 1;
        if first.is_none() {
            first = Some((u32::try_from(index + 1).unwrap_or(u32::MAX), message));
        }
    }
    // Two texts differing only in a trailing newline yield identical line
    // sequences, so the loop finds nothing and the gate would report a clean
    // artifact it has already decided is wrong.
    let Some((line, message)) = first else {
        return vec![Finding::in_file(
            path.to_path_buf(),
            "the artifact and the tree agree line for line but not byte for byte; the trailing newline differs",
            fix,
        )];
    };
    let message = if disagreeing == 1 {
        message
    } else {
        format!("{disagreeing} lines disagree with what the tree generates; the first is {message}")
    };
    vec![Finding::at(path.to_path_buf(), line, message, fix)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    const ARTIFACT: &str = "release/evidence/metadata/matrix.json";

    #[test]
    fn the_recorder_refuses_an_artifact_whose_tree_has_no_source_fingerprint() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        let artifact = Generated::evidence_text(
            ARTIFACT,
            MeasurementRecord::HostOnly,
            "{\n  \"schema_version\": 1\n}\n",
        );

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
        let artifact = Generated::document_text("docs/optimization/OP_MATRIX.toml", "rows = 0\n");

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
    fn re_recording_an_unchanged_body_restamps_the_tree_that_produced_it() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        init_repository(dir.path());
        let body = "{\n  \"schema_version\": 1\n}\n";

        let findings = settle(
            dir.path(),
            "metadata-matrix",
            &[Generated::evidence_text(
                ARTIFACT,
                MeasurementRecord::HostOnly,
                body,
            )],
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
                &[Generated::evidence_text(
                    ARTIFACT,
                    MeasurementRecord::HostOnly,
                    body
                )],
                true,
            ),
            Vec::new(),
            "Fix: re-recording an unchanged body must find nothing."
        );

        assert_ne!(
            std::fs::read_to_string(dir.path().join(ARTIFACT)).expect("Fix: read the artifact."),
            recorded,
            "Fix: a stamp kept across a moved tree names a tree the artifact is no longer \
             committed alongside, and nothing could then correct it."
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
                &[Generated::evidence_text(
                    ARTIFACT,
                    MeasurementRecord::HostOnly,
                    "{\n  \"schema_version\": 1\n}\n"
                )],
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
                &[Generated::evidence_text(
                    ARTIFACT,
                    MeasurementRecord::HostOnly,
                    "{\n  \"schema_version\": 2\n}\n"
                )],
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
            &[Generated::evidence_text(
                ARTIFACT,
                MeasurementRecord::HostOnly,
                "{\n  \"schema_version\": 2\n}\n",
            )],
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
            &[Generated::evidence_text(
                ARTIFACT,
                MeasurementRecord::HostOnly,
                "{\n  \"schema_version\": 1\n}\n",
            )],
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
            .tree
            .source_fingerprint()
            .expect("Fix: a recorded artifact names the source it was read from.")
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

        // Modify the content, then put the mtime back: only the digest can now
        // tell the two states apart.
        fs::write(&target, "version 2 modified").expect("write modified");
        fs::File::options()
            .write(true)
            .open(&target)
            .and_then(|file| file.set_modified(original_mtime))
            .expect("restore mtime");
        assert_eq!(
            fs::metadata(&target)
                .and_then(|m| m.modified())
                .expect("mtime"),
            original_mtime,
            "Fix: restore the mtime, or the test cannot prove the digest found the change."
        );

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

    /// WHY: the capture reads its files over parallel lanes, so a chunking
    /// error drops or duplicates whole ranges of the tree. One file per lane is
    /// not enough to see that: the file count here exceeds the lane bound, the
    /// files are spread across nested directories, and every one carries
    /// distinct content, so a lost chunk shows up as a file the snapshot never
    /// recorded and a modification it therefore cannot report.
    #[test]
    fn a_tree_wider_than_the_read_lanes_is_captured_whole() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        let count = 200;
        for index in 0..count {
            let nested = root.join(format!("dir_{}", index % 7));
            fs::create_dir_all(&nested).expect("Fix: create the fixture directory.");
            fs::write(
                nested.join(format!("file_{index}.txt")),
                format!("body {index}"),
            )
            .expect("Fix: write the fixture file.");
        }

        let snap = WorkspaceSnapshot::capture(root);
        assert_eq!(
            snap.files.len(),
            count,
            "Fix: every file the walk named must carry an entry; a lane whose \
             results were dropped leaves the tree partly unrecorded."
        );

        let changed = root.join("dir_3/file_199.txt");
        fs::write(&changed, "changed body").expect("Fix: modify one fixture file.");
        let violations = snap.detect_mutations(root, "test-gate", &[], false);
        assert_eq!(
            violations.len(),
            1,
            "Fix: exactly the modified file is a violation; violations={violations:?}"
        );
        assert!(
            violations[0].contains("dir_3/file_199.txt"),
            "Fix: the violation must name the file that changed; violation={}",
            violations[0]
        );
    }

    #[test]
    fn a_write_under_a_gitignored_path_is_not_this_gate_s_mutation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        crate::fixture_checkout::seeded(root);
        fs::write(root.join(".gitignore"), "scratch/\n").expect("Fix: write the ignore rule.");
        fs::create_dir_all(root.join("scratch")).expect("Fix: create the scratch directory.");

        let snap = WorkspaceSnapshot::capture(root);
        fs::write(root.join("scratch/job.log"), "another process wrote this\n")
            .expect("Fix: write the scratch file.");
        fs::write(root.join("tracked.txt"), "the gate wrote this\n")
            .expect("Fix: modify the tracked file.");

        let violations = snap.detect_mutations(root, "test-gate", &[], false);

        assert_eq!(
            violations.len(),
            1,
            "Fix: an ignored path carries nothing a reviewer reads and belongs to \
             whatever else runs on this checkout; only the tracked write is a \
             mutation; violations={violations:?}"
        );
        assert!(
            violations[0].contains("tracked.txt"),
            "Fix: the tracked write must still be reported; violation={}",
            violations[0]
        );
    }
}
