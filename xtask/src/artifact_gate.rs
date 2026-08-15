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

use serde::Serialize;

use crate::gate::{Finding, GateCtx, Report};

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
        self.findings
            .push(Finding::in_file(artifact, message, fix));
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
    findings.extend(settle(&ctx.root, gate, &artifacts, ctx.write));
    Report { findings, notes }
}

/// Compare every artifact against the tree, or write it when `write` is set.
///
/// `gate` names the subcommand in each `fix`, so a reader learns the exact
/// command that settles the disagreement rather than being told one exists.
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

/// Put one artifact on disk, reporting a write failure as a finding.
fn write_artifact(root: &Path, artifact: &Generated) -> Vec<Finding> {
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
    match fs::write(&absolute, &artifact.content) {
        Ok(()) => Vec::new(),
        Err(error) => vec![Finding::in_file(
            artifact.path.clone(),
            format!("failed to write `{}`: {error}", artifact.path.display()),
            "Make the artifact writable, then run the gate again with --write.",
        )],
    }
}

/// Name every way the committed artifact differs from what the tree generates.
///
/// The read is bounded so a corrupted or accidentally enormous artifact refuses
/// rather than being allocated whole.
fn compare_artifact(root: &Path, gate: &str, artifact: &Generated) -> Vec<Finding> {
    let absolute = root.join(&artifact.path);
    match crate::output_arg::read_text_bounded(&absolute, MAX_ARTIFACT_BYTES, "evidence artifact") {
        Ok(committed) => divergences(gate, &artifact.path, &committed, &artifact.content),
        Err(error) => vec![Finding::in_file(
            artifact.path.clone(),
            format!(
                "`{}` is the artifact this gate owns and it could not be read: {error}",
                artifact.path.display()
            ),
            format!("Run `cargo_full run --bin xtask -- {gate} --write` and commit the artifact."),
        )],
    }
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
        "Run `cargo_full run --bin xtask -- {gate} --write` and commit the artifact, or correct the tree fact the line reports."
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
