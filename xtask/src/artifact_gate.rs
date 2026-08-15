//! Regenerate an evidence artifact into memory and hold the tree to it.
//!
//! A generator that writes a file and exits zero proves nothing. The only thing
//! it ever agreed with is the file it just wrote, so the artifact and the tree
//! could drift apart for a year and no run would say so. Every gate that owns a
//! generated artifact renders it here instead, and the default action reads the
//! committed copy and names each line where the two disagree. `--write` is the
//! only path that touches the tree.
//!
//! The comparison is line by line and every divergent line is its own finding,
//! because the pinned number a gate answers to is its finding count. Collapsing
//! a thousand-line disagreement into one finding would let an artifact rot back
//! to nothing while the pin stayed level.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::gate::{Finding, GateCtx, Report};

/// What one evidence generator found in the tree, and what it would write.
///
/// Every gate that owns a generated artifact produces this and nothing else.
/// Deciding what to do with it, compare or write, belongs to one place, so a
/// gate cannot forget to compare and cannot write without being asked.
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
        Self {
            findings: Vec::new(),
            notes: Vec::new(),
            artifacts: Vec::new(),
        }
    }
}

impl Default for Inspection {
    fn default() -> Self {
        Self::new()
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
    /// the same renderer the writers use, so a difference reported here is a
    /// difference in content and never in formatting.
    ///
    /// # Errors
    ///
    /// Returns a finding naming the artifact when `value` cannot be serialized.
    pub fn json(path: impl Into<PathBuf>, value: &impl Serialize) -> Result<Self, Finding> {
        let path = path.into();
        match crate::output_arg::render_evidence_json(value) {
            Ok(content) => Ok(Self { path, content }),
            Err(error) => Err(Finding {
                file: Some(path.clone()),
                line: None,
                message: format!("`{}` could not be serialized: {error}", path.display()),
                fix: "Correct the artifact type so serde can represent it. A gate that cannot render its own artifact can never compare one.".to_string(),
            }),
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
            return vec![Finding {
                file: Some(artifact.path.clone()),
                line: None,
                message: format!("failed to create `{}`: {error}", parent.display()),
                fix: "Make the evidence directory writable, then run the gate again with --write."
                    .to_string(),
            }];
        }
    }
    match fs::write(&absolute, &artifact.content) {
        Ok(()) => Vec::new(),
        Err(error) => vec![Finding {
            file: Some(artifact.path.clone()),
            line: None,
            message: format!("failed to write `{}`: {error}", artifact.path.display()),
            fix: "Make the artifact writable, then run the gate again with --write.".to_string(),
        }],
    }
}

/// Name every way the committed artifact differs from what the tree generates.
fn compare_artifact(root: &Path, gate: &str, artifact: &Generated) -> Vec<Finding> {
    let absolute = root.join(&artifact.path);
    let committed = match fs::read_to_string(&absolute) {
        Ok(text) => text,
        Err(error) => {
            return vec![Finding {
                file: Some(artifact.path.clone()),
                line: None,
                message: format!(
                    "`{}` is the artifact this gate owns and it could not be read: {error}",
                    artifact.path.display()
                ),
                fix: format!(
                    "Run `cargo_full run --bin xtask -- {gate} --write` and commit the artifact."
                ),
            }]
        }
    };
    divergences(gate, &artifact.path, &committed, &artifact.content)
}

/// One finding per line on which `committed` and `generated` disagree.
///
/// Split out from the read so the comparison itself is provable without a
/// filesystem, and so a caller holding an artifact in memory can reuse it.
#[must_use]
pub fn divergences(
    gate: &str,
    path: &Path,
    committed: &str,
    generated: &str,
) -> Vec<Finding> {
    if committed == generated {
        return Vec::new();
    }
    let fix =
        format!("Run `cargo_full run --bin xtask -- {gate} --write` and commit the artifact, or correct the tree fact the line reports.");
    let committed_lines: Vec<&str> = committed.lines().collect();
    let generated_lines: Vec<&str> = generated.lines().collect();
    let mut findings = Vec::new();
    for index in 0..committed_lines.len().max(generated_lines.len()) {
        let line = u32::try_from(index + 1).ok();
        match (committed_lines.get(index), generated_lines.get(index)) {
            (Some(left), Some(right)) if left == right => {}
            (Some(left), Some(right)) => findings.push(Finding {
                file: Some(path.to_path_buf()),
                line,
                message: format!("the artifact says `{left}`; the tree generates `{right}`"),
                fix: fix.clone(),
            }),
            (Some(left), None) => findings.push(Finding {
                file: Some(path.to_path_buf()),
                line,
                message: format!("the artifact says `{left}`; the tree generates nothing here"),
                fix: fix.clone(),
            }),
            (None, Some(right)) => findings.push(Finding {
                file: Some(path.to_path_buf()),
                line,
                message: format!("the artifact ends before `{right}`, which the tree generates"),
                fix: fix.clone(),
            }),
            (None, None) => {}
        }
    }
    // Two texts differing only in a trailing newline produce identical line
    // sequences, so the loop above finds nothing and the gate would report a
    // clean artifact it had already decided was wrong.
    if findings.is_empty() {
        findings.push(Finding {
            file: Some(path.to_path_buf()),
            line: None,
            message: "the artifact and the tree agree line for line but not byte for byte; the trailing newline differs".to_string(),
            fix,
        });
    }
    findings
}

/// A finding for a tree fact the artifact records differently.
///
/// Measured artifacts are not regenerated for comparison, so their gates state
/// the disagreement directly instead of diffing. The wording matches
/// [`divergences`] so both halves of a report read the same way.
#[must_use]
pub fn disagreement(
    path: &Path,
    field: &str,
    artifact_says: &str,
    tree_says: &str,
    fix: impl Into<String>,
) -> Finding {
    Finding {
        file: Some(path.to_path_buf()),
        line: None,
        message: format!("`{field}`: the artifact says `{artifact_says}`; the tree says `{tree_says}`"),
        fix: fix.into(),
    }
}

/// Turn a generator's blocker sentence into a finding against its artifact.
///
/// The fourteen evidence generators each produced a `blockers` list of prose,
/// and every one of those sentences is a real judgement about the tree that
/// must survive the conversion. This is where they cross over: the sentence
/// becomes the message, and the caller supplies the corrective action the
/// sentence never carried.
#[must_use]
pub fn blocker(path: &Path, message: impl Into<String>, fix: impl Into<String>) -> Finding {
    Finding {
        file: Some(path.to_path_buf()),
        line: None,
        message: message.into(),
        fix: fix.into(),
    }
}
