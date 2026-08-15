//! The source fingerprint every recorded artifact names its tree with.
//!
//! Evidence is only evidence if it is attributable to a tree. A fingerprint is
//! either `git:<commit>:dirty=false`, which a reader can check out and
//! reproduce, or `git:<commit>:dirty=true:worktree=<digest>`, which names a
//! commit plus a digest of everything the worktree changed on top of it.
//!
//! Both halves of the contract live here. [`capture`] is the only producer in
//! this crate and it refuses rather than emit a fingerprint with an unknown
//! dirty state or an unknown worktree digest. [`issues`] is the judge every
//! reader of a recorded fingerprint uses. A recorder runs the judge over what
//! it is about to write, so the imprecisions a reader used to discover months
//! later are refused at the moment of recording.
//!
//! `release/evidence/**` is excluded from the dirty scan. Writing evidence is
//! what a generator does, so counting the artifact it just wrote as a change to
//! the tree it describes would make every recorded fingerprint dirty by
//! construction.

use std::path::Path;
use std::process::Command;

/// Largest file this module reads whole while digesting an untracked path.
const MAX_UNTRACKED_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// A way a recorded fingerprint fails to identify the source it names.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceFingerprintIssue {
    /// The recorder could not tell whether the tree was dirty and said so.
    DirtyUnknownState {
        /// The fingerprint as recorded.
        source_fingerprint: String,
    },
    /// The tree was dirty and the fingerprint carries no worktree digest.
    DirtyMissingWorktree {
        /// The fingerprint as recorded.
        source_fingerprint: String,
    },
    /// The tree was dirty and the worktree digest is the literal `unknown`.
    DirtyUnknownWorktree {
        /// The fingerprint as recorded.
        source_fingerprint: String,
    },
    /// The worktree digest is present but is not a BLAKE3 hex digest.
    DirtyInvalidWorktree {
        /// The fingerprint as recorded.
        source_fingerprint: String,
        /// The digest text that is not a digest.
        worktree: String,
    },
}

impl SourceFingerprintIssue {
    /// The half-sentence a caller appends to whatever it is naming.
    ///
    /// Every reader used to spell these four sentences itself, against its own
    /// subject, which is how two of them came to word the same defect
    /// differently. The subject is the caller's; the predicate is here.
    #[must_use]
    pub fn predicate(&self) -> String {
        match self {
            Self::DirtyUnknownState { source_fingerprint } => {
                format!("source_fingerprint `{source_fingerprint}` has unknown dirty state")
            }
            Self::DirtyMissingWorktree { source_fingerprint } => format!(
                "source_fingerprint `{source_fingerprint}` is dirty but has no worktree digest"
            ),
            Self::DirtyUnknownWorktree { source_fingerprint } => format!(
                "source_fingerprint `{source_fingerprint}` is dirty but has unknown worktree digest"
            ),
            Self::DirtyInvalidWorktree {
                source_fingerprint,
                worktree,
            } => format!(
                "source_fingerprint `{source_fingerprint}` has invalid worktree digest `{worktree}`"
            ),
        }
    }
}

/// Name every way `source_fingerprint` fails to identify a source tree.
///
/// A fingerprint that does not start with `git:` names something other than a
/// checkout and this judge has nothing to say about it.
#[must_use]
pub fn issues(source_fingerprint: &str) -> Vec<SourceFingerprintIssue> {
    let Some(rest) = source_fingerprint.strip_prefix("git:") else {
        return Vec::new();
    };
    let mut issues = Vec::new();
    if rest.contains(":dirty=unknown") {
        issues.push(SourceFingerprintIssue::DirtyUnknownState {
            source_fingerprint: source_fingerprint.to_string(),
        });
    }
    let Some(dirty_offset) = rest.find(":dirty=true") else {
        return issues;
    };
    let after_dirty = &rest[dirty_offset + ":dirty=true".len()..];
    let Some(worktree) = after_dirty.strip_prefix(":worktree=") else {
        issues.push(SourceFingerprintIssue::DirtyMissingWorktree {
            source_fingerprint: source_fingerprint.to_string(),
        });
        return issues;
    };
    if worktree == "unknown" {
        issues.push(SourceFingerprintIssue::DirtyUnknownWorktree {
            source_fingerprint: source_fingerprint.to_string(),
        });
    } else if !is_blake3_hex_digest(worktree) {
        issues.push(SourceFingerprintIssue::DirtyInvalidWorktree {
            source_fingerprint: source_fingerprint.to_string(),
            worktree: worktree.to_string(),
        });
    }
    issues
}

/// Whether `value` is a 64-character BLAKE3 hex digest.
#[must_use]
pub fn is_blake3_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// Build the fingerprint of the tree rooted at `root`, or say why there is none.
///
/// # Errors
///
/// Returns the sentence a gate reports when the tree cannot be identified: git
/// names no commit, the dirty scan failed, or the tree is dirty and the
/// worktree digest could not be taken. Every one of those used to be recorded
/// as `unknown` inside an otherwise well-formed fingerprint, which is a claim
/// about a tree that identifies no tree.
pub fn capture(root: &Path) -> Result<String, String> {
    let commit = git_text(root, &["rev-parse", "HEAD"])
        .map_err(|error| format!("git names no commit for `{}`: {error}", root.display()))?;
    if commit.is_empty() {
        return Err(format!(
            "git names no commit for `{}`: rev-parse returned nothing",
            root.display()
        ));
    }
    let status = git_bytes(root, DIRTY_STATUS_ARGS).map_err(|error| {
        format!(
            "git cannot tell whether `{}` is dirty: {error}",
            root.display()
        )
    })?;
    if status.is_empty() {
        return Ok(format!("git:{commit}:dirty=false"));
    }
    let worktree = dirty_worktree_digest(root, &status).ok_or_else(|| {
        format!(
            "`{}` is dirty and git could not produce the worktree diff that digests it",
            root.display()
        )
    })?;
    Ok(format!("git:{commit}:dirty=true:worktree={worktree}"))
}

/// The dirty scan, spelled once because the digest hashes its exact output.
const DIRTY_STATUS_ARGS: &[&str] = &[
    "status",
    "--porcelain=v1",
    "-z",
    "--untracked-files=all",
    "--",
    ".",
    ":!release/evidence/**",
];

/// Digest everything the worktree changed on top of `HEAD`.
///
/// Two dirty checkouts of the same commit are different trees, so the digest
/// covers the status lines, the tracked diff and the content of every untracked
/// file. An untracked file too large to read whole contributes its cap instead
/// of its bytes, so the digest stays bounded without silently ignoring it.
fn dirty_worktree_digest(root: &Path, status: &[u8]) -> Option<String> {
    let diff = git_bytes(
        root,
        &["diff", "--binary", "HEAD", "--", ".", ":!release/evidence/**"],
    )
    .ok()?;
    let untracked = git_bytes(
        root,
        &[
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            ".",
            ":!release/evidence/**",
        ],
    )
    .unwrap_or_default();
    Some(dirty_worktree_digest_of(root, status, &diff, &untracked))
}

/// The digest itself, separated from the four git calls that feed it.
fn dirty_worktree_digest_of(root: &Path, status: &[u8], diff: &[u8], untracked: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, b"format", b"vyre-bench-dirty-source-v1");
    hash_field(&mut hasher, b"status", status);
    hash_field(&mut hasher, b"diff", diff);
    for path in untracked
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
    {
        hash_field(&mut hasher, b"untracked-path", path);
        let path = String::from_utf8_lossy(path);
        match read_bounded(&root.join(path.as_ref())) {
            Ok(Some(bytes)) => hash_field(&mut hasher, b"untracked-content", &bytes),
            Ok(None) => hash_field(
                &mut hasher,
                b"untracked-content-oversized",
                MAX_UNTRACKED_FILE_BYTES.to_string().as_bytes(),
            ),
            Err(_) => {}
        }
    }
    hasher.finalize().to_hex().to_string()
}

/// Add one length-delimited label/value field, so two fields cannot fuse.
fn hash_field(hasher: &mut blake3::Hasher, label: &[u8], value: &[u8]) {
    hasher.update(&(label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Read `path` whole, or `None` when it exceeds [`MAX_UNTRACKED_FILE_BYTES`].
fn read_bounded(path: &Path) -> std::io::Result<Option<Vec<u8>>> {
    use std::io::Read;

    let mut reader = std::fs::File::open(path)?.take(MAX_UNTRACKED_FILE_BYTES.saturating_add(1));
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_UNTRACKED_FILE_BYTES {
        return Ok(None);
    }
    Ok(Some(bytes))
}

fn git_text(root: &Path, args: &[&str]) -> Result<String, String> {
    let stdout = git_bytes(root, args)?;
    Ok(String::from_utf8_lossy(&stdout).trim().to_string())
}

fn git_bytes(root: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_refuses_a_directory_git_names_no_commit_for() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");

        let error = capture(dir.path())
            .expect_err("Fix: a directory outside any checkout identifies no source tree.");

        assert!(
            error.contains("git names no commit"),
            "Fix: refusing to record must name what could not be resolved; error={error}"
        );
    }

    #[test]
    fn capture_marks_a_dirty_tree_dirty_with_a_worktree_digest() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        crate::fixture_checkout::seeded(dir.path());
        std::fs::write(dir.path().join("tracked.txt"), "changed\n")
            .expect("Fix: dirty the tracked file.");

        let fingerprint = capture(dir.path()).expect("Fix: a dirty checkout still names a commit.");

        assert!(
            fingerprint.contains(":dirty=true:worktree="),
            "Fix: a dirty tree must be recorded dirty; fingerprint={fingerprint}"
        );
        assert!(
            issues(&fingerprint).is_empty(),
            "Fix: the recorder must not produce a fingerprint its own judge rejects; issues={:?}",
            issues(&fingerprint)
        );
    }

    #[test]
    fn capture_records_a_clean_tree_clean() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        crate::fixture_checkout::seeded(dir.path());

        let fingerprint = capture(dir.path()).expect("Fix: a clean checkout names its commit.");

        assert!(
            fingerprint.ends_with(":dirty=false"),
            "Fix: a clean tree must be recorded clean; fingerprint={fingerprint}"
        );
    }

    #[test]
    fn a_committed_evidence_artifact_does_not_dirty_the_tree_it_describes() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        crate::fixture_checkout::seeded(dir.path());
        std::fs::create_dir_all(dir.path().join("release/evidence/metadata"))
            .expect("Fix: create the evidence directory.");
        std::fs::write(
            dir.path().join("release/evidence/metadata/matrix.json"),
            "{}\n",
        )
        .expect("Fix: write an evidence artifact.");

        let fingerprint = capture(dir.path()).expect("Fix: the checkout still names a commit.");

        assert!(
            fingerprint.ends_with(":dirty=false"),
            "Fix: writing evidence must not make the tree it describes dirty; fingerprint={fingerprint}"
        );
    }

    #[test]
    fn two_different_dirty_trees_get_two_different_digests() {
        let first = tempfile::tempdir().expect("Fix: create a temporary directory.");
        let second = tempfile::tempdir().expect("Fix: create a temporary directory.");
        crate::fixture_checkout::seeded(first.path());
        crate::fixture_checkout::seeded(second.path());
        std::fs::write(first.path().join("tracked.txt"), "one\n")
            .expect("Fix: dirty the first tree.");
        std::fs::write(second.path().join("tracked.txt"), "two\n")
            .expect("Fix: dirty the second tree.");

        let one = capture(first.path()).expect("Fix: the first checkout names a commit.");
        let two = capture(second.path()).expect("Fix: the second checkout names a commit.");

        assert_ne!(
            one.rsplit(":worktree=").next(),
            two.rsplit(":worktree=").next(),
            "Fix: a worktree digest that cannot distinguish two dirty trees identifies neither."
        );
    }

    #[test]
    fn the_judge_rejects_every_imprecision_a_recorder_could_leave() {
        assert_eq!(
            issues("git:abc:dirty=unknown"),
            vec![SourceFingerprintIssue::DirtyUnknownState {
                source_fingerprint: "git:abc:dirty=unknown".to_string()
            }]
        );
        assert_eq!(
            issues("git:abc:dirty=true"),
            vec![SourceFingerprintIssue::DirtyMissingWorktree {
                source_fingerprint: "git:abc:dirty=true".to_string()
            }]
        );
        assert_eq!(
            issues("git:abc:dirty=true:worktree=unknown"),
            vec![SourceFingerprintIssue::DirtyUnknownWorktree {
                source_fingerprint: "git:abc:dirty=true:worktree=unknown".to_string()
            }]
        );
        assert_eq!(
            issues("git:abc:dirty=true:worktree=short"),
            vec![SourceFingerprintIssue::DirtyInvalidWorktree {
                source_fingerprint: "git:abc:dirty=true:worktree=short".to_string(),
                worktree: "short".to_string()
            }]
        );
        assert!(issues("git:abc:dirty=false").is_empty());
        assert!(issues(&format!("git:abc:dirty=true:worktree={}", "a".repeat(64))).is_empty());
    }

}
