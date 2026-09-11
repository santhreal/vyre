//! The source fingerprint every recorded artifact names its tree with.
//!
//! Evidence is only evidence if it is attributable to a tree. A fingerprint is
//! either `git:<commit>:dirty=false`, which a reader can check out and
//! reproduce, or `git:<commit>:dirty=true:worktree=<digest>`, which names a
//! commit plus a digest of the content every non-evidence path differs from it
//! by.
//!
//! The digest covers content, never how git reported it. A status line, a
//! rename classification and the tracked/untracked distinction all disappear
//! the moment a change is committed, so a digest over any of them names a state
//! no commit can carry. A digest over the differing content is what
//! [`resolves_against`] recomputes from the commit an artifact is committed in:
//! the artifact records the tree it was generated from, and that tree is the one
//! the next commit captures.
//!
//! Three parts of the contract live here. [`capture`] is the only producer in
//! this crate and it refuses rather than emit a fingerprint with an unknown
//! dirty state or an unknown worktree digest. [`issues`] judges the shape of a
//! recorded fingerprint. [`resolves_against`] judges it against a commit, which
//! is what makes the recorded value checkable rather than merely well-formed.
//!
//! `release/evidence/**` is excluded throughout. Writing evidence is what a
//! generator does, so counting the artifact it just wrote as a change to the
//! tree it describes would make every recorded fingerprint dirty by
//! construction.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{LazyLock, Mutex};

/// Largest file this module digests whole.
const MAX_UNTRACKED_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// The predicate every stale-source verdict is written with.
///
/// Five readers form this sentence and a sixth recognises it, so the words are
/// declared once. Recognising the shape is what lets a gate answer a whole
/// population of stale artifacts with the one command that re-measures them,
/// instead of one finding per artifact the same re-measurement would close.
pub const STALE_SOURCE_PREDICATE: &str = "does not match current workspace source";

/// Whether a verdict says a recorded fingerprint names a tree that is no longer
/// this one.
#[must_use]
pub fn is_stale_source_verdict(verdict: &str) -> bool {
    verdict.contains(STALE_SOURCE_PREDICATE)
}

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

/// The commit a `git:` fingerprint names, when it names one.
#[must_use]
pub fn recorded_commit(source_fingerprint: &str) -> Option<&str> {
    source_fingerprint
        .strip_prefix("git:")?
        .split(':')
        .next()
        .filter(|commit| !commit.is_empty())
}

/// Build the fingerprint of the tree rooted at `root`, or say why there is none.
///
/// # Errors
///
/// Returns the sentence a gate reports when the tree cannot be identified: git
/// names no commit, or git could not state what the worktree differs from that
/// commit by. Every one of those used to be recorded as `unknown` inside an
/// otherwise well-formed fingerprint, which is a claim about a tree that
/// identifies no tree.
pub fn capture(root: &Path) -> Result<String, String> {
    let commit = git_text(root, &["rev-parse", "HEAD"])
        .map_err(|error| format!("git names no commit for `{}`: {error}", root.display()))?;
    if commit.is_empty() {
        return Err(format!(
            "git names no commit for `{}`: rev-parse returned nothing",
            root.display()
        ));
    }
    let changed = changed_in_worktree(root).ok_or_else(|| {
        format!(
            "git cannot state what `{}` differs from `{commit}` by",
            root.display()
        )
    })?;
    Ok(fingerprint_of(&commit, &changed))
}

/// Whether `source_fingerprint` names the source the commit `carrier` carries.
///
/// The recorded commit is the one the generator ran against, so the source it
/// names is that commit plus whatever was uncommitted at the time. A commit that
/// captures those changes carries exactly that source, and a reader recomputes
/// the same value from the two commits alone.
///
/// # Errors
///
/// Returns the sentence a gate reports when the fingerprint names no commit,
/// when git cannot compare the two commits, or when the recorded value is not
/// the one the carrier's source produces.
pub fn resolves_against(
    root: &Path,
    source_fingerprint: &str,
    carrier: &str,
) -> Result<(), String> {
    let Some(base) = recorded_commit(source_fingerprint) else {
        return Err(format!(
            "source_fingerprint `{source_fingerprint}` names no commit"
        ));
    };
    let expected = expected_fingerprint(root, base, carrier)?;
    if expected == source_fingerprint {
        return Ok(());
    }
    Err(format!(
        "source_fingerprint `{source_fingerprint}` does not name the source `{carrier}` carries, \
         which is `{expected}`"
    ))
}

/// The fingerprint the source `carrier` carries produces, against `base`.
///
/// Memoized on the tree and the pair. An evidence sweep judges every committed
/// artifact, and artifacts generated from one tree and committed together share
/// both commits, so the diff and its object reads are asked for once per pair
/// instead of once per artifact.
fn expected_fingerprint(root: &Path, base: &str, carrier: &str) -> Result<String, String> {
    type Key = (PathBuf, String, String);
    static EXPECTED: LazyLock<Mutex<BTreeMap<Key, String>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));

    // Two fixture checkouts can reach the same commit id from the same seeded
    // content, so the tree is part of the identity of an answer about it.
    let key = (root.to_path_buf(), base.to_string(), carrier.to_string());
    let memo = crate::lock_policy::govern_memo(&EXPECTED, MEMO_OWNER, MEMO_STATE);
    if let Some(expected) = memo.get(&key) {
        return Ok(expected.clone());
    }
    drop(memo);
    let expected = fingerprint_of(base, &changed_between(root, base, carrier)?);
    crate::lock_policy::govern_memo(&EXPECTED, MEMO_OWNER, MEMO_STATE)
        .insert(key, expected.clone());
    Ok(expected)
}

/// The subsystem the provenance memo reports as the owner on poison.
const MEMO_OWNER: &str = "the source provenance checker";

/// The state the provenance memo reports on poison.
const MEMO_STATE: &str = "the per-pair expected fingerprint memo";

/// The fingerprint text for a base commit and what the source differs from it by.
fn fingerprint_of(commit: &str, changed: &[ChangedPath]) -> String {
    if changed.is_empty() {
        return format!("git:{commit}:dirty=false");
    }
    format!("git:{commit}:dirty=true:worktree={}", digest_of(changed))
}

/// Paths no source digest covers, because writing them is what a generator does.
///
/// `release/evidence/**` is the corpus itself. The release provenance document
/// is a projection of the stamps in that corpus and holds nothing a generator
/// reads, so counting it as source made recording evidence change the source
/// the recording names: every capture dirtied the next one and no sequence of
/// commits reached a matching fingerprint.
pub(crate) const EXCLUDED_FROM_SOURCE: [&str; 2] = [
    ":!release/evidence/**",
    ":!docs/generated/release-provenance.toml",
];

/// The label the source-difference digest is taken under.
const SOURCE_DIFF_FORMAT: &[u8] = b"vyre-source-diff-v2";

/// One non-evidence path whose content differs from the base commit.
struct ChangedPath {
    /// Repository-relative path.
    path: String,
    /// What the path holds now, or `None` when it holds nothing.
    content: Option<Content>,
}

/// What a changed path holds, bounded.
enum Content {
    /// The bytes at the path.
    Bytes(Vec<u8>),
    /// The path exceeds [`MAX_UNTRACKED_FILE_BYTES`] and contributes its cap.
    Oversized,
}

/// Every non-evidence path the worktree differs from `HEAD` by.
///
/// Rename detection is off on both sides of the contract: a rename reported as
/// one new path and a rename reported as a deletion plus an addition digest
/// differently, and which one git reports depends on its own similarity
/// heuristic rather than on the source.
fn changed_in_worktree(root: &Path) -> Option<Vec<ChangedPath>> {
    let tracked = git_bytes(
        root,
        &[
            &[
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                "HEAD",
                "--",
                ".",
            ],
            &EXCLUDED_FROM_SOURCE[..],
        ]
        .concat(),
    )
    .ok()?;
    let untracked = git_bytes(
        root,
        &[
            &[
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ".",
            ],
            &EXCLUDED_FROM_SOURCE[..],
        ]
        .concat(),
    )
    .unwrap_or_default();
    Some(
        ordered_paths(&[&tracked, &untracked])
            .into_iter()
            .map(|path| {
                let content = match read_bounded(&root.join(&path)) {
                    Ok(Some(bytes)) => Some(Content::Bytes(bytes)),
                    Ok(None) => Some(Content::Oversized),
                    Err(_) => None,
                };
                ChangedPath { path, content }
            })
            .collect(),
    )
}

/// Every non-evidence path two commits differ by, as `carrier` holds it.
///
/// One object read serves every changed path: the pair can differ by hundreds
/// of files, and a `cat-file` process per file was what made judging one
/// artifact cost minutes on a network checkout.
fn changed_between(root: &Path, base: &str, carrier: &str) -> Result<Vec<ChangedPath>, String> {
    let names = git_bytes(
        root,
        &[
            &[
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                base,
                carrier,
                "--",
                ".",
            ],
            &EXCLUDED_FROM_SOURCE[..],
        ]
        .concat(),
    )
    .map_err(|error| format!("git cannot compare `{base}` with `{carrier}`: {error}"))?;
    let paths = ordered_paths(&[&names]);
    let objects: Vec<String> = paths
        .iter()
        .map(|path| format!("{carrier}:{path}"))
        .collect();
    Ok(paths
        .into_iter()
        .zip(committed_objects(root, &objects))
        .map(|(path, content)| ChangedPath { path, content })
        .collect())
}

/// What `carrier` holds at each requested `<commit>:<path>` object, in order.
///
/// One `git cat-file --batch` answers every request. The batch protocol frames
/// each response, so a missing object is reported as missing rather than
/// guessed at, and the whole set costs one process instead of two per path.
fn committed_objects(root: &Path, objects: &[String]) -> Vec<Option<Content>> {
    if objects.is_empty() {
        return Vec::new();
    }
    match batch_objects(root, objects, Some(MAX_UNTRACKED_FILE_BYTES)) {
        Ok(contents) => contents,
        // A batch that cannot start says nothing about any single object, so
        // every request is answered as holding nothing, which is what a failed
        // per-object read reported before.
        Err(_) => objects.iter().map(|_| None).collect(),
    }
}

/// The newest commit touching each committed path under `pathspec`.
///
/// One history walk answers every path. `git log -1 -- <path>` per artifact
/// walks the whole history again for each one, which is what made the
/// committed-provenance sweep cost tens of minutes on a network checkout: the
/// commits arrive newest-first, so the first mention of a path is the commit
/// that carries it.
///
/// # Errors
///
/// Returns the sentence a gate reports when git cannot run or when the log is
/// not a sequence of commit-then-paths records.
pub fn carrier_commits(root: &Path, pathspec: &str) -> Result<BTreeMap<String, String>, String> {
    let output = Command::new("git")
        .args([
            "-c",
            "core.quotePath=false",
            "log",
            "--format=%H",
            "--name-only",
            "--no-renames",
            "--",
            pathspec,
        ])
        .current_dir(root)
        .output()
        .map_err(|error| format!("git log over `{pathspec}` could not run: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git log over `{pathspec}` failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let mut carriers = BTreeMap::new();
    let mut commit = String::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if line.is_empty() {
            continue;
        }
        if is_commit_id(line) {
            commit = line.to_string();
            continue;
        }
        if commit.is_empty() {
            return Err(format!(
                "git log over `{pathspec}` reported path `{line}` before any commit"
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
/// A tracked path cannot be 40 hex characters with no separator, so the two
/// record kinds are distinguishable without one.
fn is_commit_id(line: &str) -> bool {
    line.len() == 40 && line.bytes().all(|byte| byte.is_ascii_hexdigit())
}

/// What each requested `<commit>:<path>` object holds, as text, in order.
///
/// `None` names an object git does not resolve. Committed text is read whole:
/// the size cap on `committed_objects` bounds what a source digest reads, and
/// a caller judging one artifact needs all of it.
#[must_use]
pub fn committed_texts(root: &Path, objects: &[String]) -> Vec<Option<String>> {
    if objects.is_empty() {
        return Vec::new();
    }
    match batch_objects(root, objects, None) {
        Ok(contents) => contents
            .into_iter()
            .map(|content| match content {
                Some(Content::Bytes(bytes)) => Some(String::from_utf8_lossy(&bytes).into_owned()),
                Some(Content::Oversized) | None => None,
            })
            .collect(),
        Err(_) => objects.iter().map(|_| None).collect(),
    }
}

/// Drive one `git cat-file --batch` over `objects`.
fn batch_objects(
    root: &Path,
    objects: &[String],
    cap: Option<u64>,
) -> Result<Vec<Option<Content>>, String> {
    let mut child = Command::new("git")
        .args(["cat-file", "--batch"])
        .current_dir(root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| error.to_string())?;
    let mut stdin = child.stdin.take().ok_or("git cat-file has no stdin")?;
    let mut stdout = BufReader::new(child.stdout.take().ok_or("git cat-file has no stdout")?);

    let mut contents = Vec::with_capacity(objects.len());
    for object in objects {
        // One request, one response: the reply is bounded by the object it
        // names, so writing and reading in lockstep cannot fill a pipe.
        writeln!(stdin, "{object}").map_err(|error| error.to_string())?;
        stdin.flush().map_err(|error| error.to_string())?;
        contents.push(read_batch_response(&mut stdout, cap)?);
    }
    drop(stdin);
    let _ = child.wait();
    Ok(contents)
}

/// Read one `cat-file --batch` response.
fn read_batch_response(
    stdout: &mut impl BufRead,
    cap: Option<u64>,
) -> Result<Option<Content>, String> {
    let mut header = String::new();
    if stdout
        .read_line(&mut header)
        .map_err(|error| error.to_string())?
        == 0
    {
        return Err("git cat-file closed its output early".to_string());
    }
    let header = header.trim_end();
    let mut fields = header.rsplitn(3, ' ');
    let Some(size) = fields.next().and_then(|size| size.parse::<u64>().ok()) else {
        // `<object> missing` and `<object> ambiguous` carry no body.
        return Ok(None);
    };
    if cap.is_some_and(|cap| size > cap) {
        // The body is still on the pipe and has to be consumed to keep the
        // stream framed for the next request.
        let mut sink = std::io::sink();
        let mut body = stdout.take(size.saturating_add(1));
        std::io::copy(&mut body, &mut sink).map_err(|error| error.to_string())?;
        return Ok(Some(Content::Oversized));
    }
    let size = usize::try_from(size).map_err(|error| error.to_string())?;
    let mut bytes = vec![0_u8; size];
    stdout
        .read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    let mut terminator = [0_u8; 1];
    stdout
        .read_exact(&mut terminator)
        .map_err(|error| error.to_string())?;
    Ok(Some(Content::Bytes(bytes)))
}

/// Sort and deduplicate the NUL-separated path lists, so order is the source's.
fn ordered_paths(lists: &[&[u8]]) -> Vec<String> {
    let mut paths = lists
        .iter()
        .flat_map(|list| list.split(|byte| *byte == 0))
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .collect::<Vec<_>>();
    paths.sort_unstable();
    paths.dedup();
    paths
}

/// Digest the content of every changed path, in path order.
fn digest_of(changed: &[ChangedPath]) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_field(&mut hasher, b"format", SOURCE_DIFF_FORMAT);
    for entry in changed {
        hash_field(&mut hasher, b"path", entry.path.as_bytes());
        match &entry.content {
            Some(Content::Bytes(bytes)) => hash_field(&mut hasher, b"content", bytes),
            Some(Content::Oversized) => hash_field(
                &mut hasher,
                b"content-oversized",
                MAX_UNTRACKED_FILE_BYTES.to_string().as_bytes(),
            ),
            None => hash_field(&mut hasher, b"absent", b""),
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

    /// Writing anything the source digest excludes leaves the tree clean.
    ///
    /// The set is read from `EXCLUDED_FROM_SOURCE` rather than listed here, so
    /// a path added to the exclusion is proven without anyone remembering to
    /// extend this. Iterating the constant cannot see an entry that was
    /// deleted, so each entry the corpus depends on is also asserted by name
    /// below.
    #[test]
    fn no_excluded_path_dirties_the_tree_a_record_names() {
        for pathspec in EXCLUDED_FROM_SOURCE {
            let relative = pathspec
                .strip_prefix(":!")
                .expect("Fix: every exclusion is a negative git pathspec.")
                .replace("**", "generated-artifact.json");
            let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
            crate::fixture_checkout::seeded(dir.path());
            let target = dir.path().join(&relative);
            std::fs::create_dir_all(target.parent().expect("Fix: the path names a parent."))
                .expect("Fix: create the excluded directory.");
            std::fs::write(&target, "{}\n").expect("Fix: write the excluded artifact.");

            let fingerprint = capture(dir.path()).expect("Fix: the checkout still names a commit.");

            assert!(
                fingerprint.ends_with(":dirty=false"),
                "Fix: `{relative}` is excluded from the source digest, so writing it must leave the tree clean; fingerprint={fingerprint}"
            );
        }
    }

    /// The release provenance projection is excluded, by name.
    ///
    /// Iterating the exclusion set proves whatever is in it and nothing about
    /// what was taken out, so removing this entry would leave the loop above
    /// green. Recording evidence rewrites this document, and counting it as
    /// source made every capture dirty the next one, which is a state no
    /// sequence of commits recovers from.
    #[test]
    fn the_release_provenance_projection_is_excluded_from_the_source_digest() {
        assert!(
            EXCLUDED_FROM_SOURCE.contains(&":!docs/generated/release-provenance.toml"),
            "Fix: the release provenance document is a projection of the evidence stamps; counting it as source makes recording evidence change the source the recording names"
        );
    }

    /// The exclusion covers one named projection, not every generated document.
    ///
    /// A generated document that a compile reads is source to every record
    /// that names it. Widening the exclusion to `docs/generated/**` would let
    /// a crate graph or an ownership table change under a stamp that still
    /// claimed to name the tree.
    #[test]
    fn a_generated_document_outside_the_exclusion_dirties_the_tree() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        crate::fixture_checkout::seeded(dir.path());
        let target = dir.path().join("docs/generated/crate-graph.toml");
        std::fs::create_dir_all(target.parent().expect("Fix: the path names a parent."))
            .expect("Fix: create the generated directory.");
        std::fs::write(&target, "graph = []\n").expect("Fix: write the generated document.");

        let fingerprint = capture(dir.path()).expect("Fix: the checkout still names a commit.");

        assert!(
            fingerprint.contains(":dirty=true:worktree="),
            "Fix: only the release provenance projection is excluded; every other generated document is source; fingerprint={fingerprint}"
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

    #[test]
    fn a_recording_over_added_changed_and_deleted_paths_resolves_after_it_is_committed() {
        let dir = tempfile::tempdir().expect("Fix: create a temporary directory.");
        crate::fixture_checkout::seeded(dir.path());
        std::fs::write(dir.path().join("deleted.txt"), "gone soon\n")
            .expect("Fix: write the path the recording deletes.");
        crate::fixture_checkout::commit_worktree(dir.path(), "add a second tracked path");

        // One diff carrying a change, an addition and a deletion: the deleted
        // path resolves to no object in the carrier, so the batch stream frames
        // a body-less response between two bodies.
        std::fs::write(dir.path().join("tracked.txt"), "changed\n")
            .expect("Fix: change the tracked path.");
        std::fs::write(dir.path().join("added.txt"), "new\n").expect("Fix: add an untracked path.");
        std::fs::remove_file(dir.path().join("deleted.txt"))
            .expect("Fix: delete the tracked path.");

        let fingerprint = capture(dir.path()).expect("Fix: a dirty checkout still names a commit.");
        crate::fixture_checkout::commit_worktree(dir.path(), "commit the recorded source");
        let carrier = crate::fixture_checkout::head(dir.path());

        resolves_against(dir.path(), &fingerprint, &carrier).expect(
            "Fix: the digest of what a commit carries must equal the digest of the worktree it \
             was committed from, across added, changed and deleted paths.",
        );
    }
}
