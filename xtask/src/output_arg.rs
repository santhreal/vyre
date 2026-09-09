//! Shared `--output` parsing for evidence-producing xtask commands.

use std::fmt::{Display, Write as _};
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

/// Whether `path` climbs out of the directory it is written relative to.
///
/// `earned` is how many segments it may pop before it leaves. A path built from
/// the repository root has earned none, so its first `..` is already outside; an
/// evidence entry written beside a manifest one level below the root has earned
/// exactly one. A root or prefix component makes the path absolute, which names
/// a file no clone of this repository is guaranteed to have.
#[must_use]
pub fn escapes_root(path: &Path, earned: i32) -> bool {
    let mut depth = earned;
    for component in path.components() {
        match component {
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return true;
                }
            }
            Component::Normal(_) => depth += 1,
            Component::RootDir | Component::Prefix(_) => return true,
            Component::CurDir => {}
        }
    }
    false
}

/// Read a text file, failing rather than allocating past `max_bytes`.
///
/// `context` names the cap in the error, so an operator can tell which reader
/// refused the file.
pub fn read_text_bounded(path: &Path, max_bytes: u64, context: &str) -> io::Result<String> {
    let mut reader = std::fs::File::open(path)?.take(max_bytes.saturating_add(1));
    let mut text = String::new();
    reader.read_to_string(&mut text)?;
    if text.len() as u64 > max_bytes {
        let cap = if context.is_empty() {
            "read cap".to_string()
        } else {
            format!("{context} read cap")
        };
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} exceeds {max_bytes} byte {cap}", path.display()),
        ));
    }
    Ok(text)
}

/// Resolve `path` against `base_dir` unless it is already absolute.
pub fn resolve_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        base_dir.join(candidate)
    }
}

/// Resolve a release artifact path, which is written relative to the workspace
/// root rather than to the crate that reads it.
pub fn resolve_release_artifact_path(base_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate;
    }
    if path.starts_with("release/") {
        return base_dir
            .parent()
            .map(|workspace| workspace.join(candidate))
            .unwrap_or_else(|| base_dir.join(path));
    }
    base_dir.join(candidate)
}

/// Announce a written evidence artifact and exit 1 when it carries blockers.
///
/// Every evidence command ends this way: the path is printed so a reader knows
/// what to open, and a non-empty blocker list is a failing gate. Each command
/// carried its own copy, so a command that forgot the exit would have written a
/// blocked artifact and reported success.
///
/// The blockers themselves go to stderr before the exit. A command that printed
/// only the artifact path and exited 1 left a caller with an exit code and no
/// cause, so a wrong binary reporting "not implemented" and a real gate failure
/// were indistinguishable at the terminal. Reading the reason out of the JSON is
/// not the caller's job.
pub fn report_evidence_artifact(command: &str, output: &Path, blockers: &[impl Display]) {
    println!("{command}: wrote {}", output.display());
    let Some(report) = evidence_blocker_report(command, output, blockers) else {
        return;
    };
    eprint!("{report}");
    std::process::exit(1);
}

/// Render the stderr blocker report, or `None` when the gate passed.
///
/// Separate from the printing so the contract can be asserted without a process
/// exit: every blocker reaches the reader, not just their count.
fn evidence_blocker_report(
    command: &str,
    output: &Path,
    blockers: &[impl Display],
) -> Option<String> {
    if blockers.is_empty() {
        return None;
    }
    let mut report = format!("{command}: {} blocker(s):\n", blockers.len());
    for blocker in blockers {
        let _ = writeln!(report, "  {blocker}");
    }
    let _ = writeln!(
        report,
        "Fix: resolve every blocker listed above; the full record is in {}.",
        output.display()
    );
    Some(report)
}

/// Create `path`'s parent directory, reporting the failure and exiting 1.
pub fn create_parent_dir(path: &Path) {
    let Some(parent) = path.parent() else {
        return;
    };
    if let Err(error) = std::fs::create_dir_all(parent) {
        eprintln!("Fix: failed to create `{}`: {error}", parent.display());
        std::process::exit(1);
    }
}

/// Render `value` as the exact bytes an evidence artifact holds on disk.
///
/// Pretty JSON, workspace paths normalised to repository-relative form, one
/// trailing newline. A gate that regenerates an artifact in memory and a writer
/// that puts one on disk must agree byte for byte, or every comparison reports
/// a difference that is only the serializer, so both read this.
///
/// # Errors
///
/// Returns the serializer's message when `value` cannot be represented as JSON.
pub fn render_evidence_json(value: &impl serde::Serialize) -> Result<String, String> {
    let json = serde_json::to_string_pretty(value).map_err(|error| error.to_string())?;
    let vyre_root = crate::checkout::checkout_root();
    let json = normalize_serialized_workspace_paths(&json, &vyre_root, outer_root(&vyre_root));
    Ok(format!("{json}\n"))
}

/// The enclosing release root a sibling component is named relative to.
///
/// The checkout sits some distance below a root that carries sibling
/// components, and an artifact names one of those by climbing out of the
/// repository. The distance is a property of the checkout, not a constant: a
/// detached worktree sits closer to the root than the primary checkout does.
/// Walking a fixed four parents from a two-level checkout resolved the root to
/// `/`, and substituting `/` rewrote every path separator in the artifact, so
/// `xtask-evidence/Cargo.toml` was recorded as
/// `xtask-evidence../../../..Cargo.toml`.
///
/// A candidate keeps the filesystem root plus two names, so neither the root
/// itself nor a mount point is ever substituted. A checkout with no enclosing
/// root has no sibling to name and is left alone. The prefix is always the true
/// climb from the checkout to the candidate the walk landed on.
fn outer_root(vyre_root: &Path) -> Option<(PathBuf, String)> {
    let mut candidate = vyre_root.to_path_buf();
    let mut climbed = 0usize;
    while climbed < 4 {
        let Some(parent) = candidate.parent() else {
            break;
        };
        if parent.components().count() < 3 {
            break;
        }
        candidate = parent.to_path_buf();
        climbed += 1;
    }
    (climbed > 0).then(|| (candidate, "../".repeat(climbed)))
}

/// Write `value` as pretty JSON, exiting with a `Fix:` message on failure.
///
/// The parent directory is created here, because an artifact writer that has to
/// remember to create it is one that will eventually forget.
pub fn write_json(path: &Path, value: &impl serde::Serialize) {
    create_parent_dir(path);
    let json = match render_evidence_json(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: failed to serialize `{}`: {error}", path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = std::fs::write(path, json) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn normalize_serialized_workspace_paths(
    json: &str,
    vyre_root: &Path,
    outer: Option<(PathBuf, String)>,
) -> String {
    let json = replace_serialized_root(json, vyre_root, "", ".");
    let Some((outer_root, descendant_prefix)) = outer else {
        return json;
    };
    let exact = descendant_prefix.trim_end_matches('/');
    replace_serialized_root(&json, &outer_root, &descendant_prefix, exact)
}

fn replace_serialized_root(
    json: &str,
    root: &Path,
    descendant_prefix: &str,
    exact_replacement: &str,
) -> String {
    let Ok(encoded) = serde_json::to_string(root.to_string_lossy().as_ref()) else {
        return json.to_string();
    };
    let Some(fragment) = encoded
        .strip_prefix('"')
        .and_then(|text| text.strip_suffix('"'))
    else {
        return json.to_string();
    };
    json.replace(&format!("{fragment}/"), descendant_prefix)
        .replace(&format!("{fragment}\\\\\\"), descendant_prefix)
        .replace(fragment, exact_replacement)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: `release-evidence` spawned the wrong binary for twelve of its
    /// thirteen children and every one reported "not implemented". Nothing
    /// noticed for a merge cycle, because a failing evidence command printed the
    /// artifact path and exited 1: a wrong process and a real gate failure were
    /// the same two lines. Every blocker the artifact records must reach stderr,
    /// so the cause is on the terminal and not only inside the JSON.
    #[test]
    fn every_blocker_reaches_the_reader() {
        let blockers = [
            "vyre-test-support defines 1 feature(s) but no explicit default feature policy"
                .to_string(),
            "23 release-blocking source hygiene finding(s) remain".to_string(),
        ];

        let report = evidence_blocker_report("feature-matrix", Path::new("out/m.json"), &blockers)
            .expect("Fix: a non-empty blocker list must produce a report.");

        for blocker in &blockers {
            assert!(
                report.contains(blocker.as_str()),
                "Fix: `{blocker}` never reached the reader; report was:\n{report}"
            );
        }
        assert!(report.contains("feature-matrix: 2 blocker(s):"), "{report}");
        assert!(report.contains("out/m.json"), "{report}");
    }

    /// A passing gate says nothing extra, so a clean run stays one line.
    #[test]
    fn an_empty_blocker_list_produces_no_report() {
        let empty: [String; 0] = [];
        assert_eq!(
            evidence_blocker_report("feature-matrix", Path::new("out/m.json"), &empty),
            None
        );
    }

    /// Public evidence uses repository-relative paths instead of host-private Vyre paths.
    #[test]
    fn serialized_vyre_paths_are_repository_relative() {
        let vyre_root = Path::new("/srv/Santh/libs/performance/matching/vyre");
        let json = r#"{"path":"/srv/Santh/libs/performance/matching/vyre/docs/RELEASE.md","message":"read /srv/Santh/libs/performance/matching/vyre/README.md"}"#;
        let normalized =
            normalize_serialized_workspace_paths(json, vyre_root, outer_root(vyre_root));

        assert_eq!(
            normalized,
            r#"{"path":"docs/RELEASE.md","message":"read README.md"}"#
        );
    }

    /// Sibling release components retain a stable path from the public Vyre repository.
    #[test]
    fn serialized_santh_sibling_paths_use_public_relative_locations() {
        let vyre_root = Path::new("/srv/Santh/libs/performance/matching/vyre");
        let json = r#"{"path":"/srv/Santh/tools/vyrec/README.md","root":"/srv/Santh"}"#;
        let normalized =
            normalize_serialized_workspace_paths(json, vyre_root, outer_root(vyre_root));

        assert_eq!(
            normalized,
            r#"{"path":"../../../../tools/vyrec/README.md","root":"../../../.."}"#
        );
    }

    /// WHY: closes the class "the climb out of the repository is a constant".
    /// The prefix is the distance from the checkout to its enclosing root, and a
    /// detached worktree sits closer to that root than the primary checkout. A
    /// fixed four-parent walk from a two-level checkout resolved the root to `/`
    /// and rewrote every path separator, recording `xtask-evidence/Cargo.toml`
    /// as `xtask-evidence../../../..Cargo.toml` in every generated artifact.
    #[test]
    fn the_climb_out_of_the_repository_matches_the_checkout_depth() {
        let deep = Path::new("/srv/Santh/libs/performance/matching/vyre");
        let shallow = Path::new("/srv/Santh/worktrees/vyre-device");
        assert_eq!(
            outer_root(deep),
            Some((PathBuf::from("/srv/Santh"), "../../../../".to_string()))
        );
        assert_eq!(
            outer_root(shallow),
            Some((PathBuf::from("/srv/Santh"), "../../".to_string()))
        );

        let json = r#"{"manifest":"/srv/Santh/worktrees/vyre-device/xtask-evidence/Cargo.toml","sibling":"/srv/Santh/tools/vyrec/README.md"}"#;
        assert_eq!(
            normalize_serialized_workspace_paths(json, shallow, outer_root(shallow)),
            r#"{"manifest":"xtask-evidence/Cargo.toml","sibling":"../../tools/vyrec/README.md"}"#
        );
    }

    /// WHY: a checkout with no enclosing root has no sibling component to name,
    /// and substituting a single top-level directory would replace a path
    /// separator with a climb. Nothing outside the repository is rewritten.
    #[test]
    fn a_top_level_checkout_rewrites_nothing_outside_itself() {
        let root = Path::new("/vyre");
        assert_eq!(outer_root(root), None);

        let json = r#"{"manifest":"/vyre/xtask/Cargo.toml","other":"/opt/tool/README.md"}"#;
        assert_eq!(
            normalize_serialized_workspace_paths(json, root, outer_root(root)),
            r#"{"manifest":"xtask/Cargo.toml","other":"/opt/tool/README.md"}"#
        );
    }

    /// WHY: a release artifact path is written relative to the workspace root but
    /// resolved by a crate whose base directory is one level down, so the two
    /// resolvers must disagree on exactly the `release/` prefix and agree
    /// everywhere else. Resolving a release path against the crate directory
    /// writes evidence into `xtask/release/`, where no reader looks.
    #[test]
    fn only_release_prefixed_relative_paths_climb_to_the_workspace_root() {
        let base = Path::new("/w/xtask");
        assert_eq!(
            resolve_release_artifact_path(base, "release/evidence/a.json"),
            PathBuf::from("/w/release/evidence/a.json")
        );
        assert_eq!(
            resolve_path(base, "release/evidence/a.json"),
            PathBuf::from("/w/xtask/release/evidence/a.json")
        );
        for path in ["docs/a.md", "releases/a.json", "not-release/a.json"] {
            assert_eq!(
                resolve_release_artifact_path(base, path),
                resolve_path(base, path),
                "`{path}` does not carry the release prefix and must not climb"
            );
        }
    }

    /// WHY: an absolute path is already resolved, and joining a base onto it
    /// silently produces the base again on Unix. Both resolvers must return it
    /// unchanged.
    #[test]
    fn an_absolute_path_is_returned_unchanged_by_both_resolvers() {
        let base = Path::new("/w/xtask");
        for path in ["/tmp/a.json", "/w/release/evidence/a.json"] {
            assert_eq!(resolve_path(base, path), PathBuf::from(path));
            assert_eq!(
                resolve_release_artifact_path(base, path),
                PathBuf::from(path)
            );
        }
    }
}
