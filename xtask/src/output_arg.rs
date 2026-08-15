//! Shared `--output` parsing for evidence-producing xtask commands.

use std::fmt::{Display, Write as _};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

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

pub(crate) fn parse_define(value: &str) -> (String, Option<String>) {
    match value.split_once('=') {
        Some((name, body)) => (name.to_string(), Some(body.to_string())),
        None => (value.to_string(), None),
    }
}

pub(crate) fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
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

/// Parse `--output <path>` for `command`, falling back to `default_output`.
pub fn parse_output_arg(
    args: &[String],
    command: &str,
    description: &str,
    default_output: impl FnOnce() -> PathBuf,
) -> Result<PathBuf, String> {
    let usage = format!("USAGE:\n  cargo xtask {command} [--output PATH]\n\n  {description}");
    parse_output_options(args, command, None, &usage, default_output).map(|(output, _)| output)
}

/// Parse `--output <path>` plus one valueless `flag` for `command`.
///
/// `usage` is printed verbatim for `--help`, because a command with a second
/// option documents its own option list rather than the one-option shape
/// `parse_output_arg` renders. The loop itself is the same either way, and the
/// commands that owned a copy of it each had to be corrected separately.
pub fn parse_output_and_flag_arg(
    args: &[String],
    command: &str,
    flag: &str,
    usage: &str,
    default_output: impl FnOnce() -> PathBuf,
) -> Result<(PathBuf, bool), String> {
    parse_output_options(args, command, Some(flag), usage, default_output)
}

fn parse_output_options(
    args: &[String],
    command: &str,
    flag: Option<&str>,
    usage: &str,
    default_output: impl FnOnce() -> PathBuf,
) -> Result<(PathBuf, bool), String> {
    let mut output = None;
    let mut flag_set = false;
    let mut index = 2;
    while index < args.len() {
        match args[index].as_str() {
            "--output" => {
                let Some(path) = args.get(index + 1) else {
                    return Err("Fix: --output requires a path.".to_string());
                };
                output = Some(PathBuf::from(path));
                index += 2;
            }
            "--help" | "-h" => {
                println!("{usage}");
                std::process::exit(0);
            }
            other if Some(other) == flag => {
                flag_set = true;
                index += 1;
            }
            other => return Err(format!("Fix: unknown {command} option `{other}`.")),
        }
    }
    Ok((output.unwrap_or_else(default_output), flag_set))
}

/// Take a parsed command line, or report the usage error and exit 2.
///
/// Exit 2 separates a usage error from a gate failure, which exits 1, and CI
/// reads that difference. A dozen evidence commands wrote the same match out,
/// so one that returned the wrong code would have read as a failing gate.
pub fn parsed_or_exit<T>(parsed: Result<T, String>) -> T {
    match parsed {
        Ok(value) => value,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    }
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

/// Write `value` as pretty JSON, exiting with a `Fix:` message on failure.
///
/// The parent directory is created here, because an artifact writer that has to
/// remember to create it is one that will eventually forget.
pub fn write_json(path: &Path, value: &impl serde::Serialize) {
    create_parent_dir(path);
    let json = match render_evidence_json(value) {
        Ok(json) => json,
        Err(error) => {
            eprintln!("Fix: {error} for `{}`", path.display());
            std::process::exit(1);
        }
    };
    if let Err(error) = std::fs::write(path, json) {
        eprintln!("Fix: failed to write `{}`: {error}", path.display());
        std::process::exit(1);
    }
}

fn normalize_serialized_workspace_paths(json: &str, vyre_root: &Path, santh_root: &Path) -> String {
    let json = replace_serialized_root(json, vyre_root, "", ".");
    replace_serialized_root(&json, santh_root, "../../../../", "../../../..")
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

pub(crate) fn cargo_runner(workspace_root: &Path) -> PathBuf {
    if let Some(runner) = std::env::var_os("VYRE_CARGO_RUNNER") {
        return PathBuf::from(runner);
    }
    let local = workspace_root.join("cargo_full");
    if local.is_file() {
        return local;
    }
    PathBuf::from("cargo_full")
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

    /// Locks the shared default path contract when no override is supplied.
    #[test]
    fn no_override_uses_command_default() {
        let args = vec!["xtask".to_string(), "fixture-command".to_string()];
        assert_eq!(
            parse_output_arg(&args, "fixture-command", "description", || {
                PathBuf::from("default.json")
            }),
            Ok(PathBuf::from("default.json"))
        );
    }

    /// Locks explicit output precedence so every evidence command writes to the requested artifact.
    #[test]
    fn explicit_output_overrides_command_default() {
        let args = vec![
            "xtask".to_string(),
            "fixture-command".to_string(),
            "--output".to_string(),
            "custom.json".to_string(),
        ];
        assert_eq!(
            parse_output_arg(&args, "fixture-command", "description", || {
                PathBuf::from("default.json")
            }),
            Ok(PathBuf::from("custom.json"))
        );
    }

    /// Locks the actionable missing-value diagnostic shared by all output-producing commands.
    #[test]
    fn missing_output_value_fails_with_fix() {
        let args = vec![
            "xtask".to_string(),
            "fixture-command".to_string(),
            "--output".to_string(),
        ];
        assert_eq!(
            parse_output_arg(&args, "fixture-command", "description", PathBuf::new),
            Err("Fix: --output requires a path.".to_string())
        );
    }

    /// Locks command-specific unknown-option diagnostics after parser centralization.
    #[test]
    fn unknown_option_names_owning_command() {
        let args = vec![
            "xtask".to_string(),
            "fixture-command".to_string(),
            "--wat".to_string(),
        ];
        assert_eq!(
            parse_output_arg(&args, "fixture-command", "description", PathBuf::new),
            Err("Fix: unknown fixture-command option `--wat`.".to_string())
        );
    }

    /// Public evidence uses repository-relative paths instead of host-private Vyre paths.
    #[test]
    fn serialized_vyre_paths_are_repository_relative() {
        let json = r#"{"path":"/srv/Santh/libs/performance/matching/vyre/docs/RELEASE.md","message":"read /srv/Santh/libs/performance/matching/vyre/README.md"}"#;
        let normalized = normalize_serialized_workspace_paths(
            json,
            Path::new("/srv/Santh/libs/performance/matching/vyre"),
            Path::new("/srv/Santh"),
        );

        assert_eq!(
            normalized,
            r#"{"path":"docs/RELEASE.md","message":"read README.md"}"#
        );
    }

    /// Sibling release components retain a stable path from the public Vyre repository.
    #[test]
    fn serialized_santh_sibling_paths_use_public_relative_locations() {
        let json = r#"{"path":"/srv/Santh/tools/vyrec/README.md","root":"/srv/Santh"}"#;
        let normalized = normalize_serialized_workspace_paths(
            json,
            Path::new("/srv/Santh/libs/performance/matching/vyre"),
            Path::new("/srv/Santh"),
        );

        assert_eq!(
            normalized,
            r#"{"path":"../../../../tools/vyrec/README.md","root":"../../../.."}"#
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
