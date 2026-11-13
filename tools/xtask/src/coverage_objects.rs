//! Instrumented objects a coverage run did not build.
//!
//! `cargo llvm-cov report` reads every executable in the instrumented profile
//! directory. On a persistent runner that directory also holds builds of older
//! commits, of other feature sets and of other checkouts. Their regions have no
//! profile in the current run, so each one is reported as missed lines. This
//! module selects those executables from the `compiler-artifact` records of
//! the run, so the report reads only what the run built.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

/// One `compiler-artifact` record, reduced to the fields the selection reads.
#[derive(Deserialize)]
struct Artifact {
    reason: String,
    #[serde(default)]
    filenames: Vec<PathBuf>,
    #[serde(default)]
    executable: Option<PathBuf>,
    #[serde(default)]
    profile: Option<Profile>,
}

#[derive(Deserialize)]
struct Profile {
    test: bool,
}

/// Executables in the profile directory that no artifact record of the run
/// lists, and the count of files the run lists.
#[derive(Debug, Eq, PartialEq)]
pub struct Selection {
    /// Executables to delete before the report.
    pub stale: Vec<PathBuf>,
    /// Files the run listed that exist on disk, plus the profile's examples.
    pub kept: usize,
}

/// Select the stale executables from cargo's JSON message stream.
///
/// The instrumented `deps` directory is the one directory every test
/// executable of the run was linked into, and the profile directory is its
/// parent. Executables in `examples` are kept: a test may build an example
/// with a cargo invocation of its own, which no record of the run lists.
///
/// # Errors
///
/// Returns the corrective action when a record is malformed, when no record
/// lists a test executable, or when test executables were linked into more
/// than one directory.
pub fn select(messages: &str) -> Result<Selection, String> {
    let mut used = BTreeSet::new();
    let mut deps = BTreeSet::new();
    for line in messages.lines() {
        // Cargo prints `reason` first, but a record is JSON and its key order
        // carries no meaning, so the cheap filter matches the value anywhere.
        if !line.starts_with('{') || !line.contains("\"compiler-artifact\"") {
            continue;
        }
        let artifact: Artifact = serde_json::from_str(line).map_err(|error| {
            format!("Fix: a compiler-artifact record does not parse: {error}. Pass cargo's --message-format=json output unmodified.")
        })?;
        if artifact.reason != "compiler-artifact" {
            continue;
        }
        used.extend(artifact.filenames);
        if let Some(executable) = artifact.executable {
            if artifact.profile.is_some_and(|profile| profile.test) {
                if let Some(parent) = executable.parent() {
                    deps.insert(parent.to_path_buf());
                }
            }
            used.insert(executable);
        }
    }
    let mut deps = deps.into_iter();
    let Some(deps_dir) = deps.next() else {
        return Err("Fix: no compiler-artifact record lists a test executable, so nothing identifies the stale builds. Pass the JSON messages of the coverage run.".to_string());
    };
    if let Some(other) = deps.next() {
        return Err(format!(
            "Fix: test executables were linked into `{}` and `{}`; run the coverage build with one target and one profile.",
            deps_dir.display(),
            other.display()
        ));
    }
    let profile_dir = deps_dir.parent().unwrap_or(&deps_dir).to_path_buf();
    let mut keep: BTreeSet<(u64, u64)> = used.iter().filter_map(|path| identity(path)).collect();
    keep.extend(
        executables(&profile_dir.join("examples"))
            .iter()
            .filter_map(|path| identity(path)),
    );
    let mut stale: Vec<PathBuf> = executables(&deps_dir)
        .into_iter()
        .chain(executables(&profile_dir))
        .filter(|path| identity(path).is_some_and(|id| !keep.contains(&id)))
        .collect();
    stale.sort();
    Ok(Selection {
        stale,
        kept: keep.len(),
    })
}

/// Delete every executable [`select`] returns.
///
/// # Errors
///
/// Returns the selection error, or the path that could not be deleted.
pub fn prune(messages: &str) -> Result<Selection, String> {
    let selection = select(messages)?;
    for path in &selection.stale {
        fs::remove_file(path).map_err(|error| {
            format!(
                "Fix: cannot delete stale instrumented object `{}`: {error}",
                path.display()
            )
        })?;
    }
    Ok(selection)
}

/// Regular executable files directly inside `directory`, excluding `.d`
/// dependency files. A missing directory holds none.
fn executables(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_none_or(|extension| extension != "d"))
        .filter(|path| is_executable(path))
        .collect()
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(path).is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o100 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .is_some_and(|extension| extension == "exe" || extension == "dll")
}

/// The file identity a hard link shares, so a listed name keeps its links.
#[cfg(unix)]
fn identity(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    let meta = fs::metadata(path).ok()?;
    meta.is_file().then(|| (meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn identity(path: &Path) -> Option<(u64, u64)> {
    use std::hash::{Hash, Hasher};
    let canonical = fs::canonicalize(path).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    canonical.hash(&mut hasher);
    Some((0, hasher.finish()))
}
