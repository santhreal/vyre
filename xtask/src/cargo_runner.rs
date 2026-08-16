//! The cargo a gate runs.
//!
//! Two answers to one question lived in this workspace: a `CARGO`-then-`PATH`
//! lookup copied into four gates, and a wrapper lookup in `output_arg` that ten
//! more called. A copy is where a default diverges, and these two diverged on
//! the thing that decides what gets compiled. The lookup is here now, once.
//!
//! Job count and target directory are declared outside this module, so nothing
//! here sets either.

use std::ffi::OsString;
use std::io::{BufRead, BufReader, Result as IoResult};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

/// Environment variable the workspace wrapper exports, naming itself.
const RUNNER_VARIABLE: &str = "VYRE_CARGO_RUNNER";

/// Environment variable cargo exports to every process it starts.
const CARGO_VARIABLE: &str = "CARGO";

/// The wrapper's file name at the workspace root.
const WRAPPER: &str = "cargo_full";

/// Pick the cargo to start from the environment and the workspace root.
///
/// The order is the order of authority. `VYRE_CARGO_RUNNER` is set by the
/// wrapper to its own path, so it names the checkout being judged and wins. A
/// wrapper beside `root` comes next, because the target directory is derived
/// from the wrapper's own location: a child started through a bare cargo with a
/// scrubbed environment compiles a member into a directory another checkout
/// already owns, and the two builds produce the same unit hash for different
/// source. `CARGO` follows, carrying the toolchain that started this process, so
/// a child of a `+nightly` run is not silently built by whatever is first on
/// `PATH`. The bare name is last.
fn resolve(runner: Option<OsString>, root: &Path, cargo: Option<OsString>) -> PathBuf {
    if let Some(runner) = runner.filter(|value| !value.is_empty()) {
        return PathBuf::from(runner);
    }
    let wrapper = root.join(WRAPPER);
    if wrapper.is_file() {
        return wrapper;
    }
    if let Some(cargo) = cargo.filter(|value| !value.is_empty()) {
        return PathBuf::from(cargo);
    }
    PathBuf::from("cargo")
}

/// The cargo binary this process should start for a build rooted at `root`.
#[must_use]
pub fn binary(root: &Path) -> PathBuf {
    resolve(
        std::env::var_os(RUNNER_VARIABLE),
        root,
        std::env::var_os(CARGO_VARIABLE),
    )
}

/// A cargo command rooted at `root`.
#[must_use]
pub fn command(root: &Path) -> Command {
    let mut command = Command::new(binary(root));
    command.current_dir(root);
    command
}

/// Run a long command, streaming its output and keeping its diagnostics.
///
/// A sweep or a workspace build is watched while it runs, so its output goes to
/// the terminal as it arrives. Reading the exit status alone throws the text
/// away, and a gate with no text cannot tell a compile that found a defect from
/// one whose build directory was deleted under it. Standard output is inherited
/// so the child writes straight to the terminal with no relay, and standard
/// error is read line by line and echoed as it arrives, which keeps the two in
/// the order the child produced them and cannot deadlock: nothing waits on a
/// pipe the child is not writing.
pub fn run_streaming(command: &mut Command) -> IoResult<(ExitStatus, String)> {
    command.stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let mut diagnostics = String::new();
    if let Some(stream) = child.stderr.take() {
        for line in BufReader::new(stream).lines() {
            let line = line?;
            eprintln!("{line}");
            diagnostics.push_str(&line);
            diagnostics.push('\n');
        }
    }
    let status = child.wait()?;
    Ok((status, diagnostics))
}

/// Directory segments cargo writes inside a profile directory.
const BUILD_SEGMENTS: &[&str] = &["/deps/", "/.fingerprint/", "/incremental/", "/build/"];

/// Profile directories cargo builds into.
const PROFILE_SEGMENTS: &[&str] = &["/debug/", "/release/"];

/// The first build-directory path a diagnostic names that is no longer there.
///
/// A build whose output directory is deleted while it runs fails with a
/// diagnostic naming a missing rlib, rmeta, dep-info or fingerprint file. That
/// failure measured nothing: the compiler never read the code the gate was
/// pointed at, and the error describes the disk. A gate that reports it as a
/// finding manufactures one, so the classification is answered here, once, for
/// every gate that starts a compile.
///
/// The question is asked of the path and not of the message text, because the
/// wording differs per diagnostic and per toolchain while the missing file is
/// the same fact in all of them. A path is a build path only when it carries a
/// profile directory and one of cargo's own directories inside it, so a source
/// file under a directory called `build` is not mistaken for one. A path under
/// a build directory that still exists is a real diagnostic and is left alone,
/// and so is a build path under a profile name this does not know: reporting a
/// real finding for a phantom is recoverable, and hiding a real one is not.
#[must_use]
pub fn unmeasured(output: &str) -> Option<String> {
    for token in output.split(|character: char| {
        character.is_whitespace() || matches!(character, '`' | '"' | '\'')
    }) {
        let candidate = token.trim_end_matches([',', ')', ';', ':', '.']);
        if !candidate.starts_with('/') {
            continue;
        }
        let profile = PROFILE_SEGMENTS
            .iter()
            .find_map(|segment| candidate.find(segment).map(|at| at + segment.len()));
        let Some(after_profile) = profile else {
            continue;
        };
        if !BUILD_SEGMENTS
            .iter()
            .any(|segment| candidate[after_profile.saturating_sub(1)..].contains(segment))
        {
            continue;
        }
        if Path::new(candidate).exists() {
            continue;
        }
        return Some(candidate.to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: precedence is the whole content of this module, and it cannot be
    /// asserted through `binary` without writing process environment that every
    /// other test in the binary shares. `resolve` is crate-private, so no
    /// integration test reaches it.
    #[test]
    fn the_exported_runner_outranks_every_other_answer() {
        let root = Path::new("/does/not/exist");
        let chosen = resolve(
            Some(OsString::from("/wrapper/cargo_full")),
            root,
            Some(OsString::from("/toolchain/cargo")),
        );

        assert_eq!(chosen, PathBuf::from("/wrapper/cargo_full"));
    }

    #[test]
    fn a_wrapper_beside_the_root_outranks_the_parent_toolchain() {
        let root = tempfile::tempdir().expect("temporary directory");
        std::fs::write(root.path().join(WRAPPER), "#!/bin/sh\nexec cargo \"$@\"\n")
            .expect("wrapper");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, root.path().join(WRAPPER));
    }

    #[test]
    fn the_parent_toolchain_answers_when_no_wrapper_is_beside_the_root() {
        let root = tempfile::tempdir().expect("temporary directory");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, PathBuf::from("/toolchain/cargo"));
    }

    #[test]
    fn an_empty_variable_is_not_an_answer() {
        let root = tempfile::tempdir().expect("temporary directory");

        let chosen = resolve(
            Some(OsString::new()),
            root.path(),
            Some(OsString::new()),
        );

        assert_eq!(chosen, PathBuf::from("cargo"));
    }

    #[test]
    fn a_directory_named_like_the_wrapper_is_not_the_wrapper() {
        let root = tempfile::tempdir().expect("temporary directory");
        std::fs::create_dir(root.path().join(WRAPPER)).expect("directory");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, PathBuf::from("/toolchain/cargo"));
    }

    /// WHY: this is the rule that keeps a deleted build directory from
    /// manufacturing a finding. The classifier is reached through `unmeasured`,
    /// which is public, but the cases below need a path that provably does not
    /// exist and one that provably does, so they are written where the
    /// temporary directory can supply both.
    #[test]
    fn a_diagnostic_naming_a_vanished_build_file_is_unmeasured() {
        let text = "error: couldn't read /target/debug/deps/libunicode_ident-1.rmeta: No such file or directory (os error 2)";

        assert_eq!(
            unmeasured(text).as_deref(),
            Some("/target/debug/deps/libunicode_ident-1.rmeta")
        );
    }

    /// A build file that is still there is a real diagnostic about real code.
    #[test]
    fn a_diagnostic_naming_a_present_build_file_is_measured() {
        let root = tempfile::tempdir().expect("temporary directory");
        let deps = root.path().join("debug/deps");
        std::fs::create_dir_all(&deps).expect("build directory");
        let artifact = deps.join("libthing-1.rmeta");
        std::fs::write(&artifact, "").expect("artifact");

        let text = format!("error: something about {}", artifact.display());

        assert_eq!(unmeasured(&text), None);
    }

    /// A source path is never a build path, whatever it is missing.
    #[test]
    fn a_diagnostic_about_source_is_measured() {
        let text = "error[E0432]: unresolved import `crate::reduce`\n  --> /checkout/vyre-libs/src/nn/norm/rms_norm.rs:13:5";

        assert_eq!(unmeasured(text), None);
    }

    /// The path is read out of a quoted diagnostic as well as a bare one, since
    /// cargo quotes the file in some messages and not in others.
    #[test]
    fn a_quoted_path_is_read_the_same_way() {
        let text = "error: failed to write `/target/debug/.fingerprint/xtask-1/dep-lib-xtask`";

        assert_eq!(
            unmeasured(text).as_deref(),
            Some("/target/debug/.fingerprint/xtask-1/dep-lib-xtask")
        );
    }
}
