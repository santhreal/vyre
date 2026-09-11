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

use crate::gate::{Finding, GateError};

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
    if !cfg!(windows) {
        if let Some(runner) = runner.filter(|value| !value.is_empty()) {
            return PathBuf::from(runner);
        }
        let wrapper = root.join(WRAPPER);
        if wrapper.is_file() {
            return wrapper;
        }
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
    for token in output
        .split(|character: char| character.is_whitespace() || matches!(character, '`' | '"' | '\''))
    {
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

/// One compiler diagnostic, reduced to what a finding carries.
pub struct Diagnostic {
    /// The cargo target the diagnostic was emitted for.
    pub target: Option<String>,
    /// The file the primary span names.
    pub file: Option<String>,
    /// The line the primary span starts at.
    pub line: Option<u32>,
    /// The compiler's own text.
    pub message: String,
}

impl Diagnostic {
    /// A finding at this diagnostic's own span, with the path relative to `root`.
    ///
    /// `None` when the diagnostic names no file, which leaves the placement to
    /// the caller: one gate blames the manifest it was pointed at, another
    /// states the message with no location at all.
    #[must_use]
    pub fn place(&self, root: &Path, message: &str, fix: &str) -> Option<Finding> {
        let file = PathBuf::from(self.file.as_ref()?);
        let relative = file.strip_prefix(root).unwrap_or(&file);
        Some(match self.line {
            Some(line) => Finding::at(relative, line, message, fix),
            None => Finding::in_file(relative, message, fix),
        })
    }
}

/// What one cargo invocation produced.
///
/// The answers are kept apart because they mean opposite things. `found` is
/// what the compiler said about the source. `unmeasured` names a file the build
/// needed and did not find under its own build directory, which says the run
/// never reached the source at all. `status` is judged last, because a failing
/// run that emitted no diagnostic is the one shape a diagnostic-counting gate
/// can report as clean, and what that failure means differs per gate.
pub struct Run {
    /// Diagnostics the compiler emitted at a judged level.
    pub found: Vec<Diagnostic>,
    /// A build-directory path the run named that is no longer there.
    pub unmeasured: Option<String>,
    /// How the invocation ended.
    pub status: ExitStatus,
    /// Everything the invocation wrote to standard error.
    pub stderr: String,
}

impl Run {
    /// The invocation failed and named nothing a caller can act on.
    #[must_use]
    pub fn failed_silently(&self) -> bool {
        !self.status.success() && self.found.is_empty()
    }

    /// The exit code, or `-1` when a signal ended the run.
    #[must_use]
    pub fn code(&self) -> i32 {
        self.status.code().unwrap_or(-1)
    }
}

/// Run one cargo invocation and return the diagnostics it emitted.
///
/// `--message-format=json` is the only reason this is reliable: a gate that
/// scraped human output counted the same error twice as soon as cargo repeated
/// its summary. It goes before any `--`, because everything after that
/// separator reaches the compiler driver instead of cargo, and clippy-driver
/// answers an unknown option with `Unrecognized option` and exit 101 per crate.
/// Appended blindly, it turned the clippy gate into one that could only report
/// that it had not run, so the workspace was neither clippy-clean nor dirty for
/// as long as it stood. Nothing here sets a build-affecting flag or variable,
/// because build configuration is declared once in `.cargo/config.toml`.
///
/// `judge_warnings` is for a gate whose command cannot deny them. `cargo doc`
/// takes no trailing rustdoc argument, so a broken intra-doc link arrives as a
/// warning, and recording only errors let one gate report a clean workspace
/// while rustdoc under a deny flag refused the same tree.
pub fn diagnostics(
    root: &Path,
    arguments: &[&str],
    judge_warnings: bool,
) -> Result<Run, GateError> {
    let cargo = binary(root);
    let (cargo_arguments, driver_arguments) = split_at_driver(arguments);
    let output = Command::new(&cargo)
        .args(cargo_arguments)
        .arg("--message-format=json")
        .args(driver_arguments)
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "cannot run `{} {}`: {error}",
                    cargo.display(),
                    arguments.join(" ")
                ),
                "restore the cargo_full wrapper at the workspace root",
            )
        })?;
    let found = parsed(&String::from_utf8_lossy(&output.stdout), judge_warnings);
    // A build directory deleted under a running compile fails with a diagnostic
    // naming a file that is not there. The run measured nothing, so it is
    // classified before the status is judged: reporting it as a compile error
    // would blame the source for the state of the disk.
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let unmeasured = unmeasured(&stderr).or_else(|| {
        found
            .iter()
            .find_map(|diagnostic| self::unmeasured(&diagnostic.message))
    });
    Ok(Run {
        found,
        unmeasured,
        status: output.status,
        stderr,
    })
}

/// Read every judged diagnostic out of one `--message-format=json` stream.
#[must_use]
pub fn parsed(stdout: &str, judge_warnings: bool) -> Vec<Diagnostic> {
    let mut found = Vec::new();
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if value.get("reason").and_then(serde_json::Value::as_str) != Some("compiler-message") {
            continue;
        }
        let Some(message) = value.get("message") else {
            continue;
        };
        if !judged(
            message.get("level").and_then(serde_json::Value::as_str),
            judge_warnings,
        ) {
            continue;
        }
        let primary = message
            .get("spans")
            .and_then(serde_json::Value::as_array)
            .and_then(|spans| {
                spans.iter().find(|span| {
                    span.get("is_primary")
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false)
                })
            });
        found.push(Diagnostic {
            target: value
                .get("target")
                .and_then(|target| target.get("name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            file: primary
                .and_then(|span| span.get("file_name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            line: primary
                .and_then(|span| span.get("line_start"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|line| u32::try_from(line).ok()),
            message: message
                .get("message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("the compiler reported a diagnostic with no message")
                .to_string(),
        });
    }
    found
}

/// Whether a diagnostic at `level` is one the calling gate judges.
///
/// Split out because the gate that could not fail on a broken intra-doc link
/// was proven by a test that read its command line. A command line is not a
/// verdict, and the verdict was the defect: a predicate is decidable without a
/// cargo run, so what the gate counts is what gets proven.
#[must_use]
pub fn judged(level: Option<&str>, judge_warnings: bool) -> bool {
    level == Some("error") || (judge_warnings && level == Some("warning"))
}

/// Split an argument list into what cargo reads and what the compiler driver
/// reads, at the first `--`.
#[must_use]
pub fn split_at_driver<'a>(arguments: &'a [&'a str]) -> (&'a [&'a str], &'a [&'a str]) {
    match arguments.iter().position(|argument| *argument == "--") {
        Some(at) => (&arguments[..at], &arguments[at..]),
        None => (arguments, &arguments[arguments.len()..]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: precedence is the whole content of this module, and it cannot be
    /// asserted through `binary` without writing process environment that every
    /// other test in the binary shares. `resolve` is crate-private, so no
    /// integration test reaches it.
    ///
    /// Both answers this test ranks are read only off Windows: `cargo_full` is a
    /// shell script, and `resolve` skips the exported runner and the wrapper
    /// beside the root there rather than starting something Windows cannot
    /// execute.
    #[cfg(not(windows))]
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

    #[cfg(not(windows))]
    #[test]
    fn a_wrapper_beside_the_root_outranks_the_parent_toolchain() {
        let root = tempfile::tempdir().expect("temporary directory");
        std::fs::write(root.path().join(WRAPPER), "#!/bin/sh\nexec cargo \"$@\"\n")
            .expect("wrapper");

        let chosen = resolve(None, root.path(), Some(OsString::from("/toolchain/cargo")));

        assert_eq!(chosen, root.path().join(WRAPPER));
    }

    /// WHY: the two cases above are the whole of the non-Windows ranking, and
    /// on Windows the same inputs must reach a different answer. Without this
    /// case the Windows arm of `resolve` is unasserted, and deleting the
    /// `cfg!(windows)` guard would turn no test red on any host.
    #[cfg(windows)]
    #[test]
    fn the_toolchain_answers_on_windows_over_a_runner_and_a_wrapper() {
        let root = tempfile::tempdir().expect("temporary directory");
        std::fs::write(root.path().join(WRAPPER), "#!/bin/sh\nexec cargo \"$@\"\n")
            .expect("wrapper");

        let chosen = resolve(
            Some(OsString::from(r"C:\wrapper\cargo_full")),
            root.path(),
            Some(OsString::from(r"C:\toolchain\cargo.exe")),
        );

        assert_eq!(chosen, PathBuf::from(r"C:\toolchain\cargo.exe"));
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

        let chosen = resolve(Some(OsString::new()), root.path(), Some(OsString::new()));

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

    /// WHY: `cargo doc` takes no trailing rustdoc argument, so a broken
    /// intra-doc link arrives as a warning. Recording only errors let one gate
    /// report a clean workspace while rustdoc under a deny flag refused the
    /// same tree, and that gate's own proof read its command line instead of
    /// its verdict. This decides every level the compiler emits.
    #[test]
    fn only_the_gate_that_cannot_deny_a_warning_judges_one() {
        assert!(judged(Some("error"), false), "an error is always judged");
        assert!(judged(Some("error"), true), "an error is always judged");
        assert!(
            judged(Some("warning"), true),
            "a broken intra-doc link is a warning, and cargo doc has no flag to deny it"
        );
        assert!(
            !judged(Some("warning"), false),
            "clippy denies warnings on its own command line, so they arrive as errors"
        );
        for ignored in ["note", "help", "failure-note"] {
            assert!(
                !judged(Some(ignored), true),
                "`{ignored}` explains a diagnostic and is not one"
            );
        }
        assert!(
            !judged(None, true),
            "a compiler message with no level states no verdict"
        );
    }

    /// WHY: `--message-format=json` after the `--` reaches the compiler driver
    /// rather than cargo, and clippy-driver answers an unknown option with
    /// `Unrecognized option` and exit 101 per crate. Appended blindly, it
    /// turned the clippy gate into one that could only report that it had not
    /// run.
    #[test]
    fn the_driver_separator_bounds_the_cargo_arguments() {
        let clippy = [
            "clippy",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ];
        let (cargo, driver) = split_at_driver(&clippy);
        assert_eq!(cargo, ["clippy", "--workspace", "--all-targets"]);
        assert_eq!(driver, ["--", "-D", "warnings"]);

        let check = ["check", "--workspace"];
        let (cargo, driver) = split_at_driver(&check);
        assert_eq!(cargo, ["check", "--workspace"]);
        assert!(driver.is_empty());
    }

    /// WHY: every gate that starts a compile reads its verdict out of this
    /// stream, and three of them used to parse it separately. A reader that
    /// admitted a warning under a gate that denies them, or dropped the target
    /// name, misattributed or manufactured a finding. Anything that is not a
    /// judged compiler message is not a diagnostic.
    #[test]
    fn only_a_judged_compiler_message_becomes_a_diagnostic() {
        let stream = concat!(
            r#"{"reason":"compiler-artifact","target":{"name":"all_tests"}}"#,
            "\n",
            r#"{"reason":"compiler-message","target":{"name":"all_tests"},"message":{"level":"warning","message":"unused variable","spans":[]}}"#,
            "\n",
            r#"{"reason":"compiler-message","target":{"name":"all_tests"},"message":{"level":"error","message":"no method named `target_payload` found","spans":[{"file_name":"note.rs","line_start":9,"is_primary":false},{"file_name":"tests/foo.rs","line_start":12,"is_primary":true}]}}"#,
            "\n",
            "warning: this line is not json\n",
            r#"{"reason":"build-finished","success":false}"#,
            "\n",
        );

        let errors = parsed(stream, false);
        assert_eq!(errors.len(), 1, "only the error is judged");
        assert_eq!(errors[0].target.as_deref(), Some("all_tests"));
        assert_eq!(
            errors[0].file.as_deref(),
            Some("tests/foo.rs"),
            "the primary span locates the diagnostic, not the first span"
        );
        assert_eq!(errors[0].line, Some(12));
        assert_eq!(errors[0].message, "no method named `target_payload` found");

        let both = parsed(stream, true);
        assert_eq!(
            both.len(),
            2,
            "a gate with no deny flag judges warnings too"
        );
        assert_eq!(both[0].message, "unused variable");
        assert!(
            both[0].file.is_none() && both[0].line.is_none(),
            "a diagnostic with no span states no location"
        );
    }

    /// WHY: cargo states an absolute path for a target outside the workspace
    /// and a relative one for a member, and a finding that repeats an absolute
    /// path names a file that only exists on the machine that ran the gate.
    #[test]
    fn a_placed_finding_states_its_path_relative_to_the_checkout() {
        let root = Path::new("/checkout");
        let absolute = Diagnostic {
            target: None,
            file: Some("/checkout/consumers/app/src/main.rs".to_string()),
            line: Some(7),
            message: "mismatched types".to_string(),
        };
        let placed = absolute.place(root, "message", "fix").expect("placed");
        assert_eq!(
            placed.file.as_deref(),
            Some(Path::new("consumers/app/src/main.rs"))
        );
        assert_eq!(placed.line, Some(7));

        let outside = Diagnostic {
            file: Some("/elsewhere/src/lib.rs".to_string()),
            ..absolute
        };
        assert_eq!(
            outside
                .place(root, "message", "fix")
                .expect("placed")
                .file
                .as_deref(),
            Some(Path::new("/elsewhere/src/lib.rs")),
            "a path outside the checkout is stated as the compiler gave it"
        );

        let spanless = Diagnostic {
            target: None,
            file: None,
            line: None,
            message: "linker `cc` not found".to_string(),
        };
        assert!(
            spanless.place(root, "message", "fix").is_none(),
            "a diagnostic with no file leaves the placement to the caller"
        );
    }
}
