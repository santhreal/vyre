//! Run a gate that is implemented in a crate linking vyre.
//!
//! This crate links no vyre crate, so those gates cannot be called: the owning
//! package is built on demand and executed as a child process. Delegation is a
//! property of one gate and not a category, so a delegated gate answers the
//! same contract as a local one. The child serialises its `Report` on stdout
//! and the parent renders it, which is the whole protocol; cargo's own output
//! is captured rather than inherited so a `Compiling ...` line cannot reach the
//! report.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use serde_json::Value;

use crate::gate::{Gate, GateCtx, GateError, Report};

/// Package whose binary owns the subcommand table.
const DISPATCHER_PACKAGE: &str = "xtask";

/// Path of the `xtask` dispatcher binary, resolved once per process.
///
/// `xtask` owns the subcommand table, so running a subcommand means running
/// that binary and letting it route. A sibling build-task binary must not spawn
/// its own path: it is not the dispatcher, so every subcommand it does not
/// itself implement fails against its own table as unimplemented while the
/// reported command still reads `xtask <name>`. That is what `release-evidence`
/// did to twelve of the thirteen subcommands it drives once it moved out of
/// `xtask`, and the misreported command name is why it read as those gates
/// failing.
pub fn dispatcher() -> &'static Path {
    static DISPATCHER: LazyLock<PathBuf> = LazyLock::new(|| {
        let current = std::env::current_exe().unwrap_or_else(|error| {
            eprintln!("Fix: cannot resolve the running binary: {error}. Rebuild it with `./cargo_full build -p {DISPATCHER_PACKAGE}`.");
            std::process::exit(1);
        });
        dispatcher_from(&current).unwrap_or_else(|| {
            build(DISPATCHER_PACKAGE).unwrap_or_else(|error| {
                eprintln!("{error}");
                std::process::exit(1);
            })
        })
    });
    DISPATCHER.as_path()
}

/// `current` when the running binary is already the dispatcher, `None` when a
/// sibling build-task binary is running and the dispatcher has to be built.
fn dispatcher_from(current: &Path) -> Option<PathBuf> {
    (current.file_stem()? == DISPATCHER_PACKAGE).then(|| current.to_path_buf())
}

/// Build the package that implements `name`, run it, and read back the report.
///
/// The child writes one JSON `Report` on stdout and nothing else, so anything
/// it printed on stderr belongs to the build or to a crash and is carried into
/// the error. A child that cannot run is a `GateError` rather than a clean
/// report: a gate that failed to execute has not judged the tree.
pub fn run_child_gate(package: &str, name: &str, ctx: &GateCtx) -> Result<Report, GateError> {
    let executable = build(package)?;
    let output = Command::new(&executable)
        .arg(name)
        .args(&ctx.args)
        .current_dir(&ctx.root)
        .output();
    // The copy exists only for this run, and the next gate makes its own.
    let _ = fs::remove_file(&executable);
    let output = output.map_err(|error| {
        GateError::new(
            format!("cannot run {} for `{name}`: {error}", executable.display()),
            format!("rebuild it with `./cargo_full build -p {package}`"),
        )
    })?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`{name}` exited {} in {package}: {}",
                output.status.code().unwrap_or(-1),
                stderr.trim()
            ),
            format!("run `./cargo_full run --bin xtask -- {name}` and fix what it reports"),
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        GateError::new(
            format!(
                "`{name}` in {package} did not return a report: {error}; it printed {:?} and {:?}",
                String::from_utf8_lossy(&output.stdout).trim(),
                stderr.trim()
            ),
            format!("a gate returns a Report and prints nothing; make `{name}` return its findings instead of printing them"),
        )
    })
}

/// Run a delegated binary's `main`: help, then the one gate it was asked for.
///
/// Both delegated crates are entered the same way, because `xtask` enters them
/// the same way: it hands a gate name and that gate's flags to a binary whose
/// only job is to resolve the name against its own table. Each `main` used to
/// spell that out, so the two could disagree about which exit code an
/// unimplemented name gets, and CI reads that code.
///
/// Stdout is the protocol. The child prints one JSON `Report` and nothing else,
/// so a converted gate returns everything it has to say and prints none of it.
pub fn run_delegated_main(package: &str, purpose: &str, gates: &[&dyn Gate]) -> ! {
    let args: Vec<String> = std::env::args().collect();
    if args
        .iter()
        .skip(1)
        .any(|argument| argument == "--help" || argument == "-h")
    {
        print_dispatch_help(package, purpose, gates.iter().map(|gate| gate.name()));
        std::process::exit(0);
    }
    let Some(name) = args.get(1) else {
        eprintln!("Fix: missing subcommand. Run `./cargo_full run --bin xtask -- --help`.");
        std::process::exit(1);
    };
    let Some(gate) = gates.iter().find(|gate| gate.name() == name.as_str()) else {
        eprintln!("Fix: `{name}` is not a subcommand of {package}. Run `./cargo_full run --bin xtask -- --help`.");
        std::process::exit(1);
    };
    let root = crate::checkout::checkout_root();
    let ctx = GateCtx::new(root, args[2..].to_vec());
    match gate.run(&ctx) {
        Ok(report) => match serde_json::to_string(&report) {
            Ok(json) => {
                println!("{json}");
                std::process::exit(0);
            }
            Err(error) => {
                eprintln!("Fix: cannot serialise the report of `{name}`: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

/// Print a delegated binary's help: how to invoke it, and what it dispatches.
///
/// `xtask` is the documented entry point, so this help exists to answer `--help`
/// without executing anything and to name what the binary can run. Both
/// delegated binaries print the same shape, so the shape lives here beside the
/// delegation rather than being written out again in each `main`.
///
/// `subcommands` is expected to come from the callee's own dispatch table, so
/// the printed roster cannot drift from what it will actually accept.
pub fn print_dispatch_help(
    package: &str,
    purpose: &str,
    subcommands: impl IntoIterator<Item = &'static str>,
) {
    println!("USAGE");
    println!("  ./cargo_full run -p {package} -- <subcommand> [options]");
    println!();
    println!("{purpose}");
    println!();
    println!("Run `./cargo_full run --bin xtask -- --help` for every command, and");
    println!("`./cargo_full run --bin xtask -- <subcommand> --help` for one command.");
    println!();
    println!("SUBCOMMANDS:");
    for name in subcommands {
        println!("  {name}");
    }
}

/// Cargo binary that is building this process, so the child build matches it.
fn cargo() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

/// Build one delegated crate and return the path of the binary cargo produced.
///
/// A crate that does not compile is a gate that could not run, so the compiler
/// diagnostics travel in the error rather than to this process's stderr: the
/// caller may be the sweep, which renders every gate's outcome in one place.
fn build(package: &str) -> Result<PathBuf, GateError> {
    let output = Command::new(cargo())
        .args([
            "build",
            "--quiet",
            "-p",
            package,
            "--bin",
            package,
            "--message-format=json",
        ])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run cargo to build {package}: {error}"),
                "install a cargo that the workspace configuration selects".to_string(),
            )
        })?;
    if !output.status.success() {
        let mut diagnostics = String::from_utf8_lossy(&output.stderr).into_owned();
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some(rendered) = rendered_diagnostic(line) {
                diagnostics.push_str(&rendered);
            }
        }
        return Err(GateError::new(
            format!("`{package}` does not compile:\n{}", diagnostics.trim_end()),
            format!("`{package}` must compile before the gates it implements can run"),
        ));
    }
    let built =
        executable_from(&String::from_utf8_lossy(&output.stdout), package).ok_or_else(|| {
            GateError::new(
                format!("cargo built `{package}` without reporting an executable"),
                format!("check that `{package}` declares a binary named `{package}`"),
            )
        })?;
    private_copy(&built, package)
}

/// Copy the binary cargo just built out of the shared target directory.
///
/// `target/debug/<package>` is one path, and cargo hashes a workspace member by
/// its path relative to the workspace root, so several checkouts sharing a
/// target directory compute the same unit hash and overwrite each other's
/// artifact. Measured 2026-08-15: a sweep that took minutes ran another
/// checkout's `xtask`, which reported four registered subsets as unregistered
/// because that binary's registry predated them. The copy is named for this
/// process, so the child that judges this tree is the child this build made.
fn private_copy(built: &Path, package: &str) -> Result<PathBuf, GateError> {
    let copy = std::env::temp_dir().join(format!("vyre-gate-{package}-{}", std::process::id()));
    fs::copy(built, &copy).map_err(|error| {
        GateError::new(
            format!(
                "cannot copy {} to {}: {error}",
                built.display(),
                copy.display()
            ),
            "make the temporary directory writable; a binary read out of the shared target directory may have been built by another checkout".to_string(),
        )
    })?;
    Ok(copy)
}

/// The rendered compiler message carried by one `--message-format=json` line.
fn rendered_diagnostic(line: &str) -> Option<String> {
    let value: Value = serde_json::from_str(line).ok()?;
    if value.get("reason").and_then(Value::as_str)? != "compiler-message" {
        return None;
    }
    Some(value.get("message")?.get("rendered")?.as_str()?.to_string())
}

/// The binary path cargo reported for `package`'s own binary target.
fn executable_from(stdout: &str, package: &str) -> Option<PathBuf> {
    stdout.lines().rev().find_map(|line| {
        let value: Value = serde_json::from_str(line).ok()?;
        if value.get("reason").and_then(Value::as_str)? != "compiler-artifact" {
            return None;
        }
        if value.get("target")?.get("name")?.as_str()? != package {
            return None;
        }
        Some(PathBuf::from(value.get("executable")?.as_str()?))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the dispatcher locates the delegated binary from cargo's own report
    /// instead of guessing a target directory layout. A reader that accepted any
    /// artifact line would return the path of a dependency's build script.
    #[test]
    fn executable_comes_from_the_named_binary_target() {
        let stdout = concat!(
            r#"{"reason":"compiler-artifact","target":{"name":"walkdir"},"executable":null}"#,
            "\n",
            r#"{"reason":"compiler-artifact","target":{"name":"build-script-build"},"executable":"/t/build-script-build"}"#,
            "\n",
            r#"{"reason":"compiler-artifact","target":{"name":"xtask-registry"},"executable":"/t/xtask-registry"}"#,
            "\n",
            r#"{"reason":"build-finished","success":true}"#,
        );
        assert_eq!(
            executable_from(stdout, "xtask-registry"),
            Some(PathBuf::from("/t/xtask-registry"))
        );
    }

    /// WHY: a successful build that produced no binary for the named package is
    /// a wiring mistake, not a runnable command. Returning `None` makes the
    /// dispatcher say so instead of executing whatever path came last.
    #[test]
    fn missing_binary_artifact_is_not_resolved() {
        let stdout = concat!(
            r#"{"reason":"compiler-artifact","target":{"name":"xtask"},"executable":"/t/xtask"}"#,
            "\n",
            r#"{"reason":"build-finished","success":true}"#,
        );
        assert_eq!(executable_from(stdout, "xtask-registry"), None);
    }

    /// WHY: a compiler warning on a delegated crate must reach the operator, and
    /// only the rendered field is human-readable. Every other json line is
    /// cargo bookkeeping and must stay out of the gate's output.
    #[test]
    fn only_compiler_messages_render() {
        assert_eq!(
            rendered_diagnostic(
                r#"{"reason":"compiler-message","message":{"rendered":"error: bad\n"}}"#
            ),
            Some("error: bad\n".to_string())
        );
        assert_eq!(
            rendered_diagnostic(r#"{"reason":"build-finished","success":false}"#),
            None
        );
        assert_eq!(rendered_diagnostic("not json"), None);
    }

    /// WHY: eight checkouts share one target directory, and cargo hands the
    /// artifact at `target/debug/<package>` to whichever of them asks. A child
    /// executed straight from that path is whatever checkout built last, which
    /// is how a sweep came to report four registered subsets as unregistered.
    /// The copy carries this process's id, so two gates in one process reuse it
    /// and two processes never collide.
    #[test]
    fn a_child_runs_from_a_copy_this_process_owns() {
        let temp = tempfile::tempdir().expect("Fix: fixture directory must be creatable");
        let built = temp.path().join("xtask-registry");
        std::fs::write(&built, b"binary bytes").expect("Fix: fixture binary must be writable");

        let copy = private_copy(&built, "xtask-registry").expect("the copy must be made");
        assert_ne!(copy, built, "a copy at the same path protects nothing");
        assert_eq!(
            std::fs::read(&copy).expect("the copy must be readable"),
            b"binary bytes"
        );
        assert!(copy
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.contains("xtask-registry")
                && name.ends_with(&std::process::id().to_string())));
        std::fs::remove_file(&copy).expect("the copy must be removable");
    }

    /// WHY: a sibling build-task binary is not the dispatcher, so it must build
    /// and run `xtask` rather than re-enter itself. `release-evidence` re-entered
    /// itself for thirteen subcommands and reported twelve of them as
    /// unimplemented gates, so the decision is tested per binary name rather
    /// than trusted to whichever binary happens to be running.
    #[test]
    fn only_the_dispatcher_binary_is_its_own_dispatcher() {
        assert_eq!(
            dispatcher_from(Path::new("/t/debug/xtask")),
            Some(PathBuf::from("/t/debug/xtask"))
        );
        assert_eq!(
            dispatcher_from(Path::new("/t/debug/xtask.exe")),
            Some(PathBuf::from("/t/debug/xtask.exe"))
        );
        assert_eq!(dispatcher_from(Path::new("/t/debug/xtask-evidence")), None);
        assert_eq!(dispatcher_from(Path::new("/t/debug/xtask-registry")), None);
        assert_eq!(
            dispatcher_from(Path::new("/t/debug/deps/release_evidence-9f1")),
            None
        );
    }
}
