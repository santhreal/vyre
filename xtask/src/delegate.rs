//! Run a subcommand that is implemented in a crate linking vyre.
//!
//! This crate links no vyre crate, so those subcommands cannot be called: they
//! are built on demand and executed as a child process. Cargo's own output is
//! captured rather than inherited, because the gate sweep records how many
//! lines a gate printed and a `Compiling ...` line would change the recorded
//! result of the gate it wraps. On a build failure the captured text is printed
//! and the exit code is cargo's.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

use serde_json::Value;

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
            eprintln!("Fix: cannot resolve the running binary: {error}. Rebuild it with `cargo build -p {DISPATCHER_PACKAGE}`.");
            std::process::exit(1);
        });
        dispatcher_from(&current).unwrap_or_else(|| build(DISPATCHER_PACKAGE))
    });
    DISPATCHER.as_path()
}

/// `current` when the running binary is already the dispatcher, `None` when a
/// sibling build-task binary is running and the dispatcher has to be built.
fn dispatcher_from(current: &Path) -> Option<PathBuf> {
    (current.file_stem()? == DISPATCHER_PACKAGE).then(|| current.to_path_buf())
}

/// Build `package`'s binary of the same name, then run it with `args`.
///
/// `args` is the process argument vector, so `args[0]` is the dispatcher's own
/// path and is dropped.
pub fn run(package: &str, args: &[String]) -> ! {
    let executable = build(package);
    let status = Command::new(&executable)
        .args(&args[1..])
        .status()
        .unwrap_or_else(|error| {
            eprintln!(
                "Fix: cannot run {}: {error}. Rebuild it with `cargo build -p {package}`.",
                executable.display()
            );
            std::process::exit(1);
        });
    std::process::exit(status.code().unwrap_or(1));
}

/// Run a delegated binary's `main`: help, argument count, then dispatch.
///
/// Both delegated crates are entered the same way, because `xtask` enters them
/// the same way: it hands the whole argument vector to a binary whose only job
/// is to resolve `args[1]` against its own table. Each `main` used to spell
/// that out, so the two could disagree about which exit code an unimplemented
/// subcommand gets, and CI reads that code.
pub fn run_delegated_main(
    package: &str,
    purpose: &str,
    implemented: &[(&'static str, fn(&[String]))],
) {
    let args: Vec<String> = std::env::args().collect();
    if args
        .iter()
        .skip(1)
        .any(|arg| arg == "--help" || arg == "-h")
    {
        print_dispatch_help(package, purpose, implemented.iter().map(|(name, _)| *name));
        return;
    }
    if args.len() < 2 {
        eprintln!("Fix: missing subcommand. Run `cargo xtask --help`.");
        std::process::exit(1);
    }
    if !crate::subcommands::dispatch(implemented, args[1].as_str(), &args) {
        eprintln!(
            "Fix: `{}` is not implemented in {package}. Run `cargo xtask --help`.",
            args[1]
        );
        std::process::exit(1);
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
    println!("  cargo run -p {package} -- <subcommand> [options]");
    println!();
    println!("{purpose}");
    println!();
    println!("Run `cargo xtask --help` for every workspace command, and");
    println!("`cargo xtask <subcommand> --help` for one command's options.");
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
fn build(package: &str) -> PathBuf {
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
        .unwrap_or_else(|error| {
            eprintln!("Fix: cannot run cargo to build {package}: {error}");
            std::process::exit(1);
        });
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Some(rendered) = rendered_diagnostic(line) {
                eprint!("{rendered}");
            }
        }
        eprintln!("Fix: `{package}` must compile before `{package}` subcommands can run.");
        std::process::exit(output.status.code().unwrap_or(1));
    }
    executable_from(&String::from_utf8_lossy(&output.stdout), package).unwrap_or_else(|| {
        eprintln!(
            "Fix: cargo built `{package}` without reporting an executable; \
             check that `{package}` declares a binary named `{package}`."
        );
        std::process::exit(1);
    })
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
