//! Run a subcommand that is implemented in a crate linking vyre.
//!
//! This crate links no vyre crate, so those subcommands cannot be called: they
//! are built on demand and executed as a child process. Cargo's own output is
//! captured rather than inherited, because the gate sweep records how many
//! lines a gate printed and a `Compiling ...` line would change the recorded
//! result of the gate it wraps. On a build failure the captured text is printed
//! and the exit code is cargo's.

use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

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
}
