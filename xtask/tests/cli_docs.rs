//! Executable command-line documentation contract tests.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("Fix: xtask must remain directly under the workspace root")
        .to_path_buf()
}

fn run(executable: &str, args: &[&str]) -> Output {
    Command::new(executable)
        .args(args)
        .output()
        .expect("Fix: documented CLI executable must launch")
}

/// Locks every Cargo binary, discovered subcommand, README block, and help transcript to one executable contract.
#[test]
fn workspace_cli_documentation_is_current() {
    let output = Command::new("python3")
        .arg(workspace_root().join("scripts/cli_docs.py"))
        .arg("--check")
        .output()
        .expect("Fix: CLI documentation generator must launch with python3");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair CLI contracts: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).expect("Fix: generator output must be UTF-8"),
        "cli-docs: verified 10 binaries and 72 subcommands\n"
    );
}

/// Prevents internal helper binaries from running audits or writes when a reader asks for help.
#[test]
fn every_xtask_binary_help_route_exits_zero() {
    let cases = [
        (
            env!("CARGO_BIN_EXE_audit_rule_contracts"),
            "Usage: audit_rule_contracts",
        ),
        (env!("CARGO_BIN_EXE_scaffold_rule"), "Usage: scaffold_rule"),
        (env!("CARGO_BIN_EXE_xtask"), "SUBCOMMANDS:"),
    ];
    for (executable, expected) in cases {
        let output = run(executable, &["--help"]);
        assert!(
            output.status.success(),
            "{} --help returned {:?}: {}",
            executable,
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains(expected));
    }
}

/// Prevents the historical `scaffold_rule --help` bug from creating a rule literally named `--help`.
#[test]
fn scaffold_help_is_side_effect_free() {
    let temp = tempfile::tempdir().expect("Fix: fixture workspace must be creatable");
    let cwd = temp.path().join("a/b/c/d/e/f");
    fs::create_dir_all(&cwd).expect("Fix: nested fixture directory must be creatable");
    let output = Command::new(env!("CARGO_BIN_EXE_scaffold_rule"))
        .arg("--help")
        .current_dir(&cwd)
        .output()
        .expect("Fix: scaffold help must launch");
    assert!(output.status.success());
    assert!(!temp.path().join("a/rules/launch/--help").exists());
    assert!(!temp
        .path()
        .join("a/tests/launch_rule_truth/--help")
        .exists());
}

/// Preserves status 2 for invalid CLI syntax instead of running partial audits or scaffolds.
#[test]
fn invalid_helper_arguments_return_usage_status() {
    for executable in [
        env!("CARGO_BIN_EXE_audit_rule_contracts"),
        env!("CARGO_BIN_EXE_scaffold_rule"),
    ] {
        let output = run(executable, &["--definitely-invalid"]);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{} returned {:?}: {}",
            executable,
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("Fix:"));
    }
}
