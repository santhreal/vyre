//! Executable command-line documentation contract tests.

use std::fs;
use std::process::{Command, Output};

use super::common::workspace_root;

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
    let summary = String::from_utf8(output.stdout).expect("Fix: generator output must be UTF-8");
    let root = workspace_root();
    let manifest = fs::read_to_string(root.join("docs/CLI.toml"))
        .expect("Fix: docs/CLI.toml must be readable");
    let declared = manifest
        .lines()
        .filter(|line| line.trim() == "[[binary]]")
        .count();
    let documented =
        fs::read_to_string(root.join("docs/CLI.md")).expect("Fix: docs/CLI.md must be readable");
    let expected = format!(
        "cli-docs: verified {declared} binaries and {} subcommands\n",
        documented_subcommand_count(&documented)
    );
    assert_eq!(
        summary, expected,
        "Fix: the generator must verify every binary declared in docs/CLI.toml and every \
         subcommand it wrote into docs/CLI.md"
    );
}

/// Total subcommands the generated summary table attributes to the binaries.
///
/// The count is read back out of the artifact rather than written here, so
/// registering a binary or adding a subcommand does not need this test edited,
/// and a generator that stopped verifying one of them cannot stay green.
fn documented_subcommand_count(doc: &str) -> usize {
    doc.lines()
        .filter(|line| line.starts_with("| `"))
        .filter_map(|line| line.split('|').nth(4))
        .map(str::trim)
        .filter(|cell| !cell.is_empty() && *cell != "none")
        .map(|cell| cell.split(',').count())
        .sum()
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

/// Prevents the historical `scaffold_rule --help` bug from creating a rule
/// literally named `--help`, and pins that the tree it would write is resolved
/// from the repository root instead of the process working directory. The old
/// `Path::new("../../../../../rules/launch")` climbed five levels out of the
/// checkout, so a scaffold landed in whatever tree the clone happened to sit in.
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
    assert_eq!(
        fs::read_dir(temp.path().join("a"))
            .expect("Fix: fixture directory must be readable")
            .count(),
        1,
        "help must not write anything anywhere near the working directory"
    );
    let repo_root = workspace_root();
    assert!(!repo_root.join("rules/launch/--help").exists());
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
