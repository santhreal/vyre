//! Contracts over the command-line surface every Cargo binary presents.

use std::collections::BTreeSet;
use std::fs;
use std::process::{Command, Output};

use super::common::workspace_root;

fn run(executable: &str, args: &[&str]) -> Output {
    Command::new(executable)
        .args(args)
        .output()
        .expect("Fix: documented CLI executable must launch")
}

/// Locks every Cargo binary, discovered subcommand, README block, and help
/// transcript to one executable contract.
#[test]
fn workspace_cli_documentation_is_current() {
    let root = workspace_root();
    let output = Command::new("python3")
        .arg(root.join("scripts/cli_docs.py"))
        .arg("--check")
        .output()
        .expect("Fix: CLI documentation generator must launch with python3");
    assert!(
        output.status.success(),
        "Fix: regenerate or repair CLI contracts: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let summary = String::from_utf8(output.stdout).expect("Fix: generator output must be UTF-8");
    let manifest: toml::Value = toml::from_str(
        &fs::read_to_string(root.join("docs/CLI.toml")).expect("Fix: docs/CLI.toml must be readable"),
    )
    .expect("Fix: docs/CLI.toml must be valid TOML");
    let binaries = manifest["binary"]
        .as_array()
        .expect("Fix: docs/CLI.toml must declare [[binary]] entries");
    let mut readmes: BTreeSet<&str> = BTreeSet::new();
    for entry in binaries {
        readmes.insert(
            entry["readme"]
                .as_str()
                .expect("Fix: every [[binary]] must name its README"),
        );
    }
    let documented: usize = readmes
        .iter()
        .map(|readme| {
            let text = fs::read_to_string(root.join(readme))
                .unwrap_or_else(|error| panic!("Fix: {readme} must be readable: {error}"));
            documented_subcommand_count(&text)
        })
        .sum();

    let expected = format!(
        "cli-docs: verified {} binaries and {documented} subcommands\n",
        binaries.len()
    );
    assert_eq!(
        summary, expected,
        "Fix: the generator must verify every binary declared in docs/CLI.toml and every \
         subcommand it wrote into the generated README blocks"
    );
}

/// Subcommands the generated README blocks attribute to the binaries.
///
/// The count is read back out of the artifact rather than written here, so
/// registering a binary or adding a subcommand does not need this test edited,
/// and a generator that stopped verifying one of them cannot stay green. The
/// artifact is the generated block in each crate README; the summary table this
/// used to count lived in `docs/CLI.md`, which no longer exists.
fn documented_subcommand_count(readme: &str) -> usize {
    let Some((_, after)) = readme.split_once("<!-- BEGIN GENERATED CLI CONTRACT -->") else {
        return 0;
    };
    let block = after
        .split("<!-- END GENERATED CLI CONTRACT -->")
        .next()
        .unwrap_or_default();
    block
        .lines()
        .filter_map(|line| line.trim().strip_prefix("Commands: "))
        .map(|listed| {
            let listed = listed.trim_end_matches('.');
            if listed == "none" {
                0
            } else {
                listed.split(',').count()
            }
        })
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

/// `xtask --help` lists every registered gate.
///
/// WHY: a reader reaches a subcommand through help. A help route that prints a
/// header and a truncated table, or that stops at the first row whose usage
/// string is empty, still exits 0 and still contains `SUBCOMMANDS:`, so the
/// route check above passes while the commands are unreachable. The expected
/// set is the table itself, so a subcommand added tomorrow is judged tomorrow.
#[test]
fn xtask_help_lists_every_registered_gate() {
    let output = run(env!("CARGO_BIN_EXE_xtask"), &["--help"]);
    let help = String::from_utf8_lossy(&output.stdout);
    let missing: Vec<&str> = xtask::subcommands::registry()
        .iter()
        .map(|gate| gate.name())
        .filter(|name| !help.contains(name))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: `xtask --help` omits {} registered gate(s): {}",
        missing.len(),
        missing.join(", ")
    );
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
