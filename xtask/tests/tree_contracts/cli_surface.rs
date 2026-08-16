//! Contracts over the command-line surface every Cargo binary presents.

use std::collections::BTreeSet;
use std::fs;
use std::process::{Command, Output};

use super::workspace_sources::workspace_root;

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
    let output = run(env!("CARGO_BIN_EXE_xtask"), &["cli-docs"]);
    assert!(
        output.status.success(),
        "Fix: regenerate or repair CLI contracts: {}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let summary = String::from_utf8(output.stdout).expect("Fix: gate output must be UTF-8");
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
        "cli-docs: note: verified {} binaries and {documented} subcommands\n",
        binaries.len()
    );
    assert!(
        summary.contains(&expected),
        "Fix: the gate must verify every binary declared in docs/CLI.toml and every \
         subcommand it wrote into the generated README blocks; it reported:\n{summary}"
    );
}

/// Subcommands the generated README blocks attribute to the binaries.
///
/// The count is read back out of the artifact rather than written here, so
/// registering a binary or adding a subcommand does not need this test edited,
/// and a generator that stopped verifying one of them cannot stay green. The
/// artifact is the `BEGIN GENERATED CLI CONTRACT` block in each crate README,
/// and the count is the sum of its `Commands: ` lines.
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

/// Every helper binary under `xtask/src/bin/`, with the executable cargo built
/// for it.
///
/// The roster is the source directory, not a list here. `publishable_packages`
/// shipped for a year without a help-route or exit-code case because it was
/// added to `src/bin/` and not to the array this replaced, and a list that has
/// to be edited to cover a new binary covers nothing added tomorrow.
fn helper_binaries() -> Vec<(String, std::path::PathBuf)> {
    let directory = std::path::Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .expect("Fix: a cargo binary must live in a directory")
        .to_path_buf();
    let mut binaries: Vec<(String, std::path::PathBuf)> =
        fs::read_dir(workspace_root().join("xtask/src/bin"))
            .expect("Fix: xtask/src/bin must be readable")
            .map(|entry| entry.expect("Fix: xtask/src/bin entries must be readable").path())
            .filter(|path| path.extension().is_some_and(|value| value == "rs"))
            .map(|path| {
                let name = path
                    .file_stem()
                    .expect("Fix: a source file must have a stem")
                    .to_string_lossy()
                    .into_owned();
                let executable = directory.join(&name);
                (name, executable)
            })
            .collect();
    binaries.sort();
    assert!(
        !binaries.is_empty(),
        "Fix: xtask/src/bin must hold at least one helper binary."
    );
    binaries
}

/// Prevents internal helper binaries from running audits or writes when a reader asks for help.
#[test]
fn every_xtask_binary_help_route_exits_zero() {
    for (name, executable) in helper_binaries() {
        let path = executable.to_string_lossy().into_owned();
        let output = run(&path, &["--help"]);
        assert!(
            output.status.success(),
            "{name} --help returned {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains(&format!("Usage: {name}")),
            "Fix: `{name} --help` must print its own usage line."
        );
    }
    let output = run(env!("CARGO_BIN_EXE_xtask"), &["--help"]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("SUBCOMMANDS:"));
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
    for (name, executable) in helper_binaries() {
        let output = run(&executable.to_string_lossy(), &["--definitely-invalid"]);
        assert_eq!(
            output.status.code(),
            Some(2),
            "{name} returned {:?}: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stderr).contains("Fix:"));
    }
}

/// WHY: `--help` is a help route only when it is the whole command line. A
/// binary that takes nothing has no reading of `--help extra`, and treating the
/// help word as a prefix would exit 0 while silently discarding whatever the
/// caller meant by the rest, which is the shape that let an argument typo run
/// the audit instead of reporting it.
#[test]
fn a_help_word_is_help_only_when_it_is_the_whole_command_line() {
    use xtask::cli::{request, Request};

    let line = |words: &[&str]| request(words.iter().map(|word| (*word).to_string()));

    assert_eq!(line(&[]), Request::Run);
    assert_eq!(line(&["-h"]), Request::Help);
    assert_eq!(line(&["--help"]), Request::Help);
    assert_eq!(line(&["--help", "extra"]), Request::Unknown("--help".into()));
    assert_eq!(line(&["--write"]), Request::Unknown("--write".into()));
}

/// WHY: `docs/CLI.toml` is generated by parsing each binary's help page, so the
/// exit-code block is a contract on the text and not decoration. The last row
/// is produced by the shared module rather than by the caller, which is what
/// makes an exit code the caller never wrote still true.
#[test]
fn a_no_argument_help_page_names_all_three_exit_codes() {
    let page = xtask::cli::NoArguments {
        binary: "fixture_binary",
        summary: "Do the one thing.",
        success: "the thing was done",
        failure: "the thing could not be done",
    }
    .help();

    assert_eq!(
        page,
        "Do the one thing.\n\nUsage: fixture_binary\n\nExit codes:\n  0  the thing was done\n  \
         1  the thing could not be done\n  2  command-line arguments are invalid\n"
    );
}
