//! The `cli-docs` gate: what every Cargo binary answers to `--help`, and the
//! command-line section each crate README is rendered from.
//!
//! `docs/CLI.toml` declares one row per shipped binary: the package that builds
//! it, the README that documents it, who it is for, and the hardware,
//! environment, configuration, failure and exit-code sentences a reader needs
//! before running it. The declared set is compared against the binaries cargo
//! reports, so a binary added without a row and a row naming a binary nobody
//! builds are both findings.
//!
//! The subcommand list is not declared anywhere. It is read out of the help
//! output of the built executable, because help is what a reader sees and a
//! declared list drifts from it in silence. Two help shapes exist in this
//! workspace: a `Commands:` section, and a usage block whose lines lead with the
//! program word. The shape is decided by the text, not by the binary's name.
//!
//! Every help route must exit zero with bounded, non-empty output, and a public
//! binary's per-command help routes are run too. Running them is the verdict: a
//! `--help` that executes the command, dies on a missing device, or prints
//! nothing is a defect a document cannot record.
//!
//! `xtask` is held to a stronger rule. Its help is rendered from the gate
//! registry, so the commands parsed back out of that help must be exactly the
//! registered gates plus the sweep runner. The check reads the registry in
//! process rather than matching a constant in its source, so reformatting the
//! registry cannot empty the comparison.
//!
//! Everything cargo is asked for goes through the workspace wrapper. Asking bare
//! `cargo` for the target directory while building through the wrapper is how
//! this check spent its time running another checkout's binaries: the wrapper
//! derives a per-checkout target directory, so the two answers are different
//! trees.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::sweep::RUNNER;

/// The manifest that declares every documented binary.
const MANIFEST: &str = "docs/CLI.toml";
/// Manifest schema this gate reads.
const SCHEMA_VERSION: i64 = 1;
/// Start of the section this gate owns in each README.
const BEGIN: &str = "<!-- BEGIN GENERATED CLI CONTRACT -->";
/// End of the section this gate owns in each README.
const END: &str = "<!-- END GENERATED CLI CONTRACT -->";
/// Start of the section the crate-contract generator owns, which this section
/// is inserted above when a README has no CLI section yet.
const CRATE_CONTRACT: &str = "<!-- BEGIN GENERATED CRATE CONTRACT -->";
/// Bound on the help output of one route.
const MAX_HELP_BYTES: usize = 1_048_576;
/// Readers a row may be written for.
const AUDIENCES: [&str; 2] = ["internal", "public"];
/// Fields every row must carry, beyond the optional example.
const REQUIRED: [&str; 9] = [
    "audience",
    "config",
    "environment",
    "exit_codes",
    "failure",
    "hardware",
    "name",
    "package",
    "readme",
];
/// How to regenerate what this gate owns.
const REGENERATE: &str =
    "regenerate the README sections with `./cargo_full run --bin xtask -- cli-docs --write`";

/// One `[[binary]]` row of the manifest.
struct Binary {
    /// Package that builds the binary.
    package: String,
    /// Binary target name, which is also the executable name.
    name: String,
    /// README that documents it, relative to the checkout root.
    readme: String,
    /// Whether the binary is part of the published surface.
    audience: String,
    /// What device it needs.
    hardware: String,
    /// Which environment variables it reads.
    environment: String,
    /// Where its configuration comes from.
    config: String,
    /// What it does when something is wrong.
    failure: String,
    /// What each exit status means.
    exit_codes: String,
    /// Extra invocation the README shows, already in markdown. Empty for a
    /// binary whose help route is the whole example.
    example: String,
}

impl Binary {
    /// Read one row, treating a missing or non-string key as empty. An empty
    /// required field fails the rule that names that field.
    fn from_row(row: &toml::Table) -> Self {
        let text = |key: &str| crate::toml_text::string_field(row, key);
        Self {
            package: text("package"),
            name: text("name"),
            readme: text("readme"),
            audience: text("audience"),
            hardware: text("hardware"),
            environment: text("environment"),
            config: text("config"),
            failure: text("failure"),
            exit_codes: text("exit_codes"),
            example: text("example"),
        }
    }

    /// `package/name`, the key the inventory comparison uses.
    fn key(&self) -> String {
        format!("{}/{}", self.package, self.name)
    }
}

/// The gate.
pub struct CliDocs;

impl Gate for CliDocs {
    fn name(&self) -> &'static str {
        "cli-docs"
    }

    fn help(&self) -> &'static str {
        "Run every documented help route and hold the README command sections to it"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = ctx.root.clone();
        let mut report = Report::clean();
        let manifest = read_manifest(&root)?;
        let rows: Vec<Binary> = manifest.iter().map(Binary::from_row).collect();
        report.findings.extend(row_findings(&rows));
        if !report.findings.is_empty() {
            return Ok(report);
        }

        let runner = crate::cargo_runner::binary(&root);
        let target = target_directory(&root, &runner)?;
        report
            .findings
            .extend(inventory_findings(&root, &runner, &rows)?);
        build_binaries(&root, &runner)?;

        let mut commands: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for row in &rows {
            let executable = target.join("debug").join(&row.name);
            let help = match help_output(&root, &executable, &[]) {
                Ok(text) => text,
                Err(finding) => {
                    report.find(finding);
                    commands.insert(row.key(), Vec::new());
                    continue;
                }
            };
            let discovered = commands_from_help(&help, &row.name);
            if row.name == "xtask" {
                report.findings.extend(registry_findings(&discovered));
            }
            if row.audience == "public" {
                for command in &discovered {
                    if let Err(finding) = help_output(&root, &executable, &[command]) {
                        report.find(finding);
                    }
                }
            }
            commands.insert(row.key(), discovered);
        }

        let counted: usize = commands.values().map(Vec::len).sum();
        let mut wrote = 0;
        for (readme, rows) in by_readme(&rows) {
            let path = root.join(&readme);
            let current = fs::read_to_string(&path).map_err(|error| {
                GateError::new(
                    format!("{readme} could not be read: {error}"),
                    "restore the README the manifest names, or correct the row that names it",
                )
            })?;
            let block = render_block(&rows, &commands);
            let expected = match replace_block(&current, &block) {
                Ok(text) => text,
                Err(message) => {
                    report.find(Finding::in_file(
                        &readme,
                        message,
                        "restore both markers around the generated section, or delete both and let this gate insert it",
                    ));
                    continue;
                }
            };
            if current == expected {
                continue;
            }
            if ctx.write {
                fs::write(&path, expected).map_err(|error| {
                    GateError::new(
                        format!("{readme} could not be written: {error}"),
                        "make the README writable and run this gate again",
                    )
                })?;
                wrote += 1;
            } else {
                report.find(Finding::in_file(
                    &readme,
                    "the generated command-line section does not match the help output of the binaries it documents",
                    REGENERATE,
                ));
            }
        }

        if ctx.write {
            report.note(format!("wrote {wrote} README section(s)"));
        }
        report.note(format!(
            "verified {} binaries and {counted} subcommands",
            rows.len()
        ));
        Ok(report)
    }
}

/// The `[[binary]]` rows, or the reason the manifest cannot be judged at all.
fn read_manifest(root: &Path) -> Result<Vec<toml::Table>, GateError> {
    let path = root.join(MANIFEST);
    let text = fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("{MANIFEST} could not be read: {error}"),
            "restore the manifest that declares every documented binary",
        )
    })?;
    let document: toml::Table = text.parse().map_err(|error| {
        GateError::new(
            format!("{MANIFEST} is not valid TOML: {error}"),
            "repair the manifest syntax",
        )
    })?;
    if document
        .get("schema_version")
        .and_then(toml::Value::as_integer)
        != Some(SCHEMA_VERSION)
    {
        return Err(GateError::new(
            format!("{MANIFEST} does not declare schema_version = {SCHEMA_VERSION}"),
            format!("set `schema_version = {SCHEMA_VERSION}` at the top of the manifest"),
        ));
    }
    let rows: Vec<toml::Table> = document
        .get("binary")
        .and_then(toml::Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(toml::Value::as_table)
                .cloned()
                .collect()
        })
        .unwrap_or_default();
    if rows.is_empty() {
        return Err(GateError::new(
            format!("{MANIFEST} declares no [[binary]] rows"),
            "declare one row per shipped binary",
        ));
    }
    Ok(rows)
}

/// Everything wrong with the rows themselves, independent of any build.
fn row_findings(rows: &[Binary]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for row in rows {
        let label = if row.name.is_empty() {
            row.package.clone()
        } else {
            row.key()
        };
        for field in REQUIRED {
            let value = match field {
                "audience" => &row.audience,
                "config" => &row.config,
                "environment" => &row.environment,
                "exit_codes" => &row.exit_codes,
                "failure" => &row.failure,
                "hardware" => &row.hardware,
                "name" => &row.name,
                "package" => &row.package,
                _ => &row.readme,
            };
            if value.is_empty() {
                findings.push(Finding::in_file(
                    MANIFEST,
                    format!("the row for `{label}` declares no {field}"),
                    format!("state {field} for that binary"),
                ));
            }
        }
        if !row.audience.is_empty() && !AUDIENCES.contains(&row.audience.as_str()) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!(
                    "`{label}` declares audience `{}`, which is not a reader",
                    row.audience
                ),
                format!("declare one of: {}", AUDIENCES.join(", ")),
            ));
        }
        if !seen.insert(row.key()) {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("`{label}` is declared more than once"),
                "delete the duplicate row",
            ));
        }
    }
    findings
}

/// Every disagreement between the declared rows and the binaries cargo builds.
fn inventory_findings(
    root: &Path,
    runner: &Path,
    rows: &[Binary],
) -> Result<Vec<Finding>, GateError> {
    let built = built_binaries(root, runner)?;
    let declared: BTreeSet<String> = rows.iter().map(Binary::key).collect();
    let mut findings = Vec::new();
    for key in built.difference(&declared) {
        findings.push(Finding::in_file(
            MANIFEST,
            format!("`{key}` is a workspace binary the manifest does not declare"),
            "declare the binary, or stop building it",
        ));
    }
    for key in declared.difference(&built) {
        findings.push(Finding::in_file(
            MANIFEST,
            format!("`{key}` is declared but is not a workspace binary"),
            "correct the package and name, or delete the row",
        ));
    }
    for row in rows {
        if !row.readme.is_empty() && !root.join(&row.readme).is_file() {
            findings.push(Finding::in_file(
                MANIFEST,
                format!("`{}` names {}, which does not exist", row.key(), row.readme),
                "name the README that documents the binary",
            ));
        }
    }
    Ok(findings)
}

/// `package/target` of every `bin` target in the workspace.
fn built_binaries(root: &Path, runner: &Path) -> Result<BTreeSet<String>, GateError> {
    let metadata = metadata(root, runner)?;
    let mut binaries = BTreeSet::new();
    let packages = metadata
        .get("packages")
        .and_then(|value| value.as_array())
        .ok_or_else(|| {
            GateError::new(
                "cargo metadata reported no packages",
                "run the gate from inside the workspace",
            )
        })?;
    for package in packages {
        let Some(name) = package.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(targets) = package.get("targets").and_then(|value| value.as_array()) else {
            continue;
        };
        for target in targets {
            let is_binary = target
                .get("kind")
                .and_then(|value| value.as_array())
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
            if !is_binary {
                continue;
            }
            if let Some(target_name) = target.get("name").and_then(|value| value.as_str()) {
                binaries.insert(format!("{name}/{target_name}"));
            }
        }
    }
    Ok(binaries)
}

/// The target directory the wrapper builds into.
fn target_directory(root: &Path, runner: &Path) -> Result<PathBuf, GateError> {
    let metadata = metadata(root, runner)?;
    metadata
        .get("target_directory")
        .and_then(|value| value.as_str())
        .map(PathBuf::from)
        .ok_or_else(|| {
            GateError::new(
                "cargo metadata reported no target directory",
                "run the gate from inside the workspace",
            )
        })
}

/// `cargo metadata` as the wrapper answers it.
///
/// Asked through the wrapper on purpose: the wrapper derives a per-checkout
/// target directory, so bare `cargo` names a directory holding another
/// checkout's executables.
fn metadata(root: &Path, runner: &Path) -> Result<serde_json::Value, GateError> {
    let output = Command::new(runner)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!(
                    "`{} metadata` could not be spawned: {error}",
                    runner.display()
                ),
                "install the workspace cargo wrapper, or set VYRE_CARGO_RUNNER",
            )
        })?;
    if !output.status.success() {
        return Err(GateError::new(
            format!(
                "`{} metadata` failed: {}",
                runner.display(),
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            "repair the workspace manifests until cargo can read them",
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        GateError::new(
            format!("cargo metadata is not JSON this gate can read: {error}"),
            "report the cargo version; the metadata format changed",
        )
    })
}

/// Build every workspace binary, so the help routes run the current code.
fn build_binaries(root: &Path, runner: &Path) -> Result<(), GateError> {
    let output = Command::new(runner)
        .args(["build", "--workspace", "--bins"])
        .current_dir(root)
        .output()
        .map_err(|error| {
            GateError::new(
                format!("`{} build` could not be spawned: {error}", runner.display()),
                "install the workspace cargo wrapper, or set VYRE_CARGO_RUNNER",
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    Err(GateError::new(
        format!(
            "`{} build --workspace --bins` failed: {}",
            runner.display(),
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .filter(|line| line.starts_with("error"))
                .collect::<Vec<_>>()
                .join("; ")
        ),
        "repair the command-line build before documenting it",
    ))
}

/// The help text of one route, or what is wrong with that route.
fn help_output(root: &Path, executable: &Path, command: &[&str]) -> Result<String, Finding> {
    let route = if command.is_empty() {
        "--help".to_string()
    } else {
        format!("{} --help", command.join(" "))
    };
    let name = executable
        .file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned());
    let fix = "make the help route print its own usage and exit 0 without executing the command";
    let output = Command::new(executable)
        .args(command)
        .arg("--help")
        .current_dir(root)
        .output()
        .map_err(|error| {
            Finding::new(
                format!("`{name} {route}` could not be spawned: {error}"),
                "declare the binary the manifest names, or correct the row",
            )
        })?;
    let mut text = output.stdout;
    text.extend_from_slice(&output.stderr);
    if text.len() > MAX_HELP_BYTES {
        return Err(Finding::new(
            format!(
                "`{name} {route}` printed {} bytes, over the {MAX_HELP_BYTES} byte bound",
                text.len()
            ),
            "print usage, not the content the command produces",
        ));
    }
    let text = String::from_utf8(text).map_err(|_| {
        Finding::new(
            format!("`{name} {route}` printed bytes that are not UTF-8"),
            fix,
        )
    })?;
    if !output.status.success() {
        return Err(Finding::new(
            format!(
                "`{name} {route}` exited {}",
                output
                    .status
                    .code()
                    .map_or_else(|| "on a signal".to_string(), |code| code.to_string())
            ),
            fix,
        ));
    }
    if text.trim().is_empty() {
        return Err(Finding::new(
            format!("`{name} {route}` printed nothing"),
            fix,
        ));
    }
    Ok(text.trim_end().to_string() + "\n")
}

/// Every subcommand `help` offers a reader.
///
/// A `Commands:` section wins when the help has one. Otherwise the usage block
/// is read: a usage line names the program and then one of its commands. The
/// leading word has to name this binary, because a usage line that leads with
/// another program is describing how to reach this one and its second word is
/// that program's argument: `cargo run -p structure-gate` would otherwise
/// document a command called `run`.
fn commands_from_help(help: &str, binary: &str) -> Vec<String> {
    let mut commands = BTreeSet::new();
    let mut in_section = false;
    for line in help.lines() {
        let stripped = line.trim();
        if stripped == "Commands:" || stripped == "SUBCOMMANDS:" {
            in_section = true;
            continue;
        }
        if in_section && stripped.ends_with(':') {
            in_section = false;
        }
        if in_section && !stripped.is_empty() {
            if let Some(token) = subcommand_token(stripped.split_whitespace().next()) {
                commands.insert(token);
            }
        }
    }
    if commands.is_empty() {
        commands.extend(usage_commands(help, binary));
    }
    commands.into_iter().collect()
}

/// Every subcommand named by a usage line of `binary`'s own usage block.
fn usage_commands(help: &str, binary: &str) -> BTreeSet<String> {
    let mut commands = BTreeSet::new();
    let mut program: Option<String> = None;
    for line in help.lines() {
        let mut words = line.split_whitespace().peekable();
        if words
            .peek()
            .is_some_and(|word| word.eq_ignore_ascii_case("usage:"))
        {
            words.next();
        }
        let Some(first) = words.next() else {
            continue;
        };
        match &program {
            Some(known) if known != first => continue,
            Some(_) => {}
            None if names_the_binary(first, binary) => program = Some(first.to_string()),
            None => continue,
        }
        if let Some(token) = subcommand_token(words.next()) {
            commands.insert(token);
        }
    }
    commands
}

/// Whether `word` is how `binary` is invoked.
///
/// The executable's own name, or the wrapper it is reached through, which is a
/// leading dash-separated segment of it: `vyre_new_op` prints `vyre new-op`,
/// because the operation is one command of the `vyre` front end.
fn names_the_binary(word: &str, binary: &str) -> bool {
    let normalize = |text: &str| text.replace('_', "-");
    let binary = normalize(binary);
    let word = normalize(word);
    binary == word || binary.starts_with(&format!("{word}-"))
}

/// `word` as a subcommand name, or `None` when it is not one.
fn subcommand_token(word: Option<&str>) -> Option<String> {
    let word = word?;
    if word == "help" {
        return None;
    }
    let mut characters = word.chars();
    if !characters
        .next()
        .is_some_and(|first| first.is_ascii_lowercase())
    {
        return None;
    }
    if !characters.all(|character| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-'
    }) {
        return None;
    }
    Some(word.to_string())
}

/// Every disagreement between `xtask` help and the gate registry.
///
/// The registry is read in process. Matching a constant in its source is how
/// this check emptied itself: the table became a function and the pattern found
/// nothing, so help and dispatch were compared against an empty set.
fn registry_findings(documented: &[String]) -> Vec<Finding> {
    let mut expected: BTreeSet<&str> = crate::subcommands::registry()
        .into_iter()
        .map(Gate::name)
        .collect();
    expected.insert(RUNNER);
    let documented: BTreeSet<&str> = documented.iter().map(String::as_str).collect();
    let mut findings = Vec::new();
    for name in expected.difference(&documented) {
        findings.push(Finding::new(
            format!("`xtask --help` does not offer the registered gate `{name}`"),
            "render help from the registry so every registered gate reaches a reader",
        ));
    }
    for name in documented.difference(&expected) {
        findings.push(Finding::new(
            format!("`xtask --help` offers `{name}`, which the registry does not hold"),
            "register the gate, or stop advertising it",
        ));
    }
    findings
}

/// The rows each README documents, in manifest order.
fn by_readme(rows: &[Binary]) -> Vec<(String, Vec<&Binary>)> {
    let mut order: Vec<String> = Vec::new();
    let mut grouped: BTreeMap<String, Vec<&Binary>> = BTreeMap::new();
    for row in rows {
        if !grouped.contains_key(&row.readme) {
            order.push(row.readme.clone());
        }
        grouped.entry(row.readme.clone()).or_default().push(row);
    }
    order
        .into_iter()
        .filter_map(|readme| grouped.remove(&readme).map(|rows| (readme, rows)))
        .collect()
}

/// The generated section for one README.
fn render_block(rows: &[&Binary], commands: &BTreeMap<String, Vec<String>>) -> String {
    let mut lines = vec![
        BEGIN.to_string(),
        "## Command-line interface".to_string(),
        String::new(),
        "This section is generated from `docs/CLI.toml` and executable help output.".to_string(),
    ];
    for row in rows {
        lines.extend([
            String::new(),
            format!("### `{}`", row.name),
            String::new(),
            "```console".to_string(),
            format!(
                "./cargo_full run -p {} --bin {} -- --help",
                row.package, row.name
            ),
            "```".to_string(),
        ]);
        if !row.example.trim().is_empty() {
            lines.push(String::new());
            lines.extend(row.example.trim().lines().map(str::to_string));
        }
        let listed = commands
            .get(&row.key())
            .map(|names| {
                names
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        let listed = if listed.is_empty() {
            "none".to_string()
        } else {
            listed
        };
        lines.extend([
            String::new(),
            format!("Commands: {listed}."),
            String::new(),
            format!("Hardware: {}", row.hardware),
            String::new(),
            format!("Environment: {}", row.environment),
            String::new(),
            format!("Configuration: {}", row.config),
            String::new(),
            format!("Failure behavior: {}", row.failure),
            String::new(),
            format!("Exit codes: {}", row.exit_codes),
        ]);
    }
    lines.push(END.to_string());
    lines.push(String::new());
    lines.join("\n")
}

/// `text` with the generated section replaced, inserted above the crate
/// contract, or appended.
fn replace_block(text: &str, block: &str) -> Result<String, String> {
    let start = text.find(BEGIN);
    let end = text.find(END);
    match (start, end) {
        (Some(start), Some(end)) => Ok(format!(
            "{}\n\n{}\n\n{}",
            text[..start].trim_end(),
            block.trim_end(),
            text[end + END.len()..].trim_start_matches('\n')
        )),
        (None, None) => match text.find(CRATE_CONTRACT) {
            Some(at) => Ok(format!(
                "{}\n\n{block}\n{}",
                text[..at].trim_end(),
                &text[at..]
            )),
            None => Ok(format!("{}\n\n{block}", text.trim_end())),
        },
        _ => Err("the generated command-line section has one marker and not the other".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: two help shapes ship here, and the old reader carried one hardcoded
    /// exception per binary that did not print a `Commands:` section. An
    /// exception keyed on a binary's name stops matching when the binary is
    /// renamed, and says nothing about a binary added tomorrow.
    #[test]
    fn both_help_shapes_yield_their_commands() {
        let clap = "Compile things.\n\nUsage: tool <COMMAND>\n\nCommands:\n  emit        Emit\n  dump-lexer  Dump\n  help        Print this message\n\nOptions:\n  -h, --help  Print help\n";
        assert_eq!(commands_from_help(clap, "tool"), vec!["dump-lexer", "emit"]);

        let usage = "usage: vyre-conform dispatch --backend <id>\n       vyre-conform plan [--out <plan.json>]\n       vyre-conform merge --out <merged.json>\n";
        assert_eq!(
            commands_from_help(usage, "vyre-conform"),
            vec!["dispatch", "merge", "plan"]
        );

        let wrapped = "Usage:\n  vyre new-op <id> --archetype <archetype>\n\nExamples:\n  cargo_full run -p vyre --bin vyre_new_op -- new-op primitive.arithmetic.test_op\n";
        assert_eq!(commands_from_help(wrapped, "vyre_new_op"), vec!["new-op"]);

        let none = "Audit launch-rule contracts.\n\nUsage: audit_rule_contracts\n\nExit codes:\n  0  clean\n";
        assert_eq!(
            commands_from_help(none, "audit_rule_contracts"),
            Vec::<String>::new()
        );

        // A usage line that reaches this binary through another program: the
        // second word is that program's argument, not a command here.
        let through_cargo =
            "USAGE:\n  cargo run -p structure-gate\n\nFails when a crate registers an operation.\n";
        assert_eq!(
            commands_from_help(through_cargo, "structure-gate"),
            Vec::<String>::new()
        );

        // A description line before the usage block must not become the program
        // word: `Lego-block enforcement lints` documented `enforcement`.
        let prose_first = "Lego-block enforcement lints for vyre\n\nUsage: vyre-lints [OPTIONS]\n\nOptions:\n      --workspace-root <ROOT>  Workspace root\n";
        assert_eq!(
            commands_from_help(prose_first, "vyre-lints"),
            Vec::<String>::new()
        );
    }

    /// WHY: a token that is not a subcommand in a `Commands:` section, or an
    /// option in a usage line, would be documented as one.
    #[test]
    fn only_lowercase_command_words_are_commands() {
        assert_eq!(
            subcommand_token(Some("dump-lr")),
            Some("dump-lr".to_string())
        );
        assert_eq!(subcommand_token(Some("gate1")), Some("gate1".to_string()));
        assert_eq!(subcommand_token(Some("help")), None);
        assert_eq!(subcommand_token(Some("--out")), None);
        assert_eq!(subcommand_token(Some("<COMMAND>")), None);
        assert_eq!(subcommand_token(Some("Options")), None);
        assert_eq!(subcommand_token(None), None);
    }

    /// WHY: the section is written into a file a person also edits, so the
    /// replacement has to be exact in all three states: present, absent above a
    /// generated crate contract, and absent entirely. A half-present pair of
    /// markers is a defect rather than a place to insert.
    #[test]
    fn the_generated_section_is_replaced_inserted_or_rejected() {
        let block = format!("{BEGIN}\n## Command-line interface\n{END}\n");

        let replaced = replace_block(
            &format!("# Tool\n\n{BEGIN}\nold\n{END}\n\n## Tail\n"),
            &block,
        )
        .expect("both markers present");
        assert_eq!(
            replaced,
            format!("# Tool\n\n{BEGIN}\n## Command-line interface\n{END}\n\n## Tail\n")
        );

        let inserted = replace_block(&format!("# Tool\n\n{CRATE_CONTRACT}\nrows\n"), &block)
            .expect("no markers, crate contract present");
        assert_eq!(
            inserted,
            format!(
                "# Tool\n\n{BEGIN}\n## Command-line interface\n{END}\n\n{CRATE_CONTRACT}\nrows\n"
            )
        );

        let appended = replace_block("# Tool\n", &block).expect("no markers at all");
        assert_eq!(
            appended,
            format!("# Tool\n\n{BEGIN}\n## Command-line interface\n{END}\n")
        );

        assert!(replace_block(&format!("# Tool\n\n{BEGIN}\nold\n"), &block).is_err());
    }

    /// WHY: a row missing a sentence a reader needs, declaring an audience
    /// nobody is, or declared twice, has to be one finding each rather than one
    /// error that stops at the first row.
    #[test]
    fn every_row_defect_is_its_own_finding() {
        let complete = |name: &str| Binary {
            package: "pkg".to_string(),
            name: name.to_string(),
            readme: "README.md".to_string(),
            audience: "public".to_string(),
            hardware: "none".to_string(),
            environment: "none".to_string(),
            config: "flags".to_string(),
            failure: "nonzero".to_string(),
            exit_codes: "0, 1".to_string(),
            example: String::new(),
        };
        assert!(row_findings(&[complete("a"), complete("b")]).is_empty());

        let mut broken = complete("c");
        broken.hardware = String::new();
        broken.audience = "everyone".to_string();
        let findings = row_findings(&[broken, complete("a"), complete("a")]);
        assert_eq!(findings.len(), 3, "{findings:?}");
        assert!(findings[0].message.contains("declares no hardware"));
        assert!(findings[1].message.contains("audience `everyone`"));
        assert!(findings[2].message.contains("declared more than once"));
    }

    /// WHY: the registry is the authority for what `xtask` offers, and the
    /// comparison has to fail in both directions: a gate help never mentions is
    /// unreachable, and a command help offers that nothing registers dispatches
    /// to nothing.
    #[test]
    fn help_and_registry_disagreement_fails_in_both_directions() {
        let live: Vec<String> = crate::subcommands::registry()
            .into_iter()
            .map(|gate| gate.name().to_string())
            .chain(std::iter::once(RUNNER.to_string()))
            .collect();
        assert!(registry_findings(&live).is_empty());

        let missing: Vec<String> = live.iter().skip(1).cloned().collect();
        let findings = registry_findings(&missing);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("does not offer"));

        let extra: Vec<String> = live
            .iter()
            .cloned()
            .chain(std::iter::once("not-a-gate".to_string()))
            .collect();
        let findings = registry_findings(&extra);
        assert_eq!(findings.len(), 1);
        assert!(findings[0].message.contains("`not-a-gate`"));
    }

    /// WHY: a README documenting two binaries must render one section holding
    /// both, in manifest order, and a binary with no subcommands must say so
    /// rather than render an empty list.
    #[test]
    fn one_section_renders_every_binary_the_readme_documents() {
        let row = |name: &str, example: &str| Binary {
            package: "pkg".to_string(),
            name: name.to_string(),
            readme: "README.md".to_string(),
            audience: "internal".to_string(),
            hardware: "none".to_string(),
            environment: "none".to_string(),
            config: "flags".to_string(),
            failure: "nonzero".to_string(),
            exit_codes: "0, 1".to_string(),
            example: example.to_string(),
        };
        let declared = [
            row("first", "Extra:\n\n```console\nfirst demo\n```"),
            row("second", ""),
        ];
        let mut commands = BTreeMap::new();
        commands.insert("pkg/first".to_string(), vec!["demo".to_string()]);
        commands.insert("pkg/second".to_string(), Vec::new());

        let grouped = by_readme(&declared);
        assert_eq!(grouped.len(), 1, "one README documents both binaries");
        let (readme, rows) = &grouped[0];
        assert_eq!(readme, "README.md");
        assert_eq!(
            rows.iter().map(|row| row.name.as_str()).collect::<Vec<_>>(),
            vec!["first", "second"],
            "manifest order decides the section order"
        );

        let block = render_block(rows, &commands);
        assert!(block.starts_with(BEGIN));
        assert!(block.ends_with(&format!("{END}\n")));
        assert!(block.contains("### `first`"));
        assert!(block.contains("Commands: `demo`."));
        assert!(block.contains("Commands: none."));
        assert!(block.contains("\nExtra:\n\n```console\nfirst demo\n```\n"));
    }
}
