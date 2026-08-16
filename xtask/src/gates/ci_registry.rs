//! The `ci-registry` gate: one declaration of every check CI runs.
//!
//! A gate used to be declared in three places that nothing held together. The
//! registry is Rust, the subsets are Rust, and the step that runs it is YAML.
//! Each pair could disagree in silence, and the failure that costs the most is
//! the quiet one: a gate is written, reviewed, registered, and never named by a
//! workflow, so it judges nothing forever while its name in the registry reads
//! as coverage.
//!
//! `xtask/ci-registry.toml` is the one declaration of that wiring. Every
//! registered gate has exactly one row carrying its subsets and the workflows
//! that run it, and every check CI runs that is not an xtask gate has an
//! `[[external]]` row carrying the same wiring. Pinned finding counts are a
//! different fact with a different owner and stay in `xtask/gate-baselines.toml`.
//! The gate compares that declaration against the tree it describes, in both
//! directions:
//!
//! 1. Every registered gate has exactly one row, and every row names a
//!    registered gate. A row with no gate is wiring nobody runs, which is what
//!    a retired gate leaves behind.
//! 2. Each row's `subsets` equal the subsets that contain the gate, derived
//!    from the registry at run time.
//! 3. Each row's `workflows` equal the workflows that run it, derived by
//!    reading the steps. A gate no workflow runs is reported as that, not as a
//!    list mismatch, because it is the defect this file exists for.
//! 4. Every workflow step naming an xtask subcommand or subset names a
//!    registered one, and every check a workflow runs that is not an xtask gate
//!    has an `[[external]]` row. A script a step names and the checkout does
//!    not carry is a step that fails at run time under a name that reads as
//!    coverage.
//!
//! The declaration is written by `xtask gates --write-baseline`, which derives
//! every column from the registry and the workflow steps, so no column is
//! maintained by hand and none can quietly fall behind.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::sweep::RUNNER;
use crate::subcommands::{self, SUBSETS};

/// The one declaration of every check CI runs.
pub const REGISTRY: &str = "xtask/ci-registry.toml";
/// Schema this gate reads.
pub const SCHEMA_VERSION: i64 = 1;
/// Where the workflows live.
const WORKFLOWS: &str = ".github/workflows";
/// Where a workflow that does not run is parked.
const PAUSED: &str = ".github/workflows-paused";
/// A glob a workflow may name, because a glob names a set rather than a file.
const SCRIPT_GLOBS: &[&str] = &["check_*.sh"];
/// The package whose gates `xtask` runs in process.
const XTASK: &str = "xtask";

/// One registered gate and its wiring.
///
/// `deny_unknown_fields` is load-bearing. This file used to carry `status` and
/// `owner` per row, which together let a failing gate stay legal indefinitely
/// behind a prose excuse. A row that still carries a retired field, or the
/// `findings` pin that belongs to `xtask/gate-baselines.toml`, fails to load
/// rather than being ignored.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateRow {
    /// Name as typed on the command line.
    pub name: String,
    /// Subsets that contain the gate.
    #[serde(default)]
    pub subsets: Vec<String>,
    /// Workflow files that run the gate, directly or through a subset.
    #[serde(default)]
    pub workflows: Vec<String>,
}

/// One check CI runs that is not an xtask gate.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalRow {
    /// What the step runs: a `scripts/` path, or a package run as `-p NAME`.
    pub run: String,
    /// Workflow files that run it.
    #[serde(default)]
    pub workflows: Vec<String>,
}

/// One workflow that is parked rather than run.
///
/// A workflow moved out of `.github/workflows` stops running and keeps every
/// appearance of a lane: `.github/CI_REQUIRED.md` named two parked workflows as
/// deep gates for months, and nothing was red. A pause is a decision, so it is
/// written down with the condition that ends it, or the file is deleted.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PausedRow {
    /// File name under `.github/workflows-paused`.
    pub workflow: String,
    /// Why it does not run.
    pub reason: String,
    /// What has to be true before it runs again.
    pub returns_when: String,
}

/// The declaration file.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Registry {
    /// Schema the file declares.
    pub schema_version: i64,
    /// One row per registered gate.
    #[serde(default)]
    pub gate: Vec<GateRow>,
    /// One row per check CI runs that is not an xtask gate.
    #[serde(default)]
    pub external: Vec<ExternalRow>,
    /// One row per workflow that is parked rather than run.
    #[serde(default)]
    pub paused: Vec<PausedRow>,
}

/// Where the declaration lives in `root`.
#[must_use]
pub fn registry_path(root: &Path) -> PathBuf {
    root.join(REGISTRY)
}

/// Read the declaration, or say why it could not be read.
pub fn load(root: &Path) -> Result<Registry, GateError> {
    let path = registry_path(root);
    let text = fs::read_to_string(&path).map_err(|error| {
        GateError::new(
            format!("cannot read {}: {error}", path.display()),
            format!("regenerate it with `./cargo_full run -p xtask --bin xtask -- {RUNNER} --write-baseline`"),
        )
    })?;
    let registry: Registry = toml::from_str(&text).map_err(|error| {
        GateError::new(
            format!("{REGISTRY} does not parse: {error}"),
            "a row carrying a field the schema retired fails to load rather than being ignored; delete the field",
        )
    })?;
    if registry.schema_version != SCHEMA_VERSION {
        return Err(GateError::new(
            format!(
                "{REGISTRY} declares schema_version {} and this gate reads {SCHEMA_VERSION}",
                registry.schema_version
            ),
            "regenerate the file with `--write-baseline`, or teach the gate the new schema",
        ));
    }
    Ok(registry)
}

/// What the in-repo workflows name.
#[derive(Default)]
pub struct WorkflowNames {
    /// Every `xtask <name>` subcommand a workflow invokes, and where.
    pub invoked: BTreeMap<String, BTreeSet<String>>,
    /// Every `xtask gates --subset <name>` a workflow runs, and where.
    pub subsets: BTreeMap<String, BTreeSet<String>>,
    /// Every `scripts/<path>` a workflow names, and where, with the line.
    pub scripts: Vec<(String, usize, String)>,
    /// Every `cargo_full run -p <package>` a workflow runs, and where.
    pub packages: BTreeMap<String, BTreeSet<String>>,
}

/// Read every check the workflows name.
///
/// Only lines that mention `xtask` are read for subcommands, and a token
/// beginning with `-` is not a subcommand, so `./cargo_full test -- --nocapture`
/// is not mistaken for one. A YAML comment is documentation, not a reference:
/// prose that ends a sentence with a script name invokes nothing.
pub fn workflow_names(root: &Path) -> WorkflowNames {
    let mut names = WorkflowNames::default();
    let Ok(entries) = fs::read_dir(root.join(WORKFLOWS)) else {
        return names;
    };
    let mut files: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "yml" || extension == "yaml")
        })
        .collect();
    files.sort();
    for path in files {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let file = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        for (index, line) in text.lines().enumerate() {
            read_line(&mut names, &file, index + 1, line);
        }
    }
    names
}

/// Read one workflow line into the name sets.
fn read_line(names: &mut WorkflowNames, file: &str, line_number: usize, line: &str) {
    if let Some(script) = referenced_script(line) {
        names
            .scripts
            .push((file.to_string(), line_number, script.to_string()));
    }
    let command = strip_yaml_comment(line.trim());
    let mut rest = command;
    while let Some(at) = rest.find("run -p ") {
        rest = &rest[at + "run -p ".len()..];
        let package = token(rest);
        if !package.is_empty() && package != XTASK {
            names
                .packages
                .entry(package)
                .or_default()
                .insert(file.to_string());
        }
    }
    if !command.contains(XTASK) {
        return;
    }
    let mut rest = command;
    let selects_a_subset = command.contains("--subset ");
    while let Some(at) = rest.find("-- ") {
        rest = &rest[at + 3..];
        let name = token(rest);
        if name.is_empty() || name.starts_with('-') {
            continue;
        }
        if name == RUNNER && selects_a_subset {
            continue;
        }
        names
            .invoked
            .entry(name)
            .or_default()
            .insert(file.to_string());
    }
    let mut rest = command;
    while let Some(at) = rest.find("--subset ") {
        rest = &rest[at + "--subset ".len()..];
        let name = token(rest);
        if !name.is_empty() {
            names
                .subsets
                .entry(name)
                .or_default()
                .insert(file.to_string());
        }
    }
}

/// The leading subcommand-shaped token of `text`.
fn token(text: &str) -> String {
    text.chars()
        .take_while(|character| character.is_ascii_alphanumeric() || *character == '-')
        .collect()
}

/// The script a workflow line invokes, relative to `scripts/`, or `None`.
fn referenced_script(line: &str) -> Option<&str> {
    let command = strip_yaml_comment(line.trim());
    let index = command.find("scripts/")?;
    let rest = &command[index + "scripts/".len()..];
    let name: &str = rest
        .split(|character: char| character.is_whitespace() || character == '"' || character == '\'')
        .next()?;
    let name = name.trim_end_matches(['.', ',', ')', ';', ':']);
    if name.is_empty() {
        return None;
    }
    Some(name)
}

/// The command without its YAML comment.
fn strip_yaml_comment(line: &str) -> &str {
    if line.starts_with('#') {
        return "";
    }
    match line.find(" #") {
        Some(at) => &line[..at],
        None => line,
    }
}

/// The subsets that contain each registered gate.
#[must_use]
pub fn derived_subsets() -> BTreeMap<String, BTreeSet<String>> {
    let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for subset in SUBSETS {
        for gate in subset.gates {
            map.entry((*gate).to_string())
                .or_default()
                .insert(subset.name.to_string());
        }
    }
    map
}

/// The workflows that run each registered gate, directly or through a subset.
#[must_use]
pub fn derived_workflows(
    names: &WorkflowNames,
    subsets: &BTreeMap<String, BTreeSet<String>>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let whole_registry = names.invoked.get(RUNNER).cloned().unwrap_or_default();
    for gate in subcommands::registry() {
        let name = gate.name().to_string();
        let mut files = whole_registry.clone();
        if let Some(direct) = names.invoked.get(&name) {
            files.extend(direct.iter().cloned());
        }
        for subset in subsets.get(&name).into_iter().flatten() {
            if let Some(running) = names.subsets.get(subset) {
                files.extend(running.iter().cloned());
            }
        }
        map.insert(name, files);
    }
    map
}

/// Every check a workflow runs that is not an xtask gate.
#[must_use]
pub fn derived_externals(names: &WorkflowNames) -> BTreeMap<String, BTreeSet<String>> {
    let mut map: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (package, files) in &names.packages {
        map.entry(format!("-p {package}"))
            .or_default()
            .extend(files.iter().cloned());
    }
    for (file, _, script) in &names.scripts {
        map.entry(format!("scripts/{script}"))
            .or_default()
            .insert(file.clone());
    }
    map
}

/// Every workflow parked under `.github/workflows-paused`.
#[must_use]
pub fn paused_workflows(root: &Path) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let Ok(entries) = fs::read_dir(root.join(PAUSED)) else {
        return names;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "yml" || extension == "yaml")
        {
            if let Some(name) = path.file_name() {
                names.insert(name.to_string_lossy().into_owned());
            }
        }
    }
    names
}

/// Render the declaration from derived wiring.
///
/// A pause carries prose no derivation can produce, so the writer copies the
/// rows it was given and emits an empty row for a workflow that has none. An
/// empty row is red, which is the point: the writer never invents a reason.
#[must_use]
pub fn render(
    gate_names: &[&str],
    subsets: &BTreeMap<String, BTreeSet<String>>,
    workflows: &BTreeMap<String, BTreeSet<String>>,
    externals: &BTreeMap<String, BTreeSet<String>>,
    paused: &[PausedRow],
    parked: &BTreeSet<String>,
) -> String {
    let mut text = String::from(
        "# Every check CI runs, declared once, written by\n\
         # `xtask gates --write-baseline`.\n\
         #\n\
         # A `[[gate]]` row carries the subsets that hold the gate and the\n\
         # workflows that run it. An `[[external]]` row is a check CI runs that\n\
         # is not an xtask gate. The finding count each gate is pinned at lives\n\
         # in `xtask/gate-baselines.toml`.\n\
         #\n\
         # The `ci-registry` gate compares this file against the registry, the\n\
         # subsets and the workflow steps, in both directions, so a gate no\n\
         # workflow runs and a row naming no gate are both failures.\n",
    );
    text.push_str(&format!("\nschema_version = {SCHEMA_VERSION}\n"));
    for name in gate_names {
        text.push_str("\n[[gate]]\n");
        text.push_str(&format!("name = \"{name}\"\n"));
        text.push_str(&format!(
            "subsets = {}\n",
            list(subsets.get(*name).cloned().unwrap_or_default())
        ));
        text.push_str(&format!(
            "workflows = {}\n",
            list(workflows.get(*name).cloned().unwrap_or_default())
        ));
    }
    for (run, files) in externals {
        text.push_str("\n[[external]]\n");
        text.push_str(&format!("run = \"{run}\"\n"));
        text.push_str(&format!("workflows = {}\n", list(files.clone())));
    }
    for workflow in parked {
        let recorded = paused.iter().find(|row| row.workflow == *workflow);
        text.push_str("\n[[paused]]\n");
        text.push_str(&format!("workflow = \"{workflow}\"\n"));
        text.push_str(&format!(
            "reason = \"{}\"\n",
            recorded.map(|row| row.reason.as_str()).unwrap_or_default()
        ));
        text.push_str(&format!(
            "returns_when = \"{}\"\n",
            recorded
                .map(|row| row.returns_when.as_str())
                .unwrap_or_default()
        ));
    }
    text
}

/// One TOML array of strings.
fn list(values: BTreeSet<String>) -> String {
    let inner: Vec<String> = values
        .into_iter()
        .map(|value| format!("\"{value}\""))
        .collect();
    format!("[{}]", inner.join(", "))
}

/// Every disagreement between the declaration and the tree it describes.
#[must_use]
pub fn findings(
    root: &Path,
    registry: &Registry,
    names: &WorkflowNames,
    gate_names: &[&str],
) -> Vec<Finding> {
    let mut findings = Vec::new();
    let subsets = derived_subsets();
    let workflows = derived_workflows(names, &subsets);

    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for row in &registry.gate {
        *seen.entry(row.name.as_str()).or_default() += 1;
    }
    for name in gate_names {
        match seen.get(name).copied().unwrap_or_default() {
            0 => findings.push(Finding::in_file(
                REGISTRY,
                format!("gate `{name}` has no row"),
                "add one with its measured finding count, or regenerate the file with `--write-baseline`",
            )),
            1 => {}
            count => findings.push(Finding::in_file(
                REGISTRY,
                format!("gate `{name}` has {count} rows"),
                "a gate is declared once; delete the duplicate row",
            )),
        }
    }
    for row in &registry.gate {
        if !gate_names.contains(&row.name.as_str()) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("row `{}` names no registered gate", row.name),
                "delete the row, or register the gate it names",
            ));
            continue;
        }
        let declared: BTreeSet<String> = row.subsets.iter().cloned().collect();
        let actual = subsets.get(&row.name).cloned().unwrap_or_default();
        if declared != actual {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "row `{}` declares subsets {} and the registry puts it in {}",
                    row.name,
                    list(declared),
                    list(actual)
                ),
                "the subsets are the registry's; correct the row or move the gate",
            ));
        }
        let declared: BTreeSet<String> = row.workflows.iter().cloned().collect();
        let actual = workflows.get(&row.name).cloned().unwrap_or_default();
        if actual.is_empty() {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "no workflow runs gate `{}`, so nothing in CI selects it by name",
                    row.name
                ),
                "invoke it from the workflow that owns it, add it to a subset a workflow runs, or delete the gate",
            ));
        } else if declared != actual {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "row `{}` declares workflows {} and the steps run it from {}",
                    row.name,
                    list(declared),
                    list(actual)
                ),
                "the workflows are the steps'; correct the row or change the step",
            ));
        }
    }

    for subset in SUBSETS {
        for gate in subset.gates {
            if !gate_names.contains(gate) {
                findings.push(Finding::new(
                    format!("subset `{}` names `{gate}`, which is not a registered gate", subset.name),
                    "register the gate, or take the name out of the subset",
                ));
            }
        }
        if !names.subsets.contains_key(subset.name) {
            findings.push(Finding::new(
                format!("no workflow runs subset `{}`", subset.name),
                format!("run `xtask gates --subset {}` from a workflow, or delete the subset", subset.name),
            ));
        }
    }
    if !names.invoked.contains_key(RUNNER) {
        findings.push(Finding::new(
            "no workflow runs `xtask gates`, so no workflow runs the whole registry",
            "run `xtask gates` from the workflow that owns the sweep",
        ));
    }
    for name in names.subsets.keys() {
        if !SUBSETS.iter().any(|subset| subset.name == name) {
            findings.push(Finding::new(
                format!("a workflow runs `xtask gates --subset {name}`, which is not a registered subset"),
                "correct the step, or register the subset",
            ));
        }
    }
    for name in names.invoked.keys() {
        if name != RUNNER && !gate_names.contains(&name.as_str()) {
            findings.push(Finding::new(
                format!("a workflow invokes `xtask {name}`, which is not a registered gate"),
                "correct the step, or register the gate",
            ));
        }
    }

    findings.extend(external_findings(root, registry, names));
    findings.extend(paused_findings(root, registry));
    findings
}

/// Every disagreement about a workflow that is parked rather than run.
///
/// A parked workflow is invisible: it keeps its name, its steps and its place
/// in the reader's head, and runs nothing. Either it is deleted, or the reason
/// and the condition that ends the pause are written down where the rest of the
/// CI surface is declared.
fn paused_findings(root: &Path, registry: &Registry) -> Vec<Finding> {
    let mut findings = Vec::new();
    let parked = paused_workflows(root);
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for row in &registry.paused {
        *seen.entry(row.workflow.as_str()).or_default() += 1;
    }
    for workflow in &parked {
        match seen.get(workflow.as_str()).copied().unwrap_or_default() {
            0 => findings.push(Finding::in_file(
                REGISTRY,
                format!("`{workflow}` is parked under {PAUSED} and no row records the pause"),
                "add a `[[paused]]` row naming why it does not run and what has to be true \
                 before it does, or delete the workflow",
            )),
            1 => {}
            count => findings.push(Finding::in_file(
                REGISTRY,
                format!("`{workflow}` has {count} paused rows"),
                "a pause is recorded once; delete the duplicate row",
            )),
        }
    }
    for row in &registry.paused {
        if !parked.contains(&row.workflow) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "paused row `{}` names no file under {PAUSED}",
                    row.workflow
                ),
                "delete the row; the workflow it names was restored or deleted",
            ));
            continue;
        }
        if row.reason.trim().is_empty() {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("paused row `{}` records no reason", row.workflow),
                "state why the workflow does not run, or delete the workflow",
            ));
        }
        if row.returns_when.trim().is_empty() {
            findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "paused row `{}` records no condition for its return",
                    row.workflow
                ),
                "state what has to be true before it runs again; a pause with no way back is \
                 a deletion nobody performed",
            ));
        }
    }
    findings
}

/// Every disagreement about a check CI runs that is not an xtask gate.
fn external_findings(root: &Path, registry: &Registry, names: &WorkflowNames) -> Vec<Finding> {
    let mut findings = Vec::new();
    let directory = root.join("scripts");
    for (file, line, script) in &names.scripts {
        if script.contains('*') {
            if !SCRIPT_GLOBS.contains(&script.as_str()) {
                findings.push(Finding::at(
                    format!("{WORKFLOWS}/{file}"),
                    *line as u32,
                    format!("`scripts/{script}` is not an accepted glob"),
                    "name the script, or add the glob to the accepted set",
                ));
            }
            continue;
        }
        if !directory.join(script).exists() {
            findings.push(Finding::at(
                format!("{WORKFLOWS}/{file}"),
                *line as u32,
                format!("the step runs `scripts/{script}`, which the checkout does not carry"),
                "point the step at what owns the rule now, or delete the step",
            ));
        }
    }

    let actual = derived_externals(names);
    let declared: BTreeMap<&str, BTreeSet<String>> = registry
        .external
        .iter()
        .map(|row| (row.run.as_str(), row.workflows.iter().cloned().collect()))
        .collect();
    for (run, files) in &actual {
        match declared.get(run.as_str()) {
            None => findings.push(Finding::in_file(
                REGISTRY,
                format!("a workflow runs `{run}`, which no row declares"),
                "add an `[[external]]` row naming it and the workflows that run it, or delete the step",
            )),
            Some(rows) if rows != files => findings.push(Finding::in_file(
                REGISTRY,
                format!(
                    "row `{run}` declares workflows {} and the steps run it from {}",
                    list(rows.clone()),
                    list(files.clone())
                ),
                "the workflows are the steps'; correct the row or change the step",
            )),
            Some(_) => {}
        }
    }
    for run in declared.keys() {
        if !actual.contains_key(*run) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("row `{run}` names a check no workflow runs"),
                "run it from the workflow that owns it, or delete the row",
            ));
        }
    }
    findings
}

/// Derive the declaration from the tree and write it.
///
/// Nothing here measures a gate, because no column is a measurement: the
/// subsets come from the registry and the workflows from the steps. A pause
/// carries prose no derivation can produce, so a recorded reason survives the
/// rewrite and a newly parked workflow comes out empty, and red.
fn write(root: &Path) -> Result<Report, GateError> {
    let names = workflow_names(root);
    let subsets = derived_subsets();
    let workflows = derived_workflows(&names, &subsets);
    let externals = derived_externals(&names);
    let parked = paused_workflows(root);
    let recorded = load(root).map(|registry| registry.paused).unwrap_or_default();
    let gates = subcommands::registry();
    let gate_names: Vec<&str> = gates.iter().map(|gate| gate.name()).collect();
    let text = render(&gate_names, &subsets, &workflows, &externals, &recorded, &parked);
    let path = registry_path(root);
    fs::write(&path, text).map_err(|error| GateError {
        message: format!("cannot write {}: {error}", path.display()),
        fix: "check the permissions on the xtask directory".to_string(),
    })?;
    let mut report = Report::clean();
    report.note(format!(
        "wrote {} gate row(s), {} external row(s) and {} paused row(s) to {}",
        gate_names.len(),
        externals.len(),
        parked.len(),
        REGISTRY
    ));
    Ok(report)
}

/// Hold every CI entry point to one declaration.
pub struct CiRegistry;

impl Gate for CiRegistry {
    fn name(&self) -> &'static str {
        "ci-registry"
    }

    fn help(&self) -> &'static str {
        "Hold xtask/ci-registry.toml to the gate registry, the subsets and the workflow steps, in both directions"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        if ctx.write {
            return write(&ctx.root);
        }
        let registry = load(&ctx.root)?;
        let names = workflow_names(&ctx.root);
        let gates = subcommands::registry();
        let gate_names: Vec<&str> = gates.iter().map(|gate| gate.name()).collect();
        let mut report = Report::with_findings(findings(
            &ctx.root,
            &registry,
            &names,
            &gate_names,
        ));
        report.note(format!(
            "{} gate row(s), {} external row(s), {} paused row(s), {} subset(s), {} workflow file(s) read",
            registry.gate.len(),
            registry.external.len(),
            registry.paused.len(),
            SUBSETS.len(),
            names
                .invoked
                .values()
                .chain(names.subsets.values())
                .chain(names.packages.values())
                .flatten()
                .collect::<BTreeSet<_>>()
                .len()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(pairs: &[(&str, &str)], subsets: &[(&str, &str)]) -> WorkflowNames {
        let mut out = WorkflowNames::default();
        for (name, file) in pairs {
            out.invoked
                .entry((*name).to_string())
                .or_default()
                .insert((*file).to_string());
        }
        for (name, file) in subsets {
            out.subsets
                .entry((*name).to_string())
                .or_default()
                .insert((*file).to_string());
        }
        out
    }

    /// WHY: a workflow that runs one subset runs the gates in that subset and
    /// nothing else. Recording the sweep for it credited every registered gate
    /// to that file, which both hid gates no workflow selects and named the
    /// wrong workflow on the rows it did report.
    #[test]
    fn a_subset_step_does_not_credit_the_whole_registry() {
        let mut scanned = WorkflowNames::default();
        read_line(
            &mut scanned,
            "docs-ci.yml",
            7,
            &format!("        run: ./cargo_full run -p xtask --bin xtask -- {RUNNER} --subset docs"),
        );
        assert!(!scanned.invoked.contains_key(RUNNER), "{:?}", scanned.invoked);
        assert_eq!(
            scanned.subsets.get("docs"),
            Some(&BTreeSet::from(["docs-ci.yml".to_string()]))
        );

        let mut swept = WorkflowNames::default();
        read_line(
            &mut swept,
            "gates.yml",
            7,
            &format!("        run: ./cargo_full run -p xtask --bin xtask -- {RUNNER}"),
        );
        assert_eq!(
            swept.invoked.get(RUNNER),
            Some(&BTreeSet::from(["gates.yml".to_string()]))
        );
    }

    fn registry(rows: Vec<GateRow>) -> Registry {
        Registry {
            schema_version: SCHEMA_VERSION,
            gate: rows,
            external: Vec::new(),
            paused: Vec::new(),
        }
    }

    fn row(name: &str, subsets: &[&str], workflows: &[&str]) -> GateRow {
        GateRow {
            name: name.to_string(),
            subsets: subsets.iter().map(|value| (*value).to_string()).collect(),
            workflows: workflows.iter().map(|value| (*value).to_string()).collect(),
        }
    }

    fn messages(findings: &[Finding]) -> String {
        findings
            .iter()
            .map(|finding| finding.message.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// WHY: the whole point of one declaration is that the four sources agree.
    /// A tree where they do agree must be silent, or every real finding below
    /// is indistinguishable from noise the gate always emits.
    #[test]
    fn the_checked_in_declaration_agrees_with_the_checked_in_tree() {
        let root = crate::checkout::checkout_root();
        let declaration = load(&root).expect("the declaration loads");
        let scanned = workflow_names(&root);
        let gates = subcommands::registry();
        let gate_names: Vec<&str> = gates.iter().map(|gate| gate.name()).collect();
        let found = findings(&root, &declaration, &scanned, &gate_names);
        assert!(found.is_empty(), "{}", messages(&found));
    }

    /// WHY: a gate with no row runs unpinned, so a new finding in it passes; a
    /// row with no gate is a pin nobody enforces. Both directions have to fail,
    /// and a gate declared twice is two pins for one name, whichever the reader
    /// edits.
    #[test]
    fn a_row_and_a_gate_that_do_not_pair_up_both_fail() {
        let scanned = names(&[(RUNNER, "gates.yml")], &[]);
        let missing = findings(
            Path::new("."),
            &registry(Vec::new()),
            &scanned,
            &["dep-drift"],
        );
        assert!(
            messages(&missing).contains("gate `dep-drift` has no row"),
            "{}",
            messages(&missing)
        );

        let retired = findings(
            Path::new("."),
            &registry(vec![row("retired-gate", &[], &["gates.yml"])]),
            &scanned,
            &[],
        );
        assert!(
            messages(&retired).contains("row `retired-gate` names no registered gate"),
            "{}",
            messages(&retired)
        );

        let twice = findings(
            Path::new("."),
            &registry(vec![
                row("dep-drift", &["prepublish"], &["gates.yml"]),
                row("dep-drift", &["prepublish"], &["gates.yml"]),
            ]),
            &scanned,
            &["dep-drift"],
        );
        assert!(
            messages(&twice).contains("gate `dep-drift` has 2 rows"),
            "{}",
            messages(&twice)
        );
    }

    /// WHY: the subset column is what tells a reader which sweep reaches a
    /// gate. A row that names a subset the registry does not put the gate in
    /// sends the reader to a command that never runs it.
    #[test]
    fn a_row_declaring_the_wrong_subset_fails() {
        let scanned = names(&[(RUNNER, "gates.yml")], &[]);
        let wrong = findings(
            Path::new("."),
            &registry(vec![row("dep-drift", &["docs"], &["gates.yml"])]),
            &scanned,
            &["dep-drift"],
        );
        assert!(
            messages(&wrong).contains("declares subsets [\"docs\"]"),
            "{}",
            messages(&wrong)
        );
    }

    /// WHY: this is the class the file exists for. A gate nothing runs judges
    /// nothing forever while its name reads as coverage, and it must be
    /// reported as that rather than as a column mismatch.
    #[test]
    fn a_gate_no_workflow_runs_is_reported_as_unrun() {
        let scanned = WorkflowNames::default();
        let unrun = findings(
            Path::new("."),
            &registry(vec![row("dep-drift", &[], &[])]),
            &scanned,
            &["dep-drift"],
        );
        assert!(
            messages(&unrun).contains("no workflow runs gate `dep-drift`"),
            "{}",
            messages(&unrun)
        );
        assert!(
            messages(&unrun).contains("no workflow runs `xtask gates`"),
            "{}",
            messages(&unrun)
        );
    }

    /// WHY: a step that names a gate nobody registered runs nothing under a
    /// name that reads as coverage, and the same holds for a subset.
    #[test]
    fn a_step_naming_no_registered_gate_or_subset_fails() {
        let scanned = names(
            &[(RUNNER, "gates.yml"), ("dep-drfit", "gates.yml")],
            &[("not-a-subset", "gates.yml")],
        );
        let found = findings(
            Path::new("."),
            &registry(vec![row("dep-drift", &[], &["gates.yml"])]),
            &scanned,
            &["dep-drift"],
        );
        assert!(
            messages(&found).contains("a workflow invokes `xtask dep-drfit`"),
            "{}",
            messages(&found)
        );
        assert!(
            messages(&found).contains("--subset not-a-subset"),
            "{}",
            messages(&found)
        );
    }

    /// WHY: half the CI surface is not an xtask gate. A script step and a
    /// package run are entry points too, and an undeclared one is exactly the
    /// hole the declaration closes.
    #[test]
    fn an_undeclared_external_entry_point_fails_in_both_directions() {
        let mut scanned = WorkflowNames::default();
        scanned
            .packages
            .entry("structure-gate".to_string())
            .or_default()
            .insert("gates.yml".to_string());
        let undeclared = external_findings(
            Path::new("."),
            &Registry {
                schema_version: SCHEMA_VERSION,
                gate: Vec::new(),
                external: Vec::new(),
                paused: Vec::new(),
            },
            &scanned,
        );
        assert!(
            messages(&undeclared).contains("a workflow runs `-p structure-gate`, which no row declares"),
            "{}",
            messages(&undeclared)
        );

        let unrun = external_findings(
            Path::new("."),
            &Registry {
                schema_version: SCHEMA_VERSION,
                gate: Vec::new(),
                external: vec![ExternalRow {
                    run: "-p gone".to_string(),
                    workflows: vec!["gates.yml".to_string()],
                }],
                paused: Vec::new(),
            },
            &WorkflowNames::default(),
        );
        assert!(
            messages(&unrun).contains("row `-p gone` names a check no workflow runs"),
            "{}",
            messages(&unrun)
        );
    }

    /// WHY: a step pointing at a script the checkout no longer carries fails at
    /// run time under a step name that still reads as coverage, so the
    /// declaration has to name it before CI does.
    #[test]
    fn a_step_naming_a_script_the_tree_lacks_fails() {
        let root = std::env::temp_dir().join(format!("vyre-ci-registry-{}", std::process::id()));
        fs::create_dir_all(root.join("scripts")).expect("the fixture tree is created");
        fs::write(root.join("scripts/present.sh"), "#!/bin/sh\n").expect("the script is written");
        let mut scanned = WorkflowNames::default();
        scanned.scripts.push((
            "gates.yml".to_string(),
            7,
            "present.sh".to_string(),
        ));
        scanned.scripts.push((
            "gates.yml".to_string(),
            9,
            "departed.sh".to_string(),
        ));
        let declaration = Registry {
            schema_version: SCHEMA_VERSION,
            gate: Vec::new(),
            external: vec![
                ExternalRow {
                    run: "scripts/present.sh".to_string(),
                    workflows: vec!["gates.yml".to_string()],
                },
                ExternalRow {
                    run: "scripts/departed.sh".to_string(),
                    workflows: vec!["gates.yml".to_string()],
                },
            ],
            paused: Vec::new(),
        };
        let found = external_findings(&root, &declaration, &scanned);
        fs::remove_dir_all(&root).ok();
        assert_eq!(found.len(), 1, "{}", messages(&found));
        assert!(
            found[0].message.contains("`scripts/departed.sh`"),
            "{}",
            messages(&found)
        );
    }

    /// WHY: a workflow moved out of `.github/workflows` keeps every appearance
    /// of a lane and runs nothing. The required-context document named two
    /// parked workflows as deep gates for months while nothing was red. Every
    /// direction has to fail: no row, an empty reason, no way back, and a row
    /// naming a file that is not parked at all.
    #[test]
    fn a_pause_without_a_reason_or_a_way_back_fails() {
        let root = std::env::temp_dir().join(format!("vyre-ci-paused-{}", std::process::id()));
        fs::create_dir_all(root.join(PAUSED)).expect("the fixture tree is created");
        fs::write(root.join(PAUSED).join("book.yml"), "name: book\n")
            .expect("the workflow is written");

        let unrecorded = paused_findings(&root, &registry(Vec::new()));
        assert!(
            messages(&unrecorded).contains("`book.yml` is parked under"),
            "{}",
            messages(&unrecorded)
        );

        let mut declaration = registry(Vec::new());
        declaration.paused.push(PausedRow {
            workflow: "book.yml".to_string(),
            reason: String::new(),
            returns_when: "  ".to_string(),
        });
        let empty = paused_findings(&root, &declaration);
        assert!(
            messages(&empty).contains("records no reason"),
            "{}",
            messages(&empty)
        );
        assert!(
            messages(&empty).contains("records no condition for its return"),
            "{}",
            messages(&empty)
        );

        let mut declaration = registry(Vec::new());
        declaration.paused.push(PausedRow {
            workflow: "book.yml".to_string(),
            reason: "the build path names a directory the checkout does not carry".to_string(),
            returns_when: "the path names the book this repository ships".to_string(),
        });
        declaration.paused.push(PausedRow {
            workflow: "restored.yml".to_string(),
            reason: "recorded once".to_string(),
            returns_when: "recorded once".to_string(),
        });
        let stale = paused_findings(&root, &declaration);
        fs::remove_dir_all(&root).ok();
        assert_eq!(stale.len(), 1, "{}", messages(&stale));
        assert!(
            stale[0].message.contains("paused row `restored.yml` names no file"),
            "{}",
            messages(&stale)
        );
    }

    /// WHY: the file used to carry per-row prose that let a failing gate stay
    /// legal. A retired field must fail to load rather than be ignored, a schema
    /// this gate does not read must stop it rather than be guessed at, and the
    /// `findings` pin belongs to `xtask/gate-baselines.toml`, so a pin written
    /// here is a second copy of a fact and fails to load.
    #[test]
    fn a_retired_field_and_an_unknown_schema_both_refuse_to_load() {
        let good: Registry = toml::from_str(
            "schema_version = 1\n[[gate]]\nname = \"dep-drift\"\nsubsets = []\nworkflows = [\"gates.yml\"]\n",
        )
        .expect("the current schema loads");
        assert_eq!(good.gate.len(), 1);
        for text in [
            "schema_version = 1\n[[gate]]\nname = \"dep-drift\"\nstatus = \"red\"\n",
            "schema_version = 1\n[[gate]]\nname = \"dep-drift\"\nfindings = 0\n",
        ] {
            assert!(
                toml::from_str::<Registry>(text).is_err(),
                "a row carrying a field this file does not own must not load: {text}"
            );
        }
    }

    /// WHY: a workflow explains itself in YAML comments, and prose that ends a
    /// sentence with a script name invokes nothing. Reading a comment as a step
    /// makes the gate fail on documentation.
    #[test]
    fn a_reference_comes_from_a_command_not_from_prose() {
        assert_eq!(
            referenced_script("        run: bash scripts/check_feature_msrv.sh"),
            Some("check_feature_msrv.sh")
        );
        assert_eq!(referenced_script("      # see scripts/check_feature_msrv.sh"), None);
        assert_eq!(token("dep-drift --strict"), "dep-drift");
        assert_eq!(token("--nocapture"), "--nocapture");
    }

    /// WHY: the derivation is what makes every column checkable rather than
    /// maintained by hand. A gate reachable only through a subset a workflow
    /// runs is run by that workflow, and the whole-registry sweep runs
    /// everything.
    #[test]
    fn a_gate_is_run_by_the_workflow_that_runs_its_subset() {
        let subsets = derived_subsets();
        let scanned = names(&[], &[("docs", "gates.yml")]);
        let workflows = derived_workflows(&scanned, &subsets);
        assert_eq!(
            workflows.get("docs-register").expect("a registered gate"),
            &BTreeSet::from(["gates.yml".to_string()])
        );
        let swept = derived_workflows(&names(&[(RUNNER, "sweep.yml")], &[]), &subsets);
        assert_eq!(
            swept.get("docs-register").expect("a registered gate"),
            &BTreeSet::from(["sweep.yml".to_string()])
        );
    }
}
