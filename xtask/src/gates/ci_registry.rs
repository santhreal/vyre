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
//! The declaration is written by `xtask ci-registry --write`, which derives
//! every column from the registry and the workflow steps, so no column is
//! maintained by hand and none can quietly fall behind.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::sweep::RUNNER;
use crate::subcommands;

/// The one declaration of every check CI runs.
pub const REGISTRY: &str = "xtask/ci-registry.toml";
/// Schema this gate reads.
pub const SCHEMA_VERSION: i64 = 1;
/// The command that writes the declaration.
pub const WRITER: &str = "./cargo_full run -p xtask --bin xtask -- ci-registry --write";
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

/// One workflow path the tree carries or once carried.
///
/// A workflow moved out of `.github/workflows` stops running and keeps every
/// appearance of a lane: `.github/CI_REQUIRED.md` named two parked workflows as
/// deep gates for months, and nothing was red. Deleting the file is quieter
/// still, because the wiring this file records is derived from the tree and a
/// deleted lane derives to nothing at all. Every path therefore keeps a row for
/// as long as the repository exists, and the row says what happened to it.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkflowRow {
    /// Repository-relative path, live or parked.
    pub path: String,
    /// `live`, `paused`, `superseded` or `unprotected`.
    pub state: String,
    /// Why it does not run. Required unless the state is `live`.
    #[serde(default)]
    pub reason: String,
    /// What has to be true before it runs again. Paused rows only.
    #[serde(default)]
    pub returns_when: String,
    /// The workflow path that runs the checks this one ran. Superseded rows
    /// only.
    #[serde(default)]
    pub superseded_by: String,
    /// The registered gate that carries the checks. Superseded rows only, and
    /// optional: a workflow can supersede another without a gate of its own.
    #[serde(default)]
    pub gate: String,
    /// The verification class no check covers. Unprotected rows only.
    #[serde(default)]
    pub class: String,
}

/// A workflow file that runs.
pub const LIVE: &str = "live";
/// A workflow file that is parked and expected back.
pub const PAUSED_STATE: &str = "paused";
/// A deleted workflow whose checks another workflow runs.
pub const SUPERSEDED: &str = "superseded";
/// A deleted workflow whose verification class nothing covers.
pub const UNPROTECTED: &str = "unprotected";
/// Every state a row may declare.
const STATES: &[&str] = &[LIVE, PAUSED_STATE, SUPERSEDED, UNPROTECTED];

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
    /// One row per workflow path the tree carries or once carried.
    #[serde(default)]
    pub workflow: Vec<WorkflowRow>,
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
            format!("regenerate it with `{WRITER}`"),
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
            format!("regenerate the file with `{WRITER}`, or teach the gate the new schema"),
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
    /// Every `scripts/<path>` a live workflow names, and where, with the line.
    pub scripts: Vec<(String, usize, String)>,
    /// Every `scripts/<path>` a paused workflow names, and where, with the
    /// line. A parked workflow runs nothing, so it credits no check, and a
    /// script it names still has to exist for the pause to be reversible.
    pub paused_scripts: Vec<(String, usize, String)>,
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
    for path in yaml_files(&root.join(WORKFLOWS)) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let file = file_name(&path);
        for (index, line) in text.lines().enumerate() {
            read_line(&mut names, &file, index + 1, line);
        }
    }
    for path in yaml_files(&root.join(PAUSED)) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let file = file_name(&path);
        for (index, line) in text.lines().enumerate() {
            if let Some(script) = referenced_script(line) {
                names
                    .paused_scripts
                    .push((file.clone(), index + 1, script.to_string()));
            }
        }
    }
    names
}

/// Every workflow file in one directory, sorted.
fn yaml_files(directory: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(directory) else {
        return Vec::new();
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
    files
}

/// The file name of `path`.
fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
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
pub(crate) fn referenced_script(line: &str) -> Option<&str> {
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
pub(crate) fn strip_yaml_comment(line: &str) -> &str {
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
    for subset in subcommands::subsets() {
        for gate in subset.gates {
            map.entry(gate.to_string())
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

/// Every workflow file the checkout carries, by path, with the state its
/// directory gives it.
#[must_use]
pub fn workflow_files(root: &Path) -> BTreeMap<String, &'static str> {
    let mut files = BTreeMap::new();
    for (directory, state) in [(WORKFLOWS, LIVE), (PAUSED, PAUSED_STATE)] {
        let Ok(entries) = fs::read_dir(root.join(directory)) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path
                .extension()
                .is_some_and(|extension| extension == "yml" || extension == "yaml")
            {
                continue;
            }
            let Some(name) = path.file_name() else {
                continue;
            };
            files.insert(format!("{directory}/{}", name.to_string_lossy()), state);
        }
    }
    files
}

/// Render the declaration from derived wiring.
///
/// A pause and a retirement carry prose no derivation can produce, so the
/// writer copies the rows it was given and emits an empty row for a workflow
/// that has none. An empty row is red, which is the point: the writer never
/// invents a reason. A row whose file the checkout no longer carries is copied
/// out unchanged rather than dropped, so deleting a lane cannot delete the
/// record of it.
#[must_use]
pub fn render(
    gate_names: &[&str],
    subsets: &BTreeMap<String, BTreeSet<String>>,
    workflows: &BTreeMap<String, BTreeSet<String>>,
    externals: &BTreeMap<String, BTreeSet<String>>,
    recorded: &[WorkflowRow],
    files: &BTreeMap<String, &'static str>,
) -> String {
    let mut text = String::from(
        "# Every check CI runs, declared once, written by\n\
         # `xtask ci-registry --write`.\n\
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
    let mut paths: BTreeSet<&str> = files.keys().map(String::as_str).collect();
    paths.extend(recorded.iter().map(|row| row.path.as_str()));
    for path in paths {
        let row = recorded.iter().find(|row| row.path == path);
        let state = match (row.map(|row| row.state.as_str()), files.get(path).copied()) {
            (Some(declared), None) => declared,
            (Some(declared), Some(on_disk)) if declared == on_disk => declared,
            (_, Some(on_disk)) => on_disk,
            (None, None) => "",
        };
        text.push_str("\n[[workflow]]\n");
        text.push_str(&format!("path = \"{path}\"\n"));
        text.push_str(&format!("state = \"{state}\"\n"));
        if state == LIVE {
            continue;
        }
        text.push_str(&format!(
            "reason = \"{}\"\n",
            row.map(|row| row.reason.as_str()).unwrap_or_default()
        ));
        match state {
            PAUSED_STATE => text.push_str(&format!(
                "returns_when = \"{}\"\n",
                row.map(|row| row.returns_when.as_str()).unwrap_or_default()
            )),
            SUPERSEDED => {
                text.push_str(&format!(
                    "superseded_by = \"{}\"\n",
                    row.map(|row| row.superseded_by.as_str())
                        .unwrap_or_default()
                ));
                text.push_str(&format!(
                    "gate = \"{}\"\n",
                    row.map(|row| row.gate.as_str()).unwrap_or_default()
                ));
            }
            _ => text.push_str(&format!(
                "class = \"{}\"\n",
                row.map(|row| row.class.as_str()).unwrap_or_default()
            )),
        }
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
                format!("add one with its measured finding count, or regenerate the file with `{WRITER}`"),
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

    let subsets = subcommands::subsets();
    for subset in &subsets {
        for gate in &subset.gates {
            if !gate_names.contains(gate) {
                findings.push(Finding::new(
                    format!(
                        "subset `{}` names `{gate}`, which is not a registered gate",
                        subset.name
                    ),
                    "register the gate, or take the name out of the subset",
                ));
            }
        }
        if !names.subsets.contains_key(subset.name) {
            findings.push(Finding::new(
                format!("no workflow runs subset `{}`", subset.name),
                format!(
                    "run `xtask gates --subset {}` from a workflow, or delete the subset",
                    subset.name
                ),
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
        if !subsets.iter().any(|subset| subset.name == name) {
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
    findings.extend(workflow_findings(root, registry, gate_names));
    findings
}

/// Every disagreement about a workflow path the tree carries or once carried.
///
/// A parked workflow is invisible: it keeps its name, its steps and its place
/// in the reader's head, and runs nothing. A deleted one is quieter still,
/// because the wiring this file records is derived from the tree, so the lane
/// leaves no trace to go red. Every path keeps a row, and the row says which of
/// four things happened to it: it runs, it is parked with a way back, another
/// workflow runs its checks, or its verification class is uncovered.
fn workflow_findings(root: &Path, registry: &Registry, gate_names: &[&str]) -> Vec<Finding> {
    let mut findings = Vec::new();
    let files = workflow_files(root);
    let runs = derived_workflows(&workflow_names(root), &derived_subsets());
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    for row in &registry.workflow {
        *seen.entry(row.path.as_str()).or_default() += 1;
    }
    for (path, state) in &files {
        match seen.get(path.as_str()).copied().unwrap_or_default() {
            0 => findings.push(Finding::in_file(
                REGISTRY,
                format!("`{path}` is in the checkout and no `[[workflow]]` row records it"),
                format!("add a row declaring it `{state}`"),
            )),
            1 => {}
            count => findings.push(Finding::in_file(
                REGISTRY,
                format!("`{path}` has {count} rows"),
                "a workflow is recorded once; delete the duplicate row",
            )),
        }
    }
    for row in &registry.workflow {
        let on_disk = files.get(row.path.as_str()).copied();
        if !STATES.contains(&row.state.as_str()) {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("`{}` declares state `{}`", row.path, row.state),
                format!("a workflow is one of {}", STATES.join(", ")),
            ));
            continue;
        }
        let declared_present = row.state == LIVE || row.state == PAUSED_STATE;
        match (declared_present, on_disk) {
            (true, None) => {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "`{}` is declared `{}` and the checkout does not carry it",
                        row.path, row.state
                    ),
                    format!(
                        "restore the file, or record where its checks went: `{SUPERSEDED}` naming \
                         the workflow that runs them, or `{UNPROTECTED}` naming the class nothing \
                         covers"
                    ),
                ));
                continue;
            }
            (false, Some(state)) => {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "`{}` is declared `{}` and the checkout carries it",
                        row.path, row.state
                    ),
                    format!("the lane is back; declare it `{state}`"),
                ));
                continue;
            }
            (true, Some(state)) if state != row.state => {
                findings.push(Finding::in_file(
                    REGISTRY,
                    format!(
                        "`{}` is declared `{}` and sits under the `{state}` directory",
                        row.path, row.state
                    ),
                    "the directory decides; correct the row or move the file",
                ));
                continue;
            }
            _ => {}
        }
        if row.state == LIVE {
            continue;
        }
        if row.reason.trim().is_empty() {
            findings.push(Finding::in_file(
                REGISTRY,
                format!("`{}` records no reason", row.path),
                "state why it does not run",
            ));
        }
        match row.state.as_str() {
            PAUSED_STATE => {
                if row.returns_when.trim().is_empty() {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!("`{}` records no condition for its return", row.path),
                        "state what has to be true before it runs again; a pause with no way \
                         back is a deletion nobody performed",
                    ));
                }
            }
            SUPERSEDED => {
                let successor = files.get(row.superseded_by.as_str()).copied();
                let file = row
                    .superseded_by
                    .rsplit('/')
                    .next()
                    .unwrap_or_default()
                    .to_string();
                match successor {
                    None => findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{}` is superseded by `{}`, which the checkout does not carry",
                            row.path, row.superseded_by
                        ),
                        format!(
                            "name the workflow that runs its checks, or declare it \
                             `{UNPROTECTED}` with the class nothing covers"
                        ),
                    )),
                    Some(LIVE) => {}
                    Some(state) => findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{}` is superseded by `{}`, which is `{state}` and runs nothing",
                            row.path, row.superseded_by
                        ),
                        format!(
                            "coverage terminates at a lane CI runs today; name a `{LIVE}` \
                             workflow, or declare it `{UNPROTECTED}` with the class nothing \
                             covers"
                        ),
                    )),
                }
                if row.gate.is_empty() {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!("`{}` names no gate that carries its checks", row.path),
                        format!(
                            "name the registered gate the successor runs, or declare the lane \
                             `{UNPROTECTED}` with the class nothing covers"
                        ),
                    ));
                } else if !gate_names.contains(&row.gate.as_str()) {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{}` names gate `{}`, which is not registered",
                            row.path, row.gate
                        ),
                        "name the gate that carries the checks, or leave the field empty",
                    ));
                } else if successor == Some(LIVE)
                    && !runs
                        .get(row.gate.as_str())
                        .is_some_and(|workflows| workflows.contains(&file))
                {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!(
                            "`{}` says gate `{}` runs in `{}`, and that workflow does not run it",
                            row.path, row.gate, row.superseded_by
                        ),
                        "the steps decide; name the workflow whose steps run the gate, or run \
                         the gate from the one named here",
                    ));
                }
            }
            _ => {
                if row.class.trim().is_empty() {
                    findings.push(Finding::in_file(
                        REGISTRY,
                        format!("`{}` records no uncovered class", row.path),
                        "name the verification nothing performs, so the gap is readable",
                    ));
                }
            }
        }
    }
    findings
}

/// Every disagreement about a check CI runs that is not an xtask gate.
fn external_findings(root: &Path, registry: &Registry, names: &WorkflowNames) -> Vec<Finding> {
    let mut findings = Vec::new();
    let directory = root.join("scripts");
    for (parent, scripts) in [(WORKFLOWS, &names.scripts), (PAUSED, &names.paused_scripts)] {
        for (file, line, script) in scripts {
            if script.contains('*') {
                if !SCRIPT_GLOBS.contains(&script.as_str()) {
                    findings.push(Finding::at(
                        format!("{parent}/{file}"),
                        *line as u32,
                        format!("`scripts/{script}` is not an accepted glob"),
                        "name the script, or add the glob to the accepted set",
                    ));
                }
                continue;
            }
            if !directory.join(script).exists() {
                findings.push(Finding::at(
                    format!("{parent}/{file}"),
                    *line as u32,
                    format!("the step runs `scripts/{script}`, which the checkout does not carry"),
                    "point the step at what owns the rule now, or delete the step",
                ));
            }
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
/// subsets come from the registry and the workflows from the steps. A pause and
/// a retirement carry prose no derivation can produce, so a recorded row
/// survives the rewrite, a newly parked workflow comes out empty and red, and a
/// row whose file is gone keeps the state it was given rather than being
/// dropped.
fn write(root: &Path) -> Result<Report, GateError> {
    let names = workflow_names(root);
    let subsets = derived_subsets();
    let workflows = derived_workflows(&names, &subsets);
    let externals = derived_externals(&names);
    let files = workflow_files(root);
    let recorded = match load(root) {
        Ok(registry) => registry.workflow,
        Err(_) if !registry_path(root).exists() => Vec::new(),
        Err(error) => {
            return Err(GateError::new(
                format!(
                    "{REGISTRY} carries the pause and retirement reasons and cannot be read, so \
                     a rewrite would drop them: {}",
                    error.message
                ),
                "repair the file, or delete it to regenerate a declaration that states no reason"
                    .to_string(),
            ));
        }
    };
    let gates = subcommands::registry();
    let gate_names: Vec<&str> = gates.iter().map(|gate| gate.name()).collect();
    let text = render(
        &gate_names,
        &subsets,
        &workflows,
        &externals,
        &recorded,
        &files,
    );
    let path = registry_path(root);
    fs::write(&path, text).map_err(|error| GateError {
        message: format!("cannot write {}: {error}", path.display()),
        fix: "check the permissions on the xtask directory".to_string(),
    })?;
    let mut rows = files.len();
    rows += recorded
        .iter()
        .filter(|row| !files.contains_key(row.path.as_str()))
        .count();
    let mut report = Report::clean();
    report.cover_complete("ci registry rows", gate_names.len());
    report.produced(REGISTRY);
    report.note(format!(
        "wrote {} gate row(s), {} external row(s) and {rows} workflow row(s) to {}",
        gate_names.len(),
        externals.len(),
        REGISTRY
    ));
    Ok(report)
}

/// Hold every CI entry point to one declaration.
pub struct CiRegistry;

impl crate::gate::GateBehavior for CiRegistry {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        if ctx.write {
            return write(&ctx.root);
        }
        let registry = load(&ctx.root)?;
        let names = workflow_names(&ctx.root);
        let gates = subcommands::registry();
        let gate_names: Vec<&str> = gates.iter().map(|gate| gate.name()).collect();
        let mut report = Report::with_findings(findings(&ctx.root, &registry, &names, &gate_names));
        report.cover_complete("ci registry rows", registry.gate.len());
        report.produced(REGISTRY);
        report.note(format!(
            "{} gate row(s), {} external row(s), {} workflow row(s), {} subset(s), {} workflow file(s) read",
            registry.gate.len(),
            registry.external.len(),
            registry.workflow.len(),
            subcommands::subsets().len(),
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
            &format!(
                "        run: ./cargo_full run -p xtask --bin xtask -- {RUNNER} --subset docs"
            ),
        );
        assert!(
            !scanned.invoked.contains_key(RUNNER),
            "{:?}",
            scanned.invoked
        );
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
            workflow: Vec::new(),
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
        Finding::messages(findings)
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
                workflow: Vec::new(),
            },
            &scanned,
        );
        assert!(
            messages(&undeclared)
                .contains("a workflow runs `-p structure-gate`, which no row declares"),
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
                workflow: Vec::new(),
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
        scanned
            .scripts
            .push(("gates.yml".to_string(), 7, "present.sh".to_string()));
        scanned
            .scripts
            .push(("gates.yml".to_string(), 9, "departed.sh".to_string()));
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
            workflow: Vec::new(),
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

    fn workflow_row(path: &str, state: &str) -> WorkflowRow {
        WorkflowRow {
            path: path.to_string(),
            state: state.to_string(),
            reason: String::new(),
            returns_when: String::new(),
            superseded_by: String::new(),
            gate: String::new(),
            class: String::new(),
        }
    }

    /// WHY: a workflow moved out of `.github/workflows` keeps every appearance
    /// of a lane and runs nothing. The required-context document named two
    /// parked workflows as deep gates for months while nothing was red. Every
    /// direction has to fail: no row, an empty reason, no way back, and a row
    /// naming a file the checkout does not carry.
    #[test]
    fn a_pause_without_a_reason_or_a_way_back_fails() {
        let root = std::env::temp_dir().join(format!("vyre-ci-paused-{}", std::process::id()));
        fs::create_dir_all(root.join(PAUSED)).expect("the fixture tree is created");
        fs::write(root.join(PAUSED).join("book.yml"), "name: book\n")
            .expect("the workflow is written");
        let parked = format!("{PAUSED}/book.yml");

        let unrecorded = workflow_findings(&root, &registry(Vec::new()), &[]);
        assert!(
            messages(&unrecorded).contains(&format!("`{parked}` is in the checkout")),
            "{}",
            messages(&unrecorded)
        );

        let mut declaration = registry(Vec::new());
        let mut row = workflow_row(&parked, PAUSED_STATE);
        row.returns_when = "  ".to_string();
        declaration.workflow.push(row);
        let empty = workflow_findings(&root, &declaration, &[]);
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
        let mut row = workflow_row(&parked, PAUSED_STATE);
        row.reason = "the build path names a directory the checkout does not carry".to_string();
        row.returns_when = "the path names the book this repository ships".to_string();
        declaration.workflow.push(row);
        declaration.workflow.push(workflow_row(
            &format!("{PAUSED}/restored.yml"),
            PAUSED_STATE,
        ));
        let stale = workflow_findings(&root, &declaration, &[]);
        fs::remove_dir_all(&root).ok();
        assert!(
            messages(&stale).contains("`.github/workflows-paused/restored.yml` is declared `paused` and the checkout does not carry it"),
            "{}",
            messages(&stale)
        );
    }

    /// WHY: the wiring in this file is derived from the tree, so deleting a
    /// workflow deletes every trace of it and nothing goes red. Seven lanes
    /// went that way in one commit. The row outlives the file: a deletion
    /// leaves the row naming a file nobody carries until someone records where
    /// the checks went, and the writer copies that row out rather than dropping
    /// it.
    #[test]
    fn a_deleted_workflow_leaves_a_row_that_fails() {
        let root = std::env::temp_dir().join(format!("vyre-ci-deleted-{}", std::process::id()));
        fs::create_dir_all(root.join(WORKFLOWS)).expect("the fixture tree is created");
        let live = root.join(WORKFLOWS).join("adversarial.yml");
        fs::write(&live, "name: Adversarial\n").expect("the workflow is written");
        let path = format!("{WORKFLOWS}/adversarial.yml");

        let mut declaration = registry(Vec::new());
        declaration.workflow.push(workflow_row(&path, LIVE));
        assert!(
            workflow_findings(&root, &declaration, &[]).is_empty(),
            "a live workflow with a row is clean"
        );

        fs::remove_file(&live).expect("the workflow is deleted");
        let deleted = workflow_findings(&root, &declaration, &[]);
        assert!(
            messages(&deleted).contains(&format!(
                "`{path}` is declared `live` and the checkout does not carry it"
            )),
            "{}",
            messages(&deleted)
        );

        let written = render(
            &[],
            &BTreeMap::new(),
            &BTreeMap::new(),
            &BTreeMap::new(),
            &declaration.workflow,
            &workflow_files(&root),
        );
        fs::remove_dir_all(&root).ok();
        assert!(written.contains(&format!("path = \"{path}\"")), "{written}");
    }

    /// WHY: the reasons a lane is parked or retired exist only in the committed
    /// declaration; the generator derives everything else. Rewriting over a file
    /// that failed to parse replaces every reason with an empty field and reads
    /// as a clean regeneration, so a rewrite reads the file first and refuses
    /// when it is present and unreadable. An absent file states nothing, so the
    /// first write is allowed.
    #[test]
    fn a_rewrite_refuses_to_drop_the_reasons_it_cannot_read() {
        let root = std::env::temp_dir().join(format!("vyre-ci-rewrite-{}", std::process::id()));
        fs::create_dir_all(root.join(WORKFLOWS)).expect("the fixture tree is created");
        fs::create_dir_all(root.join(XTASK)).expect("the xtask directory is created");
        fs::write(root.join(WORKFLOWS).join("gates.yml"), "name: gates\n")
            .expect("the workflow is written");

        write(&root).expect("an absent declaration is written from the checkout");
        let generated = fs::read_to_string(registry_path(&root)).expect("the file is written");
        fs::write(registry_path(&root), "schema_version = 1\nworkflow = 3\n")
            .expect("the broken declaration is written");
        let refused = write(&root).expect_err("an unreadable declaration is not rewritten");
        let after = fs::read_to_string(registry_path(&root)).expect("the file is still there");
        fs::remove_dir_all(&root).ok();

        assert!(generated.contains("gates.yml"), "{generated}");
        assert!(refused.message.contains("cannot be read"), "{refused:?}");
        assert_eq!(after, "schema_version = 1\nworkflow = 3\n");
    }

    /// WHY: the three ways a lane can end are the three a row may declare, and
    /// each carries the fact that makes it readable. A retirement that names a
    /// successor nobody carries, or an uncovered class nobody states, records
    /// nothing while reading as a decision. Coverage also has to terminate: a
    /// row whose successor is itself parked, or whose successor runs some other
    /// gate, reads as covered while nothing performs the checks, and the
    /// committed declaration retired the randomized-order lane into another
    /// paused lane exactly that way.
    #[test]
    fn a_retirement_states_where_the_checks_went() {
        let root = std::env::temp_dir().join(format!("vyre-ci-retired-{}", std::process::id()));
        fs::create_dir_all(root.join(WORKFLOWS)).expect("the fixture tree is created");
        fs::create_dir_all(root.join(PAUSED)).expect("the parked directory is created");
        fs::write(
            root.join(WORKFLOWS).join("gates.yml"),
            "jobs:\n  run:\n    steps:\n      - run: ./cargo_full run -p xtask --bin xtask -- catalog\n",
        )
        .expect("the workflow is written");
        fs::write(root.join(PAUSED).join("parked.yml"), "name: parked\n")
            .expect("the parked workflow is written");
        let live = || workflow_row(&format!("{WORKFLOWS}/gates.yml"), LIVE);
        let parked = || {
            let mut row = workflow_row(&format!("{PAUSED}/parked.yml"), PAUSED_STATE);
            row.reason = "the toolchain it needs fails to install".to_string();
            row.returns_when = "the install is fixed".to_string();
            row
        };

        let mut declaration = registry(Vec::new());
        declaration.workflow.push(live());
        declaration.workflow.push(parked());
        let mut superseded = workflow_row(&format!("{WORKFLOWS}/catalog.yml"), SUPERSEDED);
        superseded.reason = "the gate registry runs the catalog check".to_string();
        superseded.superseded_by = format!("{WORKFLOWS}/departed.yml");
        superseded.gate = "not-a-gate".to_string();
        declaration.workflow.push(superseded);
        let mut unprotected = workflow_row(&format!("{WORKFLOWS}/mutation.yml"), UNPROTECTED);
        unprotected.reason = "the budget was never validated".to_string();
        declaration.workflow.push(unprotected);

        let found = messages(&workflow_findings(&root, &declaration, &["catalog"]));
        assert!(
            found.contains("which the checkout does not carry"),
            "{found}"
        );
        assert!(found.contains("names gate `not-a-gate`"), "{found}");
        assert!(found.contains("records no uncovered class"), "{found}");

        let mut into_a_parked_lane = registry(Vec::new());
        into_a_parked_lane.workflow.push(live());
        into_a_parked_lane.workflow.push(parked());
        let mut laundered = workflow_row(&format!("{WORKFLOWS}/random-order.yml"), SUPERSEDED);
        laundered.reason = "the other randomized-order lane carries it".to_string();
        laundered.superseded_by = format!("{PAUSED}/parked.yml");
        laundered.gate = "catalog".to_string();
        into_a_parked_lane.workflow.push(laundered);
        let dead_end = messages(&workflow_findings(&root, &into_a_parked_lane, &["catalog"]));
        assert!(
            dead_end.contains("is `paused` and runs nothing"),
            "{dead_end}"
        );

        let mut wrong_lane = registry(Vec::new());
        wrong_lane.workflow.push(live());
        wrong_lane.workflow.push(parked());
        let mut elsewhere = workflow_row(&format!("{WORKFLOWS}/catalog.yml"), SUPERSEDED);
        elsewhere.reason = "the gate registry runs the catalog check".to_string();
        elsewhere.superseded_by = format!("{WORKFLOWS}/gates.yml");
        elsewhere.gate = "docs-check".to_string();
        wrong_lane.workflow.push(elsewhere);
        let unrun = messages(&workflow_findings(
            &root,
            &wrong_lane,
            &["catalog", "docs-check"],
        ));
        assert!(
            unrun.contains("and that workflow does not run it"),
            "{unrun}"
        );

        let mut declaration = registry(Vec::new());
        declaration.workflow.push(live());
        declaration.workflow.push(parked());
        let mut superseded = workflow_row(&format!("{WORKFLOWS}/catalog.yml"), SUPERSEDED);
        superseded.reason = "the gate registry runs the catalog check".to_string();
        superseded.superseded_by = format!("{WORKFLOWS}/gates.yml");
        superseded.gate = "catalog".to_string();
        declaration.workflow.push(superseded);
        let mut unprotected = workflow_row(&format!("{WORKFLOWS}/mutation.yml"), UNPROTECTED);
        unprotected.reason = "the budget was never validated".to_string();
        unprotected.class = "mutation coverage of the verifier".to_string();
        declaration.workflow.push(unprotected);
        let clean = workflow_findings(&root, &declaration, &["catalog"]);
        fs::remove_dir_all(&root).ok();
        assert!(clean.is_empty(), "{}", messages(&clean));
    }

    /// WHY: a parked lane is a lane someone means to restart, and the script it
    /// calls rots while it waits. The reference is judged in both directories,
    /// and a parked file still runs nothing, so it credits no `[[external]]`
    /// row and cannot make a deleted check read as covered.
    #[test]
    fn a_paused_workflow_naming_a_missing_script_is_reported() {
        let root = std::env::temp_dir().join(format!("vyre-ci-parked-{}", std::process::id()));
        fs::create_dir_all(root.join(PAUSED)).expect("the fixture tree is created");
        fs::create_dir_all(root.join("scripts")).expect("the script directory is created");
        fs::write(
            root.join(PAUSED).join("mutation-testing.yml"),
            "jobs:\n  run:\n    steps:\n      - run: scripts/mutation_budget.sh\n",
        )
        .expect("the parked workflow is written");

        let names = workflow_names(&root);
        let found = messages(&external_findings(&root, &registry(Vec::new()), &names));
        let credited = derived_externals(&names);
        fs::remove_dir_all(&root).ok();
        assert!(
            found.contains(
                "the step runs `scripts/mutation_budget.sh`, which the checkout does not carry"
            ),
            "{found}"
        );
        assert!(credited.is_empty(), "{credited:?}");
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

    /// WHY: a reference is read out of a shell command, and the same file
    /// explains itself in YAML comments. Prose that ends a sentence with a
    /// script name invokes nothing, so reading it as a reference makes the gate
    /// fail on documentation. A quoted path, a nested path, a trailing
    /// semicolon and a glob are all forms a step is written in, and the reader
    /// takes the command even when a comment follows it on the same line.
    #[test]
    fn a_reference_comes_from_a_command_not_from_prose() {
        assert_eq!(
            referenced_script("        run: bash scripts/check_feature_msrv.sh"),
            Some("check_feature_msrv.sh")
        );
        assert_eq!(
            referenced_script("        run: bash scripts/lib/cargo_runner.sh --strict"),
            Some("lib/cargo_runner.sh")
        );
        assert_eq!(
            referenced_script("        run: bash \"scripts/check_public_api.sh\";"),
            Some("check_public_api.sh")
        );
        assert_eq!(
            referenced_script("        run: bash scripts/check_*.sh"),
            Some("check_*.sh")
        );
        assert_eq!(
            referenced_script("        run: bash scripts/gate.sh # see scripts/other.sh."),
            Some("gate.sh")
        );
        assert_eq!(
            referenced_script("      # all on scripts/cargo_runner.sh."),
            None
        );
        assert_eq!(
            referenced_script("      # see scripts/check_feature_msrv.sh"),
            None
        );
        assert_eq!(referenced_script("        run: cargo test"), None);
        assert_eq!(token("dep-drift --strict"), "dep-drift");
        assert_eq!(token("--nocapture"), "--nocapture");
        assert_eq!(token(""), "");
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
