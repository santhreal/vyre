//! The `ci-steps` gate: every cargo selector a workflow names resolves.
//!
//! A workflow step names packages, targets and features as plain text, and
//! nothing reads that text until CI runs the step. Two failure modes come out
//! of that, and only one of them is loud.
//!
//! The loud one: a selector the tree cannot satisfy. `conform.yml` ran twelve
//! `--test` targets against `-p vyre-primitives` after they moved to
//! `vyre-libs` with their domains, and passed `--features all-lego` in the same
//! commit that deleted the feature. cargo can only refuse a step like that, so
//! the lane is red for a reason that a manifest read would have caught before
//! the push.
//!
//! The quiet one is worse. A `[[test]]` whose `required-features` are not
//! enabled is not an error: cargo skips the target in silence and the step
//! exits zero. A step whose features are wrong therefore passes forever while
//! naming an assertion nobody runs, which is indistinguishable from coverage in
//! every place a reader looks.
//!
//! Both directions are one rule, because they are one question: does the step
//! run what its text says it runs. The gate resolves every `-p`, `--package`,
//! `--test`, `--bench`, `--example`, `--bin`, `--features` and `-F` token in
//! every workflow step against the workspace manifests and the tracked sources,
//! and reports a token the tree cannot satisfy and a named target the step's
//! feature set leaves skipped.
//!
//! Paused workflows are read too. A workflow parked under
//! `.github/workflows-paused` that names a target the checkout no longer
//! carries cannot return, whatever its row says, and a pause with no way back
//! is a deletion nobody performed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Where the workflows that run live.
pub const WORKFLOWS: &str = ".github/workflows";
/// Where a workflow that does not run is parked.
pub const PAUSED: &str = ".github/workflows-paused";

/// Where a check CI runs can also live.
pub const SCRIPTS: &str = "scripts";

/// A kind of cargo target a step can select by name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// `--test`
    Test,
    /// `--bench`
    Bench,
    /// `--example`
    Example,
    /// `--bin`
    Bin,
}

impl Kind {
    /// The flag that selects this kind.
    fn flag(self) -> &'static str {
        match self {
            Kind::Test => "--test",
            Kind::Bench => "--bench",
            Kind::Example => "--example",
            Kind::Bin => "--bin",
        }
    }

    /// The manifest table that declares one explicitly.
    fn table(self) -> &'static str {
        match self {
            Kind::Test => "test",
            Kind::Bench => "bench",
            Kind::Example => "example",
            Kind::Bin => "bin",
        }
    }

    /// The directory holding implicitly discovered targets of this kind.
    fn directory(self) -> &'static str {
        match self {
            Kind::Test => "tests",
            Kind::Bench => "benches",
            Kind::Example => "examples",
            Kind::Bin => "src/bin",
        }
    }

    /// The `[package]` key that turns implicit discovery off.
    fn autodiscovery(self) -> &'static str {
        match self {
            Kind::Test => "autotests",
            Kind::Bench => "autobenches",
            Kind::Example => "autoexamples",
            Kind::Bin => "autobins",
        }
    }

    /// Every kind, in flag order.
    fn all() -> [Kind; 4] {
        [Kind::Test, Kind::Bench, Kind::Example, Kind::Bin]
    }
}

/// One cargo target a step can name, with the features it needs to exist.
#[derive(Debug, Default)]
pub struct Target {
    /// Features that must all be enabled or cargo skips the target in silence.
    pub required_features: Vec<String>,
}

/// One workspace member, as far as a workflow selector can see it.
#[derive(Debug)]
pub struct Package {
    /// Name as `-p` spells it.
    pub name: String,
    /// Features it declares, each with the features it turns on.
    pub features: BTreeMap<String, Vec<String>>,
    /// Optional dependencies, each of which is an implicit feature.
    pub optional: BTreeSet<String>,
    /// Named targets, by kind.
    pub targets: BTreeMap<Kind, BTreeMap<String, Target>>,
    /// The binary `cargo run` picks when the package ships more than one.
    pub default_run: Option<String>,
}

impl Package {
    /// Whether the package declares `name` as a feature.
    #[must_use]
    pub fn has_feature(&self, name: &str) -> bool {
        self.features.contains_key(name) || self.optional.contains(name)
    }

    /// Every feature `requested` turns on, following the feature graph.
    #[must_use]
    pub fn closure(&self, requested: &[String]) -> BTreeSet<String> {
        let mut enabled = BTreeSet::new();
        let mut pending: Vec<String> = requested.to_vec();
        while let Some(feature) = pending.pop() {
            let (owner, feature) = match feature.split_once('/') {
                Some((owner, rest)) => (Some(owner.to_string()), rest.to_string()),
                None => (None, feature),
            };
            // A `dep/feature` request enables the dependency's feature, and the
            // implicit feature of the same name here when the dependency is
            // optional. It never enables a feature of this package.
            if let Some(owner) = owner {
                if self.optional.contains(&owner) {
                    enabled.insert(owner);
                }
                continue;
            }
            if !enabled.insert(feature.clone()) {
                continue;
            }
            if let Some(edges) = self.features.get(&feature) {
                pending.extend(edges.iter().cloned());
            }
        }
        enabled
    }
}

/// Read every workspace member a workflow selector can name.
pub fn packages(tree: &Tree) -> Result<BTreeMap<String, Package>, GateError> {
    let mut packages = BTreeMap::new();
    for member in tree.member_manifests()? {
        let mut features: BTreeMap<String, Vec<String>> = BTreeMap::new();
        if let Some(table) = member.manifest.get("features").and_then(toml::Value::as_table) {
            for (name, value) in table {
                let edges = value
                    .as_array()
                    .map(|array| {
                        array
                            .iter()
                            .filter_map(|entry| entry.as_str().map(str::to_string))
                            .collect()
                    })
                    .unwrap_or_default();
                features.insert(name.clone(), edges);
            }
        }
        let optional = crate::manifest_walk::optional_dependency_names(&member.manifest)
            .into_iter()
            .collect();
        let mut targets: BTreeMap<Kind, BTreeMap<String, Target>> = BTreeMap::new();
        for kind in Kind::all() {
            targets.insert(kind, named_targets(tree, &member, kind));
        }
        let default_run = member
            .manifest
            .get("package")
            .and_then(|package| package.get("default-run"))
            .and_then(toml::Value::as_str)
            .map(str::to_string);
        packages.insert(
            member.name.clone(),
            Package {
                name: member.name,
                features,
                optional,
                targets,
                default_run,
            },
        );
    }
    Ok(packages)
}

/// Every target of one kind the member carries, declared or discovered.
fn named_targets(
    tree: &Tree,
    member: &crate::gates::scan::Member,
    kind: Kind,
) -> BTreeMap<String, Target> {
    let mut targets: BTreeMap<String, Target> = BTreeMap::new();
    let discovers = member
        .manifest
        .get("package")
        .and_then(|package| package.get(kind.autodiscovery()))
        .and_then(toml::Value::as_bool)
        .unwrap_or(true);
    if discovers {
        let directory = format!("{}/{}/", member.path, kind.directory());
        for path in tree.paths() {
            let Some(text) = path.to_str() else { continue };
            let Some(rest) = text.strip_prefix(&directory) else {
                continue;
            };
            // `tests/name.rs` is one target and so is `tests/name/main.rs`; a
            // deeper file is a module of the second and names no target.
            let name = match rest.strip_suffix("/main.rs") {
                Some(name) => name,
                None => match rest.strip_suffix(".rs") {
                    Some(name) => name,
                    None => continue,
                },
            };
            if !name.contains('/') {
                targets.entry(name.to_string()).or_default();
            }
        }
        if kind == Kind::Bin && tree.exists(&format!("{}/src/main.rs", member.path)) {
            targets.entry(member.name.clone()).or_default();
        }
    }
    let declared = member
        .manifest
        .get(kind.table())
        .and_then(toml::Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    for entry in declared {
        let Some(name) = entry.get("name").and_then(toml::Value::as_str) else {
            continue;
        };
        let required = entry
            .get("required-features")
            .and_then(toml::Value::as_array)
            .map(|array| {
                array
                    .iter()
                    .filter_map(|value| value.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        targets.insert(
            name.to_string(),
            Target {
                required_features: required,
            },
        );
    }
    targets
}

/// One cargo invocation read out of a workflow or a script.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Step {
    /// File the command is in, relative to the checkout.
    pub origin: String,
    /// Line the command starts on.
    pub line: u32,
    /// Whether the command is `cargo run`, which must resolve to one binary.
    pub runs: bool,
    /// Packages the command selects.
    pub packages: Vec<String>,
    /// Whether it selects every workspace member.
    pub whole_workspace: bool,
    /// Named targets, by kind, in the order the command names them.
    pub targets: Vec<(Kind, String)>,
    /// Features the command asks for.
    pub features: Vec<String>,
    /// Whether it passes `--all-features`.
    pub all_features: bool,
    /// Whether it passes `--no-default-features`.
    pub no_default_features: bool,
}

impl Step {
    /// Whether the command selects anything this gate can resolve.
    fn selects_anything(&self) -> bool {
        self.runs
            || !self.packages.is_empty()
            || !self.targets.is_empty()
            || !self.features.is_empty()
            || self.whole_workspace
    }

    /// The binary the command names, if it names one.
    fn binary(&self) -> Option<&str> {
        self.targets
            .iter()
            .find(|(kind, _)| *kind == Kind::Bin)
            .map(|(_, name)| name.as_str())
    }
}

/// Read every cargo invocation one workflow or script file makes.
///
/// A command reaches the runner as one string however the file wraps it. A
/// shell continuation is one command: `ci.yml` names its package on one line
/// and forty test targets on the lines below it, so a reader that stops at the
/// newline sees forty targets belonging to no package and cannot check either.
/// A workflow writes the same command as a YAML block scalar instead, where the
/// continuation is the indentation and no backslash appears at all; a reader
/// that joins backslashes only takes the first line of such a step and throws
/// every selector after it away, which is the whole population in
/// `conform.yml`. Both forms are joined here.
///
/// Everything after a bare `--` belongs to the test binary rather than to
/// cargo, so `-- --nocapture` is not read as a selector.
#[must_use]
pub fn steps(origin: &str, text: &str) -> Vec<Step> {
    let mut steps = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut index = 0;
    while index < lines.len() {
        if let Some((next, commands)) = block_scalar(&lines, index) {
            for (line, command) in commands {
                if let Some(step) = read_command(origin, line, &command) {
                    steps.push(step);
                }
            }
            index = next;
            continue;
        }
        let start = index;
        let mut command = String::new();
        loop {
            let line = code(lines[index]);
            let continues = line.ends_with('\\');
            command.push_str(line.trim_end_matches('\\'));
            command.push(' ');
            index += 1;
            if !continues || index >= lines.len() {
                break;
            }
        }
        if let Some(step) = read_command(origin, start + 1, &command) {
            steps.push(step);
        }
    }
    steps
}

/// One line with its shell comment removed.
fn code(line: &str) -> &str {
    let line = line.trim();
    if line.starts_with('#') {
        return "";
    }
    match line.find(" #") {
        Some(at) => &line[..at],
        None => line,
    }
}

/// The commands a `run:` block scalar starting at `index` issues, and the line
/// after the block.
///
/// A folded block (`>`) is one command however many lines it spans, because
/// YAML joins them with a space before the runner ever sees it. A literal block
/// (`|`) is a shell script, so each line is its own command and a backslash
/// still continues one.
fn block_scalar(lines: &[&str], index: usize) -> Option<(usize, Vec<(usize, String)>)> {
    let opener = lines[index];
    let column = opener.len() - opener.trim_start().len();
    let declaration = opener.trim_start().trim_start_matches("- ").trim();
    let indicator = declaration.strip_prefix("run:")?.trim();
    if !indicator.starts_with('>') && !indicator.starts_with('|') {
        return None;
    }
    let folded = indicator.starts_with('>');
    let mut body = Vec::new();
    let mut end = index + 1;
    while end < lines.len() {
        let line = lines[end];
        let blank = line.trim().is_empty();
        if !blank && line.len() - line.trim_start().len() <= column {
            break;
        }
        body.push((end + 1, if blank { "" } else { code(line) }));
        end += 1;
    }
    let mut commands = Vec::new();
    if folded {
        let mut joined = String::new();
        let mut first = index + 1;
        for (line, text) in body {
            if text.is_empty() {
                if !joined.trim().is_empty() {
                    commands.push((first, std::mem::take(&mut joined)));
                }
                first = line + 1;
                continue;
            }
            if joined.is_empty() {
                first = line;
            }
            joined.push_str(text);
            joined.push(' ');
        }
        if !joined.trim().is_empty() {
            commands.push((first, joined));
        }
    } else {
        let mut carried: Option<(usize, String)> = None;
        for (line, text) in body {
            let continues = text.ends_with('\\');
            let (first, mut command) = carried.take().unwrap_or((line, String::new()));
            command.push_str(text.trim_end_matches('\\'));
            command.push(' ');
            if continues {
                carried = Some((first, command));
            } else {
                commands.push((first, command));
            }
        }
        if let Some(rest) = carried {
            commands.push(rest);
        }
    }
    Some((end, commands))
}

/// Read one command into the selectors it names.
///
/// Both the `cargo` and the `./cargo_full` spellings are read, because the
/// wrapper forwards its arguments unchanged and a step written either way fails
/// the same way. A word between the program and its subcommand means the line
/// is prose about cargo rather than a call to it, and so does a `#` before the
/// program: a commented-out step runs nothing, and reading one as a call
/// reported a package the checkout had already dropped.
fn read_command(origin: &str, line: usize, command: &str) -> Option<Step> {
    let tokens: Vec<&str> = command.split_whitespace().collect();
    let start = tokens.iter().position(|token| {
        let bare = token.trim_start_matches("./").trim_matches(['"', '\'']);
        bare == "cargo" || bare == "cargo_full"
    })?;
    if tokens[..start].iter().any(|token| token.starts_with('#')) {
        return None;
    }
    let tokens = &tokens[start + 1..];
    let verb = tokens
        .iter()
        .position(|token| !token.starts_with('-'))
        .filter(|at| {
            tokens[..*at]
                .iter()
                .all(|token| token.starts_with('-') || token.contains('='))
        })?;
    let mut step = Step {
        origin: origin.to_string(),
        line: u32::try_from(line).unwrap_or(u32::MAX),
        runs: tokens[verb] == "run",
        ..Step::default()
    };
    let mut index = verb + 1;
    while index < tokens.len() {
        let token = tokens[index];
        index += 1;
        if token == "--" {
            break;
        }
        let (flag, inline) = match token.split_once('=') {
            Some((flag, value)) => (flag, Some(value.to_string())),
            None => (token, None),
        };
        let mut value = || -> Option<String> {
            if let Some(value) = inline.clone() {
                return Some(value);
            }
            let next = tokens.get(index)?;
            if next.starts_with('-') {
                return None;
            }
            index += 1;
            Some((*next).to_string())
        };
        match flag {
            "-p" | "--package" => {
                if let Some(name) = value() {
                    step.packages.push(strip_quotes(&name));
                }
            }
            "--workspace" | "--all" => step.whole_workspace = true,
            "--all-features" => step.all_features = true,
            "--no-default-features" => step.no_default_features = true,
            "--features" | "-F" => {
                if let Some(list) = value() {
                    for feature in strip_quotes(&list).split([',', ' ']) {
                        if !feature.is_empty() {
                            step.features.push(feature.to_string());
                        }
                    }
                }
            }
            "--test" | "--bench" | "--example" | "--bin" => {
                let kind = Kind::all()
                    .into_iter()
                    .find(|kind| kind.flag() == flag)
                    .unwrap_or(Kind::Test);
                if let Some(name) = value() {
                    step.targets.push((kind, strip_quotes(&name)));
                }
            }
            _ => {}
        }
    }
    step.selects_anything().then_some(step)
}

/// A token without the quoting a YAML command puts around it.
fn strip_quotes(value: &str) -> String {
    value.trim_matches(|character| character == '"' || character == '\'').to_string()
}

/// A token a generator or a matrix fills in, which the text does not carry.
fn templated(token: &str) -> bool {
    token.contains('$') || token.contains('{')
}

/// Every selector in `step` the tree cannot satisfy.
///
/// A step naming several packages satisfies a feature or a target when any one
/// of them carries it, because that is what cargo does: `--features x` fails
/// with `none of the selected packages contains this feature` and `--test x`
/// with `no test target named x`, so the selector is refused only when no
/// selected package declares it. Requiring every named package to carry it
/// would report a step cargo runs as broken.
#[must_use]
pub fn findings(step: &Step, packages: &BTreeMap<String, Package>) -> Vec<Finding> {
    let mut findings = Vec::new();
    let mut named = Vec::new();
    for name in &step.packages {
        if templated(name) {
            continue;
        }
        match packages.get(name) {
            Some(package) => named.push(package),
            None => findings.push(Finding::at(
                step.origin.clone(),
                step.line,
                format!("the step runs `-p {name}`, which is not a workspace member"),
                "name the member that owns the target now, or delete the step",
            )),
        }
    }
    findings.extend(run_findings(step, packages, &named));

    for feature in &step.features {
        if templated(feature) {
            continue;
        }
        let (owner, name) = match feature.split_once('/') {
            Some((owner, name)) => (Some(owner), name),
            None => (None, feature.as_str()),
        };
        if let Some(owner) = owner {
            // A `dep/feature` request names another crate's feature, which this
            // manifest does not declare and this gate does not resolve.
            if packages.contains_key(owner) {
                continue;
            }
            if named.iter().any(|package| package.optional.contains(owner)) {
                continue;
            }
        }
        if named.is_empty() {
            continue;
        }
        if named.iter().any(|package| package.has_feature(name)) {
            continue;
        }
        findings.push(Finding::at(
            step.origin.clone(),
            step.line,
            format!(
                "the step runs `--features {feature}`, which {} does not declare",
                named
                    .iter()
                    .map(|package| package.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            "name a feature the manifest declares, or declare it",
        ));
    }

    for (kind, name) in &step.targets {
        if templated(name) {
            continue;
        }
        let owners: Vec<&Package> = if named.is_empty() {
            packages.values().collect()
        } else {
            named.clone()
        };
        let found: Vec<&Package> = owners
            .iter()
            .copied()
            .filter(|package| {
                package
                    .targets
                    .get(kind)
                    .is_some_and(|targets| targets.contains_key(name))
            })
            .collect();
        if found.is_empty() {
            findings.push(Finding::at(
                step.origin.clone(),
                step.line,
                format!(
                    "the step runs `{} {name}`, which {} does not carry",
                    kind.flag(),
                    if named.is_empty() {
                        "no workspace member".to_string()
                    } else {
                        named
                            .iter()
                            .map(|package| package.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    }
                ),
                "name the target the tree carries, or delete the selector",
            ));
            continue;
        }
        for package in found {
            let Some(target) = package
                .targets
                .get(kind)
                .and_then(|targets| targets.get(name))
            else {
                continue;
            };
            if target.required_features.is_empty() || step.all_features {
                continue;
            }
            let mut requested = step.features.clone();
            if !step.no_default_features {
                requested.push("default".to_string());
            }
            let enabled = package.closure(&requested);
            let missing: Vec<&str> = target
                .required_features
                .iter()
                .filter(|feature| !enabled.contains(feature.as_str()))
                .map(String::as_str)
                .collect();
            if missing.is_empty() {
                continue;
            }
            findings.push(Finding::at(
                step.origin.clone(),
                step.line,
                format!(
                    "the step runs `{} {name}`, whose required-features {} are not enabled, so cargo skips it and the step passes without running it",
                    kind.flag(),
                    missing.join(", ")
                ),
                "add the features to the step, or drop the target from it",
            ));
        }
    }

    findings
}

/// Why a `cargo run` does not resolve to exactly one binary.
///
/// A package that ships more than one binary and declares no `default-run`
/// makes `cargo run -p <package>` exit 101 with "could not determine which
/// binary to run". Nothing in the workspace changes and no test fails, because
/// the defect is in the shape of the manifest rather than in a line of code.
/// Adding a second file under `xtask/src/bin` failed nineteen hosted jobs at
/// their first step that way.
fn run_findings(
    step: &Step,
    packages: &BTreeMap<String, Package>,
    named: &[&Package],
) -> Vec<Finding> {
    if !step.runs {
        return Vec::new();
    }
    let binary = step.binary().filter(|name| !templated(name));
    let templated_package = step.packages.iter().any(|name| templated(name));
    let reason = match (named.first(), binary) {
        _ if templated_package => return Vec::new(),
        (Some(package), Some(binary)) => {
            let bins = package.targets.get(&Kind::Bin);
            if bins.is_some_and(|bins| bins.contains_key(binary)) {
                return Vec::new();
            }
            format!(
                "`{}` ships no binary `{binary}`",
                package.name
            )
        }
        (Some(package), None) => {
            let bins = package.targets.get(&Kind::Bin).map_or(0, BTreeMap::len);
            match bins {
                0 => format!("`{}` ships no binary at all", package.name),
                1 => return Vec::new(),
                _ if package.default_run.is_some() => return Vec::new(),
                count => format!(
                    "`{}` ships {count} binaries and declares no `default-run`, so cargo cannot decide which to build",
                    package.name
                ),
            }
        }
        (None, Some(binary)) => {
            let owners: Vec<&str> = packages
                .values()
                .filter(|package| {
                    package
                        .targets
                        .get(&Kind::Bin)
                        .is_some_and(|bins| bins.contains_key(binary))
                })
                .map(|package| package.name.as_str())
                .collect();
            match owners.len() {
                0 => format!("no workspace member ships a binary `{binary}`"),
                1 => return Vec::new(),
                _ => format!(
                    "binary `{binary}` is shipped by {}, so the command needs `-p` to say which",
                    owners.join(", ")
                ),
            }
        }
        (None, None) => {
            if step.packages.is_empty() {
                "the command names neither a package nor a binary, so what it runs depends on the working directory".to_string()
            } else {
                return Vec::new();
            }
        }
    };
    vec![Finding::at(
        step.origin.clone(),
        step.line,
        format!("this run step does not resolve to one binary: {reason}"),
        "name the binary in the command, or declare `default-run` in the package that ships more than one",
    )]
}

/// Read every command the files under `directory` issue, at any depth.
///
/// `scripts/` keeps its shared shell functions and its TOML reader one level
/// down, in `scripts/lib`. A single-level read skipped them, so a cargo
/// invocation written in a helper every script sources sat outside the gate.
fn read_steps(root: &Path, directory: &str, extensions: &[&str]) -> Vec<Step> {
    let mut files = Vec::new();
    collect_step_files(&root.join(directory), directory, extensions, &mut files);
    files.sort();
    let mut steps = Vec::new();
    for (origin, path) in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        steps.extend(self::steps(&origin, &text));
    }
    steps
}

/// Every file under `directory` the caller reads, paired with the path a
/// finding names it by.
fn collect_step_files(
    directory: &Path,
    origin: &str,
    extensions: &[&str],
    files: &mut Vec<(String, std::path::PathBuf)>,
) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().map(|name| name.to_string_lossy().into_owned()) else {
            continue;
        };
        let named = format!("{origin}/{name}");
        if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            collect_step_files(&path, &named, extensions, files);
            continue;
        }
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extensions.contains(&extension))
        {
            files.push((named, path));
        }
    }
}

/// Every cargo selector a workflow or script names resolves against the tree.
pub struct CiSteps;

/// Where a command that CI runs can be written, and what it is written in.
const SOURCES: &[(&str, &[&str])] = &[
    (WORKFLOWS, &["yml", "yaml"]),
    (PAUSED, &["yml", "yaml"]),
    (SCRIPTS, &["sh", "py"]),
];

impl Gate for CiSteps {
    fn name(&self) -> &'static str {
        "ci-steps"
    }

    fn help(&self) -> &'static str {
        "Resolve every package, target, binary and feature a workflow or script names against the workspace manifests"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let packages = packages(&tree)?;
        let mut report = Report::clean();
        let mut selectors = 0;
        let mut steps = 0;
        let mut files = BTreeSet::new();
        for (directory, extensions) in SOURCES {
            for step in read_steps(&ctx.root, directory, extensions) {
                selectors += step.packages.len() + step.targets.len() + step.features.len();
                steps += 1;
                files.insert(step.origin.clone());
                for finding in findings(&step, &packages) {
                    if *directory == PAUSED {
                        report.find(Finding::at(
                            step.origin.clone(),
                            step.line,
                            format!("{} while the workflow is paused", finding.message),
                            "a paused workflow that cannot run is a deletion nobody performed; delete it, or repair the step",
                        ));
                    } else {
                        report.find(finding);
                    }
                }
            }
        }
        report.note(format!(
            "{selectors} selector(s) across {steps} command(s) in {} file(s), against {} workspace member(s)",
            files.len(),
            packages.len()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn package(name: &str) -> Package {
        Package {
            name: name.to_string(),
            features: BTreeMap::new(),
            optional: BTreeSet::new(),
            targets: BTreeMap::new(),
            default_run: None,
        }
    }

    fn with_test(mut package: Package, name: &str, required: &[&str]) -> Package {
        package.targets.entry(Kind::Test).or_default().insert(
            name.to_string(),
            Target {
                required_features: required.iter().map(|value| (*value).to_string()).collect(),
            },
        );
        package
    }

    fn with_bins(mut package: Package, names: &[&str]) -> Package {
        for name in names {
            package
                .targets
                .entry(Kind::Bin)
                .or_default()
                .insert((*name).to_string(), Target::default());
        }
        package
    }

    fn set(packages: Vec<Package>) -> BTreeMap<String, Package> {
        packages
            .into_iter()
            .map(|package| (package.name.clone(), package))
            .collect()
    }

    fn messages(findings: &[Finding]) -> String {
        findings
            .iter()
            .map(|finding| finding.message.clone())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// WHY: the whole rule rests on reading a command the way cargo reads it. A
    /// shell continuation is one command, so a package named on the first line
    /// owns the targets named on the lines below it; everything after a bare
    /// `--` is the test binary's and is not a selector.
    #[test]
    fn a_continued_command_is_one_command_and_stops_at_the_double_dash() {
        let read = steps(
            "ci.yml",
            "        run: |\n          cargo test \\\n            -p vyre-libs \\\n            --features graph \\\n            --test one \\\n            --test two -- --nocapture --test three\n",
        );
        assert_eq!(read.len(), 1, "{read:?}");
        let step = &read[0];
        assert_eq!(step.packages, vec!["vyre-libs".to_string()]);
        assert_eq!(step.features, vec!["graph".to_string()]);
        assert_eq!(
            step.targets,
            vec![
                (Kind::Test, "one".to_string()),
                (Kind::Test, "two".to_string())
            ]
        );
    }

    /// WHY: the workflows write their cargo commands as YAML folded scalars,
    /// where the continuation is the indentation and no backslash appears. A
    /// reader that joins backslashes only took the first line of such a step
    /// and threw every selector after it away, so the gate reported the
    /// workflows clean while the population it exists to check was invisible.
    /// A blank line ends one folded command and starts the next.
    #[test]
    fn a_folded_block_is_one_command_and_carries_its_selectors() {
        let read = steps(
            "conform.yml",
            "      - name: Verify\n        run: >-\n          ./cargo_full test -p vyre-primitives\n          --features hardware,cpu-parity\n          --test registry_oob_clean\n\n      - name: Next\n        run: ./cargo_full test -p vyre-libs\n",
        );
        assert_eq!(read.len(), 2, "{read:?}");
        let folded = &read[0];
        assert_eq!(folded.packages, vec!["vyre-primitives".to_string()]);
        assert_eq!(
            folded.features,
            vec!["hardware".to_string(), "cpu-parity".to_string()]
        );
        assert_eq!(
            folded.targets,
            vec![(Kind::Test, "registry_oob_clean".to_string())]
        );
        assert_eq!(folded.line, 3);
        assert_eq!(read[1].packages, vec!["vyre-libs".to_string()]);

        let packages = set(vec![with_test(package("vyre-primitives"), "other", &[])]);
        let found = findings(folded, &packages);
        assert!(
            messages(&found).contains(
                "`--test registry_oob_clean`, which vyre-primitives does not carry"
            ),
            "{}",
            messages(&found)
        );
    }

    /// WHY: this is the loud half, and it is the one that shipped. Twelve
    /// `--test` targets ran against the package they had moved out of, and a
    /// deleted feature was still named in the same file.
    #[test]
    fn a_selector_the_tree_cannot_satisfy_fails() {
        let packages = set(vec![with_test(package("vyre-libs"), "sweep", &[])]);
        let step = read_command("conform.yml", 35, "cargo test -p vyre-primitives --features all-lego --test sweep")
            .expect("a cargo command");
        let found = findings(&step, &packages);
        assert!(
            messages(&found).contains("`-p vyre-primitives`, which is not a workspace member"),
            "{}",
            messages(&found)
        );

        let step = read_command("conform.yml", 35, "cargo test -p vyre-libs --features all-lego --test sweep")
            .expect("a cargo command");
        let found = findings(&step, &packages);
        assert!(
            messages(&found).contains("`--features all-lego`, which vyre-libs does not declare"),
            "{}",
            messages(&found)
        );

        let step = read_command("conform.yml", 35, "cargo test -p vyre-libs --test departed")
            .expect("a cargo command");
        let found = findings(&step, &packages);
        assert!(
            messages(&found).contains("`--test departed`, which vyre-libs does not carry"),
            "{}",
            messages(&found)
        );
    }

    /// WHY: the quiet half. cargo does not fail on a target whose
    /// required-features are absent, it skips it, so the step exits zero while
    /// running none of the assertions it names. Enabling the feature through
    /// the default set, through the flag, or through `--all-features` must all
    /// count, or the rule fails on steps that are correct.
    #[test]
    fn a_target_whose_required_features_are_absent_is_reported_as_skipped() {
        let mut libs = with_test(package("vyre-libs"), "sweep", &["graph"]);
        libs.features.insert("graph".to_string(), Vec::new());
        libs.features
            .insert("default".to_string(), vec!["graph".to_string()]);
        let mut without_default = with_test(package("vyre-libs"), "sweep", &["graph"]);
        without_default.features.insert("graph".to_string(), Vec::new());

        let skipped = findings(
            &read_command("ci.yml", 60, "cargo test -p vyre-libs --test sweep").expect("a command"),
            &set(vec![without_default]),
        );
        assert!(
            messages(&skipped).contains("required-features graph are not enabled"),
            "{}",
            messages(&skipped)
        );

        let by_default = findings(
            &read_command("ci.yml", 60, "cargo test -p vyre-libs --test sweep").expect("a command"),
            &set(vec![libs]),
        );
        assert!(by_default.is_empty(), "{}", messages(&by_default));

        let mut named = with_test(package("vyre-libs"), "sweep", &["graph"]);
        named.features.insert("graph".to_string(), Vec::new());
        let by_flag = findings(
            &read_command("ci.yml", 60, "cargo test -p vyre-libs --features graph --test sweep")
                .expect("a command"),
            &set(vec![named]),
        );
        assert!(by_flag.is_empty(), "{}", messages(&by_flag));

        let mut every = with_test(package("vyre-libs"), "sweep", &["graph"]);
        every.features.insert("graph".to_string(), Vec::new());
        let by_all = findings(
            &read_command("ci.yml", 60, "cargo test -p vyre-libs --all-features --test sweep")
                .expect("a command"),
            &set(vec![every]),
        );
        assert!(by_all.is_empty(), "{}", messages(&by_all));
    }

    /// WHY: a feature turns on other features, so a required feature can be
    /// reached through a chain the step never names. Resolving only the named
    /// tokens would report a skip on a step that runs.
    #[test]
    fn a_feature_reached_through_the_graph_counts_as_enabled() {
        let mut libs = with_test(package("vyre-libs"), "sweep", &["graph"]);
        libs.features
            .insert("all-lego".to_string(), vec!["reasoning".to_string()]);
        libs.features
            .insert("reasoning".to_string(), vec!["graph".to_string()]);
        libs.features.insert("graph".to_string(), Vec::new());
        let found = findings(
            &read_command(
                "ci.yml",
                60,
                "cargo test -p vyre-libs --no-default-features --features all-lego --test sweep",
            )
            .expect("a command"),
            &set(vec![libs]),
        );
        assert!(found.is_empty(), "{}", messages(&found));
    }

    /// WHY: a matrix step names its package through an expression the tree
    /// cannot resolve, and reporting one would make the gate fail on every
    /// matrix in the file.
    #[test]
    fn a_matrix_expression_is_not_a_selector_the_tree_can_refuse() {
        let step = read_command("bench.yml", 20, "cargo test -p ${{ matrix.package }} --test ${{ matrix.suite }}")
            .expect("a command");
        assert!(findings(&step, &BTreeMap::new()).is_empty());
    }

    /// WHY: a package that ships two binaries and declares no `default-run`
    /// makes `cargo run -p <package>` exit 101 before running anything, and a
    /// second file under `src/bin` is how that lands. Naming the binary, and
    /// declaring `default-run`, both resolve it.
    #[test]
    fn a_cargo_run_that_resolves_to_no_single_binary_fails() {
        let two = with_bins(package("xtask"), &["xtask", "publishable_packages"]);
        let ambiguous = findings(
            &read_command("scripts/release.sh", 4, "./cargo_full run -p xtask -- gates")
                .expect("a command"),
            &set(vec![two]),
        );
        assert!(
            messages(&ambiguous).contains("ships 2 binaries and declares no `default-run`"),
            "{}",
            messages(&ambiguous)
        );

        let mut declared = with_bins(package("xtask"), &["xtask", "publishable_packages"]);
        declared.default_run = Some("xtask".to_string());
        let resolved = findings(
            &read_command("scripts/release.sh", 4, "./cargo_full run -p xtask -- gates")
                .expect("a command"),
            &set(vec![declared]),
        );
        assert!(resolved.is_empty(), "{}", messages(&resolved));

        let named = findings(
            &read_command(
                "scripts/release.sh",
                4,
                "./cargo_full run -p xtask --bin xtask -- gates",
            )
            .expect("a command"),
            &set(vec![with_bins(
                package("xtask"),
                &["xtask", "publishable_packages"],
            )]),
        );
        assert!(named.is_empty(), "{}", messages(&named));

        let absent = findings(
            &read_command("scripts/release.sh", 4, "cargo run --bin departed").expect("a command"),
            &set(vec![with_bins(package("xtask"), &["xtask"])]),
        );
        assert!(
            messages(&absent).contains("no workspace member ships a binary `departed`"),
            "{}",
            messages(&absent)
        );
    }

    /// WHY: prose about cargo is not a call to it. A comment naming a failure
    /// mode, and a sentence with the word cargo in it, must read as no command
    /// at all, or the gate fails on documentation.
    #[test]
    fn prose_about_cargo_is_not_a_command() {
        assert!(read_command("ci.yml", 1, "# cargo run -p departed fails here").is_none());
        assert!(read_command("ci.yml", 1, "the cargo invocation below runs the gates").is_none());
        assert!(read_command("ci.yml", 1, "cargo test -p vyre-libs").is_some());
    }

    /// WHY: `scripts/lib` holds the shell functions every script sources and
    /// the readers the workflows call. A read that stops at the top level
    /// leaves those commands unresolved while reporting the directory covered.
    #[test]
    fn a_command_in_a_nested_script_is_read() {
        let directory = tempfile::tempdir().expect("a temporary directory");
        let root = directory.path();
        std::fs::create_dir_all(root.join("scripts/lib")).expect("the script tree");
        std::fs::write(root.join("scripts/top.sh"), "cargo test -p vyre-libs\n")
            .expect("the top script");
        std::fs::write(
            root.join("scripts/lib/cargo_runner.sh"),
            "cargo test -p vyre-nested\n",
        )
        .expect("the nested script");

        let origins: Vec<String> = read_steps(root, "scripts", &["sh", "py"])
            .into_iter()
            .map(|step| step.origin)
            .collect();

        assert_eq!(
            origins,
            vec![
                "scripts/lib/cargo_runner.sh".to_string(),
                "scripts/top.sh".to_string()
            ]
        );
    }

    /// WHY: the tree is the case that has to hold. Every selector in every
    /// workflow and every script, live and paused, resolves against the
    /// manifests, or the gate reports a defect already in the checkout.
    #[test]
    fn every_selector_in_the_checkout_resolves() {
        let root = crate::checkout::checkout_root();
        let tree = Tree::open(&root).expect("the tree lists");
        let resolved = packages(&tree).expect("the manifests parse");
        let mut found = Vec::new();
        for (directory, extensions) in SOURCES {
            for step in read_steps(&root, directory, extensions) {
                found.extend(findings(&step, &resolved));
            }
        }
        assert!(found.is_empty(), "{}", messages(&found));
    }
}
