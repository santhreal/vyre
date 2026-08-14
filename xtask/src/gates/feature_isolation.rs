//! `cargo xtask feature-isolation`  -  every declared feature compiles alone.
//!
//! A crate that builds under its default features and under `--all-features`
//! can still be uncompilable under one feature on its own. `--all-features` is a
//! union, so a feature whose prerequisites happen to be enabled by some other
//! feature passes there; the default build never turns a granular feature on at
//! all. The consumer who writes `features = ["matching-regex"]` in their own
//! manifest is the only one who sees the break, and they see it as a compile
//! error in a crate they did not write.
//!
//! The judged axis is every (workspace member, feature) pair plus one
//! `--no-default-features` probe per member, derived from the tracked manifests
//! at run time. Nothing here names a member or a feature: a new member or a new
//! feature joins the axis on the commit that declares it and turns this gate red
//! until `xtask/feature-isolation.toml` records a decision for it. A hardcoded
//! roster would go stale in silence, which is the same failure as having no gate.
//!
//! Two modes, because the two costs are three orders of magnitude apart:
//!
//!   - Default: the declaration-agreement check. It reads manifests and the data
//!     file and fails on a missing row, a stale row, a duplicate row, or a
//!     `blocked` row without a real technical reason. No cargo, so the sweep
//!     runs on every change.
//!   - `--sweep`: compiles every pair and fails when an outcome disagrees with
//!     the recorded one. This is the expensive half and CI owns it. `--member
//!     NAME` narrows the compiling to one package for the developer who just
//!     added a feature; the agreement half still judges the whole axis, because
//!     a per-member view of a completeness check is not one.
//!
//! A pair recorded `blocked` must carry a reason that names the technical
//! constraint. `--sweep --write` records a newly failing pair as
//! `UNREVIEWED: <code> at <file>:<line>`, which the agreement check rejects by
//! name, so regenerating the file cannot launder an unfixed break into an
//! accepted one.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use serde::Deserialize;

/// Stand-in feature name for the per-member `--no-default-features` probe.
///
/// Cargo restricts a feature name to alphanumerics plus `-`, `_`, `+` and `.`,
/// so parentheses cannot collide with a declared feature.
pub const BASELINE: &str = "(none)";

/// Prefix `--write` gives a newly failing pair, rejected by the agreement check.
const UNREVIEWED: &str = "UNREVIEWED";

/// Recorded outcome: the pair compiles on its own.
const COMPILES: &str = "compiles";

/// Recorded outcome: the pair cannot compile on its own, for a stated reason.
const BLOCKED: &str = "blocked";

/// Largest manifest this gate will read.
const MAX_MANIFEST_BYTES: u64 = 1_048_576;

/// One judged point on the feature axis.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Pair {
    /// Package name as `cargo -p` takes it.
    pub member: String,
    /// Declared feature, or [`BASELINE`] for the `--no-default-features` probe.
    pub feature: String,
}

impl Pair {
    /// The cargo selection this pair stands for, as a reader would type it.
    #[must_use]
    pub fn label(&self) -> String {
        if self.feature == BASELINE {
            format!("{} --no-default-features", self.member)
        } else {
            format!(
                "{} --no-default-features --features {}",
                self.member, self.feature
            )
        }
    }
}

/// One row of `xtask/feature-isolation.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    /// Package name the row judges.
    pub member: String,
    /// Feature the row judges, or [`BASELINE`].
    pub feature: String,
    /// `compiles` or `blocked`.
    pub outcome: String,
    /// Technical constraint that makes a `blocked` pair impossible to compile
    /// alone. Required on `blocked`, forbidden on `compiles`.
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RowFile {
    #[serde(default)]
    pair: Vec<Row>,
}

fn data_path(root: &Path) -> PathBuf {
    root.join("xtask/feature-isolation.toml")
}

/// Every (member, feature) pair the tracked manifests declare right now.
///
/// # Errors
///
/// Returns the reason the workspace manifests could not be read as the axis.
pub fn derive_pairs(root: &Path) -> Result<Vec<Pair>, String> {
    let manifest = root.join("Cargo.toml");
    let text = read_manifest(&manifest)?;
    let parsed: toml::Value = toml::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", manifest.display()))?;
    let members = parsed
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| format!("{} declares no [workspace] members", manifest.display()))?;
    if members.is_empty() {
        return Err(format!(
            "{} lists no workspace members, so the axis would be empty",
            manifest.display()
        ));
    }

    let mut pairs = Vec::new();
    for member in members {
        let directory = member
            .as_str()
            .ok_or_else(|| format!("{} has a non-string member entry", manifest.display()))?;
        let member_manifest = root.join(directory).join("Cargo.toml");
        let member_text = read_manifest(&member_manifest)?;
        let member_parsed: toml::Value = toml::from_str(&member_text)
            .map_err(|error| format!("cannot parse {}: {error}", member_manifest.display()))?;
        let name = member_parsed
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml::Value::as_str)
            .ok_or_else(|| {
                format!(
                    "{} has no [package].name, so no cargo selection names it",
                    member_manifest.display()
                )
            })?;
        pairs.push(Pair {
            member: name.to_string(),
            feature: BASELINE.to_string(),
        });
        for feature in enable_able_features(&member_parsed) {
            pairs.push(Pair {
                member: name.to_string(),
                feature,
            });
        }
    }
    pairs.sort();
    Ok(pairs)
}

/// Every feature name `--features` accepts for one member, `default` excluded.
///
/// The `[features]` table is not the whole answer. An optional dependency that
/// no feature names with the `dep:` prefix also gets a feature of its own, so a
/// manifest writing `remote-cache = ["ureq"]` publishes a second, undeclared
/// `ureq` feature that a consumer can enable and that nothing here would judge.
/// Reading only the declared table would leave exactly that shape unjudged.
fn enable_able_features(manifest: &toml::Value) -> BTreeSet<String> {
    let table = manifest.get("features").and_then(toml::Value::as_table);
    let mut features: BTreeSet<String> = table
        .map(|table| {
            table
                .keys()
                .filter(|name| *name != "default")
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    let mut named_with_prefix = BTreeSet::new();
    for value in table.into_iter().flat_map(toml::value::Table::values) {
        for entry in value.as_array().into_iter().flatten() {
            let Some(entry) = entry.as_str().and_then(|entry| entry.strip_prefix("dep:")) else {
                continue;
            };
            named_with_prefix.insert(entry.split('/').next().unwrap_or(entry).to_string());
        }
    }

    for dependency in optional_dependencies(manifest) {
        if !named_with_prefix.contains(&dependency) {
            features.insert(dependency);
        }
    }
    features
}

/// Keys of every `optional = true` dependency that can carry an implicit feature.
///
/// The key is what names the feature, not the `package` field, so a renamed
/// dependency contributes the rename. Dev-dependencies cannot be optional, so
/// only the normal, build and target tables are read.
fn optional_dependencies(manifest: &toml::Value) -> BTreeSet<String> {
    let mut tables = Vec::new();
    for section in ["dependencies", "build-dependencies"] {
        tables.extend(manifest.get(section).and_then(toml::Value::as_table));
    }
    for platform in manifest
        .get("target")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(toml::value::Table::values)
    {
        for section in ["dependencies", "build-dependencies"] {
            tables.extend(platform.get(section).and_then(toml::Value::as_table));
        }
    }
    tables
        .into_iter()
        .flatten()
        .filter(|(_, value)| {
            value
                .get("optional")
                .and_then(toml::Value::as_bool)
                .unwrap_or(false)
        })
        .map(|(name, _)| name.clone())
        .collect()
}

fn read_manifest(path: &Path) -> Result<String, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("cannot stat {}: {error}", path.display()))?;
    if metadata.len() > MAX_MANIFEST_BYTES {
        return Err(format!(
            "{} is {} bytes, past the {MAX_MANIFEST_BYTES}-byte manifest bound",
            path.display(),
            metadata.len()
        ));
    }
    fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

/// Rows recorded in `xtask/feature-isolation.toml`.
///
/// # Errors
///
/// Returns the reason the data file could not be read as rows.
pub fn load_rows(root: &Path) -> Result<Vec<Row>, String> {
    let path = data_path(root);
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "cannot read {}: {error}. Regenerate it with `cargo run -p xtask -- feature-isolation --sweep --write`.",
            path.display()
        )
    })?;
    let parsed: RowFile = toml::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    Ok(parsed.pair)
}

/// A reason has to name a constraint a reader can go and check.
///
/// "not fixed yet" in any of its dressings is a schedule, not a constraint, and
/// an exemption justified by a schedule is an exemption with no expiry.
fn is_real_reason(reason: &str) -> bool {
    const EXCUSES: [&str; 12] = [
        "unreviewed",
        "not fixed",
        "not yet",
        "todo",
        "tbd",
        "wip",
        "for now",
        "later",
        "pending",
        "unknown",
        "temporar",
        "revisit",
    ];
    let folded = reason.to_ascii_lowercase();
    reason.trim().len() >= 30
        && !reason.contains('\n')
        && !EXCUSES.iter().any(|excuse| folded.contains(excuse))
}

/// Every disagreement between the derived axis and the recorded rows.
///
/// This is the fast half of the gate. It reads no cargo output, so it is the
/// half that can run on every change, and it is what makes a new feature red by
/// default instead of unjudged.
#[must_use]
pub fn agreement_failures(pairs: &[Pair], rows: &[Row]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut recorded: BTreeMap<(&str, &str), &Row> = BTreeMap::new();

    for row in rows {
        let key = (row.member.as_str(), row.feature.as_str());
        if recorded.insert(key, row).is_some() {
            failures.push(format!(
                "`{}` is recorded more than once; the later row is dead weight, delete it",
                Pair {
                    member: row.member.clone(),
                    feature: row.feature.clone(),
                }
                .label()
            ));
        }
        if row.outcome != COMPILES && row.outcome != BLOCKED {
            failures.push(format!(
                "`{} {}` records outcome `{}`; use `{COMPILES}` or `{BLOCKED}`",
                row.member, row.feature, row.outcome
            ));
            continue;
        }
        let reason = row.reason.as_deref().unwrap_or("").trim();
        if row.outcome == BLOCKED && !is_real_reason(reason) {
            failures.push(format!(
                "`{}` is recorded `{BLOCKED}` with no real reason (`{reason}`); state the technical constraint on one line, not a schedule",
                Pair {
                    member: row.member.clone(),
                    feature: row.feature.clone(),
                }
                .label()
            ));
        }
        if row.outcome == COMPILES && !reason.is_empty() {
            failures.push(format!(
                "`{}` compiles and still carries a reason; a reason belongs only on a `{BLOCKED}` row",
                Pair {
                    member: row.member.clone(),
                    feature: row.feature.clone(),
                }
                .label()
            ));
        }
    }

    for pair in pairs {
        if !recorded.contains_key(&(pair.member.as_str(), pair.feature.as_str())) {
            failures.push(format!(
                "`{}` has no row in xtask/feature-isolation.toml; a new {} is unjudged until one is recorded",
                pair.label(),
                if pair.feature == BASELINE {
                    "member"
                } else {
                    "feature"
                }
            ));
        }
    }
    for row in rows {
        let live = pairs
            .iter()
            .any(|pair| pair.member == row.member && pair.feature == row.feature);
        if !live {
            failures.push(format!(
                "xtask/feature-isolation.toml records `{}`, which no manifest declares any more; delete the stale row",
                Pair {
                    member: row.member.clone(),
                    feature: row.feature.clone(),
                }
                .label()
            ));
        }
    }

    failures
}

/// What compiling one pair produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// Whether cargo accepted the selection.
    pub compiles: bool,
    /// First reported error, as `<code> at <file>:<line>`, when it did not.
    pub first_error: Option<String>,
}

/// The first `error` diagnostic in a `--message-format=json` stream, rendered as
/// `<code> at <file>:<line>`.
///
/// The location is the point of the gate's report: "vyre-libs fails" sends a
/// reader to a crate, and `E0432 at vyre-libs/src/matching/regex.rs:12` sends
/// them to the line.
#[must_use]
pub fn first_error(stdout: &str) -> Option<String> {
    for line in stdout.lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if message.get("reason").and_then(serde_json::Value::as_str) != Some("compiler-message") {
            continue;
        }
        let diagnostic = message.get("message")?;
        if diagnostic.get("level").and_then(serde_json::Value::as_str) != Some("error") {
            continue;
        }
        let code = diagnostic
            .get("code")
            .and_then(|code| code.get("code"))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("error");
        let span = diagnostic
            .get("spans")
            .and_then(serde_json::Value::as_array)
            .and_then(|spans| spans.first());
        let location = match span {
            Some(span) => format!(
                "{}:{}",
                span.get("file_name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("?"),
                span.get("line_start")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0)
            ),
            None => "no span".to_string(),
        };
        return Some(format!("{code} at {location}"));
    }
    None
}

/// Compile one pair once.
fn check_once(root: &Path, cargo: &str, pair: &Pair) -> Observation {
    let mut command = Command::new(cargo);
    command.current_dir(root).args([
        "check",
        "--locked",
        "-p",
        &pair.member,
        "--no-default-features",
    ]);
    if pair.feature != BASELINE {
        command.args(["--features", &pair.feature]);
    }
    command.args(["--all-targets", "--message-format=json"]);
    let output = command.output().unwrap_or_else(|error| {
        eprintln!(
            "Fix: cannot run `{cargo} check` for `{}`: {error}",
            pair.label()
        );
        process::exit(1);
    });
    let compiles = output.status.success();
    Observation {
        first_error: (!compiles)
            .then(|| first_error(&String::from_utf8_lossy(&output.stdout)))
            .flatten(),
        compiles,
    }
}

/// Compile one pair, and confirm a failure by compiling it again.
///
/// Whether a selection compiles is a property of the tree, so a failure that
/// does not repeat came from outside it. Measured twice on this fleet: a
/// concurrently rebuilt shared target directory removes a dependency's `.rlib`
/// mid-check and rustc reports E0463 in a crate whose manifest declares that
/// dependency, which is indistinguishable in one run from a feature that really
/// forgot an edge. A gate that records the first answer publishes those as
/// blocked pairs, and a gate that publishes false reds gets ignored, which is
/// the outcome this whole axis exists to prevent. Only a failure is retried, so
/// a green sweep pays nothing.
fn compile(root: &Path, cargo: &str, pair: &Pair) -> Observation {
    let first = check_once(root, cargo, pair);
    if first.compiles {
        return first;
    }
    check_once(root, cargo, pair)
}

/// Render the data file from observed outcomes, keeping reviewed reasons.
#[must_use]
fn render(observed: &[(Pair, Observation)], previous: &[Row]) -> String {
    let mut text = String::from(
        "# Recorded compile-alone outcome of every (workspace member, feature) pair.\n\
         #\n\
         # The axis is derived from the tracked manifests at run time, never from this\n\
         # file: `feature = \"(none)\"` is the per-member `--no-default-features` probe,\n\
         # and every other row is one declared feature enabled on its own. A pair with\n\
         # no row here, a row naming a pair no manifest declares, and a row whose\n\
         # outcome disagrees with the sweep are each a failure.\n\
         #\n\
         # `outcome = \"blocked\"` needs a one-line technical constraint in `reason`.\n\
         # A feature that merely needs another feature is not blocked: give it the\n\
         # missing edge in its own [features] table so `--features x` enables what x\n\
         # needs, which fixes the crate for a downstream consumer and not only for\n\
         # this sweep.\n\
         #\n\
         # Regenerate: `cargo run -p xtask --bin xtask -- feature-isolation --sweep --write`.\n\
         # Check agreement: `cargo run -p xtask --bin xtask -- feature-isolation`.\n",
    );
    for (pair, observation) in observed {
        text.push_str("\n[[pair]]\n");
        text.push_str(&format!("member = \"{}\"\n", pair.member));
        text.push_str(&format!("feature = \"{}\"\n", pair.feature));
        if observation.compiles {
            text.push_str(&format!("outcome = \"{COMPILES}\"\n"));
            continue;
        }
        text.push_str(&format!("outcome = \"{BLOCKED}\"\n"));
        let kept = previous
            .iter()
            .find(|row| row.member == pair.member && row.feature == pair.feature)
            .and_then(|row| row.reason.clone())
            .filter(|reason| is_real_reason(reason));
        let reason = kept.unwrap_or_else(|| {
            format!(
                "{UNREVIEWED}: {}",
                observation
                    .first_error
                    .as_deref()
                    .unwrap_or("cargo check failed")
            )
        });
        text.push_str(&format!("reason = {}\n", toml_string(&reason)));
    }
    text
}

/// A TOML basic string, so a reason containing a quote or a backslash round-trips.
fn toml_string(value: &str) -> String {
    let mut quoted = String::with_capacity(value.len() + 2);
    quoted.push('"');
    for character in value.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            other => quoted.push(other),
        }
    }
    quoted.push('"');
    quoted
}

fn cargo_binary() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

fn report(failures: &[String], fix: &str) {
    if failures.is_empty() {
        return;
    }
    eprintln!("feature-isolation: {} disagreement(s):", failures.len());
    for failure in failures {
        eprintln!("  - {failure}");
    }
    eprintln!("Fix: {fix}");
    process::exit(1);
}

/// Compile each pair in turn, printing the outcome as it goes.
///
/// Both the sweep and `--write` need exactly this, and a sweep with no progress
/// output is indistinguishable from a hang for its whole multi-hour run.
fn observe(root: &Path, pairs: &[Pair]) -> Vec<(Pair, Observation)> {
    let cargo = cargo_binary();
    let mut observed = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let observation = compile(root, &cargo, pair);
        println!(
            "[{}/{}] {}: {}",
            index + 1,
            pairs.len(),
            pair.label(),
            match (&observation.first_error, observation.compiles) {
                (_, true) => COMPILES.to_string(),
                (Some(error), false) => error.clone(),
                (None, false) => BLOCKED.to_string(),
            }
        );
        observed.push((pair.clone(), observation));
    }
    observed
}

/// Run the gate.
///
/// `args` is the process argument vector, so the dispatcher path and the
/// subcommand name are dropped before the flags are read.
pub fn run(args: &[String]) {
    let root = crate::checkout::checkout_root();
    let flags = args.iter().skip(2).map(String::as_str).collect::<Vec<_>>();
    let list = flags.contains(&"--list");
    let sweep = flags.contains(&"--sweep");
    let write = flags.contains(&"--write");
    let mut member = None;
    let mut rest = flags.iter();
    while let Some(argument) = rest.next() {
        match *argument {
            "--list" | "--sweep" | "--write" => {}
            "--member" => {
                member = rest.next().copied();
                if member.is_none() {
                    eprintln!("Fix: `--member` needs the package name to restrict the sweep to.");
                    process::exit(1);
                }
            }
            other => {
                eprintln!(
                    "Fix: `feature-isolation` takes [--list] [--sweep [--write]] [--member NAME]; `{other}` is not one of them."
                );
                process::exit(1);
            }
        }
    }
    if write && !sweep {
        eprintln!(
            "Fix: `--write` records observed outcomes, so it needs `--sweep` to observe them."
        );
        process::exit(1);
    }
    if write && member.is_some() {
        eprintln!(
            "Fix: `--write` rewrites the whole file, so it cannot run from one member's observations; drop `--member`."
        );
        process::exit(1);
    }

    let pairs = derive_pairs(&root).unwrap_or_else(|error| {
        eprintln!("Fix: {error}");
        process::exit(1);
    });

    // The agreement half always judges the whole axis: it costs milliseconds,
    // and a per-member view of a completeness check is not a completeness check.
    let selected = match member {
        None => pairs.clone(),
        Some(name) => {
            let selected = pairs
                .iter()
                .filter(|pair| pair.member == name)
                .cloned()
                .collect::<Vec<_>>();
            if selected.is_empty() {
                eprintln!("Fix: no workspace member is named `{name}`; `--list` prints the axis.");
                process::exit(1);
            }
            selected
        }
    };

    if list {
        for pair in &selected {
            println!("{}", pair.label());
        }
        println!(
            "feature-isolation: {} pair(s) derived from the tracked manifests",
            selected.len()
        );
        return;
    }

    if write {
        let previous = load_rows(&root).unwrap_or_default();
        let observed = observe(&root, &selected);
        let path = data_path(&root);
        fs::write(&path, render(&observed, &previous)).unwrap_or_else(|error| {
            eprintln!("Fix: cannot write {}: {error}", path.display());
            process::exit(1);
        });
        println!("wrote {}", path.display());
        return;
    }

    let rows = load_rows(&root).unwrap_or_else(|error| {
        eprintln!("Fix: {error}");
        process::exit(1);
    });
    report(
        &agreement_failures(&pairs, &rows),
        "record a row for every derived pair in xtask/feature-isolation.toml and delete every row no manifest declares; `cargo run -p xtask --bin xtask -- feature-isolation --sweep --write` regenerates it from an observed sweep.",
    );

    if !sweep {
        println!(
            "feature-isolation: {} declared pair(s) agree with the manifests",
            pairs.len()
        );
        return;
    }

    let mut failures = Vec::new();
    for (pair, observation) in observe(&root, &selected) {
        let recorded_compiles = rows
            .iter()
            .find(|row| row.member == pair.member && row.feature == pair.feature)
            .is_some_and(|row| row.outcome == COMPILES);
        match (recorded_compiles, observation.compiles) {
            (true, false) => failures.push(format!(
                "`{}` is recorded `{COMPILES}` and fails with {}",
                pair.label(),
                observation
                    .first_error
                    .as_deref()
                    .unwrap_or("no parsed diagnostic")
            )),
            (false, true) => failures.push(format!(
                "`{}` is recorded `{BLOCKED}` and now compiles; set outcome = \"{COMPILES}\" and drop its reason",
                pair.label()
            )),
            _ => {}
        }
    }
    report(
        &failures,
        "give the feature the missing edge in its own [features] table so enabling it enables what it needs, or move the source behind the cfg that matches; record a row as blocked only for a constraint inherent to the crate.",
    );
    println!(
        "feature-isolation: {} pair(s) compile as recorded",
        selected.len()
    );
}
