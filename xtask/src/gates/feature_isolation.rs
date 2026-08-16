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
//! The judged axis is derived from the tracked manifests at run time and has
//! four kinds of point. Every (workspace member, feature) pair, one feature on
//! its own. One `--no-default-features` probe per member. One plain default
//! build per member, which is the selection a consumer writing
//! `cargo check -p <member>` gets and the one every other probe skips past. And
//! every selection a workspace edge actually asks of a sibling, because a break
//! in a combination this workspace itself requests stops a build nobody had to
//! opt into. Nothing here names a member or a feature: a new member, feature or
//! edge selection joins the axis on the commit that declares it and turns this
//! gate red until `xtask/feature-isolation.toml` records a decision for it. A
//! hardcoded roster would go stale in silence, which is the same failure as
//! having no gate.
//!
//! Two modes, because the two costs are three orders of magnitude apart:
//!
//!   - Default: the declaration-agreement check. It reads manifests and the data
//!     file and fails on a missing row, a stale row, a duplicate row, or a
//!     `blocked` row without a real technical reason. No cargo, so the sweep
//!     runs on every change.
//!   - `--sweep`: compiles every selection and fails when an outcome disagrees
//!     with the recorded one. This is the expensive half and CI owns it.
//!     `--member NAME` narrows the compiling to one package and
//!     `--only-unrecorded` to the selections that have no row yet, for the
//!     developer who just added a feature or an edge; the agreement half still
//!     judges the whole axis, because a per-member view of a completeness check
//!     is not one. `--write` merges what this run observed over the rows already
//!     recorded, so adding one row does not cost a full sweep.
//!
//! A pair recorded `blocked` must carry a reason that names the technical
//! constraint. `--sweep --write` records a newly failing pair as
//! `UNREVIEWED: <code> at <file>:<line>`, which the agreement check rejects by
//! name, so regenerating the file cannot launder an unfixed break into an
//! accepted one.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::gate::{Gate, GateCtx, GateError, Report};
use crate::toml_text::quote;

/// Stand-in feature name for the per-member `--no-default-features` probe.
///
/// Cargo restricts a feature name to alphanumerics plus `-`, `_`, `+` and `.`,
/// so parentheses cannot collide with a declared feature.
pub const BASELINE: &str = "(none)";

/// Selection element standing for the crate's own default features.
///
/// `(default)` alone is the plain `cargo check -p <member>` build. Leading a
/// comma-joined list, it is an edge that keeps defaults and adds features on
/// top, which is what `default-features` left unset means in a manifest.
pub const DEFAULTS: &str = "(default)";

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
    /// The cargo feature flags this selection stands for.
    ///
    /// One column spells every kind the axis judges, so a row is readable
    /// without knowing which kind produced it: `(none)` is
    /// `--no-default-features`, `(default)` is no feature flags at all, a bare
    /// name is that one feature alone, and a comma-joined list is those features
    /// together, with defaults kept when the list opens with `(default)`.
    #[must_use]
    pub fn cargo_flags(&self) -> Vec<String> {
        if self.feature == BASELINE {
            return vec!["--no-default-features".to_string()];
        }
        let mut defaults = false;
        let mut requested = Vec::new();
        for element in self.feature.split(',') {
            if element == DEFAULTS {
                defaults = true;
            } else {
                requested.push(element);
            }
        }
        let mut flags = Vec::new();
        if !defaults {
            flags.push("--no-default-features".to_string());
        }
        if !requested.is_empty() {
            flags.push("--features".to_string());
            flags.push(requested.join(","));
        }
        flags
    }

    /// The cargo selection this pair stands for, as a reader would type it.
    #[must_use]
    pub fn label(&self) -> String {
        let flags = self.cargo_flags();
        if flags.is_empty() {
            format!("{} (default features)", self.member)
        } else {
            format!("{} {}", self.member, flags.join(" "))
        }
    }

    /// Canonical spelling of one selection, so two edges asking for the same
    /// thing in a different order are one judged point rather than two.
    #[must_use]
    fn spelled(defaults: bool, features: &BTreeSet<String>) -> String {
        let joined = features
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join(",");
        match (defaults, joined.is_empty()) {
            (true, true) => DEFAULTS.to_string(),
            (false, true) => BASELINE.to_string(),
            (true, false) => format!("{DEFAULTS},{joined}"),
            (false, false) => joined,
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

/// Every selection the tracked manifests put on the axis right now.
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

    let mut manifests = BTreeMap::new();
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
            })?
            .to_string();
        manifests.insert(name, member_parsed);
    }

    let mut selections: BTreeSet<Pair> = BTreeSet::new();
    for (name, member_parsed) in &manifests {
        selections.insert(Pair {
            member: name.clone(),
            feature: BASELINE.to_string(),
        });
        selections.insert(Pair {
            member: name.clone(),
            feature: DEFAULTS.to_string(),
        });
        for feature in enable_able_features(member_parsed) {
            selections.insert(Pair {
                member: name.clone(),
                feature,
            });
        }
    }
    selections.extend(edge_selections(&manifests));

    let mut pairs = selections.into_iter().collect::<Vec<_>>();
    pairs.sort();
    Ok(pairs)
}

/// Every selection one workspace member asks of another.
///
/// The one-feature-at-a-time axis judges what a consumer could write. It does
/// not judge what this workspace does write, and the two are different
/// questions: `vyre-libs` asking `vyre-primitives` for `graph`,
/// `inventory-registry` and `text` together is a selection no single-feature
/// probe covers, and a break inside it stops every build that resolves that edge
/// without a wider one unifying the missing feature back in. Cargo unifies
/// features across a build, so a whole-workspace `cargo check` hides such a
/// break behind whichever unrelated member happened to enable the rest.
///
/// Dev-dependency edges count. The layer gate exempts them because a test may
/// depend upward, but a dev selection that does not compile stops
/// `cargo test -p` exactly as hard as a normal one.
fn edge_selections(manifests: &BTreeMap<String, toml::Value>) -> BTreeSet<Pair> {
    let mut selections = BTreeSet::new();
    for member_parsed in manifests.values() {
        for table in dependency_tables(member_parsed) {
            for (key, spec) in table {
                let Some(spec) = spec.as_table() else {
                    continue;
                };
                let package = spec
                    .get("package")
                    .and_then(toml::Value::as_str)
                    .unwrap_or(key);
                if !manifests.contains_key(package) {
                    continue;
                }
                let features: BTreeSet<String> = spec
                    .get("features")
                    .and_then(toml::Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string)
                    .collect();
                let defaults = spec
                    .get("default-features")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(true);
                selections.insert(Pair {
                    member: package.to_string(),
                    feature: Pair::spelled(defaults, &features),
                });
            }
        }
    }
    selections
}

/// Every dependency table of one manifest, including per-target and dev tables.
fn dependency_tables(manifest: &toml::Value) -> Vec<&toml::value::Table> {
    let sections = ["dependencies", "build-dependencies", "dev-dependencies"];
    let mut tables = Vec::new();
    for section in sections {
        tables.extend(manifest.get(section).and_then(toml::Value::as_table));
    }
    for platform in manifest
        .get("target")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(toml::value::Table::values)
    {
        for section in sections {
            tables.extend(platform.get(section).and_then(toml::Value::as_table));
        }
    }
    tables
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
/// dependency contributes the rename. Cargo rejects `optional` on a
/// dev-dependency, so reading the dev tables here adds nothing and costs nothing.
fn optional_dependencies(manifest: &toml::Value) -> BTreeSet<String> {
    dependency_tables(manifest)
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
                match pair.feature.as_str() {
                    BASELINE => "member",
                    DEFAULTS => "default build",
                    feature if feature.contains(',') => "edge selection",
                    _ => "feature",
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

/// Every disagreement between what a sweep compiled and what the rows record.
///
/// The expensive half's judgement, separated from the compiling so it can be
/// held to both directions without a cargo run. A recorded `compiles` that
/// fails is the break the axis exists to catch. A recorded `blocked` that now
/// compiles is the other half: a reason that has stopped being true keeps a
/// selection exempt, and the exemption then covers the next break in it.
///
/// A selection the rows do not mention at all is the agreement half's finding,
/// and it is skipped here so one omission is reported once, under the fix that
/// closes it, rather than a second time as a `blocked` row that compiles.
#[must_use]
pub fn sweep_failures(rows: &[Row], observed: &[(Pair, Observation)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (pair, observation) in observed {
        let Some(row) = rows
            .iter()
            .find(|row| row.member == pair.member && row.feature == pair.feature)
        else {
            continue;
        };
        match (row.outcome == COMPILES, observation.compiles) {
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
fn check_once(root: &Path, cargo: &str, pair: &Pair) -> Result<Observation, GateError> {
    let mut command = Command::new(cargo);
    command
        .current_dir(root)
        .args(["check", "--locked", "-p", &pair.member])
        .args(pair.cargo_flags())
        .args(["--all-targets", "--message-format=json"]);
    let output = command.output().map_err(|error| {
        GateError::new(
            format!("cannot run `{cargo} check` for `{}`: {error}", pair.label()),
            "install a cargo the sweep can run, or set CARGO to one",
        )
    })?;
    let compiles = output.status.success();
    Ok(Observation {
        first_error: (!compiles)
            .then(|| first_error(&String::from_utf8_lossy(&output.stdout)))
            .flatten(),
        compiles,
    })
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
fn compile(root: &Path, cargo: &str, pair: &Pair) -> Result<Observation, GateError> {
    let first = check_once(root, cargo, pair)?;
    if first.compiles {
        return Ok(first);
    }
    check_once(root, cargo, pair)
}

/// Render the data file for the whole axis, from observations where this run has
/// them and the previously recorded row otherwise.
///
/// Merging is what makes recording one new selection affordable. A write that
/// only kept what it just observed forced a full sweep to add a single row, and
/// a gate whose data costs hours to touch is a gate whose data goes stale, which
/// is the failure this whole axis exists to prevent. Iterating the derived axis
/// rather than the rows also drops a row for a selection no manifest declares any
/// more, so a write cannot leave a stale row behind.
#[must_use]
pub fn render(axis: &[Pair], observed: &[(Pair, Observation)], previous: &[Row]) -> String {
    let mut text = String::from(
        "# Recorded compile outcome of every feature selection this workspace judges.\n\
         #\n\
         # The axis is derived from the tracked manifests at run time, never from this\n\
         # file. The `feature` column spells the selection: `(none)` is the per-member\n\
         # `--no-default-features` probe, `(default)` is the plain `cargo check -p`\n\
         # build, a bare name is that one feature enabled alone, and a comma-joined\n\
         # list is a selection a workspace edge asks of a sibling, with defaults kept\n\
         # when the list opens with `(default)`. A selection with no row here, a row\n\
         # naming a selection no manifest declares, and a row whose outcome disagrees\n\
         # with the sweep are each a failure.\n\
         #\n\
         # `outcome = \"blocked\"` needs a one-line technical constraint in `reason`.\n\
         # A feature that merely needs another feature is not blocked: give it the\n\
         # missing edge in its own [features] table so `--features x` enables what x\n\
         # needs, which fixes the crate for a downstream consumer and not only for\n\
         # this sweep.\n\
         #\n\
         # Regenerate: `cargo run -p xtask --bin xtask -- feature-isolation --sweep --write`.\n\
         # Record only what has no row yet: add `--only-unrecorded`.\n\
         # Check agreement: `cargo run -p xtask --bin xtask -- feature-isolation`.\n",
    );
    for pair in axis {
        let recorded = previous
            .iter()
            .find(|row| row.member == pair.member && row.feature == pair.feature);
        text.push_str("\n[[pair]]\n");
        text.push_str(&format!("member = \"{}\"\n", pair.member));
        text.push_str(&format!("feature = \"{}\"\n", pair.feature));
        let Some(observation) = observed
            .iter()
            .find(|(observed_pair, _)| observed_pair == pair)
            .map(|(_, observation)| observation)
        else {
            match recorded {
                Some(row) if row.outcome == COMPILES => {
                    text.push_str(&format!("outcome = \"{COMPILES}\"\n"));
                }
                Some(row) => {
                    text.push_str(&format!("outcome = \"{}\"\n", row.outcome));
                    if let Some(reason) = row.reason.as_deref() {
                        text.push_str(&format!("reason = {}\n", quote(reason)));
                    }
                }
                None => {
                    text.push_str(&format!("outcome = \"{BLOCKED}\"\n"));
                    text.push_str(&format!(
                        "reason = {}\n",
                        quote(&format!("{UNREVIEWED}: never observed"))
                    ));
                }
            }
            continue;
        };
        if observation.compiles {
            text.push_str(&format!("outcome = \"{COMPILES}\"\n"));
            continue;
        }
        text.push_str(&format!("outcome = \"{BLOCKED}\"\n"));
        let kept = recorded
            .and_then(|row| row.reason.clone())
            .filter(|reason| is_real_reason(reason));
        let reason = kept.unwrap_or_else(|| {
            format!(
                "{UNREVIEWED}: {}",
                observation
                    .first_error
                    .as_deref()
                    .unwrap_or("the feature check reported no error line")
            )
        });
        text.push_str(&format!("reason = {}\n", quote(&reason)));
    }
    text
}

fn cargo_binary() -> String {
    std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string())
}

/// Turn one half's disagreements into findings under the fix that closes them.
fn record(report: &mut Report, failures: Vec<String>, fix: &str) {
    report.findings.extend(
        Report::from_messages(failures, fix).findings,
    );
}

/// Compile each pair in turn, recording the outcome as it goes.
///
/// Both the sweep and `--write` need exactly this. The per-pair line is a note
/// rather than a print, because a gate returns everything it has to say.
fn observe(
    root: &Path,
    pairs: &[Pair],
    report: &mut Report,
) -> Result<Vec<(Pair, Observation)>, GateError> {
    let cargo = cargo_binary();
    let mut observed = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let observation = compile(root, &cargo, pair)?;
        report.note(format!(
            "[{}/{}] {}: {}",
            index + 1,
            pairs.len(),
            pair.label(),
            match (&observation.first_error, observation.compiles) {
                (_, true) => COMPILES.to_string(),
                (Some(error), false) => error.clone(),
                (None, false) => BLOCKED.to_string(),
            }
        ));
        observed.push((pair.clone(), observation));
    }
    Ok(observed)
}

/// What an unrecorded or stale row costs, and how to close it.
const DECLARATION_FIX: &str = "record a row for every derived selection in xtask/feature-isolation.toml and delete every row no manifest declares; `xtask feature-isolation --sweep --write --only-unrecorded` observes just the selections that have none";

/// What a row that disagrees with the compiler costs, and how to close it.
const COMPILE_FIX: &str = "give the feature the missing edge in its own [features] table so enabling it enables what it needs, or move the source behind the cfg that matches; record a row as blocked only for a constraint inherent to the crate";

/// Holds every feature selection the manifests declare to its recorded compile outcome.
pub struct FeatureIsolation;

impl Gate for FeatureIsolation {
    fn name(&self) -> &'static str {
        "feature-isolation"
    }

    fn help(&self) -> &'static str {
        "Hold every feature selection the manifests declare to its recorded compile outcome; --sweep also compiles each pair, --write records what it observed, --member NAME narrows the sweep, --list prints the axis"
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let mut report = Report::clean();
        let list = ctx.has("--list");
        let sweep = ctx.has("--sweep");
        let only_unrecorded = ctx.has("--only-unrecorded");
        let mut member = None;
        let mut rest = ctx.args.iter();
        while let Some(argument) = rest.next() {
            match argument.as_str() {
                "--list" | "--sweep" | "--write" | "--only-unrecorded" => {}
                "--member" => {
                    member = rest.next().map(String::as_str);
                    if member.is_none() {
                        return Err(GateError::new(
                            "`--member` was passed without a package name",
                            "name the package to restrict the sweep to",
                        ));
                    }
                }
                other => {
                    return Err(GateError::new(
                        format!("`{other}` is not an argument this gate takes"),
                        "pass [--list] [--sweep [--write] [--only-unrecorded]] [--member NAME]",
                    ));
                }
            }
        }
        if ctx.write && !sweep {
            return Err(GateError::new(
                "`--write` records observed outcomes and nothing was observed",
                "pass `--sweep` so there is an observation to record",
            ));
        }
        if only_unrecorded && !sweep {
            return Err(GateError::new(
                "`--only-unrecorded` narrows what the sweep compiles and no sweep was asked for",
                "pass `--sweep`",
            ));
        }

        let pairs = derive_pairs(root).map_err(|error| {
            GateError::new(error, "repair the manifests the axis is derived from")
        })?;

        // The agreement half always judges the whole axis: it costs milliseconds,
        // and a per-member view of a completeness check is not a completeness check.
        let mut selected = match member {
            None => pairs.clone(),
            Some(name) => {
                let selected = pairs
                    .iter()
                    .filter(|pair| pair.member == name)
                    .cloned()
                    .collect::<Vec<_>>();
                if selected.is_empty() {
                    return Err(GateError::new(
                        format!("no workspace member is named `{name}`"),
                        "run with `--list` to print the axis",
                    ));
                }
                selected
            }
        };
        if only_unrecorded {
            let recorded = load_rows(root).unwrap_or_default();
            selected.retain(|pair| {
                !recorded
                    .iter()
                    .any(|row| row.member == pair.member && row.feature == pair.feature)
            });
            if selected.is_empty() {
                report.note("every selection on the axis already has a row");
                return Ok(report);
            }
        }

        if list {
            for pair in &selected {
                report.note(pair.label());
            }
            report.note(format!(
                "{} pair(s) derived from the tracked manifests",
                selected.len()
            ));
            return Ok(report);
        }

        if ctx.write {
            let previous = load_rows(root).unwrap_or_default();
            let observed = observe(root, &selected, &mut report)?;
            let path = data_path(root);
            fs::write(&path, render(&pairs, &observed, &previous)).map_err(|error| {
                GateError::new(
                    format!("cannot write {}: {error}", path.display()),
                    "make the feature-isolation record writable",
                )
            })?;
            report.note(format!("wrote {}", path.display()));
            return Ok(report);
        }

        let rows = load_rows(root)
            .map_err(|error| GateError::new(error, "repair xtask/feature-isolation.toml"))?;
        record(
            &mut report,
            agreement_failures(&pairs, &rows),
            DECLARATION_FIX,
        );

        if !sweep {
            report.note(format!(
                "{} declared pair(s) agree with the manifests",
                pairs.len()
            ));
            return Ok(report);
        }

        let observed = observe(root, &selected, &mut report)?;
        let failures = sweep_failures(&rows, &observed);
        record(&mut report, failures, COMPILE_FIX);
        report.note(format!("{} pair(s) compiled", selected.len()));
        Ok(report)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the sweep half is what turns the record from a claim into a
    /// measurement, and it has to fail in both directions. A selection recorded
    /// `compiles` that fails is the break the axis exists to catch, and it is the
    /// exact state this workspace was in: two test suites in vyre-pass-engine and
    /// one in vyre-registry-link imported a feature-gated namespace, so six
    /// selections recorded as compiling did not. A selection recorded `blocked`
    /// that now compiles is the other direction: a reason that stopped being true
    /// keeps the selection exempt, and the exemption then covers the next break in
    /// it.
    #[test]
    fn a_recorded_outcome_that_disagrees_with_the_build_is_reported_both_ways() {
        let pair = |member: &str, feature: &str| Pair {
            member: member.to_string(),
            feature: feature.to_string(),
        };
        let row = |member: &str, feature: &str, outcome: &str| Row {
            member: member.to_string(),
            feature: feature.to_string(),
            outcome: outcome.to_string(),
            reason: (outcome == BLOCKED).then(|| "the vendored driver has no Linux build".to_string()),
        };
        let rows = vec![
            row("vyre-pass-engine", "all-solvers", COMPILES),
            row("vyre-libs", "matching-regex", BLOCKED),
            row("vyre-libs", "visual", COMPILES),
        ];
        let observed = vec![
            (
                pair("vyre-pass-engine", "all-solvers"),
                Observation {
                    compiles: false,
                    first_error: Some(
                        "E0433 at vyre-pass-engine/tests/scope_rewrite_owner_contract.rs:19"
                            .to_string(),
                    ),
                },
            ),
            (
                pair("vyre-libs", "matching-regex"),
                Observation { compiles: true, first_error: None },
            ),
            (
                pair("vyre-libs", "visual"),
                Observation { compiles: true, first_error: None },
            ),
            // No row at all: the agreement half owns that, and counting it here
            // too would report one omission as two unrelated failures.
            (
                pair("vyre-libs", "unrecorded"),
                Observation { compiles: true, first_error: None },
            ),
        ];

        let failures = sweep_failures(&rows, &observed);
        assert_eq!(
            failures,
            vec![
                "`vyre-pass-engine --no-default-features --features all-solvers` is recorded `compiles` and fails with E0433 at vyre-pass-engine/tests/scope_rewrite_owner_contract.rs:19".to_string(),
                "`vyre-libs --no-default-features --features matching-regex` is recorded `blocked` and now compiles; set outcome = \"compiles\" and drop its reason".to_string(),
            ],
            "a recorded outcome the build contradicts is reported, in the direction it was contradicted, and an agreeing pair is not"
        );
    }
}
