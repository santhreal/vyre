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
//! Two halves, because the two costs are three orders of magnitude apart:
//!
//!   - The declaration check reads the manifests and the data file and fails on
//!     a missing row, a stale row, a duplicate row, or a `blocked` row without a
//!     real technical reason. No cargo, so it runs on every change.
//!   - The measurement compiles every selection. `--sweep` asks for it,
//!     `--member NAME` narrows it to one package and `--only-unrecorded` to the
//!     selections that have no row yet; the declaration check still judges the
//!     whole axis, because a per-member view of a completeness check is not one.
//!
//! Whether a selection compiles is never stored. A measurement written into a
//! tracked file is stale the moment a feature edge moves, and the file cannot
//! tell a measured outcome from one typed in, so the outcome is produced by the
//! run that compiles it, held in run state, and never deserialized. A
//! measurement therefore fails closed on absence: a sweep that skips part of the
//! axis reports the rest as `unmeasured: N` and exits non-zero, rather than
//! reporting an agreement it did not observe. A copy of the data file that still
//! carries `measured`, or a row that still records `outcome = "compiles"`, is
//! rejected outright, because two records of the same fact is how the stale one
//! survives.
//!
//! A pair that cannot compile must carry a reason naming the technical
//! constraint. `--sweep --write` records a newly failing pair as
//! `UNREVIEWED: <code> at <file>:<line>`, which the declaration check rejects by
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
///
/// A row declares a judged selection, and an exemption when the selection
/// cannot compile. It states no compile outcome: that is a measurement, and a
/// measurement belongs to the run that took it rather than to a tracked file
/// which carries it forward past every edge that could have invalidated it.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Row {
    /// Package name the row judges.
    pub member: String,
    /// Feature the row judges, or [`BASELINE`].
    pub feature: String,
    /// `blocked` when the selection is exempt from compiling, absent otherwise.
    #[serde(default)]
    pub outcome: Option<String>,
    /// Technical constraint that makes a `blocked` pair impossible to compile
    /// alone. Required on `blocked`, forbidden without it.
    #[serde(default)]
    pub reason: Option<String>,
}

impl Row {
    /// Whether the row exempts its selection from having to compile.
    #[must_use]
    pub fn blocked(&self) -> bool {
        self.outcome.as_deref() == Some(BLOCKED)
    }
}

#[derive(Debug, Deserialize)]
struct RowFile {
    #[serde(default)]
    pair: Vec<Row>,
}

fn data_path(root: &Path) -> PathBuf {
    root.join("xtask/feature-isolation.toml")
}

/// Every tracked manifest of this workspace, by package name.
///
/// # Errors
///
/// Returns the reason a manifest could not be read or names no package.
pub fn workspace_manifests(root: &Path) -> Result<BTreeMap<String, toml::Value>, String> {
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
    Ok(manifests)
}

/// Every selection the tracked manifests put on the axis right now.
///
/// # Errors
///
/// Returns the reason the workspace manifests could not be read as the axis.
pub fn derive_pairs(root: &Path) -> Result<Vec<Pair>, String> {
    let manifests = workspace_manifests(root)?;

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

/// Feature that compiles a CPU host oracle into a library.
///
/// An oracle is a second implementation kept to disagree with the one under
/// test. It belongs to a parity check, not to a build a consumer gets by
/// writing `cargo add`.
const HOST_ORACLE_FEATURE: &str = "cpu-parity";

/// Runtime dependency tables of one manifest, dev and build tables excluded.
///
/// The question here is what a consumer links, so a dev dependency is out of
/// scope: it compiles for this workspace's own tests and reaches no released
/// artifact.
fn runtime_dependency_tables(manifest: &toml::Value) -> Vec<&toml::value::Table> {
    let mut tables = Vec::new();
    tables.extend(manifest.get("dependencies").and_then(toml::Value::as_table));
    for platform in manifest
        .get("target")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flat_map(toml::value::Table::values)
    {
        tables.extend(platform.get("dependencies").and_then(toml::Value::as_table));
    }
    tables
}

/// The dependency spec one manifest holds for a dependency key, if any.
fn dependency_spec<'a>(manifest: &'a toml::Value, key: &str) -> Option<&'a toml::value::Table> {
    runtime_dependency_tables(manifest)
        .into_iter()
        .find_map(|table| table.get(key))
        .and_then(toml::Value::as_table)
}

/// Package a dependency key resolves to, following a `package` rename.
fn dependency_package(manifest: &toml::Value, key: &str) -> String {
    dependency_spec(manifest, key)
        .and_then(|spec| spec.get("package"))
        .and_then(toml::Value::as_str)
        .unwrap_or(key)
        .to_string()
}

/// One reached `(member, feature)` state and the state that reached it.
type Reach = BTreeMap<(String, String), Option<(String, String)>>;

/// Every `(member, feature)` a plain `cargo add <root>` build turns on.
///
/// Cargo activates a member's own `default` list, everything that list names
/// transitively, and the features each runtime dependency edge asks for,
/// including that dependency's own defaults unless the edge disables them. The
/// walk keeps the state that reached each state so a failure can print the
/// path instead of the destination.
fn default_feature_reach(manifests: &BTreeMap<String, toml::Value>, root: &str) -> Reach {
    let mut reached: Reach = BTreeMap::new();
    let mut built: BTreeSet<String> = BTreeSet::new();
    let mut queue: Vec<((String, String), Option<(String, String)>)> =
        vec![((root.to_string(), "default".to_string()), None)];
    // A `dep?/feature` entry turns nothing on until something else activates
    // `dep`, so it waits here until the walk reaches that dependency, and is
    // discarded when the walk ends without it.
    let mut deferred: Vec<((String, String), (String, String))> = Vec::new();

    loop {
        while let Some((state, from)) = queue.pop() {
            if reached.contains_key(&state) {
                continue;
            }
            reached.insert(state.clone(), from);
            let (member, feature) = state.clone();
            let Some(parsed) = manifests.get(&member) else {
                continue;
            };

            // Reaching a member at all builds it, and building it builds every
            // non-optional runtime dependency with the features that edge names.
            if built.insert(member.clone()) {
                for table in runtime_dependency_tables(parsed) {
                    for (key, spec) in table {
                        let optional = spec
                            .get("optional")
                            .and_then(toml::Value::as_bool)
                            .unwrap_or(false);
                        if optional {
                            continue;
                        }
                        queue.extend(
                            edge_activations(manifests, parsed, key)
                                .into_iter()
                                .map(|next| (next, Some(state.clone()))),
                        );
                    }
                }
            }

            for entry in parsed
                .get("features")
                .and_then(toml::Value::as_table)
                .and_then(|table| table.get(&feature))
                .and_then(toml::Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(toml::Value::as_str)
            {
                let (strong, weak) = feature_entry_activations(manifests, parsed, &member, entry);
                queue.extend(strong.into_iter().map(|next| (next, Some(state.clone()))));
                deferred.extend(weak.into_iter().map(|next| (next, state.clone())));
            }
        }

        let (ready, waiting): (Vec<_>, Vec<_>) = deferred
            .into_iter()
            .partition(|(target, _)| built.contains(&target.0));
        deferred = waiting;
        if ready.is_empty() {
            return reached;
        }
        queue.extend(ready.into_iter().map(|(target, from)| (target, Some(from))));
    }
}

/// States a dependency edge turns on, when the dependency is a tracked member.
fn edge_activations(
    manifests: &BTreeMap<String, toml::Value>,
    manifest: &toml::Value,
    key: &str,
) -> Vec<(String, String)> {
    let package = dependency_package(manifest, key);
    if !manifests.contains_key(&package) {
        return Vec::new();
    }
    let Some(spec) = dependency_spec(manifest, key) else {
        return vec![(package, "default".to_string())];
    };
    let mut states = Vec::new();
    if spec
        .get("default-features")
        .and_then(toml::Value::as_bool)
        .unwrap_or(true)
    {
        states.push((package.clone(), "default".to_string()));
    }
    for feature in spec
        .get("features")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(toml::Value::as_str)
    {
        states.push((package.clone(), feature.to_string()));
    }
    states
}

/// States one entry of a `[features]` list turns on, strong ones first.
///
/// An entry is a sibling feature, `dep:key` for an optional dependency,
/// `key/feature` or the weak `key?/feature` for a feature of a dependency, or a
/// bare key naming an optional dependency that no entry spells with `dep:`.
/// `key/feature` also activates the dependency itself. The weak form activates
/// nothing on its own, so it is returned separately and the caller holds it
/// until something else activates that dependency.
type Activations = (Vec<(String, String)>, Vec<(String, String)>);

fn feature_entry_activations(
    manifests: &BTreeMap<String, toml::Value>,
    manifest: &toml::Value,
    member: &str,
    entry: &str,
) -> Activations {
    if let Some(key) = entry.strip_prefix("dep:") {
        return (edge_activations(manifests, manifest, key), Vec::new());
    }
    if let Some((key, feature)) = entry.split_once('/') {
        let weak = key.ends_with('?');
        let key = key.trim_end_matches('?');
        let package = dependency_package(manifest, key);
        if !manifests.contains_key(&package) {
            return (Vec::new(), Vec::new());
        }
        if weak {
            return (Vec::new(), vec![(package, feature.to_string())]);
        }
        let mut states = vec![(package, feature.to_string())];
        states.extend(edge_activations(manifests, manifest, key));
        return (states, Vec::new());
    }
    if manifest
        .get("features")
        .and_then(toml::Value::as_table)
        .is_some_and(|table| table.contains_key(entry))
    {
        return (vec![(member.to_string(), entry.to_string())], Vec::new());
    }
    (edge_activations(manifests, manifest, entry), Vec::new())
}

/// Members whose `[features]` table declares the host-oracle feature.
fn host_oracle_members(manifests: &BTreeMap<String, toml::Value>) -> BTreeSet<String> {
    manifests
        .iter()
        .filter(|(_, parsed)| {
            parsed
                .get("features")
                .and_then(toml::Value::as_table)
                .is_some_and(|table| table.contains_key(HOST_ORACLE_FEATURE))
        })
        .map(|(name, _)| name.clone())
        .collect()
}

/// Whether a manifest describes a crate someone outside this workspace can add.
///
/// `publish = false` names a member that exists only for this workspace, and a
/// parity harness is exactly that: measuring the GPU path against the CPU one
/// is its purpose, so it links the oracle on purpose. The distinction is read
/// from the manifest, so a member that becomes publishable is judged on the
/// commit that publishes it.
fn is_publishable(manifest: &toml::Value) -> bool {
    manifest
        .get("package")
        .and_then(|package| package.get("publish"))
        .and_then(toml::Value::as_bool)
        .unwrap_or(true)
}

/// Every default build of a publishable member that reaches a host oracle.
///
/// Three vyre-libs domain features named `cpu-parity` in the default set, so
/// `cargo add vyre-libs` compiled the CPU oracles into the shipped library and
/// the oracle symbols stood in for the domain edges this gate exists to find.
/// The roster of oracle-declaring members is read from the manifests, so a new
/// crate with the feature is judged on the commit that adds it.
#[must_use]
pub fn host_oracle_reach_failures(manifests: &BTreeMap<String, toml::Value>) -> Vec<String> {
    let declaring = host_oracle_members(manifests);
    if declaring.is_empty() {
        return vec![format!(
            "no tracked manifest declares a `{HOST_ORACLE_FEATURE}` feature, so this check judges nothing. Delete it with the last oracle feature, or restore the feature it guards."
        )];
    }

    let mut failures = Vec::new();
    for (root, parsed) in manifests {
        if !is_publishable(parsed) {
            continue;
        }
        let reached = default_feature_reach(manifests, root);
        for owner in &declaring {
            let target = (owner.clone(), HOST_ORACLE_FEATURE.to_string());
            if !reached.contains_key(&target) {
                continue;
            }
            let mut path = Vec::new();
            let mut step = Some(target.clone());
            while let Some(state) = step {
                path.push(format!("{}:{}", state.0, state.1));
                step = reached.get(&state).cloned().flatten();
            }
            path.reverse();
            failures.push(format!(
                "the default build of `{root}` turns on `{owner}/{HOST_ORACLE_FEATURE}` through {}",
                path.join(" -> ")
            ));
        }
    }
    failures
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
/// Returns the reason the data file could not be read as declarations,
/// including a copy that still stores a compile outcome.
pub fn load_rows(root: &Path) -> Result<Vec<Row>, String> {
    let path = data_path(root);
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "cannot read {}: {error}. Regenerate it with `cargo run -p xtask -- feature-isolation --write`.",
            path.display()
        )
    })?;
    parse_rows(&path, &text)
}

/// Declarations from the text of a data file.
///
/// # Errors
///
/// Returns the reason the text is not a set of declarations.
pub fn parse_rows(path: &Path, text: &str) -> Result<Vec<Row>, String> {
    let document: toml::Value = toml::from_str(text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    reject_stored_measurement(path, &document)?;
    let parsed: RowFile = toml::from_str(text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    Ok(parsed.pair)
}

/// Refuse a data file that still stores what a run is supposed to measure.
///
/// The provenance moved into run state, so a surviving copy of it is not a
/// harmless leftover. `measured = true` reads as a measurement nobody took this
/// run, and `outcome = "compiles"` is a green exemption the file has no standing
/// to grant; either one is a second record of a fact the run already owns, and a
/// second record is what goes stale. Rejecting the file is what makes the move
/// complete rather than optional.
fn reject_stored_measurement(path: &Path, document: &toml::Value) -> Result<(), String> {
    let Some(rows) = document.get("pair").and_then(toml::Value::as_array) else {
        return Ok(());
    };
    for row in rows {
        let label = format!(
            "{} {}",
            row.get("member")
                .and_then(toml::Value::as_str)
                .unwrap_or("?"),
            row.get("feature")
                .and_then(toml::Value::as_str)
                .unwrap_or("?")
        );
        if row.get("measured").is_some() {
            return Err(format!(
                "{} row `{label}` still carries `measured`; that column moved into the run, so delete it from every row and let a sweep observe the outcome",
                path.display()
            ));
        }
        if row.get("outcome").and_then(toml::Value::as_str) == Some(COMPILES) {
            return Err(format!(
                "{} row `{label}` records outcome = \"{COMPILES}\"; that column moved into the run, so a row declares the pair and a `{BLOCKED}` exemption with a reason, and nothing else",
                path.display()
            ));
        }
    }
    Ok(())
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

/// Every disagreement between the derived axis and the recorded declarations.
///
/// This is the cheap half of the gate. It reads no cargo output, so it is the
/// half that can run on every change, and it is what makes a new feature red by
/// default instead of unjudged.
///
/// It judges the shape of a declaration and nothing about compiling. A row
/// states no compile outcome, so there is no claim here for a run to have
/// observed: whether a selection holds is [`unmeasured_failures`] and
/// [`sweep_failures`], and both need this run to have compiled it.
#[must_use]
pub fn agreement_failures(pairs: &[Pair], rows: &[Row]) -> Vec<String> {
    let mut failures = Vec::new();
    let mut recorded: BTreeMap<(&str, &str), &Row> = BTreeMap::new();

    for row in rows {
        let label = Pair {
            member: row.member.clone(),
            feature: row.feature.clone(),
        }
        .label();
        let key = (row.member.as_str(), row.feature.as_str());
        if recorded.insert(key, row).is_some() {
            failures.push(format!(
                "`{label}` is recorded more than once; the later row is dead weight, delete it"
            ));
        }
        match row.outcome.as_deref() {
            None | Some(BLOCKED) => {}
            Some(outcome) => {
                failures.push(format!(
                    "`{} {}` records outcome `{outcome}`; the only outcome a row may state is `{BLOCKED}`, because whether a selection compiles is measured by the run and never stored",
                    row.member, row.feature
                ));
                continue;
            }
        }
        let reason = row.reason.as_deref().unwrap_or("").trim();
        if row.blocked() && !is_real_reason(reason) {
            failures.push(format!(
                "`{label}` is recorded `{BLOCKED}` with no real reason (`{reason}`); state the technical constraint on one line, not a schedule"
            ));
        }
        if !row.blocked() && !reason.is_empty() {
            failures.push(format!(
                "`{label}` is not recorded `{BLOCKED}` and still carries a reason; a reason belongs only on a `{BLOCKED}` row"
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

/// How many selections a measuring run names before it stops listing them.
///
/// The count is the finding; the names are what makes it actionable. A narrowed
/// sweep can leave nearly two hundred selections unobserved, and a report that
/// prints all of them buries the disagreements the sweep did find.
const NAMED_UNMEASURED: usize = 8;

/// Every selection a measuring run left unobserved.
///
/// Fail closed on absence. The outcome of a selection is a measurement, and an
/// absent measurement is not an agreement: a sweep narrowed by `--member` or
/// `--only-unrecorded` compiled a handful of selections and used to report
/// itself green, which reads as the axis holding when nothing observed all but
/// those few. So the unobserved remainder is a finding, and the run exits
/// non-zero.
///
/// This is asked of a run that set out to measure. A declaration-only
/// invocation compiles nothing on purpose and claims nothing about compiling:
/// with the outcome column out of the data file there is no stored green left
/// for it to launder, which is what the column removal bought.
#[must_use]
pub fn unmeasured_failures(pairs: &[Pair], observed: &[(Pair, Observation)]) -> Vec<String> {
    let missing = pairs
        .iter()
        .filter(|pair| {
            !observed
                .iter()
                .any(|(observed_pair, _)| observed_pair == *pair)
        })
        .map(Pair::label)
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Vec::new();
    }
    let mut message = format!(
        "unmeasured: {} of {} selection(s) were not compiled by this run, so nothing observed whether they hold: {}",
        missing.len(),
        pairs.len(),
        missing
            .iter()
            .take(NAMED_UNMEASURED)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ")
    );
    if missing.len() > NAMED_UNMEASURED {
        message.push_str(&format!(", and {} more", missing.len() - NAMED_UNMEASURED));
    }
    vec![message]
}

/// Every disagreement between what this run compiled and what the rows declare.
///
/// The expensive half's judgement, separated from the compiling so it can be
/// held to both directions without a cargo run. A selection declared with no
/// exemption that fails is the break the axis exists to catch. A selection
/// recorded `blocked` that now compiles is the other half: a reason that has
/// stopped being true keeps a selection exempt, and the exemption then covers
/// the next break in it.
///
/// A selection the rows do not mention at all is the declaration half's
/// finding, and it is skipped here so one omission is reported once, under the
/// fix that closes it, rather than a second time as a `blocked` row that
/// compiles.
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
        match (row.blocked(), observation.compiles) {
            (false, false) => failures.push(format!(
                "`{}` is declared with no exemption and fails with {}",
                pair.label(),
                observation
                    .first_error
                    .as_deref()
                    .unwrap_or("no parsed diagnostic")
            )),
            (true, true) => failures.push(format!(
                "`{}` is recorded `{BLOCKED}` and now compiles; delete its outcome and its reason",
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

/// Arguments the sweep passes cargo to compile one pair.
///
/// `--lib` and not `--all-targets`. The question is whether the crate's own
/// source compiles under the selection, and a test or bench target drags in
/// dev-dependencies. Cargo unifies features across that graph, so a
/// dev-dependency that depends on this crate with its full feature set turns
/// on the very feature the probe removed, and the missing edge compiles. Every
/// break this axis exists to catch is invisible to `--all-targets`.
#[must_use]
pub fn check_args(pair: &Pair) -> Vec<String> {
    let mut args = vec![
        "check".to_string(),
        "--locked".to_string(),
        "-p".to_string(),
        pair.member.clone(),
    ];
    args.extend(pair.cargo_flags());
    args.push("--lib".to_string());
    args.push("--message-format=json".to_string());
    args
}

/// Compile one pair once, on `toolchain` when one is named.
///
/// The binary comes from [`crate::cargo_runner::runner`], which owns which
/// cargo this tooling spawns and in which directory. A gate that resolves its
/// own binary picks a different one from every other gate the moment the
/// environment differs, and the environment differs on CI, where `CARGO` is
/// unset for a step that does not run under cargo.
fn check_once(
    root: &Path,
    cargo: &str,
    toolchain: &str,
    pair: &Pair,
) -> Result<Observation, GateError> {
    let mut command = Command::new(cargo);
    command.current_dir(root);
    if !toolchain.is_empty() {
        command.arg(format!("+{toolchain}"));
    }
    command.args(check_args(pair));
    let output = command.output().map_err(|error| {
        GateError::new(
            format!("cannot run `{cargo} check` for `{}`: {error}", pair.label()),
            "install a cargo the sweep can run, or restore the cargo_full wrapper at the workspace root",
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
fn compile(
    root: &Path,
    cargo: &str,
    toolchain: &str,
    pair: &Pair,
) -> Result<Observation, GateError> {
    let first = check_once(root, cargo, toolchain, pair)?;
    if first.compiles {
        return Ok(first);
    }
    check_once(root, cargo, toolchain, pair)
}

/// Render the data file for the whole axis: the declarations, plus the
/// exemptions this run either observed or inherited.
///
/// Merging is what makes recording one new selection affordable. Iterating the
/// derived axis rather than the rows also drops a row for a selection no
/// manifest declares any more, so a write cannot leave a stale row behind. No
/// compile outcome is written: a selection expected to compile is a bare pair,
/// and the only thing a row can state is that the selection is exempt and why.
#[must_use]
pub fn render(axis: &[Pair], observed: &[(Pair, Observation)], previous: &[Row]) -> String {
    let mut text = String::from(
        "# Every feature selection this workspace judges.\n\
         #\n\
         # The axis is derived from the tracked manifests at run time, never from this\n\
         # file. The `feature` column spells the selection: `(none)` is the per-member\n\
         # `--no-default-features` probe, `(default)` is the plain `cargo check -p`\n\
         # build, a bare name is that one feature enabled alone, and a comma-joined\n\
         # list is a selection a workspace edge asks of a sibling, with defaults kept\n\
         # when the list opens with `(default)`. A selection with no row here, and a row\n\
         # naming a selection no manifest declares, are each a failure.\n\
         #\n\
         # A row states no compile outcome. Whether a selection compiles is measured by\n\
         # the run that compiles it and is never stored here, because a stored\n\
         # measurement is stale the moment a feature edge moves. A sweep that leaves\n\
         # part of the axis uncompiled reports the rest as unmeasured and fails, so a\n\
         # bare row claims nothing and exempts nothing.\n\
         #\n\
         # `outcome = \"blocked\"` exempts a selection that cannot compile, and needs a\n\
         # one-line technical constraint in `reason`. A feature that merely needs\n\
         # another feature is not blocked: give it the missing edge in its own\n\
         # [features] table so `--features x` enables what x needs, which fixes the\n\
         # crate for a downstream consumer and not only for this sweep.\n\
         #\n\
         # Declare the axis: `cargo run -p xtask --bin xtask -- feature-isolation --write`.\n\
         # Measure it: `cargo run -p xtask --bin xtask -- feature-isolation --sweep`.\n",
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
            // Not compiled in this run. An exemption is a review, so it carries
            // forward verbatim; nothing else on a row is a decision, and a
            // selection with no exemption is written as the bare declaration it
            // is rather than as an outcome nobody observed.
            if let Some(row) = recorded.filter(|row| row.blocked()) {
                text.push_str(&format!("outcome = \"{BLOCKED}\"\n"));
                if let Some(reason) = row.reason.as_deref() {
                    text.push_str(&format!("reason = {}\n", quote(reason)));
                }
            }
            continue;
        };
        if observation.compiles {
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

/// Declarations already recorded, treating a file that does not exist yet as
/// none and a file that still stores an outcome as a hard failure.
fn recorded_rows(root: &Path) -> Result<Vec<Row>, GateError> {
    if !data_path(root).exists() {
        return Ok(Vec::new());
    }
    load_rows(root).map_err(|error| GateError::new(error, STALE_FIX))
/// The version `[workspace.package].rust-version` advertises.
///
/// The manifest is the single owner of the MSRV. A second copy in a workflow,
/// a script or a toolchain file disagrees with it on the commit that bumps one
/// of them, and the sweep then measures a compiler nobody publishes.
fn advertised_msrv(root: &Path) -> Result<String, GateError> {
    let manifest = root.join("Cargo.toml");
    let text = read_manifest(&manifest)
        .map_err(|error| GateError::new(error, "repair the workspace manifest"))?;
    let parsed: toml::Value = toml::from_str(&text).map_err(|error| {
        GateError::new(
            format!("cannot parse {}: {error}", manifest.display()),
            "repair the workspace manifest",
        )
    })?;
    let version = parsed
        .get("workspace")
        .and_then(|workspace| workspace.get("package"))
        .and_then(|package| package.get("rust-version"))
        .and_then(toml::Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if version.is_empty() {
        return Err(GateError::new(
            format!(
                "{} declares no [workspace.package].rust-version, and the MSRV sweep measures that version",
                manifest.display()
            ),
            "declare the minimum supported Rust version in the workspace manifest",
        ));
    }
    Ok(version)
}

/// Install the advertised MSRV toolchain unless rustup already carries it.
///
/// The sweep runs `cargo +<msrv>`, which needs the toolchain present. This ran
/// as a workflow step that read the manifest with a second reader; doing it
/// here keeps one owner of both the version and the sweep, and a developer gets
/// the same setup CI gets.
fn ensure_msrv_toolchain(version: &str, report: &mut Report) -> Result<(), GateError> {
    let listed = Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot run `rustup toolchain list`: {error}"),
                "install rustup, or run the sweep on a host that has the MSRV toolchain",
            )
        })?;
    let installed = String::from_utf8_lossy(&listed.stdout)
        .lines()
        .any(|line| line.starts_with(version));
    if installed {
        return Ok(());
    }
    let install = Command::new("rustup")
        .args(["toolchain", "install", "--profile", "minimal", version])
        .output()
        .map_err(|error| {
            GateError::new(
                format!("cannot install the `{version}` toolchain: {error}"),
                format!("run `rustup toolchain install --profile minimal {version}`"),
            )
        })?;
    if !install.status.success() {
        return Err(GateError::new(
            format!(
                "`rustup toolchain install --profile minimal {version}` failed: {}",
                String::from_utf8_lossy(&install.stderr).trim()
            ),
            "install the advertised MSRV toolchain, or correct [workspace.package].rust-version",
        ));
    }
    report.note(format!("installed the advertised MSRV toolchain {version}"));
    Ok(())
}

/// Turn one half's disagreements into findings under the fix that closes them.
fn record(report: &mut Report, failures: Vec<String>, fix: &str) {
    report
        .findings
        .extend(Report::from_messages(failures, fix).findings);
}

/// Compile each pair in turn, recording the outcome as it goes.
///
/// Both the sweep and `--write` need exactly this. The per-pair line is a note
/// rather than a print, because a gate returns everything it has to say.
/// `toolchain` names the rustup toolchain the MSRV mode measures, and is empty
/// for the default one.
fn observe(
    root: &Path,
    pairs: &[Pair],
    toolchain: &str,
    report: &mut Report,
) -> Result<Vec<(Pair, Observation)>, GateError> {
    let cargo = crate::cargo_runner::runner(root);
    let cargo = cargo.to_string_lossy().into_owned();
    let mut observed = Vec::with_capacity(pairs.len());
    for (index, pair) in pairs.iter().enumerate() {
        let observation = compile(root, &cargo, toolchain, pair)?;
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
const DECLARATION_FIX: &str = "record a row for every derived selection in xtask/feature-isolation.toml and delete every row no manifest declares; `xtask feature-isolation --write` rewrites the file from the derived axis";

/// What a selection a measuring run left uncompiled costs, and how to close it.
const MEASUREMENT_FIX: &str = "compile the whole axis in one run with `xtask feature-isolation --sweep`; a narrowed sweep observes a few selections and the rest stay unobserved, and this gate reports no outcome it did not measure";

/// What a data file that still stores a compile outcome costs, and how to close it.
const STALE_FIX: &str = "delete every `measured` key and every `outcome = \"compiles\"` row from xtask/feature-isolation.toml, then regenerate it with `xtask feature-isolation --write`; the compile outcome is produced by the run that compiles the selection";

/// What a row that disagrees with the compiler costs, and how to close it.
const COMPILE_FIX: &str = "give the feature the missing edge in its own [features] table so enabling it enables what it needs, or move the source behind the cfg that matches; record a row as blocked only for a constraint inherent to the crate";

/// What a default build reaching a host oracle costs, and how to close it.
const ORACLE_FIX: &str = "drop the cpu-parity edge from the feature that names it and gate the oracle call with cfg(any(test, feature = \"cpu-parity\")) instead; a default build must not compile a CPU reference implementation into the shipped library";

/// What a run that compiled nothing on purpose judged.
fn declaration_note(pairs: usize) -> String {
    format!(
        "{pairs} declared pair(s) agree with the manifests; this run compiled none of them, so it judges the declarations only"
    )
}

/// What a sweep compiled out of the axis it set out to measure.
fn sweep_note(compiled: usize, pairs: usize) -> String {
    format!("{compiled} of {pairs} declared pair(s) compiled by this run")
}

/// Holds every feature selection the manifests declare to its recorded compile outcome.
pub struct FeatureIsolation;

impl Gate for FeatureIsolation {
    fn name(&self) -> &'static str {
        "feature-isolation"
    }

    fn help(&self) -> &'static str {
        "Hold every feature selection the manifests declare to a decision; --write records the derived axis, --sweep compiles each pair and reports every selection it left unmeasured, --msrv compiles it on the advertised minimum supported Rust version, --member NAME and --only-unrecorded narrow the sweep, --list prints the axis"
    }

    fn usage(&self) -> &'static [&'static str] {
        &[
            "--sweep compiles each declared selection instead of reading the recorded axis",
            "--msrv compiles the sweep on the minimum supported Rust version the manifest advertises",
            "--member NAME narrows the sweep to one workspace member",
            "--only-unrecorded sweeps the selections that carry no recorded outcome",
            "--list prints the feature axis instead of judging it",
        ]
    }

    fn generates(&self) -> bool {
        true
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let root = &ctx.root;
        let mut report = Report::clean();
        let list = ctx.has("--list");
        let sweep = ctx.has("--sweep");
        let msrv = ctx.has("--msrv");
        let only_unrecorded = ctx.has("--only-unrecorded");
        let mut member = None;
        let mut rest = ctx.args.iter();
        while let Some(argument) = rest.next() {
            match argument.as_str() {
                "--list" | "--sweep" | "--msrv" | "--write" | "--only-unrecorded" => {}
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
                        "pass [--list] [--sweep [--msrv] [--write] [--only-unrecorded]] [--member NAME]",
                    ));
                }
            }
        }
        if only_unrecorded && !sweep {
            return Err(GateError::new(
                "`--only-unrecorded` narrows what a sweep compiles and no sweep was asked for",
                "pass `--sweep`",
            ));
        }
        if msrv && !sweep {
            return Err(GateError::new(
                "`--msrv` names the compiler the sweep measures and no sweep was asked for",
                "pass `--sweep`",
            ));
        }
        if msrv && ctx.write {
            return Err(GateError::new(
                "the record holds outcomes measured on the default toolchain, and `--msrv` measures another one",
                "record the axis with `--sweep --write`, and judge the MSRV with `--sweep --msrv`",
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
            let recorded = recorded_rows(root)?;
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
            let previous = recorded_rows(root)?;
            let observed = if sweep {
                observe(root, &selected, "", &mut report)?
            } else {
                Vec::new()
            };
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

        let rows = load_rows(root).map_err(|error| GateError::new(error, STALE_FIX))?;
        record(
            &mut report,
            agreement_failures(&pairs, &rows),
            DECLARATION_FIX,
        );
        let manifests = workspace_manifests(root).map_err(|error| {
            GateError::new(error, "repair the manifests the axis is derived from")
        })?;
        record(
            &mut report,
            host_oracle_reach_failures(&manifests),
            ORACLE_FIX,
        );

        if !sweep {
            report.note(declaration_note(pairs.len()));
            return Ok(report);
        }

        let toolchain = if msrv {
            let version = advertised_msrv(root)?;
            ensure_msrv_toolchain(&version, &mut report)?;
            version
        } else {
            String::new()
        };
        let observed = observe(root, &selected, &toolchain, &mut report)?;
        record(&mut report, sweep_failures(&rows, &observed), COMPILE_FIX);
        record(
            &mut report,
            unmeasured_failures(&pairs, &observed),
            MEASUREMENT_FIX,
        );
        report.note(match toolchain.is_empty() {
            true => sweep_note(observed.len(), pairs.len()),
            false => format!(
                "{} of {} pair(s) compiled on {toolchain}",
                observed.len(),
                pairs.len()
            ),
        });
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
        let row = |member: &str, feature: &str, blocked: bool| Row {
            member: member.to_string(),
            feature: feature.to_string(),
            outcome: blocked.then(|| BLOCKED.to_string()),
            reason: blocked.then(|| "the vendored driver has no Linux build".to_string()),
        };
        let rows = vec![
            row("vyre-pass-engine", "all-solvers", false),
            row("vyre-libs", "matching-regex", true),
            row("vyre-libs", "visual", false),
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
                Observation {
                    compiles: true,
                    first_error: None,
                },
            ),
            (
                pair("vyre-libs", "visual"),
                Observation {
                    compiles: true,
                    first_error: None,
                },
            ),
            // No row at all: the agreement half owns that, and counting it here
            // too would report one omission as two unrelated failures.
            (
                pair("vyre-libs", "unrecorded"),
                Observation {
                    compiles: true,
                    first_error: None,
                },
            ),
        ];

        let failures = sweep_failures(&rows, &observed);
        assert_eq!(
            failures,
            vec![
                "`vyre-pass-engine --no-default-features --features all-solvers` is declared with no exemption and fails with E0433 at vyre-pass-engine/tests/scope_rewrite_owner_contract.rs:19".to_string(),
                "`vyre-libs --no-default-features --features matching-regex` is recorded `blocked` and now compiles; delete its outcome and its reason".to_string(),
            ],
            "a declared outcome the build contradicts is reported, in the direction it was contradicted, and an agreeing pair is not"
        );
    }
}
