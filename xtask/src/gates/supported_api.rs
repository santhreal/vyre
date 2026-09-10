//! The supported-API classification gate and the manifest generated from it.
//!
//! Every workspace member declares one publication class in its own
//! `[package.metadata.vyre]` table, and every publishable member declares, in
//! the same table, the contract that requires it on the registry. This gate
//! reads the committed public-API snapshots under `docs/public-api/` and
//! classifies every exported item by seam, stability, feature, and wire
//! compatibility.
//!
//! [`SUPPORTED_API_MANIFEST`] (`docs/SUPPORTED_API.toml`) records the resulting
//! classification.
//!
//! Three things here can fail, and each of them used not to.
//!
//! A snapshot line the classifier cannot place is a finding. The classifier
//! previously ended in a catch-all that labelled any unrecognized line `item`,
//! so `pub const fn f()` was recorded as a constant named `fn f()` and no line
//! shape could ever be reported as unclassified. The grammar below is closed:
//! attributes, `impl`, every `pub` item keyword with its modifiers, enum
//! variants and struct fields. Anything else stops the gate.
//!
//! A member that becomes publishable is a finding until its contract is
//! recorded. The roster is Cargo's own answer, so a new publishable member
//! arrives here automatically, and [`PublicationContract`] gives it nowhere to
//! hide: every contract names a target the tree can be asked about, and a
//! `closure` contract has to reach a package with an external contract of its
//! own or it justifies nothing.
//!
//! A per-item feature and wire classification is derived, not stated. The
//! feature comes from the `#[cfg(feature = …)]` a crate root puts on the module
//! or re-export the item is reached through, and wire-bound means the owning
//! type carries a serde derive in the crate's own source.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::crate_registry::{
    load_registry, CrateRecord, REGISTRY, VALID_PUBLICATION_CLASSES,
};
use crate::gates::public_api::{roster, SNAPSHOT_DIR};
use crate::gates::scan::{Member, Tree};

/// The rendered supported-API manifest path.
pub const SUPPORTED_API_MANIFEST: &str = "docs/SUPPORTED_API.toml";
/// The command that rewrites the supported-API manifest.
pub const WRITE_COMMAND: &str = "xtask supported-api --write";
/// Manifest schema version.
pub const SCHEMA_VERSION: u32 = 2;
/// The support matrix a `target-support` contract is answered from.
pub const SUPPORT_MATRIX: &str = "docs/generated/platform-support-matrix.toml";

/// One classified exported public API item.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ClassifiedItem {
    /// Canonical item path or signature.
    pub path: String,
    /// Classified item kind.
    pub kind: String,
    /// Stability classification (`stable`, `extension`, `backend`, `internal`, `tooling`, `private`).
    pub stability: String,
    /// Cargo feature expression the item is reached under, or `unconditional`.
    pub feature: String,
    /// Wire compatibility classification (`wire-bound`, `api-only`).
    pub wire_compatibility: String,
}

/// One classified publishable package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackageClassification {
    /// Package name.
    pub name: String,
    /// Repository-relative directory.
    pub path: String,
    /// Name of the seam the package owns, as the architecture manifest states it.
    pub seam: String,
    /// Publication class from crate registry.
    pub publication_class: String,
    /// The declared contract that requires this package on the registry.
    pub publication_contract: String,
    /// Stability level.
    pub stability: String,
    /// Classified exported items in stable order.
    pub items: Vec<ClassifiedItem>,
}

/// The stability tier a publication class promises.
#[must_use]
pub fn stability_of(publication_class: &str) -> Option<&'static str> {
    match publication_class {
        "stable-consumer-sdk" => Some("stable"),
        "extension-sdk" => Some("extension"),
        "concrete-backend" => Some("backend"),
        "internal-engine" => Some("internal"),
        "conformance-tooling" => Some("tooling"),
        "private-test-support" => Some("private"),
        _ => None,
    }
}

/// Why a package is on the registry, as the manifest states it.
///
/// Each form names something the tree can be asked about, so a contract that
/// stops being true is a finding rather than a sentence nobody re-reads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PublicationContract {
    /// A package under `consumers/` depends on this crate.
    ConsumerSeam(String),
    /// An extension example under `examples/` depends on this crate.
    ExtensionPoint(String),
    /// The support matrix claims the driver this crate implements.
    TargetSupport(String),
    /// A publishable workspace member depends on this crate.
    Closure(String),
}

impl PublicationContract {
    /// Parse a declared contract, or return why it is not one.
    ///
    /// # Errors
    ///
    /// Returns the corrective sentence for a value in no recognized form.
    pub fn parse(value: &str) -> Result<Self, String> {
        let Some((form, target)) = value.split_once(':') else {
            return Err(format!(
                "`{value}` names no form; write one of {}",
                Self::FORMS.join(", ")
            ));
        };
        if target.is_empty() {
            return Err(format!("`{value}` names no target after `{form}:`"));
        }
        match form {
            "consumer-seam" => Ok(Self::ConsumerSeam(target.to_string())),
            "extension-point" => Ok(Self::ExtensionPoint(target.to_string())),
            "target-support" => Ok(Self::TargetSupport(target.to_string())),
            "closure" => Ok(Self::Closure(target.to_string())),
            other => Err(format!(
                "`{other}` is not a contract form; write one of {}",
                Self::FORMS.join(", ")
            )),
        }
    }

    /// Every accepted contract form, for a diagnostic.
    pub const FORMS: [&'static str; 4] = [
        "consumer-seam:<package under consumers/>",
        "extension-point:<package under examples/>",
        "target-support:<driver id in the support matrix>",
        "closure:<publishable workspace member that depends on this crate>",
    ];

    /// Whether this contract stands on its own rather than on another package's.
    #[must_use]
    pub fn is_external(&self) -> bool {
        !matches!(self, Self::Closure(_))
    }
}

/// Everything outside the workspace manifests that a contract is answered from.
#[derive(Debug, Default)]
pub struct ContractEvidence {
    /// Package name to the crates its `consumers/` manifest depends on.
    pub consumer_dependencies: BTreeMap<String, BTreeSet<String>>,
    /// Package name to the crates its `examples/` manifest depends on.
    pub example_dependencies: BTreeMap<String, BTreeSet<String>>,
    /// Driver ids the support matrix claims.
    pub claimed_drivers: BTreeSet<String>,
    /// Package name to the workspace members that depend on it.
    pub reverse_dependencies: BTreeMap<String, BTreeSet<String>>,
}

/// Every dependency name one manifest declares, across every dependency table.
fn declared_dependencies(manifest: &toml::Table) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut collect = |table: Option<&toml::Value>| {
        if let Some(table) = table.and_then(toml::Value::as_table) {
            names.extend(table.keys().cloned());
        }
    };
    for kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
        collect(manifest.get(kind));
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            let Some(target) = target.as_table() else {
                continue;
            };
            for kind in ["dependencies", "dev-dependencies", "build-dependencies"] {
                collect(target.get(kind));
            }
        }
    }
    names
}

/// Every dependency name one manifest declares as a normal or build edge.
fn production_dependencies(manifest: &toml::Table) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut collect = |table: Option<&toml::Value>| {
        if let Some(table) = table.and_then(toml::Value::as_table) {
            names.extend(table.keys().cloned());
        }
    };
    for kind in ["dependencies", "build-dependencies"] {
        collect(manifest.get(kind));
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for target in targets.values() {
            let Some(target) = target.as_table() else {
                continue;
            };
            for kind in ["dependencies", "build-dependencies"] {
                collect(target.get(kind));
            }
        }
    }
    names
}

/// Read every out-of-workspace manifest and support-matrix fact a contract cites.
///
/// # Errors
///
/// Returns the reason a directory or manifest under the checkout could not be read.
pub fn collect_contract_evidence(tree: &Tree) -> Result<ContractEvidence, GateError> {
    let mut evidence = ContractEvidence::default();
    for (directory, into) in [("consumers", true), ("examples", false)] {
        let absolute = tree.absolute(directory);
        let Ok(entries) = fs::read_dir(&absolute) else {
            continue;
        };
        for entry in entries.flatten() {
            let manifest = entry.path().join("Cargo.toml");
            if !manifest.is_file() {
                continue;
            }
            let relative = format!(
                "{directory}/{}/Cargo.toml",
                entry.file_name().to_string_lossy()
            );
            let table = tree.read_toml(&relative)?;
            let Some(name) = table
                .get("package")
                .and_then(|package| package.get("name"))
                .and_then(toml::Value::as_str)
            else {
                continue;
            };
            let dependencies = declared_dependencies(&table);
            if into {
                evidence
                    .consumer_dependencies
                    .insert(name.to_string(), dependencies);
            } else {
                evidence
                    .example_dependencies
                    .insert(name.to_string(), dependencies);
            }
        }
    }
    if tree.exists(SUPPORT_MATRIX) {
        let matrix = tree.read_toml(SUPPORT_MATRIX)?;
        if let Some(drivers) = matrix.get("drivers").and_then(toml::Value::as_array) {
            for driver in drivers {
                let claimed = driver
                    .get("claimed")
                    .and_then(toml::Value::as_bool)
                    .unwrap_or(false);
                let Some(id) = driver.get("id").and_then(toml::Value::as_str) else {
                    continue;
                };
                if claimed {
                    evidence.claimed_drivers.insert(id.to_string());
                }
            }
        }
    }
    for member in tree.member_manifests()? {
        for dependency in production_dependencies(&member.manifest) {
            evidence
                .reverse_dependencies
                .entry(dependency)
                .or_default()
                .insert(member.name.clone());
        }
    }
    Ok(evidence)
}

/// The contract a member declares, if it declares one.
#[must_use]
pub fn declared_contract(member: &Member) -> Option<&str> {
    member
        .manifest
        .get("package")?
        .get("metadata")?
        .get("vyre")?
        .get("publication_contract")?
        .as_str()
}

/// The publication class a member declares, if it declares one.
#[must_use]
pub fn declared_class(member: &Member) -> Option<&str> {
    member
        .manifest
        .get("package")?
        .get("metadata")?
        .get("vyre")?
        .get("publication_class")?
        .as_str()
}

/// What a caller does about a missing or wrong publication contract.
const CONTRACT_FIX: &str = "declare `publication_contract` in [package.metadata.vyre] naming why the registry needs this package, or set `publish = false`";

/// Hold every member's publication declaration to the tree.
///
/// Returns the parsed contract of each publishable member, in package order.
pub fn contract_findings(
    tree: &Tree,
    report: &mut Report,
) -> Result<BTreeMap<String, PublicationContract>, GateError> {
    let members = tree.member_manifests()?;
    let evidence = collect_contract_evidence(tree)?;
    let mut contracts: BTreeMap<String, PublicationContract> = BTreeMap::new();

    for member in &members {
        let manifest = format!("{}/Cargo.toml", member.path);
        let declared = declared_contract(member);
        if !member.publishable() {
            if declared.is_some() {
                report.find(Finding::in_file(
                    manifest.clone(),
                    format!(
                        "`{}` is `publish = false` and still declares a publication_contract",
                        member.name
                    ),
                    "delete `publication_contract`, or publish the package the contract describes",
                ));
            }
            continue;
        }
        let Some(declared) = declared else {
            report.find(Finding::in_file(
                manifest.clone(),
                format!(
                    "publishable package `{}` declares no publication_contract",
                    member.name
                ),
                CONTRACT_FIX,
            ));
            continue;
        };
        let contract = match PublicationContract::parse(declared) {
            Ok(contract) => contract,
            Err(reason) => {
                report.find(Finding::in_file(
                    manifest.clone(),
                    format!(
                        "`{}` declares an unreadable contract: {reason}",
                        member.name
                    ),
                    CONTRACT_FIX,
                ));
                continue;
            }
        };
        let unmet = match &contract {
            PublicationContract::ConsumerSeam(target) => {
                evidence.consumer_dependencies.get(target).map_or(
                    Some(format!("no package under `consumers/` is named `{target}`")),
                    |dependencies| {
                        (!dependencies.contains(&member.name))
                            .then(|| format!("consumer `{target}` declares no dependency on it"))
                    },
                )
            }
            PublicationContract::ExtensionPoint(target) => {
                evidence.example_dependencies.get(target).map_or(
                    Some(format!("no package under `examples/` is named `{target}`")),
                    |dependencies| {
                        (!dependencies.contains(&member.name))
                            .then(|| format!("example `{target}` declares no dependency on it"))
                    },
                )
            }
            PublicationContract::TargetSupport(target) => {
                (!evidence.claimed_drivers.contains(target))
                    .then(|| format!("`{SUPPORT_MATRIX}` claims no driver `{target}`"))
            }
            PublicationContract::Closure(target) => {
                let dependents = evidence.reverse_dependencies.get(&member.name);
                if !dependents.is_some_and(|names| names.contains(target)) {
                    Some(format!(
                        "`{target}` has no normal or build dependency on it"
                    ))
                } else if !members
                    .iter()
                    .any(|other| &other.name == target && other.publishable())
                {
                    Some(format!("`{target}` is not a publishable workspace member"))
                } else {
                    None
                }
            }
        };
        if let Some(reason) = unmet {
            report.find(Finding::in_file(
                manifest,
                format!("`{}` declares `{declared}` and {reason}", member.name),
                CONTRACT_FIX,
            ));
            continue;
        }
        contracts.insert(member.name.clone(), contract);
    }

    for (package, contract) in &contracts {
        if let Some(reason) = unrooted(package, contract, &contracts) {
            report.find(Finding::in_file(
                format!(
                    "{}/Cargo.toml",
                    members
                        .iter()
                        .find(|member| &member.name == package)
                        .map_or(package.clone(), |member| member.path.clone())
                ),
                reason,
                "point the closure contract at a package whose own contract is a consumer seam, an extension point or a claimed target",
            ));
        }
    }

    Ok(contracts)
}

/// Why a package's contract chain justifies nothing, if it does not.
///
/// A `closure` contract borrows another package's reason, so a chain of them
/// has to end at a package with an external contract. A chain that loops or
/// runs into a package with no contract at all justifies publishing nobody.
fn unrooted(
    package: &str,
    contract: &PublicationContract,
    contracts: &BTreeMap<String, PublicationContract>,
) -> Option<String> {
    let mut seen = BTreeSet::from([package.to_string()]);
    let mut chain = vec![package.to_string()];
    let mut current = contract;
    loop {
        let PublicationContract::Closure(next) = current else {
            return None;
        };
        chain.push(next.clone());
        if !seen.insert(next.clone()) {
            return Some(format!(
                "the closure contract of `{package}` loops: {}",
                chain.join(" -> ")
            ));
        }
        let Some(following) = contracts.get(next) else {
            return Some(format!(
                "the closure contract of `{package}` reaches `{next}`, which records no contract of its own"
            ));
        };
        current = following;
    }
}

/// The Cargo feature expression each crate-root path segment is reached under.
///
/// A crate root gates its public surface with `#[cfg(feature = …)]` on the
/// module or re-export that carries it, so the segment the attribute sits above
/// is the widest thing the item can be attributed to without a compiler. A
/// segment with no attribute is unconditional.
#[must_use]
pub fn feature_map(crate_root: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let mut pending: Option<String> = None;
    for line in crate_root.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }
        if let Some(expression) = cfg_feature_expression(trimmed) {
            pending = Some(expression);
            continue;
        }
        if trimmed.starts_with("#[") || trimmed.starts_with("#!") {
            continue;
        }
        let Some(expression) = pending.take() else {
            continue;
        };
        for segment in exported_segments(trimmed) {
            map.entry(segment).or_insert_with(|| expression.clone());
        }
    }
    map
}

/// The feature expression a `#[cfg(...)]` attribute line states, if it states one.
fn cfg_feature_expression(line: &str) -> Option<String> {
    let inner = line.strip_prefix("#[cfg(")?.strip_suffix(")]")?;
    if !inner.contains("feature") {
        return None;
    }
    let mut features: Vec<&str> = Vec::new();
    let mut rest = inner;
    while let Some(at) = rest.find("feature") {
        rest = &rest[at + "feature".len()..];
        let Some(open) = rest.find('"') else { break };
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else { break };
        features.push(&after[..close]);
        rest = &after[close + 1..];
    }
    if features.is_empty() {
        return None;
    }
    features.sort_unstable();
    features.dedup();
    Some(features.join(" | "))
}

/// The crate-root path segments one module-scope declaration publishes.
///
/// `pub mod name;` publishes `name`. `pub use path::{a, b};` publishes `a` and
/// `b`, and `pub use path::name;` publishes `name`, because a re-export at the
/// crate root is where a consumer's path starts.
fn exported_segments(line: &str) -> Vec<String> {
    if let Some(rest) = line.strip_prefix("pub mod ") {
        let name = rest.trim_end_matches(&[';', ' ', '{'][..]).trim();
        return if name.is_empty() {
            Vec::new()
        } else {
            vec![name.to_string()]
        };
    }
    let Some(rest) = line.strip_prefix("pub use ") else {
        return Vec::new();
    };
    let rest = rest.trim_end_matches(&[';', ' '][..]).trim();
    if let Some(open) = rest.find('{') {
        let inner = rest[open + 1..].trim_end_matches('}');
        return inner
            .split(',')
            .filter_map(|name| {
                let name = name.trim().rsplit(" as ").next()?.trim();
                (!name.is_empty() && name != "self").then(|| name.to_string())
            })
            .collect();
    }
    rest.rsplit("::")
        .next()
        .map(|name| {
            let name = name.rsplit(" as ").next().unwrap_or(name).trim();
            vec![name.to_string()]
        })
        .unwrap_or_default()
}

/// Type names one crate serializes, taken from the derives in its own source.
///
/// A wire-bound item is one whose owning type crosses a serialized boundary, and
/// the derive is where the tree says so. Deleting a derive therefore reclassifies
/// the item and the manifest has to be regenerated, which is the point: the old
/// rule was a substring match on the path and could not notice either.
#[must_use]
pub fn serialized_types(sources: &[String]) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for source in sources {
        let mut derives_serde = false;
        for line in source.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("#[derive(") || trimmed.starts_with("#[serde") {
                derives_serde |= trimmed.contains("Serialize") || trimmed.contains("Deserialize");
                continue;
            }
            if trimmed.starts_with("#[") {
                continue;
            }
            if derives_serde {
                if let Some(name) = declared_type_name(trimmed) {
                    names.insert(name);
                }
                derives_serde = false;
            }
        }
    }
    names
}

/// The type name a `struct`, `enum` or `union` declaration states.
fn declared_type_name(line: &str) -> Option<String> {
    let mut words = line.split_whitespace();
    let mut keyword = words.next()?;
    if keyword.starts_with("pub") {
        keyword = words.next()?;
    }
    if !matches!(keyword, "struct" | "enum" | "union") {
        return None;
    }
    let name = words.next()?;
    let name = name
        .split(|character: char| !(character.is_alphanumeric() || character == '_'))
        .next()?;
    (!name.is_empty()).then(|| name.to_string())
}

/// Item modifiers that may sit between `pub` and the item keyword.
const MODIFIERS: [&str; 5] = ["unsafe", "const", "async", "default", "extern"];

/// Classify a single public API snapshot line.
///
/// Returns `None` for a line in no form the snapshot grammar produces, which the
/// caller reports rather than filing under a catch-all kind.
#[must_use]
pub fn classify_line(
    line: &str,
    publication_class: &str,
    features: &BTreeMap<String, String>,
    serialized: &BTreeSet<String>,
) -> Option<ClassifiedItem> {
    let stability = stability_of(publication_class)
        .unwrap_or("unknown")
        .to_string();
    let mut trimmed = line.trim();
    while let Some(rest) = trimmed.strip_prefix("#[") {
        let close = rest.find(']')?;
        trimmed = rest[close + 1..].trim_start();
    }
    if trimmed.is_empty() {
        return None;
    }

    let (kind, path) = if trimmed.starts_with("impl ") || trimmed.starts_with("impl<") {
        ("impl", trimmed)
    } else {
        let rest = trimmed.strip_prefix("pub ")?;
        let (kind, path) = item_kind(rest)?;
        (kind, path)
    };

    let feature = features
        .iter()
        .find(|(segment, _)| path_enters(path, segment))
        .map_or_else(|| "unconditional".to_string(), |(_, value)| value.clone());
    let wire_compatibility = if owning_type(path).is_some_and(|name| serialized.contains(name)) {
        "wire-bound"
    } else {
        "api-only"
    }
    .to_string();

    Some(ClassifiedItem {
        path: path.to_string(),
        kind: kind.to_string(),
        stability,
        feature,
        wire_compatibility,
    })
}

/// The kind and remaining path of one `pub …` snapshot line.
///
/// The grammar is closed. A `pub` line either opens with an item keyword, after
/// any modifiers, or names an enum variant or a struct field, which the snapshot
/// writes as a bare path with a `: type` suffix for a field. Nothing else is a
/// line this extractor emits, so nothing else is accepted.
fn item_kind(rest: &str) -> Option<(&'static str, &str)> {
    let mut rest = rest.trim_start();
    if let Some(tail) = rest.strip_prefix("proc macro ") {
        return Some(("macro", tail.trim()));
    }
    loop {
        let Some((word, tail)) = rest.split_once(' ') else {
            break;
        };
        let keyword = match word {
            "mod" => "mod",
            "use" => "use",
            "struct" => "struct",
            "enum" => "enum",
            "fn" => "fn",
            "trait" => "trait",
            "type" => "type",
            "const" => "const",
            "static" => "static",
            "macro" => "macro",
            "union" => "union",
            _ => {
                if MODIFIERS.contains(&word) || word.starts_with('"') {
                    rest = tail.trim_start();
                    continue;
                }
                break;
            }
        };
        // `const fn` is a function; the modifier loop already consumed the rest.
        if keyword == "const" {
            if let Some(after) = tail.trim_start().strip_prefix("fn ") {
                return Some(("fn", after.trim()));
            }
        }
        return Some((keyword, tail.trim()));
    }
    let path = rest.trim();
    // The remainder must open with the path itself. A variant carries a tuple
    // payload or a discriminant after it and a field carries `: type`, so the
    // leading token is taken up to the first of those, and an unconsumed word
    // in front of it means the line is in no form the snapshot writes.
    let head = path.split([' ', '(', '<']).next().unwrap_or(path);
    if !head.trim_end_matches(':').contains("::") {
        return None;
    }
    Some(if path.contains(": ") {
        ("field", path)
    } else {
        ("variant", path)
    })
}

/// Whether a rendered path is reached through one crate-root segment.
fn path_enters(path: &str, segment: &str) -> bool {
    let Some((_, after_crate)) = path.split_once("::") else {
        return false;
    };
    after_crate == segment
        || after_crate
            .strip_prefix(segment)
            .is_some_and(|rest| rest.starts_with("::") || rest.starts_with('('))
}

/// The type name an item hangs off, which is its last capitalized path segment.
fn owning_type(path: &str) -> Option<&str> {
    let head = path.split(['(', '<', ' ']).next().unwrap_or(path);
    head.split("::")
        .filter(|segment| segment.starts_with(char::is_uppercase))
        .last()
}

/// Render the complete supported-API manifest as valid TOML.
#[must_use]
pub fn render_manifest(packages: &[PackageClassification]) -> String {
    let mut out = String::new();
    out.push_str("# Generated from the committed public-API snapshots, the member manifests and\n");
    out.push_str(&format!(
        "# docs/CRATE_OWNERSHIP.toml by `{WRITE_COMMAND}`.\n\n"
    ));
    out.push_str(&format!("schema_version = {SCHEMA_VERSION}\n\n"));

    for pkg in packages {
        out.push_str("[[package]]\n");
        out.push_str(&format!("name = \"{}\"\n", pkg.name));
        out.push_str(&format!("path = \"{}\"\n", pkg.path));
        out.push_str(&format!("seam = \"{}\"\n", pkg.seam));
        out.push_str(&format!(
            "publication_class = \"{}\"\n",
            pkg.publication_class
        ));
        out.push_str(&format!(
            "publication_contract = \"{}\"\n",
            pkg.publication_contract
        ));
        out.push_str(&format!("stability = \"{}\"\n", pkg.stability));
        out.push_str(&format!("exported_item_count = {}\n", pkg.items.len()));
        out.push('\n');
        for item in &pkg.items {
            out.push_str("  [[package.item]]\n");
            out.push_str(&format!(
                "  path = {}\n",
                crate::toml_text::quote(&item.path)
            ));
            out.push_str(&format!("  kind = \"{}\"\n", item.kind));
            out.push_str(&format!("  stability = \"{}\"\n", item.stability));
            out.push_str(&format!("  feature = \"{}\"\n", item.feature));
            out.push_str(&format!(
                "  wire_compatibility = \"{}\"\n\n",
                item.wire_compatibility
            ));
        }
    }
    out
}

/// Every Rust source of one crate, as text.
fn crate_sources(tree: &Tree, directory: &str) -> Result<Vec<String>, GateError> {
    let root = format!("{directory}/src");
    if !tree.exists(&root) {
        return Ok(Vec::new());
    }
    let mut sources = Vec::new();
    for path in tree.rust(&[root.as_str()])? {
        sources.push(tree.read(&path)?);
    }
    Ok(sources)
}

/// Classify all publishable packages given the workspace tree and registry records.
pub fn classify_workspace(
    tree: &Tree,
    records: &[CrateRecord],
    contracts: &BTreeMap<String, PublicationContract>,
    report: &mut Report,
) -> Result<Vec<PackageClassification>, GateError> {
    let snapshotted = roster(tree)?;
    let by_package: BTreeMap<&str, &CrateRecord> = records
        .iter()
        .map(|rec| (rec.package.as_str(), rec))
        .collect();
    let declared: BTreeMap<String, String> = tree
        .member_manifests()?
        .iter()
        .filter_map(|member| {
            declared_contract(member).map(|value| (member.name.clone(), value.to_string()))
        })
        .collect();

    let mut classifications = Vec::new();

    for pkg in &snapshotted {
        let Some(record) = by_package.get(pkg.package.as_str()) else {
            report.find(Finding::in_file(
                REGISTRY,
                format!(
                    "publishable package `{}` has no row in {REGISTRY}",
                    pkg.package
                ),
                format!(
                    "add a [[crate]] row for `{}` with an explicit publication_class",
                    pkg.package
                ),
            ));
            continue;
        };

        if record.publication_class.is_empty() {
            report.find(Finding::in_file(
                format!("{}/Cargo.toml", pkg.directory),
                format!("package `{}` declares no publication_class", pkg.package),
                format!("declare one of: {}", VALID_PUBLICATION_CLASSES.join(", ")),
            ));
            continue;
        }

        let Some(stability) = stability_of(&record.publication_class) else {
            report.find(Finding::in_file(
                format!("{}/Cargo.toml", pkg.directory),
                format!(
                    "package `{}` declares unknown publication_class `{}`",
                    pkg.package, record.publication_class
                ),
                format!("declare one of: {}", VALID_PUBLICATION_CLASSES.join(", ")),
            ));
            continue;
        };

        if !contracts.contains_key(&pkg.package) {
            // `contract_findings` already reported why, and a package with no
            // proven contract is not classified into the supported surface.
            continue;
        }

        let sources = crate_sources(tree, &pkg.directory)?;
        let features = feature_map(&tree.read(format!("{}/src/lib.rs", pkg.directory))?);
        let serialized = serialized_types(&sources);

        let snapshot_file = PathBuf::from(SNAPSHOT_DIR).join(format!("{}.txt", pkg.package));
        let snapshot_text = tree.read(&snapshot_file).unwrap_or_default();
        let mut items = Vec::new();
        let mut unclassified = Vec::new();
        for line in snapshot_text.lines().filter(|line| !line.trim().is_empty()) {
            match classify_line(line, &record.publication_class, &features, &serialized) {
                Some(item) => items.push(item),
                None => unclassified.push(line),
            }
        }
        for line in unclassified.iter().take(4) {
            report.find(Finding::in_file(
                snapshot_file.clone(),
                format!(
                    "`{}` exports an item the supported-API classifier cannot place: `{line}`",
                    pkg.package
                ),
                "extend the snapshot grammar in xtask/src/gates/supported_api.rs to classify the line, or stop exporting the item",
            ));
        }
        if unclassified.len() > 4 {
            report.find(Finding::in_file(
                snapshot_file.clone(),
                format!(
                    "`{}` exports {} further unclassified item(s)",
                    pkg.package,
                    unclassified.len() - 4
                ),
                "extend the snapshot grammar in xtask/src/gates/supported_api.rs to classify the lines",
            ));
        }
        items.sort();
        items.dedup();

        classifications.push(PackageClassification {
            name: pkg.package.clone(),
            path: record.path.clone(),
            seam: record.seam.clone(),
            publication_class: record.publication_class.clone(),
            publication_contract: declared.get(&pkg.package).cloned().unwrap_or_default(),
            stability: stability.to_string(),
            items,
        });
    }

    classifications.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(classifications)
}

/// The `supported-api` gate behavior.
pub struct SupportedApi;

impl crate::gate::GateBehavior for SupportedApi {
    fn usage(&self) -> &'static [&'static str] {
        &["--write    regenerate docs/SUPPORTED_API.toml from snapshots, manifests and crate ownership"]
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let mut report = Report::clean();
        report.produced(SUPPORTED_API_MANIFEST);

        let records = load_registry(&tree, &mut report)?;
        report.cover_complete("workspace manifests", records.len());
        if !report.findings.is_empty() {
            return Ok(report);
        }

        let contracts = contract_findings(&tree, &mut report)?;
        let classifications = classify_workspace(&tree, &records, &contracts, &mut report)?;
        if !report.findings.is_empty() {
            return Ok(report);
        }

        let rendered = render_manifest(&classifications);
        let manifest_path = ctx.root.join(SUPPORTED_API_MANIFEST);

        if ctx.write {
            fs::write(&manifest_path, &rendered).map_err(|error| {
                GateError::new(
                    format!("cannot write `{SUPPORTED_API_MANIFEST}`: {error}"),
                    "make the documentation directory writable",
                )
            })?;
            report.note(format!(
                "supported-api: wrote {SUPPORTED_API_MANIFEST} for {} publishable package(s)",
                classifications.len()
            ));
            return Ok(report);
        }

        let actual = fs::read_to_string(&manifest_path).unwrap_or_default();
        let actual_normalized: Vec<&str> = actual.lines().map(str::trim_end).collect();
        let rendered_normalized: Vec<&str> = rendered.lines().map(str::trim_end).collect();

        if actual_normalized != rendered_normalized {
            report.find(Finding::in_file(
                SUPPORTED_API_MANIFEST,
                format!("`{SUPPORTED_API_MANIFEST}` is out of date with public API snapshots or publication classes"),
                format!("run `{WRITE_COMMAND}` to regenerate the manifest"),
            ));
        }

        report.note(format!(
            "supported-api: verified {} publishable package(s)",
            classifications.len()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_features() -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    fn no_serde() -> BTreeSet<String> {
        BTreeSet::new()
    }

    #[test]
    fn classify_line_identifies_kinds_and_stability() {
        let item = classify_line(
            "pub struct vyre::compiler::CompileRequest",
            "stable-consumer-sdk",
            &empty_features(),
            &no_serde(),
        )
        .expect("a struct line classifies");
        assert_eq!(item.kind, "struct");
        assert_eq!(item.stability, "stable");
        assert_eq!(item.wire_compatibility, "api-only");
        assert_eq!(item.feature, "unconditional");

        let wire_item = classify_line(
            "pub fn vyre_foundation::ir::Program::to_wire_bytes()",
            "internal-engine",
            &empty_features(),
            &BTreeSet::from(["Program".to_string()]),
        )
        .expect("a method line classifies");
        assert_eq!(wire_item.kind, "fn");
        assert_eq!(wire_item.stability, "internal");
        assert_eq!(wire_item.wire_compatibility, "wire-bound");
    }

    #[test]
    fn classify_line_covers_every_line_shape_the_extractor_emits() {
        let cases = [
            ("pub mod vyre::foo", "mod"),
            ("pub use vyre::foo::bar", "use"),
            ("pub enum vyre::foo::Bar", "enum"),
            ("pub trait vyre::foo::Baz", "trait"),
            ("pub type vyre::foo::T = u32", "type"),
            ("pub const vyre::foo::C: u32 = 1", "const"),
            ("pub static vyre::foo::S: u32 = 1", "static"),
            ("pub proc macro vyre::foo::bar!()", "macro"),
            ("pub const fn vyre::foo::f() -> u32", "fn"),
            ("pub unsafe fn vyre::foo::g()", "fn"),
            ("impl core::fmt::Debug for vyre::foo::Bar", "impl"),
            ("impl<'a> vyre::foo::Reader<'a>", "impl"),
            ("#[non_exhaustive] pub enum vyre::foo::Kind", "enum"),
            ("pub vyre::foo::Kind::Variant", "variant"),
            ("pub vyre::foo::Kind::Variant::field: u32", "field"),
        ];
        for (line, expected) in cases {
            let item = classify_line(line, "extension-sdk", &empty_features(), &no_serde())
                .unwrap_or_else(|| panic!("`{line}` must classify"));
            assert_eq!(item.kind, expected, "for `{line}`");
        }
    }

    /// A line shape the grammar does not cover must stop the gate rather than
    /// land under a catch-all kind, which is what the previous classifier did to
    /// every `impl` line and to `pub const fn`.
    #[test]
    fn a_line_outside_the_grammar_is_unclassified() {
        assert!(classify_line(
            "reachable but unwritten shape",
            "extension-sdk",
            &empty_features(),
            &no_serde()
        )
        .is_none());
        assert!(classify_line(
            "pub gizmo vyre::foo::Bar",
            "extension-sdk",
            &empty_features(),
            &no_serde()
        )
        .is_none());
        assert!(classify_line(
            "pub bare_name",
            "extension-sdk",
            &empty_features(),
            &no_serde()
        )
        .is_none());
    }

    #[test]
    fn feature_map_reads_the_cfg_a_crate_root_puts_on_its_exports() {
        let root = r#"
pub mod always;
#[cfg(feature = "graph")]
pub mod graph;
#[cfg(any(feature = "math", feature = "math-kernels"))]
pub use vyre_libs_math::math;
#[cfg(feature = "text")]
pub use vyre_libs_text::{text, TextError};
"#;
        let map = feature_map(root);
        assert_eq!(map.get("graph").map(String::as_str), Some("graph"));
        assert_eq!(
            map.get("math").map(String::as_str),
            Some("math | math-kernels")
        );
        assert_eq!(map.get("text").map(String::as_str), Some("text"));
        assert_eq!(map.get("TextError").map(String::as_str), Some("text"));
        assert!(!map.contains_key("always"));
    }

    #[test]
    fn a_feature_gated_segment_reaches_the_items_under_it() {
        let features = BTreeMap::from([("graph".to_string(), "graph".to_string())]);
        let item = classify_line(
            "pub fn vyre_libs::graph::bfs::frontier() -> u32",
            "stable-consumer-sdk",
            &features,
            &no_serde(),
        )
        .expect("classifies");
        assert_eq!(item.feature, "graph");

        let ungated = classify_line(
            "pub fn vyre_libs::always::run() -> u32",
            "stable-consumer-sdk",
            &features,
            &no_serde(),
        )
        .expect("classifies");
        assert_eq!(ungated.feature, "unconditional");
    }

    #[test]
    fn wire_bound_follows_the_serde_derive_rather_than_the_name() {
        let sources = vec![
            "#[derive(Debug, Serialize, Deserialize)]\npub struct Envelope {}\n".to_string(),
            "#[derive(Debug)]\npub struct WireLooking {}\n".to_string(),
        ];
        let serialized = serialized_types(&sources);
        assert!(serialized.contains("Envelope"));
        assert!(!serialized.contains("WireLooking"));

        let bound = classify_line(
            "pub fn vyre::a::Envelope::seal(&self)",
            "stable-consumer-sdk",
            &empty_features(),
            &serialized,
        )
        .expect("classifies");
        assert_eq!(bound.wire_compatibility, "wire-bound");

        let unbound = classify_line(
            "pub fn vyre::a::WireLooking::seal(&self)",
            "stable-consumer-sdk",
            &empty_features(),
            &serialized,
        )
        .expect("classifies");
        assert_eq!(unbound.wire_compatibility, "api-only");
    }

    #[test]
    fn a_contract_form_outside_the_four_is_rejected() {
        assert!(PublicationContract::parse("because-we-said-so:vyre").is_err());
        assert!(PublicationContract::parse("closure").is_err());
        assert!(PublicationContract::parse("closure:").is_err());
        assert_eq!(
            PublicationContract::parse("closure:vyre-libs"),
            Ok(PublicationContract::Closure("vyre-libs".to_string()))
        );
    }

    /// A `closure` contract borrows another package's reason, so a ring of them
    /// justifies publishing nobody and every member of the ring is reported.
    #[test]
    fn a_closure_chain_must_reach_an_external_contract() {
        let rooted = BTreeMap::from([
            (
                "vyre".to_string(),
                PublicationContract::ConsumerSeam("app".to_string()),
            ),
            (
                "vyre-runtime".to_string(),
                PublicationContract::Closure("vyre".to_string()),
            ),
        ]);
        assert!(unrooted("vyre-runtime", &rooted["vyre-runtime"], &rooted).is_none());

        let ring = BTreeMap::from([
            (
                "a".to_string(),
                PublicationContract::Closure("b".to_string()),
            ),
            (
                "b".to_string(),
                PublicationContract::Closure("a".to_string()),
            ),
        ]);
        assert!(unrooted("a", &ring["a"], &ring).is_some());

        let dangling = BTreeMap::from([(
            "a".to_string(),
            PublicationContract::Closure("gone".to_string()),
        )]);
        assert!(unrooted("a", &dangling["a"], &dangling).is_some());
    }

    #[test]
    fn render_manifest_records_the_contract_and_the_schema() {
        let pkgs = vec![PackageClassification {
            name: "vyre".to_string(),
            path: "vyre".to_string(),
            seam: "compiler-facade".to_string(),
            publication_class: "stable-consumer-sdk".to_string(),
            publication_contract: "consumer-seam:vyre-graphics-app".to_string(),
            stability: "stable".to_string(),
            items: vec![ClassifiedItem {
                path: "vyre::compiler::CompileRequest".to_string(),
                kind: "struct".to_string(),
                stability: "stable".to_string(),
                feature: "unconditional".to_string(),
                wire_compatibility: "api-only".to_string(),
            }],
        }];
        let rendered = render_manifest(&pkgs);
        assert!(rendered.contains("schema_version = 2"));
        assert!(rendered.contains("name = \"vyre\""));
        assert!(rendered.contains("publication_class = \"stable-consumer-sdk\""));
        assert!(rendered.contains("publication_contract = \"consumer-seam:vyre-graphics-app\""));
    }
}
