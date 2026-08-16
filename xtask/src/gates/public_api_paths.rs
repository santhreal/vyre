//! `cargo xtask public-api-paths`  -  one public path per published item.
//!
//! A crate that declares `pub mod inner` and then re-exports what `inner` holds
//! publishes every one of those items twice. Both paths compile, both appear in
//! the rendered documentation, and nothing in the source says which one a
//! consumer is supposed to write. The cost is not cosmetic: two paths for one
//! item is two names for one fact, so a reader cannot tell a moved item from a
//! renamed one, an import in this tree picks whichever path its author happened
//! to see first, and a deprecation has to be written twice or it is a lie at one
//! of the two paths.
//!
//! The judged axis is the committed snapshots under `docs/public-api`, which the
//! snapshot check keeps equal to the surface each crate really publishes. This
//! gate needs no compiler of its own for that reason, and it judges the surface
//! a consumer sees rather than the `pub use` statements that produce it, so a
//! second path arriving through a glob, a prelude, or a re-exported private
//! module counts the same as an explicit one.
//!
//! A snapshot names an item by its path and its signature, and those are equal
//! for a re-export and for two sibling modules that declare the same name: a
//! terminal id table per grammar, a per-op `OP_ID`, and a lint entry point named
//! `scan_tree` in each lint all read as one item at several paths. They are the
//! opposite shape - one name over several facts - and no rewrite of a snapshot
//! line separates them, which left a quarter of the measured count unclosable.
//! So the crate source decides: a name the crate declares twice is two items,
//! and only a name declared at most once can be one item at two paths.
//!
//! `xtask/public-api-paths.toml` records one measurement per crate. A crate with
//! no row is a finding, so a newly published crate is red until someone measures
//! it rather than silently unjudged. `--write` lowers a recorded number to what
//! this run measured and never raises one: a count that grew is the regression
//! this gate exists to catch, and a write that recorded it would launder the
//! regression into the baseline.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan;
use crate::output_arg::read_text_bounded;
use crate::toml_text::quote;

/// Largest snapshot this gate will read. The widest committed one is under 300 KiB.
const MAX_SNAPSHOT_BYTES: u64 = 4_194_304;

/// Schema this gate accepts in its data file.
const SCHEMA_VERSION: u64 = 1;

/// Example duplicated items named in one crate's finding.
const EXAMPLES_PER_CRATE: usize = 4;

/// One recorded measurement.
#[derive(Debug, Clone, Deserialize)]
pub struct Row {
    /// Package name, matching the snapshot file stem.
    pub name: String,
    /// Items that crate publishes at more than one path.
    pub duplicate_paths: usize,
}

#[derive(Debug, Deserialize)]
struct RowFile {
    schema_version: u64,
    #[serde(default, rename = "crate")]
    crates: Vec<Row>,
}

/// One published item, identified by everything about it except its module path.
///
/// The tail carries the type and member names, so `DiskCache::get` stays distinct
/// from `InMemoryPipelineCache::get`, and the remainder of the snapshot line
/// carries the signature, so an inherent method and a trait method of the same
/// name are two items rather than one item at two paths.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Item {
    tail: String,
    signature: String,
}

fn data_path(root: &Path) -> PathBuf {
    root.join("xtask/public-api-paths.toml")
}

fn snapshot_dir(root: &Path) -> PathBuf {
    root.join("docs/public-api")
}

/// Rows recorded in `xtask/public-api-paths.toml`.
///
/// # Errors
///
/// Returns the reason the data file could not be read as rows.
pub fn load_rows(root: &Path) -> Result<Vec<Row>, String> {
    let path = data_path(root);
    let text = read_text_bounded(&path, MAX_SNAPSHOT_BYTES, "public-api-paths data file")
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let file: RowFile =
        toml::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    if file.schema_version != SCHEMA_VERSION {
        return Err(format!(
            "{}: schema_version is {}, this gate reads {SCHEMA_VERSION}",
            path.display(),
            file.schema_version
        ));
    }
    Ok(file.crates)
}

/// Split one snapshot line into the item it declares and the module path it is
/// reachable through.
///
/// `modules` is every module path the same snapshot declares, which is what
/// tells a module segment from an item segment. Case cannot: an `io_uring` FFI
/// struct is `io_sqring_offsets`, so reading a snake_case segment as a module
/// merged its fields with the identically named fields of `io_cqring_offsets`
/// and reported six items nobody could delete.
///
/// Returns `None` for a line that declares nothing crate-rooted, which is how a
/// blank line and a `pub use` of a foreign crate's item leave the axis.
fn split_line(line: &str, modules: &BTreeSet<String>) -> Option<(Item, String)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let (start, path) = crate_rooted_path(trimmed)?;
    let end = start + path.len();
    let mut segments = path.split("::");
    segments.next();
    let segments: Vec<&str> = segments.collect();
    if segments.is_empty() {
        return None;
    }
    let first_item = (0..segments.len())
        .rev()
        .map(|at| (at, segments[..at].join("::")))
        .find(|(_, prefix)| prefix.is_empty() || modules.contains(prefix))
        .map_or(segments.len() - 1, |(at, _)| at);
    let tail = segments[first_item..].join("::");
    let module = segments[..first_item].join("::");
    let mut signature = String::with_capacity(trimmed.len());
    signature.push_str(&trimmed[..start]);
    signature.push('|');
    signature.push_str(&trimmed[end..]);
    Some((Item { tail, signature }, module))
}

/// Every module path one snapshot declares, relative to the crate root.
fn declared_modules(snapshot: &str) -> BTreeSet<String> {
    snapshot
        .lines()
        .filter_map(|line| {
            let path = line.trim().strip_prefix("pub mod ")?;
            let (_, rest) = path.split_once("::")?;
            Some(rest.trim().to_string())
        })
        .collect()
}

/// The first crate-rooted path in one snapshot line, with its byte offset.
///
/// A snapshot line names paths from several crates: `impl core::fmt::Debug for
/// vyre_runtime::tenant::TenantHandle` declares an item of this crate under a
/// trait from another. The subject is the first path rooted at a `vyre` crate,
/// which is what the extractor emits first for every line shape it produces.
fn crate_rooted_path(line: &str) -> Option<(usize, &str)> {
    let mut search = 0;
    while let Some(offset) = line[search..].find("vyre") {
        let start = search + offset;
        let leading_word = start > 0
            && line[..start]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_alphanumeric() || ch == '_' || ch == ':');
        let end = start
            + line[start..]
                .find(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == ':'))
                .unwrap_or(line.len() - start);
        let candidate = line[start..end].trim_end_matches(':');
        if !leading_word && candidate.contains("::") {
            return Some((start, candidate));
        }
        search = start + 4;
    }
    None
}

/// Every item one snapshot publishes at more than one path, with those paths.
///
/// A `prelude` republishes on purpose: it is the one list a caller imports from,
/// and the item still has exactly one owner. So a prelude path is not counted as
/// a second path, while the owner path it republishes is. An item reachable only
/// through a prelude keeps that path, because then the prelude is its owner and a
/// second owner elsewhere is still the defect this measures.
#[must_use]
pub fn duplicates(snapshot: &str) -> BTreeMap<String, BTreeSet<String>> {
    let modules = declared_modules(snapshot);
    let mut paths: BTreeMap<Item, BTreeSet<String>> = BTreeMap::new();
    for line in snapshot.lines() {
        if let Some((item, module)) = split_line(line, &modules) {
            paths.entry(item).or_default().insert(module);
        }
    }
    paths
        .into_iter()
        .map(|(item, modules)| {
            let owned: BTreeSet<String> = modules
                .iter()
                .filter(|module| !republishes(module))
                .cloned()
                .collect();
            let owners = if owned.is_empty() { modules } else { owned };
            (item.tail, owners)
        })
        .filter(|(_, modules)| modules.len() > 1)
        .collect()
}

/// Whether a module path republishes what another module owns, which every
/// language-wide `prelude` does by definition.
fn republishes(module: &str) -> bool {
    module
        .rsplit("::")
        .next()
        .is_some_and(|segment| segment == "prelude")
}

/// Item names the crate declares more than once, so several sibling modules each
/// publish their own, and the shared name is not one item at several paths.
///
/// A name is read from a module-scope `pub` declaration or from a name slot of a
/// module-scope macro invocation, because both plant a definition in the module
/// the file is. Sibling primitives that generate their entry points through one
/// wrapper macro each own their own function, and reporting them as one item at
/// two paths would name a defect the tree does not have.
#[must_use]
pub fn names_declared_more_than_once(sources: &[String]) -> BTreeSet<String> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for source in sources {
        for name in declared_names(source) {
            *counts.entry(name).or_default() += 1;
        }
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count > 1)
        .map(|(name, _)| name)
        .collect()
}

/// Every name one source file declares at module scope, by `pub` item or by a
/// macro invocation slot.
fn declared_names(source: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut in_invocation = false;
    for line in source.lines() {
        if in_invocation {
            if line.starts_with('}') {
                in_invocation = false;
            } else if let Some(name) = macro_slot_name(line) {
                names.push(name);
            }
            continue;
        }
        if let Some(name) = declared_item_name(line) {
            names.push(name);
            continue;
        }
        in_invocation = opens_module_scope_invocation(line);
    }
    names
}

/// Whether a line opens a macro invocation at module scope: `some_macro! {` with
/// no indentation, so a macro called inside a function body is not read.
fn opens_module_scope_invocation(line: &str) -> bool {
    let Some(bang) = line.find('!') else {
        return false;
    };
    if line.starts_with(char::is_whitespace) || !line.trim_end().ends_with('{') {
        return false;
    }
    let head = &line[..bang];
    !head.is_empty()
        && head
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// The name a macro invocation slot plants, for a `role: name` line. A `from`
/// clause names the function the generated one wraps, so only the head is read.
fn macro_slot_name(line: &str) -> Option<String> {
    let trimmed = line.trim();
    let colon = trimmed.find(':')?;
    let role = &trimmed[..colon];
    if role.is_empty()
        || !role
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return None;
    }
    let value = trimmed[colon + 1..].trim();
    let head = value
        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_'))
        .next()?;
    if head.is_empty() || !head.starts_with(|ch: char| ch.is_ascii_lowercase() || ch == '_') {
        return None;
    }
    Some(head.to_string())
}

/// Item keywords a module-scope declaration can carry.
const ITEM_KEYWORDS: [&str; 9] = [
    "mod", "fn", "const", "static", "struct", "enum", "trait", "type", "union",
];

/// The name one source line declares, when it declares a public item.
///
/// Visibility, `unsafe`, `extern "C"`, `async`, `const` and `mut` may sit between
/// the `pub` and the keyword, so the scan walks the words rather than matching a
/// fixed prefix.
fn declared_item_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with("pub") {
        return None;
    }
    let mut words = trimmed
        .split(|byte: char| byte.is_whitespace() || matches!(byte, '(' | ')' | '<' | ':' | ','))
        .filter(|word| !word.is_empty());
    let mut keyword = None;
    for word in words.by_ref() {
        if ITEM_KEYWORDS.contains(&word) {
            keyword = Some(word);
            break;
        }
        if !matches!(
            word,
            "pub" | "crate" | "super" | "self" | "in" | "unsafe" | "async" | "extern" | "default"
        ) && !word.starts_with('"')
        {
            return None;
        }
    }
    keyword?;
    // `const` and `static` open an item of their own and also qualify a function,
    // so the first keyword is not always the item's.
    let mut name = words.next()?;
    while ITEM_KEYWORDS.contains(&name) {
        name = words.next()?;
    }
    let name = name.trim_end_matches(&[';', '{', ',', ')'][..]);
    if name == "mut" {
        return None;
    }
    let mut characters = name.chars();
    let first = characters.next()?;
    if !(first.is_ascii_alphabetic() || first == '_') {
        return None;
    }
    if !characters.all(|value| value.is_ascii_alphanumeric() || value == '_') {
        return None;
    }
    Some(name.to_string())
}

/// Every Rust source of one crate, as text.
///
/// The tree walk already excludes build output and version control, and a crate
/// with no `src` directory is a crate that publishes nothing this gate can judge.
fn crate_sources(tree: &scan::Tree, directory: &str) -> Result<Vec<String>, GateError> {
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

/// What an unreadable snapshot axis costs, and how to restore it.
const SNAPSHOT_FIX: &str = "restore the committed snapshots under docs/public-api; `xtask public-api-snapshot --write --crate <name>` regenerates one from the crate it belongs to";

/// Every committed snapshot, keyed by the package it belongs to.
fn snapshots(root: &Path) -> Result<BTreeMap<String, PathBuf>, GateError> {
    let directory = snapshot_dir(root);
    let entries = std::fs::read_dir(&directory).map_err(|error| {
        GateError::new(format!("{}: {error}", directory.display()), SNAPSHOT_FIX)
    })?;
    let mut found = BTreeMap::new();
    for entry in entries {
        let path = entry
            .map_err(|error| {
                GateError::new(format!("{}: {error}", directory.display()), SNAPSHOT_FIX)
            })?
            .path();
        if path.extension().is_some_and(|extension| extension == "txt") {
            if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                found.insert(stem.to_string(), path);
            }
        }
    }
    if found.is_empty() {
        return Err(GateError::new(
            format!("{}: no snapshot files", directory.display()),
            SNAPSHOT_FIX,
        ));
    }
    Ok(found)
}

/// Render the data file from the measurements, keeping a recorded number that is
/// lower than what this run measured.
#[must_use]
pub fn render(measured: &BTreeMap<String, usize>, previous: &[Row]) -> String {
    let recorded: BTreeMap<&str, usize> = previous
        .iter()
        .map(|row| (row.name.as_str(), row.duplicate_paths))
        .collect();
    let mut out = String::new();
    out.push_str(
        "# Items each crate publishes at more than one path, measured from the\n\
         # committed snapshots under docs/public-api. A number here only goes down:\n\
         # `xtask public-api-paths --write` lowers a row to what it measured and\n\
         # never raises one, so a count that grew stays red until the second path\n\
         # is deleted.\n\
         schema_version = 1\n",
    );
    for (name, count) in measured {
        let pinned = recorded
            .get(name.as_str())
            .map_or(*count, |previous| (*previous).min(*count));
        let _ = write!(
            out,
            "\n[[crate]]\nname = {}\nduplicate_paths = {pinned}\n",
            quote(name)
        );
    }
    out
}

/// What a second path costs, and how to close it.
const FIX: &str = "delete the second path: a submodule that exists because a file was split stays private and the owning module re-exports what it holds, or the module is the public path and the parent re-export goes";

/// What an unmeasured or stale row costs, and how to close it.
const ROW_FIX: &str = "record a row for every committed snapshot in xtask/public-api-paths.toml and delete every row no snapshot answers to; `xtask public-api-paths --write` measures them";

/// Holds every published item to one public path, per crate, against a pin.
pub struct PublicApiPaths;

impl crate::gate::GateBehavior for PublicApiPaths {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let files = snapshots(&ctx.root)?;
        let rows = load_rows(&ctx.root).map_err(|error| GateError::new(error, ROW_FIX))?;
        let tree = scan::Tree::open(&ctx.root)?;
        let directories: BTreeMap<String, String> = tree
            .member_manifests()?
            .into_iter()
            .map(|member| (member.name, member.path))
            .collect();
        let mut report = Report::clean();
        report.produced("xtask/public-api-paths.toml");
        report.cover_complete("public api snapshot files", files.len());
        let mut measured = BTreeMap::new();
        let mut examples = BTreeMap::new();
        let mut shared_names = 0;
        for (name, path) in &files {
            let text = read_text_bounded(path, MAX_SNAPSHOT_BYTES, "public-api snapshot").map_err(
                |error| GateError::new(format!("{}: {error}", path.display()), SNAPSHOT_FIX),
            )?;
            let directory = directories.get(name).ok_or_else(|| {
                GateError::new(
                    format!("docs/public-api/{name}.txt names no workspace member"),
                    SNAPSHOT_FIX,
                )
            })?;
            let sources = crate_sources(&tree, directory)?;
            let shared = names_declared_more_than_once(&sources);
            let mut found = duplicates(&text);
            let before = found.len();
            found.retain(|item, _| {
                !shared.contains(item.split("::").next().unwrap_or(item.as_str()))
            });
            shared_names += before - found.len();
            measured.insert(name.clone(), found.len());
            examples.insert(name.clone(), found);
        }

        if ctx.write {
            let rendered = render(&measured, &rows);
            std::fs::write(data_path(&ctx.root), rendered).map_err(|error| {
                GateError::new(
                    format!("{}: {error}", data_path(&ctx.root).display()),
                    ROW_FIX,
                )
            })?;
        }
        let rows = if ctx.write {
            load_rows(&ctx.root).map_err(|error| GateError::new(error, ROW_FIX))?
        } else {
            rows
        };

        let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
        for row in &rows {
            if seen
                .insert(row.name.as_str(), row.duplicate_paths)
                .is_some()
            {
                report.find(Finding::in_file(
                    "xtask/public-api-paths.toml",
                    format!("{} has more than one row", row.name),
                    ROW_FIX,
                ));
            }
            if !files.contains_key(&row.name) {
                report.find(Finding::in_file(
                    "xtask/public-api-paths.toml",
                    format!(
                        "{} has a row but docs/public-api/{}.txt does not exist",
                        row.name, row.name
                    ),
                    ROW_FIX,
                ));
            }
        }

        let mut total = 0;
        for (name, count) in &measured {
            total += *count;
            let Some(pinned) = seen.get(name.as_str()) else {
                report.find(Finding::in_file(
                    "xtask/public-api-paths.toml",
                    format!(
                        "{name} publishes {count} item(s) at more than one path and records no measurement"
                    ),
                    ROW_FIX,
                ));
                continue;
            };
            if count == pinned {
                continue;
            }
            let relation = if count > pinned { "above" } else { "below" };
            let mut message = format!(
                "{name} publishes {count} item(s) at more than one path, {relation} its recorded {pinned}"
            );
            if count > pinned {
                let named: Vec<String> = examples
                    .get(name)
                    .into_iter()
                    .flatten()
                    .take(EXAMPLES_PER_CRATE)
                    .map(|(item, modules)| {
                        format!(
                            "{item} through {}",
                            modules
                                .iter()
                                .map(|module| if module.is_empty() {
                                    "the crate root".to_string()
                                } else {
                                    module.clone()
                                })
                                .collect::<Vec<_>>()
                                .join(" and ")
                        )
                    })
                    .collect();
                if !named.is_empty() {
                    let _ = write!(message, ": {}", named.join(", "));
                }
            }
            let fix = if count > pinned {
                FIX
            } else {
                "lower the recorded number to what this run measured; `xtask public-api-paths --write` does it"
            };
            report.find(Finding::in_file(
                format!("docs/public-api/{name}.txt"),
                message,
                fix,
            ));
        }
        report.note(format!(
            "measured {} snapshot(s); {total} item(s) reachable at more than one path; {shared_names} shared name(s) each declared by several sibling modules",
            files.len()
        ));
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// WHY: the shape this gate exists for. One type declared in a submodule and
    /// re-exported by its parent is one item a consumer can write two ways, and
    /// the report has to name both ways or the reader cannot delete either.
    #[test]
    fn a_type_reachable_through_two_modules_is_one_duplicate() {
        let found = duplicates(
            "pub mod vyre_x::inner\npub struct vyre_x::inner::Thing\npub struct vyre_x::Thing\npub struct vyre_x::inner::Other\n",
        );
        assert_eq!(found.len(), 1, "one item is published twice: {found:?}");
        let modules = &found["Thing"];
        assert!(
            modules.contains("inner") && modules.contains(""),
            "both paths must be named: {modules:?}"
        );
    }

    /// WHY: the naive measurement keys on the last path segment, which makes
    /// every `new`, `get` and `len` in a crate one item at dozens of paths. A
    /// gate that reports hundreds of items nobody can delete gets ignored.
    #[test]
    fn one_method_name_on_two_types_is_not_a_duplicate() {
        let found = duplicates(
            "pub fn vyre_x::a::One::get(&self) -> u32\npub fn vyre_x::b::Two::get(&self) -> u32\n",
        );
        assert!(found.is_empty(), "two types, two methods: {found:?}");
    }

    /// WHY: the item identity has to carry the signature. An inherent method and
    /// a trait method of the same name on the same type are two items, and
    /// collapsing them would report a duplicate that deleting a path cannot fix.
    #[test]
    fn one_name_with_two_signatures_is_two_items() {
        let found = duplicates(
            "pub fn vyre_x::a::One::get(&self) -> u32\npub fn vyre_x::b::One::get(&self) -> u64\n",
        );
        assert!(found.is_empty(), "two signatures, two items: {found:?}");
    }

    /// WHY: a snapshot line names foreign paths before the subject. Taking the
    /// first path in the line would make `core::fmt::Debug` the subject of every
    /// trait impl and measure nothing about this crate.
    #[test]
    fn a_foreign_trait_is_not_the_subject_of_the_line() {
        let found = duplicates(
            "pub mod vyre_x::inner\nimpl core::fmt::Debug for vyre_x::inner::Thing\nimpl core::fmt::Debug for vyre_x::Thing\n",
        );
        assert_eq!(
            found.keys().collect::<Vec<_>>(),
            vec!["Thing"],
            "the subject is the vyre-rooted path: {found:?}"
        );
    }

    /// WHY: a write that recorded a grown count would launder the regression
    /// this gate exists to catch into the baseline, and the next reader would
    /// have no way to know the surface got worse.
    #[test]
    fn a_write_lowers_a_recorded_number_and_never_raises_one() {
        let previous = vec![Row {
            name: "vyre-x".to_string(),
            duplicate_paths: 3,
        }];
        let grown = render(&BTreeMap::from([("vyre-x".to_string(), 9)]), &previous);
        assert!(
            grown.contains("duplicate_paths = 3"),
            "a grown count must not be recorded: {grown}"
        );
        let shrunk = render(&BTreeMap::from([("vyre-x".to_string(), 1)]), &previous);
        assert!(
            shrunk.contains("duplicate_paths = 1"),
            "a shrunk count must be recorded: {shrunk}"
        );
        let fresh = render(&BTreeMap::from([("vyre-y".to_string(), 7)]), &previous);
        assert!(
            fresh.contains("name = \"vyre-y\"") && fresh.contains("duplicate_paths = 7"),
            "a crate with no row gets its first measurement: {fresh}"
        );
        assert!(
            !fresh.contains("vyre-x"),
            "a row no snapshot answers to is dropped: {fresh}"
        );
    }

    /// WHY: a module segment is known only from a `pub mod` line, so a free
    /// function or constant published under one keeps its own last segment as
    /// the item and the module prefix as the path.
    #[test]
    fn a_free_function_is_identified_by_its_own_name() {
        let found = duplicates(
            "pub mod vyre_x::inner\npub fn vyre_x::inner::helper() -> u32\npub fn vyre_x::helper() -> u32\n",
        );
        assert_eq!(
            found.keys().collect::<Vec<_>>(),
            vec!["helper"],
            "a free function published twice: {found:?}"
        );
    }

    /// WHY: the module set comes from the snapshot's own `pub mod` lines, so a
    /// segment no snapshot declares as a module is part of the item. Reading a
    /// snake_case segment as a module merged the fields of `io_sqring_offsets`
    /// with the identically named fields of `io_cqring_offsets` and reported six
    /// duplicates nobody could delete.
    #[test]
    fn an_undeclared_segment_is_part_of_the_item_not_a_module() {
        let found = duplicates(
            "pub struct field vyre_x::io_sqring_offsets::head: u32\npub struct field vyre_x::io_cqring_offsets::head: u32\n",
        );
        assert!(
            found.is_empty(),
            "two structs, two fields of the same name: {found:?}"
        );
    }

    /// WHY: `vyre_libs::prelude` is the seam a dialect imports a sibling's item
    /// from, and lego-audit check 4 reports the direct path instead. Counting the
    /// seam as a second owner asked for the one import rule the tree enforces to be
    /// deleted.
    #[test]
    fn a_prelude_republish_is_not_a_second_owner() {
        let found = duplicates(
            "pub mod vyre_x::prelude\npub mod vyre_x::math\npub struct vyre_x::math::MatmulBias\npub struct vyre_x::prelude::MatmulBias\n",
        );
        assert!(found.is_empty(), "the seam owns nothing: {found:?}");
    }

    /// WHY: the exclusion must not hide a real second owner, and an item whose only
    /// path is a prelude has that prelude as its owner.
    #[test]
    fn a_prelude_hides_no_second_owner_and_owns_what_only_it_publishes() {
        let two_owners = duplicates(
            "pub mod vyre_x::prelude\npub mod vyre_x::math\npub mod vyre_x::nn\npub struct vyre_x::math::MatmulBias\npub struct vyre_x::nn::MatmulBias\npub struct vyre_x::prelude::MatmulBias\n",
        );
        assert_eq!(
            two_owners.get("MatmulBias").map(BTreeSet::len),
            Some(2),
            "two real owners survive the seam: {two_owners:?}"
        );
        let no_owner = duplicates(
            "pub mod vyre_x::prelude\npub mod vyre_x::inner::prelude\npub fn vyre_x::prelude::helper() -> u32\npub fn vyre_x::inner::prelude::helper() -> u32\n",
        );
        assert_eq!(
            no_owner.get("helper").map(BTreeSet::len),
            Some(2),
            "with no owner outside a prelude the two paths are the finding: {no_owner:?}"
        );
    }

    /// WHY: the shape that made a quarter of the measured count unclosable. Four
    /// grammars declare their own `TOK_LPAREN`, twelve ops their own `OP_ID`, and
    /// every lint its own `scan_tree`; the snapshot line is identical for those and
    /// for a re-export, so the crate source has to decide.
    #[test]
    fn a_name_two_sibling_modules_declare_is_not_one_item() {
        let shared = names_declared_more_than_once(&[
            "pub const TOK_LPAREN: u32 = 6;\n".to_string(),
            "pub const TOK_LPAREN: u32 = 11;\n".to_string(),
        ]);
        assert!(
            shared.contains("TOK_LPAREN"),
            "two declarations are two items: {shared:?}"
        );
    }

    /// WHY: the filter must not swallow the shape the gate exists for. One
    /// declaration re-exported by a parent is still one item at two paths.
    #[test]
    fn a_name_the_crate_declares_once_stays_a_duplicate() {
        let shared = names_declared_more_than_once(&[
            "pub fn pack_u32_slice(values: &[u32]) -> Vec<u8> { Vec::new() }\n".to_string(),
            "pub use crate::wire::pack_u32_slice;\n".to_string(),
        ]);
        assert!(
            shared.is_empty(),
            "a re-export is not a second declaration: {shared:?}"
        );
    }

    /// WHY: `csr_bidirectional` and `csr_forward_or_changed` each generate their own
    /// `cpu_ref_closure` through one wrapper macro, so no `pub fn` line carries the
    /// name and the snapshot line is the same one a re-export writes. Reading only
    /// `pub` lines reported two owned functions as one item at two paths.
    #[test]
    fn a_name_two_sibling_modules_plant_through_a_macro_is_not_one_item() {
        let invocation = |body: &str| {
            format!("define_csr_closure_entry_points! {{\n    allocating: {body} {{\n        /// doc\n    }},\n    hooked: cpu_ref_closure_into_with_step_hook,\n}}\n")
        };
        let shared = names_declared_more_than_once(&[
            invocation("cpu_ref_closure"),
            invocation("cpu_ref_closure"),
        ]);
        assert!(
            shared.contains("cpu_ref_closure"),
            "two macro invocations plant two functions: {shared:?}"
        );
        assert!(
            shared.contains("cpu_ref_closure_into_with_step_hook"),
            "a slot without a block is still a name slot: {shared:?}"
        );
    }

    /// WHY: a macro called inside a function body plants nothing at module scope,
    /// and a `from` clause names the function the generated one wraps rather than a
    /// second declaration.
    #[test]
    fn a_macro_inside_a_body_declares_nothing_and_a_from_clause_names_one_function() {
        let shared = names_declared_more_than_once(&[
            "pub fn run() {\n    assert_eq!(1, 1);\n    build! {\n        allocating: sneaky {\n        },\n    }\n}\n".to_string(),
            "pub fn run_again() {\n    build! {\n        allocating: sneaky {\n        },\n    }\n}\n".to_string(),
        ]);
        assert!(
            shared.is_empty(),
            "a macro in a body is not a module-scope declaration: {shared:?}"
        );
        let wrapped = names_declared_more_than_once(&[
            "define! {\n    allocating: cpu_ref_closure from try_cpu_ref_closure {\n    },\n}\n"
                .to_string(),
            "pub fn try_cpu_ref_closure() {}\n".to_string(),
        ]);
        assert!(
            wrapped.is_empty(),
            "the wrapped name is declared once: {wrapped:?}"
        );
    }

    /// WHY: the declaration scan reads the words of a line, because visibility,
    /// `unsafe`, `extern` and `async` all sit between `pub` and the keyword, and a
    /// fixed prefix match saw only the plainest form.
    #[test]
    fn every_declaration_form_is_read() {
        for line in [
            "pub mod thing;",
            "pub(crate) mod thing;",
            "pub fn thing() -> u32 {",
            "pub const fn thing() -> u32 {",
            "pub async fn thing() {",
            "pub unsafe fn thing() {",
            "pub extern \"C\" fn thing() {",
            "    pub struct thing<T> {",
            "pub enum thing {",
            "pub trait thing: Other {",
            "pub type thing = u32;",
            "pub static thing: u32 = 0;",
            "pub union thing {",
        ] {
            assert_eq!(
                declared_item_name(line).as_deref(),
                Some("thing"),
                "`{line}` declares thing"
            );
        }
    }

    /// WHY: a line that only mentions an item must not count as declaring it, or
    /// one `pub use` beside one declaration would read as two declarations and the
    /// duplicate it publishes would leave the axis.
    #[test]
    fn a_line_that_declares_nothing_is_not_a_declaration() {
        for line in [
            "pub use crate::wire::thing;",
            "/// pub fn thing() -> u32",
            "pub static mut COUNT: u32 = 0;",
            "        pub mod $module {",
            "let text = \"pub fn thing()\";",
        ] {
            assert_eq!(
                declared_item_name(line),
                None,
                "`{line}` declares no item name"
            );
        }
    }
}
