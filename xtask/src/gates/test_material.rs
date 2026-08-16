//! Test material does not ship inside a publishable crate's `src/` tree.
//!
//! A shipping crate's `src/` is product. Test material there is not a style
//! question: it compiles into the published artifact, it is carried by every
//! consumer that never runs the suite, and once it is `pub` it is public API that
//! cannot be moved. The tree already carries the shapes that go wrong: a module
//! nothing has referenced since the initial import, and a module reachable in a
//! default build that only the crate's own suites use.
//!
//! The name selects the candidate and the content decides the verdict. A stem
//! segment out of [`TOKENS`] is what makes a file worth reading, and nothing more
//! than that: `bitset/test_bit.rs` is the bit-test operation and product code
//! calls it, `bellman_shortest_path.rs` is not a candidate at all because the
//! match is on segments split at `_`, `-` and `.` rather than on a substring. A
//! rule that convicted on the name would demand renaming three frozen public
//! paths, which is why the verdict is one of four readings of the tree:
//!
//! - the module declaration chain carries a cfg that can only be true in a test
//!   build, so a release build never compiles it;
//! - a line outside `#[cfg(test)]` in some publishable crate's `src/` references
//!   it, so it is product under a name that reads like test material;
//! - it is compiled only behind a feature that is off by default and something
//!   references it, so it is opt-in material a consumer chooses;
//! - otherwise it ships in a default build and only test code, or nothing at
//!   all, refers to it. That is the finding.
//!
//! The second rule is the dependency direction: no `[dependencies]`,
//! `[build-dependencies]` or `[target.*.dependencies]` entry may name the test
//! support crate. A dev edge is how a suite reaches it; any other edge links it
//! into the artifact.

use std::collections::{BTreeMap, BTreeSet};

use crate::gate::{Finding, Gate, GateCtx, GateError, Report};
use crate::gates::scan::{cfg_test_lines, is_test_only_attribute, scan_code, Member, Tree};

/// Stem segments that make a file test material by name.
const TOKENS: &[&str] = &[
    "test", "tests", "fixture", "fixtures", "oracle", "oracles", "mock", "mocks", "stub", "stubs",
    "sample", "samples", "golden", "harness",
];

/// The crate whose whole subject is test support.
const SUPPORT_CRATE: &str = "vyre-test-support";

/// Manifest tables whose entries link into the published artifact.
const SHIPPING_TABLES: &[&str] = &["dependencies", "build-dependencies"];

/// Every manifest table that declares an edge to another crate.
const DEPENDENCY_TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

/// What the module declaration chain above one file says about it.
#[derive(Debug, Default)]
struct Gating {
    /// Whether some declaration on the chain can only be true in a test build.
    test_only: bool,
    /// Every feature named by a cfg on the chain.
    features: BTreeSet<String>,
    /// Whether every declaration on the chain was found.
    declared: bool,
}

/// One file whose name says test material and whose content decides.
#[derive(Debug)]
struct Candidate {
    /// Repository-relative path of the file.
    file: String,
    /// Directory of the member that carries it.
    member: String,
    /// Crate name of that member, which a referrer has to declare an edge to.
    crate_name: String,
    /// The module's own identifier.
    module: String,
    /// Identifiers the module exports, which a caller would name.
    exports: BTreeSet<String>,
    /// What the declaration chain says.
    gating: Gating,
}

/// Where one reference to a candidate came from.
#[derive(Debug, Default)]
struct Reach {
    /// A line outside `#[cfg(test)]` in some publishable member's `src/`.
    product: Option<String>,
    /// Any reference at all, including a test line in its own member.
    any: Option<String>,
}

/// Test material in a shipping `src/` tree, and shipping edges to test support.
pub struct TestMaterialPlacement;

impl Gate for TestMaterialPlacement {
    fn name(&self) -> &'static str {
        "test-material-placement"
    }

    fn help(&self) -> &'static str {
        "Whether a publishable crate's src tree ships test material, and whether any non-dev dependency names the test support crate"
    }

    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let members = tree.member_manifests()?;
        let mut report = Report::clean();

        for member in &members {
            for table in SHIPPING_TABLES {
                if names_support(member.manifest.get(*table)) {
                    report.find(Finding::in_file(
                        format!("{}/Cargo.toml", member.path),
                        format!(
                            "[{table}] names `{SUPPORT_CRATE}`, so test support links into the published artifact"
                        ),
                        format!("move the edge to [dev-dependencies]; `{SUPPORT_CRATE}` is reached by a suite, never by product code"),
                    ));
                }
            }
            if let Some(targets) = member
                .manifest
                .get("target")
                .and_then(toml::Value::as_table)
            {
                for (triple, table) in targets {
                    for name in SHIPPING_TABLES {
                        if names_support(table.get(*name)) {
                            report.find(Finding::in_file(
                                format!("{}/Cargo.toml", member.path),
                                format!(
                                    "[target.{triple}.{name}] names `{SUPPORT_CRATE}`, so test support links into the published artifact"
                                ),
                                format!("move the edge to [dev-dependencies]; `{SUPPORT_CRATE}` is reached by a suite, never by product code"),
                            ));
                        }
                    }
                }
            }
        }

        let publishable: Vec<&Member> = members
            .iter()
            .filter(|member| member.publishable())
            .collect();
        let mut candidates = Vec::new();
        for member in &publishable {
            let defaults = default_features(member);
            let prefix = format!("{}/src/", member.path);
            for path in tree.paths() {
                let Some(file) = path.to_str() else { continue };
                if !file.starts_with(&prefix) || !file.ends_with(".rs") {
                    continue;
                }
                let Some(module) = module_name(file) else {
                    continue;
                };
                if !named_for_testing(&module) {
                    continue;
                }
                let text = tree.read(file)?;
                candidates.push(Candidate {
                    file: file.to_string(),
                    member: member.path.clone(),
                    crate_name: member.name.clone(),
                    exports: exports(&text),
                    gating: chain(&tree, &member.path, file, &defaults)?,
                    module,
                });
            }
        }

        let reach = references(&tree, &members, &publishable, &candidates)?;
        let nowhere = Reach::default();
        for (index, candidate) in candidates.iter().enumerate() {
            // A file no declaration reaches is not compiled at all, and naming it
            // here would report the same file as `source-reachability` under a
            // different rule.
            if !candidate.gating.declared || candidate.gating.test_only {
                continue;
            }
            let found = reach.get(&index).unwrap_or(&nowhere);
            if found.product.is_some() {
                continue;
            }
            if !candidate.gating.features.is_empty() && found.any.is_some() {
                continue;
            }
            let (message, fix) = if let Some(site) = &found.any {
                (
                    format!(
                        "`{}` is compiled into a default build of `{}` and only test code refers to it, first at {site}",
                        candidate.module, candidate.member
                    ),
                    format!(
                        "move it under the crate's `tests/` tree, or into `{SUPPORT_CRATE}` when a second crate's suite needs it; a shipping module a suite alone calls is carried by every consumer that never runs the suite"
                    ),
                )
            } else {
                (
                    format!(
                        "`{}` is compiled into a default build of `{}` and nothing in the checkout refers to it",
                        candidate.module, candidate.member
                    ),
                    "delete it; a module no caller names cannot be exercised, and it is published all the same".to_string(),
                )
            };
            report.find(Finding::in_file(&candidate.file, message, fix));
        }

        report.note(format!(
            "{} candidate file(s) across {} publishable member(s), {} compiled outside a test build",
            candidates.len(),
            publishable.len(),
            candidates
                .iter()
                .filter(|candidate| candidate.gating.declared && !candidate.gating.test_only)
                .count()
        ));
        Ok(report)
    }
}

/// Whether a dependency table names the test support crate.
fn names_support(table: Option<&toml::Value>) -> bool {
    table
        .and_then(toml::Value::as_table)
        .is_some_and(|table| table.contains_key(SUPPORT_CRATE))
}

/// The module identifier one source file declares, or `None` for a crate root.
fn module_name(file: &str) -> Option<String> {
    let (directory, name) = file.rsplit_once('/')?;
    let stem = name.strip_suffix(".rs")?;
    if stem == "lib" || stem == "main" {
        return None;
    }
    if stem == "mod" {
        let parent = directory
            .rsplit_once('/')
            .map_or(directory, |split| split.1);
        return (parent != "src").then(|| parent.to_string());
    }
    Some(stem.to_string())
}

/// Whether a module identifier reads as test material.
fn named_for_testing(module: &str) -> bool {
    module
        .split(['_', '-', '.'])
        .any(|segment| TOKENS.contains(&segment))
}

/// Every top-level identifier a module declares, which is what a caller names.
///
/// Visibility is not consulted. The question is whether anything refers to the
/// module at all, and a name that turns out to be unreachable from the file that
/// spells it is a compile error rather than a placement finding.
fn exports(text: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let Ok(parsed) = syn::parse_file(text) else {
        return found;
    };
    for item in parsed.items {
        let ident = match item {
            syn::Item::Fn(node) => Some(node.sig.ident),
            syn::Item::Struct(node) => Some(node.ident),
            syn::Item::Enum(node) => Some(node.ident),
            syn::Item::Union(node) => Some(node.ident),
            syn::Item::Trait(node) => Some(node.ident),
            syn::Item::TraitAlias(node) => Some(node.ident),
            syn::Item::Type(node) => Some(node.ident),
            syn::Item::Const(node) => Some(node.ident),
            syn::Item::Static(node) => Some(node.ident),
            syn::Item::Mod(node) => Some(node.ident),
            syn::Item::Macro(node) => node.ident,
            _ => None,
        };
        if let Some(ident) = ident {
            found.insert(ident.to_string());
        }
    }
    found
}

/// What every `mod` declaration between the crate root and one file says.
fn chain(
    tree: &Tree,
    member: &str,
    file: &str,
    defaults: &BTreeSet<String>,
) -> Result<Gating, GateError> {
    let mut gating = Gating {
        declared: true,
        ..Gating::default()
    };
    let relative = file
        .strip_prefix(&format!("{member}/src/"))
        .unwrap_or(file)
        .strip_suffix(".rs")
        .unwrap_or(file);
    let mut segments: Vec<&str> = relative.split('/').collect();
    if segments.last() == Some(&"mod") {
        segments.pop();
    }
    let mut parents: Vec<String> = vec![format!("{member}/src/lib.rs")];
    for depth in 1..segments.len() {
        let directory = segments[..depth].join("/");
        let inline = format!("{member}/src/{directory}.rs");
        let out_of_line = format!("{member}/src/{directory}/mod.rs");
        parents.push(if tree.has(&inline) {
            inline
        } else {
            out_of_line
        });
    }
    for (depth, parent) in parents.iter().enumerate() {
        let Some(name) = segments.get(depth) else {
            break;
        };
        if !tree.has(parent) {
            gating.declared = false;
            break;
        }
        let text = tree.read(parent)?;
        let Some(attributes) = declaration(&text, name) else {
            gating.declared = false;
            break;
        };
        if is_test_only_attribute(&attributes) {
            gating.test_only = true;
        }
        for feature in features(&attributes) {
            if !defaults.contains(&feature) {
                gating.features.insert(feature);
            }
        }
    }
    Ok(gating)
}

/// The attribute text above one out-of-line `mod` declaration, joined into one
/// line, or `None` when the parent declares no such module.
///
/// Joining is what reads a multi-line `#[cfg(any(...))]`, which is the shape
/// `vyre-libs` gates most of its modules with; a per-line predicate sees only
/// `#[cfg(any(` and concludes the module is unconditional.
fn declaration(text: &str, module: &str) -> Option<String> {
    let wanted = format!("mod {module};");
    let mut attributes = String::new();
    let mut depth = 0i32;
    for line in text.lines() {
        let code = scan_code(line).code.trim();
        if depth > 0 {
            attributes.push(' ');
            attributes.push_str(code);
            depth += bracket_delta(code);
            continue;
        }
        if declares(code, &wanted) {
            return Some(attributes);
        }
        if code.starts_with("#[") {
            depth = bracket_delta(code);
            if depth > 0 {
                attributes.clear();
                attributes.push_str(code);
            } else if code.starts_with("#[cfg") {
                attributes.clear();
                attributes.push_str(code);
            }
            continue;
        }
        if !code.is_empty() {
            attributes.clear();
        }
    }
    None
}

/// Whether one line of code is the wanted `mod` declaration, whatever its
/// visibility.
fn declares(code: &str, wanted: &str) -> bool {
    match code.strip_suffix(wanted) {
        None => false,
        Some(prefix) => {
            let prefix = prefix.trim();
            prefix.is_empty() || prefix == "pub" || prefix.starts_with("pub(")
        }
    }
}

/// Whether one line of code declares a module at all, whatever its visibility.
fn is_mod_declaration(code: &str) -> bool {
    let rest = code.trim();
    let rest = rest.strip_prefix("pub").map_or(rest, str::trim_start);
    let rest = if rest.starts_with('(') {
        rest.split_once(')')
            .map_or(rest, |(_, tail)| tail.trim_start())
    } else {
        rest
    };
    rest.starts_with("mod ")
}

/// `(` and `[` minus `)` and `]`, which is how far an attribute is still open.
fn bracket_delta(code: &str) -> i32 {
    let opens = i32::try_from(code.matches(['(', '[']).count()).unwrap_or(0);
    let closes = i32::try_from(code.matches([')', ']']).count()).unwrap_or(0);
    opens - closes
}

/// Every feature a cfg attribute names.
fn features(attributes: &str) -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let mut rest = attributes;
    while let Some(at) = rest.find("feature") {
        rest = &rest[at + "feature".len()..];
        let Some(open) = rest.find('"') else { break };
        let after = &rest[open + 1..];
        let Some(close) = after.find('"') else { break };
        found.insert(after[..close].to_string());
        rest = &after[close + 1..];
    }
    found
}

/// The default feature closure one member declares.
fn default_features(member: &Member) -> BTreeSet<String> {
    let table = member
        .manifest
        .get("features")
        .and_then(toml::Value::as_table);
    let Some(table) = table else {
        return BTreeSet::new();
    };
    let mut closure = BTreeSet::new();
    let mut pending = vec!["default".to_string()];
    while let Some(name) = pending.pop() {
        if !closure.insert(name.clone()) {
            continue;
        }
        let Some(entries) = table.get(&name).and_then(toml::Value::as_array) else {
            continue;
        };
        for entry in entries.iter().filter_map(toml::Value::as_str) {
            let entry = entry.strip_prefix("dep:").unwrap_or(entry);
            if let Some((_, feature)) = entry.split_once('/') {
                pending.push(feature.to_string());
            } else {
                pending.push(entry.to_string());
            }
        }
    }
    closure.remove("default");
    closure
}

/// Where each candidate is referenced from, in one pass over the checkout.
///
/// A file counts as a referrer only when the crate that carries it can name the
/// candidate's crate: itself, or a member that declares an edge to it. Matching
/// on the identifier alone made any occurrence of the word a reference, so this
/// gate's own unit tests, which spell a candidate module name in an assertion,
/// were read as the caller keeping a dead `vyre-libs` module alive.
fn references(
    tree: &Tree,
    members: &[Member],
    publishable: &[&Member],
    candidates: &[Candidate],
) -> Result<BTreeMap<usize, Reach>, GateError> {
    let mut wanted: BTreeMap<&str, Vec<(usize, bool)>> = BTreeMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        wanted
            .entry(candidate.module.as_str())
            .or_default()
            .push((index, true));
        for export in &candidate.exports {
            wanted
                .entry(export.as_str())
                .or_default()
                .push((index, false));
        }
    }
    let shipping: Vec<String> = publishable
        .iter()
        .map(|member| format!("{}/src/", member.path))
        .collect();

    let mut found: BTreeMap<usize, Reach> = BTreeMap::new();
    for path in tree.paths() {
        let Some(file) = path.to_str() else { continue };
        if !file.ends_with(".rs") {
            continue;
        }
        if candidates.iter().any(|candidate| candidate.file == file) {
            continue;
        }
        let Some(owner) = owner(members, file) else {
            continue;
        };
        let reachable = reachable_crates(owner);
        let visible: Vec<bool> = candidates
            .iter()
            .map(|candidate| reachable.contains(&candidate.crate_name))
            .collect();
        if !visible.iter().any(|seen| *seen) {
            continue;
        }
        let text = tree.read(file)?;
        let lines: Vec<&str> = text.lines().collect();
        let test_only = cfg_test_lines(&lines);
        let in_shipping_src = shipping.iter().any(|prefix| file.starts_with(prefix));
        // A `mod` declaration is how a module is wired, not evidence that
        // anything calls it. A `pub use` is different: in a publishable crate it
        // is published API, recorded in the committed snapshot and judged by the
        // public-API gates. Convicting on it would make this rule decide what a
        // crate may publish, and it would convict `vyre-spec` for exporting one
        // of its three test-vector types out of a file whose name happens to
        // carry a token.
        let codes: Vec<&str> = lines
            .iter()
            .map(|line| {
                let code = scan_code(line).code;
                if is_mod_declaration(code.trim()) {
                    ""
                } else {
                    code
                }
            })
            .collect();
        let words: BTreeSet<&str> = codes.iter().flat_map(|code| identifiers(code)).collect();
        for (number, code) in codes.iter().enumerate() {
            for word in identifiers(code) {
                let Some(entries) = wanted.get(word) else {
                    continue;
                };
                for (index, is_module) in entries.iter().filter(|(index, _)| visible[*index]) {
                    // An exported name counts only where the module itself is
                    // named, because that is what an import or a path through it
                    // spells. Counting the name alone made any dependent crate's
                    // own `SAMPLE` a reference, which excuses the material
                    // instead of convicting it.
                    if !is_module && !words.contains(candidates[*index].module.as_str()) {
                        continue;
                    }
                    let site = format!("{file}:{}", number + 1);
                    let entry = found.entry(*index).or_default();
                    if entry.any.is_none() {
                        entry.any = Some(site.clone());
                    }
                    if entry.product.is_none() && in_shipping_src && !test_only[number] {
                        entry.product = Some(site);
                    }
                }
            }
        }
    }
    Ok(found)
}

/// The member that carries one file, which is the deepest member path above it.
fn owner<'a>(members: &'a [Member], file: &str) -> Option<&'a Member> {
    members
        .iter()
        .filter(|member| file.starts_with(&format!("{}/", member.path)))
        .max_by_key(|member| member.path.len())
}

/// Every workspace crate one member's code may name: itself and its edges.
///
/// A dev edge counts, because a suite in another crate reaching this module is
/// exactly the case the rule is about, and it is still not product.
fn reachable_crates(member: &Member) -> BTreeSet<String> {
    let mut names = BTreeSet::from([member.name.clone()]);
    let mut tables: Vec<&toml::Value> = DEPENDENCY_TABLES
        .iter()
        .filter_map(|table| member.manifest.get(*table))
        .collect();
    if let Some(targets) = member
        .manifest
        .get("target")
        .and_then(toml::Value::as_table)
    {
        for platform in targets.values() {
            tables.extend(
                DEPENDENCY_TABLES
                    .iter()
                    .filter_map(|table| platform.get(*table)),
            );
        }
    }
    for table in tables {
        if let Some(entries) = table.as_table() {
            names.extend(entries.keys().cloned());
            names.extend(
                entries
                    .values()
                    .filter_map(|entry| entry.get("package"))
                    .filter_map(toml::Value::as_str)
                    .map(str::to_string),
            );
        }
    }
    names
}

/// Every Rust identifier in one line of code.
fn identifiers(code: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let bytes = code.as_bytes();
    let mut start = None;
    for index in 0..=bytes.len() {
        let word =
            index < bytes.len() && (bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_');
        match (word, start) {
            (true, None) => start = Some(index),
            (false, Some(from)) => {
                found.push(&code[from..index]);
                start = None;
            }
            _ => {}
        }
    }
    found
}

/// WHY: the readers below decide the verdict and none is reachable from an
/// integration test, because the gate exposes one report over one tree and that
/// tree contains no instance of most of the shapes. The stem filter is the one
/// that has already been wrong twice: a substring match convicts
/// `bellman_shortest_path.rs` on the word inside `shortest`, and a per-line
/// attribute read calls a module gated by a multi-line `cfg(any(...))`
/// unconditional.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stem_is_test_material_by_segment_and_never_by_substring() {
        assert!(named_for_testing("test_parity_oracles"));
        assert!(named_for_testing("golden_sample"));
        assert!(named_for_testing("fixtures"));
        assert!(!named_for_testing("bellman_shortest_path"));
        assert!(!named_for_testing("latest"));
        assert!(!named_for_testing("sampler"));
    }

    #[test]
    fn a_module_identifier_comes_from_the_directory_for_a_mod_file() {
        assert_eq!(
            module_name("vyre-libs/src/test_parity_oracles.rs").as_deref(),
            Some("test_parity_oracles")
        );
        assert_eq!(
            module_name("vyre-driver/src/parity_harness/mod.rs").as_deref(),
            Some("parity_harness")
        );
        assert_eq!(module_name("vyre-libs/src/lib.rs"), None);
        assert_eq!(module_name("vyre-libs/src/mod.rs"), None);
    }

    #[test]
    fn a_multi_line_cfg_above_a_declaration_is_read_whole() {
        let text =
            "#[cfg(any(\n    feature = \"graph\",\n    feature = \"nn\"\n))]\npub mod fixtures;\n";
        let attributes = declaration(text, "fixtures").expect("the declaration is found");
        assert!(
            attributes.contains("feature = \"graph\""),
            "got {attributes}"
        );
        assert!(!is_test_only_attribute(&attributes), "got {attributes}");
        assert_eq!(
            features(&attributes),
            BTreeSet::from(["graph".to_string(), "nn".to_string()])
        );
    }

    #[test]
    fn a_test_only_declaration_is_told_from_a_feature_gated_one() {
        let test_only = declaration("#[cfg(test)]\nmod fixtures;\n", "fixtures")
            .expect("the declaration is found");
        assert!(is_test_only_attribute(&test_only));
        let optional = declaration(
            "#[cfg(any(test, feature = \"test-fixtures\"))]\npub mod fixtures;\n",
            "fixtures",
        )
        .expect("the declaration is found");
        assert!(!is_test_only_attribute(&optional));
        assert_eq!(
            features(&optional),
            BTreeSet::from(["test-fixtures".to_string()])
        );
    }

    #[test]
    fn an_undeclared_module_is_reported_as_undeclared_rather_than_unconditional() {
        assert!(declaration("pub mod other;\n", "fixtures").is_none());
    }

    #[test]
    fn every_top_level_item_name_counts_whatever_its_visibility() {
        let found = exports(
            "pub fn shown() {}\nfn hidden() {}\npub struct Shown;\npub(crate) const N: usize = 1;\nimpl Shown {}\n",
        );
        assert_eq!(
            found.iter().map(String::as_str).collect::<Vec<_>>(),
            vec!["N", "Shown", "hidden", "shown"]
        );
    }

    #[test]
    fn identifiers_are_split_at_every_non_word_byte() {
        assert_eq!(
            identifiers("use crate::fixture_bytes::pack_u32(x);"),
            vec!["use", "crate", "fixture_bytes", "pack_u32", "x"]
        );
    }

    #[test]
    fn a_declaration_is_not_a_reference_under_any_visibility() {
        assert!(is_mod_declaration("mod fixture_bytes;"));
        assert!(is_mod_declaration("pub mod test_parity_oracles;"));
        assert!(is_mod_declaration("pub(crate) mod fixture_bytes;"));
        assert!(is_mod_declaration("pub(super) mod fixtures;"));
        assert!(!is_mod_declaration("pub use fixture_bytes::pack_u32;"));
        assert!(!is_mod_declaration("let bytes = fixture_bytes::all();"));
    }

    /// WHY: the reference index matches an identifier, so before this filter any
    /// crate that spelled the word counted as a caller. This gate's own unit
    /// tests name candidate modules, which made a dead `vyre-libs` module look
    /// called from `xtask`. A crate that declares no edge to another crate
    /// cannot name anything inside it, whatever words its source contains.
    #[test]
    fn only_a_crate_that_declares_an_edge_can_be_a_referrer() {
        let root = crate::checkout::checkout_root();
        let tree = Tree::open(&root).expect("Fix: the checkout must be listable");
        let members = tree
            .member_manifests()
            .expect("Fix: every member manifest must parse");
        let named = |path: &str| {
            let member = owner(&members, &format!("{path}/src/lib.rs"))
                .expect("Fix: the member must own its own source");
            reachable_crates(member)
        };
        let from_xtask = named("xtask");
        assert!(from_xtask.contains("xtask"), "a crate names itself");
        assert!(
            !from_xtask.contains("vyre-libs"),
            "xtask declares no vyre edge: {from_xtask:?}"
        );
        assert!(named("vyre-libs").contains("vyre-foundation"));
        assert_eq!(
            owner(&members, "release/changes/unreleased/x.toml").map(|member| member.path.as_str()),
            None,
            "a file under no member has no owner"
        );
    }
}
