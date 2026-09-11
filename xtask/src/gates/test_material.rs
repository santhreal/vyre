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
//! segment out of `TOKENS` is what makes a file worth reading, and nothing more
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
//!
//! The third rule is the source counterpart of the second. No line a release
//! build compiles, in a publishable crate's `src/`, may name the test support
//! crate. The manifest rule alone cannot see this: a crate whose only edge is
//! the dev one still compiles a production `use` of that crate in its own test
//! build, so the leak reaches every suite and is reported nowhere. An operation
//! registration carrying its fixture bytes is product code, compiled into the
//! artifact, so the byte encoder it calls has to be the production one.
//! Registrations in three `vyre-libs` crates reached the test crate for a
//! little-endian pack `vyre-primitives` already owned.

use std::collections::{BTreeMap, BTreeSet};

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::{
    cfg_test_lines, is_test_only_attribute, scan_code, Code, CodeCursor, Member, Tree,
};

/// Stem segments that make a file test material by name.
const TOKENS: &[&str] = &[
    "test", "tests", "fixture", "fixtures", "oracle", "oracles", "mock", "mocks", "stub", "stubs",
    "sample", "samples", "golden", "harness",
];

/// The crate whose whole subject is test support.
const SUPPORT_CRATE: &str = "vyre-test-support";

/// How a source line spells that crate. A manifest names the package and a
/// `use` names the crate, so the two rules match different strings.
const SUPPORT_IDENT: &str = "vyre_test_support";

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

/// Test material in a shipping `src/` tree, shipping edges to test support, and
/// production lines that name it.
pub struct TestMaterialPlacement;

impl crate::gate::GateBehavior for TestMaterialPlacement {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let members = tree.member_manifests()?;
        let mut report = Report::clean();
        report.cover_complete("workspace members", members.len());

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

        for member in publishable
            .iter()
            .filter(|member| member.name != SUPPORT_CRATE)
        {
            let defaults = default_features(member);
            let named = path_declarations(&tree, &member.path)?;
            let prefix = format!("{}/src/", member.path);
            for path in tree.paths() {
                let Some(file) = path.to_str() else { continue };
                if !file.starts_with(&prefix) || !file.ends_with(".rs") {
                    continue;
                }
                if chain(&tree, &member.path, file, &defaults, &named)?.test_only {
                    continue;
                }
                for line in support_references(&tree.read(file)?) {
                    report.find(Finding::at(
                        file,
                        line,
                        format!(
                            "names `{SUPPORT_IDENT}` outside a test build, so test support is product code in `{}`",
                            member.name
                        ),
                        format!("call the production owner instead; the edge to `{SUPPORT_CRATE}` is a dev edge, so product code cannot rely on it being linked"),
                    ));
                }
            }
        }
        let mut candidates = Vec::new();
        for member in &publishable {
            let defaults = default_features(member);
            let named = path_declarations(&tree, &member.path)?;
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
                    gating: chain(&tree, &member.path, file, &defaults, &named)?,
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

/// One out-of-line module declaration that names its own file.
///
/// A file reached through `#[path]` is a child of the module that declares it
/// rather than of the directory holding it, so the layout walk in `chain`
/// cannot reach it.
#[derive(Debug)]
struct PathDeclaration {
    /// The file carrying the `mod` item.
    parent: String,
    /// The attribute text above that item, joined into one line.
    attributes: String,
}

/// Every file in one member's `src/` tree that a `#[path]` attribute names.
///
/// The checkout carries 1621 of these declarations, and the suites written
/// `#[cfg(test)] #[path = "<name>_tests.rs"] mod tests;` are why they are read.
/// The layout walk looks for `mod <stem>;` in the parent the directory implies,
/// finds nothing, and reads that as a file no declaration reaches; the support
/// rule then reported five test-only suites as production source naming the
/// test support crate.
fn path_declarations(
    tree: &Tree,
    member: &str,
) -> Result<BTreeMap<String, PathDeclaration>, GateError> {
    let prefix = format!("{member}/src/");
    let mut named = BTreeMap::new();
    for path in tree.paths() {
        let Some(parent) = path.to_str() else {
            continue;
        };
        if !parent.starts_with(&prefix) || !parent.ends_with(".rs") {
            continue;
        }
        let directory = parent.rsplit_once('/').map_or("", |(head, _)| head);
        for declaration in mod_declarations(&tree.read(parent)?) {
            let Some(value) = declaration.path else {
                continue;
            };
            let Some(target) = joined(directory, &value) else {
                continue;
            };
            named.insert(
                target,
                PathDeclaration {
                    parent: parent.to_string(),
                    attributes: declaration.attributes,
                },
            );
        }
    }
    Ok(named)
}

/// A repository-relative path for a `#[path]` value read from `directory`.
///
/// The attribute resolves against the directory holding the file that carries
/// the `mod` item. Some declarations climb out of that directory with `..`, so
/// the segments are folded rather than concatenated.
fn joined(directory: &str, value: &str) -> Option<String> {
    let mut segments: Vec<&str> = directory
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    for part in value.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                segments.pop()?;
            }
            other => segments.push(other),
        }
    }
    (!segments.is_empty()).then(|| segments.join("/"))
}

/// Fold one declaration's attributes into the reading so far.
fn absorb(gating: &mut Gating, attributes: &str, defaults: &BTreeSet<String>) {
    if is_test_only_attribute(attributes) {
        gating.test_only = true;
    }
    for feature in features(attributes) {
        if !defaults.contains(&feature) {
            gating.features.insert(feature);
        }
    }
}

/// What every `mod` declaration above one file says about it.
///
/// `#[path]` links are followed first, up to the file that a directory segment
/// names, and every attribute along the way is folded in. The walk then runs
/// over that file's own segments. A cycle is a compile error rather than a tree
/// this gate has to survive, and the visited set keeps a malformed one bounded.
fn chain(
    tree: &Tree,
    member: &str,
    file: &str,
    defaults: &BTreeSet<String>,
    named: &BTreeMap<String, PathDeclaration>,
) -> Result<Gating, GateError> {
    let mut gating = Gating {
        declared: true,
        ..Gating::default()
    };
    let mut declaring = file.to_string();
    let mut seen = BTreeSet::new();
    while let Some(declaration) = named.get(&declaring) {
        if !seen.insert(declaring.clone()) {
            break;
        }
        absorb(&mut gating, &declaration.attributes, defaults);
        declaring = declaration.parent.clone();
    }
    let relative = declaring
        .strip_prefix(&format!("{member}/src/"))
        .unwrap_or(&declaring)
        .strip_suffix(".rs")
        .unwrap_or(&declaring);
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
        absorb(&mut gating, &attributes, defaults);
    }
    Ok(gating)
}

/// The attribute text above one out-of-line `mod` declaration, joined into one
/// line, or `None` when the parent declares no such module.
fn declaration(text: &str, module: &str) -> Option<String> {
    mod_declarations(text)
        .into_iter()
        .find(|declaration| declaration.module == module)
        .map(|declaration| declaration.attributes)
}

/// One out-of-line `mod` declaration and what sits above it.
#[derive(Debug)]
struct ModDeclaration {
    /// The identifier the declaration binds.
    module: String,
    /// The cfg attribute text above it, joined into one line.
    attributes: String,
    /// The file a `#[path]` attribute names, as the attribute spells it.
    path: Option<String>,
}

/// Every out-of-line `mod` declaration in one file, in source order.
///
/// Attributes are joined across lines, which is what reads a multi-line
/// `#[cfg(any(...))]`; a per-line predicate sees only `#[cfg(any(` and concludes
/// the module is unconditional. A `#[path]` attribute is recorded beside the
/// cfg text rather than replacing it, so a declaration carrying both is read as
/// gated and relocated at once.
fn mod_declarations(text: &str) -> Vec<ModDeclaration> {
    let mut found = Vec::new();
    let mut attributes = String::new();
    let mut path: Option<String> = None;
    let mut depth = 0i32;
    for line in text.lines() {
        let code = scan_code(line).code.trim();
        if depth > 0 {
            attributes.push(' ');
            attributes.push_str(code);
            depth += bracket_delta(code);
            continue;
        }
        if let Some(module) = declared_module(code) {
            found.push(ModDeclaration {
                module,
                attributes: std::mem::take(&mut attributes),
                path: path.take(),
            });
            continue;
        }
        if code.starts_with("#[") {
            depth = bracket_delta(code);
            if depth > 0 {
                attributes.clear();
                attributes.push_str(code);
            } else if let Some(named) = path_attribute(code) {
                path = Some(named);
            } else if code.starts_with("#[cfg") {
                attributes.clear();
                attributes.push_str(code);
            }
            continue;
        }
        if !code.is_empty() {
            attributes.clear();
            path = None;
        }
    }
    found
}

/// The identifier one line declares as an out-of-line module, whatever its
/// visibility, or `None` for anything else.
fn declared_module(code: &str) -> Option<String> {
    let rest = code.trim();
    let rest = rest.strip_prefix("pub").map_or(rest, str::trim_start);
    let rest = if rest.starts_with('(') {
        rest.split_once(')')
            .map_or(rest, |(_, tail)| tail.trim_start())
    } else {
        rest
    };
    let name = rest.strip_prefix("mod ")?.trim().strip_suffix(';')?.trim();
    let named = !name.is_empty()
        && name
            .chars()
            .all(|letter| letter.is_alphanumeric() || letter == '_');
    named.then(|| name.to_string())
}

/// The file a `#[path]` attribute names, if the line is one.
fn path_attribute(code: &str) -> Option<String> {
    let inner = code.strip_prefix("#[")?.strip_suffix(']')?.trim();
    let value = inner
        .strip_prefix("path")?
        .trim_start()
        .strip_prefix('=')?
        .trim();
    let value = value.strip_prefix('"')?.strip_suffix('"')?;
    (!value.is_empty()).then(|| value.to_string())
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
    attributes
        .split("feature")
        .filter_map(|segment| {
            let after = segment.trim_start();
            let after = after.strip_prefix('=')?.trim_start();
            let name = after.strip_prefix('"')?;
            let close = name.find('"')?;
            Some(name[..close].to_string())
        })
        .collect()
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
    code.split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
        .filter(|segment| !segment.is_empty())
        .collect()
}

/// Every line of one file that names the test support crate where a release
/// build compiles it.
///
/// The walk is over code spans, so a doc comment and a string literal that
/// spell the crate name are not references. Reading the line's text instead
/// convicted a diagnostic that quoted the crate name, which is the defect
/// `CodeCursor` owns: an opaque span is taken whole and its newlines still
/// advance the line, so a multi-line literal does not shift every number after
/// it. A `#[cfg(test)]` block is where a suite is meant to reach the crate, so
/// a line inside one does not count.
fn support_references(text: &str) -> Vec<u32> {
    let lines: Vec<&str> = text.lines().collect();
    let test_only = cfg_test_lines(&lines);
    let mut code: Vec<String> = vec![String::new(); lines.len()];
    let mut cursor = CodeCursor::new(text);
    let mut line = 0usize;
    while let Some((offset, span)) = cursor.step() {
        match span {
            Code::Opaque(taken) => {
                line += taken.bytes().filter(|byte| *byte == b'\n').count();
                cursor.seek(offset + taken.len());
            }
            Code::Byte(byte) => {
                if byte == b'\n' {
                    line += 1;
                } else if let Some(row) = code.get_mut(line) {
                    row.push(char::from(byte));
                }
                cursor.seek(offset + 1);
            }
        }
    }
    code.iter()
        .enumerate()
        .filter(|(number, _)| !test_only.get(*number).copied().unwrap_or(false))
        .filter(|(_, row)| identifiers(row).contains(&SUPPORT_IDENT))
        .map(|(number, _)| u32::try_from(number).unwrap_or(u32::MAX).saturating_add(1))
        .collect()
}

/// WHY: the readers below decide the verdict and none is reachable from an
/// integration test, because the gate exposes one report over one tree and that
/// tree contains no instance of most of the shapes. The stem filter is the one
/// that has already been wrong twice: a substring match convicts
/// `bellman_shortest_path.rs` on the word inside `shortest`, and a per-line
/// attribute read calls a module gated by a multi-line `cfg(any(...))`
/// unconditional. The third is the `#[path]` link: the layout walk alone reads
/// a relocated module as declared nowhere, and the support rule then convicts a
/// suite a release build never compiles.
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
    fn a_support_reference_counts_only_where_a_release_build_compiles_it() {
        let text = concat!(
            "use vyre_test_support::test_parity_oracles::u32_bytes;\n",
            "//! vyre_test_support is named in prose here.\n",
            "const NAME: &str = \"vyre_test_support\";\n",
            "#[cfg(test)]\n",
            "mod tests {\n",
            "    use vyre_test_support::test_parity_oracles::f32_bytes;\n",
            "}\n",
        );
        assert_eq!(support_references(text), vec![1]);
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

    /// WHY: `#[path]` moves a module out of the position the directory implies.
    /// The layout walk alone finds no `mod <stem>;` for the file, reads that as
    /// a module no declaration reaches, and the support rule convicts a
    /// `#[cfg(test)]` suite as production source. Both attribute orders ship in
    /// the checkout, so both are read here.
    #[test]
    fn a_relocated_declaration_carries_both_its_cfg_and_its_file() {
        let cfg_first =
            mod_declarations("#[cfg(test)]\n#[path = \"target_tests.rs\"]\nmod tests;\n");
        assert_eq!(cfg_first.len(), 1, "{cfg_first:?}");
        assert_eq!(cfg_first[0].module, "tests");
        assert_eq!(cfg_first[0].path.as_deref(), Some("target_tests.rs"));
        assert!(is_test_only_attribute(&cfg_first[0].attributes));

        let path_first = mod_declarations(
            "#[path = \"materialize_tests.rs\"]\n#[cfg(test)]\nmod materialize_tests;\n",
        );
        assert_eq!(path_first.len(), 1, "{path_first:?}");
        assert_eq!(path_first[0].module, "materialize_tests");
        assert_eq!(path_first[0].path.as_deref(), Some("materialize_tests.rs"));
        assert!(is_test_only_attribute(&path_first[0].attributes));
    }

    /// WHY: a declaration with no `#[path]` must stay unrelocated. Recording a
    /// stale value on the next declaration would move an unrelated module.
    #[test]
    fn a_plain_declaration_names_no_file_and_does_not_inherit_one() {
        let found = mod_declarations("#[path = \"a_tests.rs\"]\nmod a;\nmod b;\n");
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!(found[0].path.as_deref(), Some("a_tests.rs"));
        assert_eq!(found[1].path, None);
    }

    /// WHY: some declarations climb out of the directory holding them, so the
    /// value is folded against it rather than appended to it.
    #[test]
    fn a_relocated_file_resolves_against_the_directory_that_declares_it() {
        assert_eq!(
            joined("vyre-megakernel/src", "target_tests.rs").as_deref(),
            Some("vyre-megakernel/src/target_tests.rs")
        );
        assert_eq!(
            joined("a/src/one/two", "../../../tests/internal/mod.rs").as_deref(),
            Some("a/tests/internal/mod.rs")
        );
        assert_eq!(joined("a", "../../out.rs"), None);
    }

    /// WHY: the variant space is every relocated declaration the checkout
    /// carries, enumerated from the tree rather than named here, so a suite
    /// added under a new `#[path]` is covered the day it lands. Before the link
    /// was followed every one of these read as compiled by a release build.
    #[test]
    fn every_test_only_relocated_module_in_the_checkout_reads_as_test_only() {
        let root = crate::checkout::checkout_root();
        let tree = Tree::open(&root).expect("Fix: the checkout must be listable");
        let members = tree
            .member_manifests()
            .expect("Fix: every member manifest must parse");
        let mut judged = 0usize;
        let mut unread = Vec::new();
        for member in members.iter().filter(|member| member.publishable()) {
            let defaults = default_features(member);
            let named =
                path_declarations(&tree, &member.path).expect("Fix: every source must be readable");
            for (file, declaration) in &named {
                if !is_test_only_attribute(&declaration.attributes) {
                    continue;
                }
                judged += 1;
                let gating = chain(&tree, &member.path, file, &defaults, &named)
                    .expect("Fix: every parent source must be readable");
                if !gating.test_only {
                    unread.push(format!("{file} declared by {}", declaration.parent));
                }
            }
        }
        assert!(
            judged > 0,
            "Fix: the checkout must carry a `#[cfg(test)] #[path = ...]` module for this to judge"
        );
        assert!(
            unread.is_empty(),
            "{} of {judged} relocated test-only module(s) read as compiled by a release build:\n{}",
            unread.len(),
            unread.join("\n")
        );
    }
}
