//! Every item this crate publishes is reachable at one public path.
//!
//! WHY. A module declared `pub` whose parent also re-exports what it holds
//! publishes each of those items twice. Both paths compile and both render in
//! the documentation, so a reader cannot tell a moved item from a renamed one,
//! an import picks whichever path its author saw first, and a deprecation
//! written at one path is a lie at the other. `schema_registry` was published
//! that way: `pub mod schema_registry` beside `pub use schema_registry::{..}`
//! at the crate root, eight items at two paths each.
//!
//! This reads the module tree from `vyre-spec/src` at run time rather than
//! naming the modules, so a module added tomorrow is judged on arrival. The
//! judged property is structural: a `pub use` in a publicly reachable module
//! whose leading path resolves to another publicly reachable module of this
//! crate is a second path for everything that statement carries. A private
//! module re-exported by its parent is the shape this crate wants, and it is
//! not a finding: the re-export is then the only path.
//!
//! What it does not catch: a glob re-export, a prelude, and one private module
//! re-exported by two separate public modules. Those need the rendered surface
//! rather than the declarations, which is what the `public-api-paths` gate
//! measures over the committed `docs/public-api` snapshots. This contract is
//! the half that holds without a snapshot, so it fails the instant the second
//! path is written instead of when the snapshot is next refreshed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vyre_test_support::monorepo::vyre_workspace_root;

/// Crate directory whose module tree this contract judges.
const CRATE_DIR: &str = "vyre-spec";

/// Below this many modules the walk has stopped reading the tree, and an empty
/// offender set would prove nothing. This crate carries over fifty.
const MINIMUM_MODULES: usize = 40;

/// Publicly reachable modules the walk must find, for the same reason.
const MINIMUM_PUBLIC_MODULES: usize = 5;

/// `pub use` statements the crate root must carry, for the same reason: the
/// root re-exports its private split files and there are hundreds of items in
/// those statements.
const MINIMUM_ROOT_REEXPORTS: usize = 30;

/// One module of the crate, as its source file declares it.
#[derive(Debug)]
struct Module {
    /// Path relative to the crate root, empty for the root itself.
    path: Vec<String>,
    /// Whether every declaration from the root down to this one is `pub`.
    public: bool,
    /// Root-relative source file, for a failure that names where to look.
    file: String,
    /// Leading module path of every `pub use` the file states, as segments.
    reexports: Vec<Vec<String>>,
}

#[test]
fn no_item_is_published_at_two_paths() {
    let modules = module_tree(&vyre_workspace_root());
    let known: BTreeSet<&[String]> = modules
        .iter()
        .map(|module| module.path.as_slice())
        .collect();
    let public: BTreeSet<&[String]> = modules
        .iter()
        .filter(|module| module.public)
        .map(|module| module.path.as_slice())
        .collect();

    let mut offenders: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for module in modules.iter().filter(|module| module.public) {
        for target in &module.reexports {
            let Some(source) = resolve(&module.path, target, &known) else {
                continue;
            };
            if source == module.path || !public.contains(source.as_slice()) {
                continue;
            }
            offenders.entry(render(&source)).or_default().push(format!(
                "{}: pub use {}",
                module.file,
                target.join("::")
            ));
        }
    }

    assert_eq!(
        offenders,
        BTreeMap::new(),
        "Fix: these modules are `pub` and another public module of the same crate \
         also re-exports what they hold, so every one of those items has two \
         public paths. Drop `pub` from the module and keep the re-export, or keep \
         the module public and delete the re-export, migrating every caller in \
         the same change. Offenders: {offenders:?}"
    );
}

#[test]
fn the_walk_actually_reaches_the_crate() {
    let modules = module_tree(&vyre_workspace_root());
    let public = modules.iter().filter(|module| module.public).count();
    let root_reexports = modules
        .iter()
        .find(|module| module.path.is_empty())
        .map(|module| module.reexports.len())
        .unwrap_or_default();

    assert!(
        modules.len() >= MINIMUM_MODULES
            && public >= MINIMUM_PUBLIC_MODULES
            && root_reexports >= MINIMUM_ROOT_REEXPORTS,
        "Fix: the walk found {} module(s), {public} of them publicly reachable, \
         and {root_reexports} `pub use` statement(s) at the crate root, below the \
         floor of {MINIMUM_MODULES}/{MINIMUM_PUBLIC_MODULES}/\
         {MINIMUM_ROOT_REEXPORTS}. The walk is not reading `{CRATE_DIR}/src`, so \
         an empty offender set proves nothing.",
        modules.len()
    );
}

#[test]
fn the_scanner_reads_module_scope_and_ignores_bodies_and_comments() {
    let source = r##"
pub mod public_child;
pub(crate) mod crate_child;
mod private_child;
// pub mod commented_child;
/* pub mod block_commented_child;
pub use block_commented::Item; */
pub use private_child::Held;
pub use crate::public_child::Other as Renamed;
pub use self::crate_child::{First, Second};
fn body() {
    pub mod nested_in_body;
    pub use nested_in_body::Thing;
}
"##;

    assert_eq!(
        declared_modules(source),
        vec![
            ("public_child".to_owned(), true),
            ("crate_child".to_owned(), false),
            ("private_child".to_owned(), false),
        ]
    );
    assert_eq!(
        declared_reexports(source),
        vec![
            vec!["private_child".to_owned(), "Held".to_owned()],
            vec![
                "crate".to_owned(),
                "public_child".to_owned(),
                "Other".to_owned()
            ],
            vec!["self".to_owned(), "crate_child".to_owned()],
        ]
    );
}

#[test]
fn a_re_export_resolves_to_the_longest_module_prefix_it_names() {
    let known: BTreeSet<&[String]> = BTreeSet::new();
    let root: Vec<String> = Vec::new();
    assert_eq!(resolve(&root, &segments("super::anything"), &known), None);

    let inner = vec!["inner".to_owned()];
    let leaf = vec!["inner".to_owned(), "leaf".to_owned()];
    let known: BTreeSet<&[String]> = [root.as_slice(), inner.as_slice(), leaf.as_slice()]
        .into_iter()
        .collect();

    assert_eq!(
        resolve(&root, &segments("inner::leaf::Item"), &known),
        Some(leaf.clone())
    );
    assert_eq!(
        resolve(&root, &segments("crate::inner"), &known),
        Some(inner.clone())
    );
    assert_eq!(
        resolve(&inner, &segments("self::leaf"), &known),
        Some(leaf.clone())
    );
    // `super::Item` names the parent module even though `Item` is not a module.
    assert_eq!(
        resolve(&leaf, &segments("super::Item"), &known),
        Some(inner.clone())
    );
    // A head no module answers to names another crate or an item already in
    // scope, and neither is a path of this crate.
    assert_eq!(resolve(&root, &segments("serde::Serialize"), &known), None);
    assert_eq!(resolve(&leaf, &segments("serde::Serialize"), &known), None);
    // The whole statement resolving to a module is the module alias shape.
    assert_eq!(resolve(&root, &segments("inner"), &known), Some(inner));
}


/// Every module of the crate, root first, in declaration order.
///
/// A module whose declaration carries `#[path]` or `#[cfg(test)]` resolves to no
/// sibling file. Such a module is skipped when it is not publicly reachable,
/// which is what `#[cfg(test)] mod tests` is, and panics when it is: a public
/// module the walk cannot open is public surface this contract would silently
/// stop judging.
fn module_tree(root: &Path) -> Vec<Module> {
    let crate_src = root.join(CRATE_DIR).join("src");
    let mut modules = Vec::new();
    let mut pending = vec![(Vec::new(), true, crate_src.join("lib.rs"))];
    while let Some((path, public, file)) = pending.pop() {
        let source = std::fs::read_to_string(&file)
            .unwrap_or_else(|error| panic!("Fix: cannot read `{}`: {error}", file.display()));
        let directory = file.parent().unwrap_or(crate_src.as_path()).to_path_buf();
        for (name, declared_public) in declared_modules(&source) {
            let child_public = public && declared_public;
            let mut child_path = path.clone();
            child_path.push(name.clone());
            let flat = directory.join(format!("{name}.rs"));
            let nested = directory.join(&name).join("mod.rs");
            let child_file = if flat.is_file() {
                flat
            } else if nested.is_file() {
                nested
            } else {
                assert!(
                    !child_public,
                    "Fix: `{}` declares `pub mod {name}` and neither `{}` nor `{}` \
                     exists, so this contract cannot read a module that is public \
                     surface. Declare it in a sibling file.",
                    file.display(),
                    flat.display(),
                    nested.display()
                );
                continue;
            };
            pending.push((child_path, child_public, child_file));
        }
        let relative = file
            .strip_prefix(root)
            .unwrap_or(&file)
            .to_string_lossy()
            .into_owned();
        modules.push(Module {
            path,
            public,
            file: relative,
            reexports: declared_reexports(&source),
        });
    }
    modules.sort_by(|left, right| left.path.cmp(&right.path));
    modules
}

/// The module path one `pub use` names, or `None` when it names no module of
/// this crate.
///
/// The longest prefix of the statement that is a known module wins, because the
/// trailing segments are the items: `a::b::Item` is the module `a::b`. A head of
/// `crate`, `self` or `super` navigates to a module before any segment is read,
/// so `super::Item` names the parent. A bare head does not: in this edition it
/// resolves to another crate or to a name already in scope, neither of which is
/// a path of this crate.
fn resolve(from: &[String], target: &[String], known: &BTreeSet<&[String]>) -> Option<Vec<String>> {
    let (mut current, rest, navigated) = match target.first()?.as_str() {
        "crate" => (Vec::new(), &target[1..], true),
        "self" => (from.to_vec(), &target[1..], true),
        "super" => {
            let (_, parent) = from.split_last()?;
            (parent.to_vec(), &target[1..], true)
        }
        _ => (from.to_vec(), target, false),
    };
    let mut found = (navigated && known.contains(current.as_slice())).then(|| current.clone());
    for segment in rest {
        current.push(segment.clone());
        if !known.contains(current.as_slice()) {
            break;
        }
        found = Some(current.clone());
    }
    found
}

/// A module path rendered as a consumer writes it.
fn render(path: &[String]) -> String {
    if path.is_empty() {
        "the crate root".to_owned()
    } else {
        path.join("::")
    }
}

/// Segments of one path, for the scanner tests.
fn segments(path: &str) -> Vec<String> {
    path.split("::")
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Module declarations at module scope, as name and whether the declaration is
/// `pub` without a restricting scope.
///
/// A declaration at module scope in this workspace starts at column zero: the
/// only indented `mod` is the `#[cfg(test)] mod tests` block every file carries,
/// and the `module-layout` gate holds that shape. Comments are stripped first so
/// a commented-out declaration is not read as one.
fn declared_modules(source: &str) -> Vec<(String, bool)> {
    let mut found = Vec::new();
    for line in module_scope_lines(source) {
        let public = line.starts_with("pub ");
        let rest = line
            .strip_prefix("pub ")
            .or_else(|| strip_restricted_pub(line))
            .unwrap_or(line);
        let Some(name) = rest.strip_prefix("mod ") else {
            continue;
        };
        let name = name.trim_end_matches(['{', ';']).trim();
        if !name.is_empty() && name.chars().all(|ch| ch.is_alphanumeric() || ch == '_') {
            found.push((name.to_owned(), public));
        }
    }
    found
}

/// Leading path segments of every `pub use` at module scope, item segments
/// included: the caller keeps the longest module prefix and discards the rest.
///
/// Only the opening line of a statement is read. A brace list spans lines and a
/// rename follows the path, so the module prefix is always on the first line.
fn declared_reexports(source: &str) -> Vec<Vec<String>> {
    module_scope_lines(source)
        .filter_map(|line| line.strip_prefix("pub use "))
        .map(|rest| {
            let head = rest.split_once('{').map_or(rest, |(before, _)| before);
            let head = head.split_once(" as ").map_or(head, |(before, _)| before);
            segments(head.trim_end_matches([';', ' ']))
        })
        .filter(|path: &Vec<String>| !path.is_empty())
        .collect()
}

/// Lines of `source` that start at column zero, with comments removed.
fn module_scope_lines(source: &str) -> impl Iterator<Item = &str> + '_ {
    strip_block_comments(source)
        .filter(|line| {
            !line.is_empty() && !line.starts_with(char::is_whitespace) && !line.starts_with("//")
        })
        .map(|line| line.trim_end())
}

/// Lines of `source` outside every `/* .. */` comment.
///
/// A block comment is dropped whole rather than blanked line by line, because a
/// declaration on the same line as the opening `/*` is commented out too.
fn strip_block_comments(source: &str) -> impl Iterator<Item = &str> + '_ {
    let mut inside = false;
    source.lines().filter_map(move |line| {
        if inside {
            if line.contains("*/") {
                inside = false;
            }
            return None;
        }
        match line.split_once("/*") {
            Some((before, after)) => {
                inside = !after.contains("*/");
                Some(before)
            }
            None => Some(line),
        }
    })
}

/// The remainder of a `pub(..) ` declaration, which publishes nothing outside
/// this crate and is therefore not a public path.
fn strip_restricted_pub(line: &str) -> Option<&str> {
    line.strip_prefix("pub(")?
        .split_once(") ")
        .map(|(_, rest)| rest)
}
