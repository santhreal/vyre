//! The published-module scan reads a facade's module re-exports.
//!
//! `vyre` and `vyre-libs` publish their domain modules with `pub use`, so a scan
//! that reads `pub mod` alone finds none of them and every published module whose
//! name the layout rules ban loses the exemption that keeps a consumer's import
//! path stable.
//!
//! Both expectations are read from the committed snapshots when the test runs. A
//! facade that republishes a different set of modules still has to appear, and no
//! name written here goes stale in silence.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use structure_gate::{scan, workspace_root};

/// Directory holding one public-API snapshot per publishable package.
const SNAPSHOT_DIR: &str = "docs/public-api";

/// Every line of every committed public-API snapshot.
fn snapshot_lines(root: &Path) -> Vec<String> {
    let directory = root.join(SNAPSHOT_DIR);
    let entries = fs::read_dir(&directory).unwrap_or_else(|error| {
        panic!(
            "Fix: make `{}` readable: {error}.",
            directory.display()
        )
    });
    let mut lines = Vec::new();
    for entry in entries {
        let path = entry
            .unwrap_or_else(|error| panic!("Fix: read one entry of the snapshot directory: {error}."))
            .path();
        if !path.extension().is_some_and(|extension| extension == "txt") {
            continue;
        }
        let text = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("Fix: read `{}` as UTF-8: {error}.", path.display()));
        lines.extend(text.lines().map(str::to_string));
    }
    lines
}

/// The name a path binds, which is its final segment.
fn last_segment(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

/// Whether any snapshot declares an item other than a module named `name`.
///
/// A path that continues past the name, `::name::`, names the module holding the
/// item rather than an item called `name`, and a longer name that merely starts
/// with these characters is a different name. A `pub use` line states no kind at
/// all, which is the hole this test covers, so it is evidence for neither answer.
fn declares_non_module(lines: &[String], name: &str) -> bool {
    let needle = format!("::{name}");
    lines
        .iter()
        .filter(|line| !line.starts_with("pub mod ") && !line.starts_with("pub use "))
        .any(|line| {
            line.match_indices(&needle).any(|(offset, _)| {
                let after = &line[offset + needle.len()..];
                !after.starts_with("::")
                    && !after.starts_with(|character: char| {
                        character.is_alphanumeric() || character == '_'
                    })
            })
        })
}

/// Every re-export the snapshots state a public path for, globs excluded.
///
/// A glob publishes no path of its own: the snapshot renders it with the source
/// module embedded in `<<>>`, and the names it carries are listed separately.
fn reexported_paths(lines: &[String]) -> Vec<&str> {
    lines
        .iter()
        .filter_map(|line| line.strip_prefix("pub use "))
        .map(str::trim)
        .filter(|path| path.contains("::") && !path.contains('<'))
        .collect()
}

/// A module a facade republishes with `pub use` is published, and a re-exported
/// item that is not a module is not.
///
/// WHY: the scan feeds the banned-name exemption. Missing a published module
/// reports a name a consumer already imports, and admitting a re-exported type
/// or function exempts an unpublished module that happens to share its name.
#[test]
fn a_facade_module_re_export_reaches_the_published_module_set() {
    let root = workspace_root();
    let lines = snapshot_lines(&root);
    let workspace = scan(&root);
    let published: BTreeSet<&str> = workspace
        .published_modules
        .iter()
        .map(String::as_str)
        .collect();
    let declared: BTreeSet<&str> = lines
        .iter()
        .filter_map(|line| line.strip_prefix("pub mod "))
        .map(str::trim)
        .collect();
    let module_names: BTreeSet<&str> = declared.iter().copied().map(last_segment).collect();
    let reexports = reexported_paths(&lines);

    assert!(
        !declared.is_empty() && !reexports.is_empty(),
        "the snapshots state {} declared module(s) and {} re-export(s), so this contract is \
         passing vacuously",
        declared.len(),
        reexports.len()
    );
    let dropped: Vec<&str> = declared
        .iter()
        .copied()
        .filter(|path| !published.contains(path))
        .collect();
    assert!(
        dropped.is_empty(),
        "the scan dropped {} module(s) the snapshots declare with `pub mod`: {dropped:?}",
        dropped.len()
    );

    let modules: Vec<&str> = reexports
        .iter()
        .copied()
        .filter(|path| {
            let name = last_segment(path);
            module_names.contains(name) && !declares_non_module(&lines, name)
        })
        .collect();
    assert!(
        !modules.is_empty(),
        "no snapshot re-exports a name the snapshots declare a module for and no other item, so \
         the facade half of this contract is passing vacuously"
    );
    let missing: Vec<&str> = modules
        .iter()
        .copied()
        .filter(|path| !published.contains(path))
        .collect();
    assert!(
        missing.is_empty(),
        "docs/public-api publishes {} module(s) through a `pub use` re-export that the scan did \
         not read, so each would be reported as a banned name: {missing:?}",
        missing.len()
    );

    let others: Vec<&str> = reexports
        .iter()
        .copied()
        .filter(|path| !module_names.contains(last_segment(path)))
        .collect();
    assert!(
        !others.is_empty(),
        "every snapshot re-export names a declared module, so the adversarial half of this \
         contract is passing vacuously"
    );
    let admitted: Vec<&str> = others
        .iter()
        .copied()
        .filter(|path| published.contains(path))
        .collect();
    assert!(
        admitted.is_empty(),
        "the scan counted {} re-export(s) as published modules that the snapshots declare no \
         module for, so an unpublished module of the same name would be exempt: {admitted:?}",
        admitted.len()
    );
}
