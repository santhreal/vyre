//! A directory module has one spelling.
//!
//! Rust accepts two layouts for a module that owns children: `foo/mod.rs`, and
//! a `foo.rs` sitting beside a `foo/` directory. Both compile, and a tree that
//! uses both has two places to look for the same module's own items. This one
//! settled on `mod.rs` in 476 directories and carried 110 of the other pair at
//! its worst, which is what made a reader guess.
//!
//! The rule is the convention the tree already keeps, so the gate is green the
//! day it lands and goes red on the first file that reintroduces the pair. It
//! reads the layout from the source list rather than a recorded count, so a
//! directory added tomorrow is judged the same way as one that has been here
//! all along.
//!
//! `mod.rs` is exempt because `foo/mod.rs` beside a `foo/` child directory is
//! the accepted layout, not the pair.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::gate::{Finding, GateCtx, GateError, Report};
use crate::gates::scan::Tree;

/// Reports a `foo.rs` that sits beside a `foo/` directory.
pub struct ModuleLayout;

impl crate::gate::GateBehavior for ModuleLayout {
    fn run(&self, ctx: &GateCtx) -> Result<Report, GateError> {
        let tree = Tree::open(&ctx.root)?;
        let sources = tree.all_rust();
        let mut report = Report::clean();
        report.cover_complete("rust source files", sources.len());
        for finding in dual_module_findings(&sources) {
            report.find(finding);
        }
        Ok(report)
    }
}

/// The `foo.rs` files that sit beside a `foo/` directory of modules.
///
/// The directory set is derived from the source paths themselves, so a
/// directory counts only when it actually holds Rust source. An empty
/// directory, or one holding a fixture, is not a module and does not make its
/// neighbour a pair.
fn dual_module_findings(sources: &[PathBuf]) -> Vec<Finding> {
    let directories: BTreeSet<&Path> = sources.iter().filter_map(|path| path.parent()).collect();
    let mut findings = Vec::new();
    for path in sources {
        let (Some(stem), Some(parent)) = (path.file_stem(), path.parent()) else {
            continue;
        };
        if stem == "mod" {
            continue;
        }
        let sibling = parent.join(stem);
        if directories.iter().any(|dir| dir.starts_with(&sibling)) {
            findings.push(Finding::in_file(
                path.clone(),
                format!(
                    "sits beside the module directory `{}`, so this module is spelled two ways",
                    sibling.to_string_lossy().replace('\\', "/")
                ),
                "move the file to `<name>/mod.rs`; this tree spells a directory module one way, \
                 and the pair leaves two places to look for the same module's own items",
            ));
        }
    }
    findings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(list: &[&str]) -> Vec<PathBuf> {
        list.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn a_file_beside_its_own_directory_is_a_finding() {
        let findings = dual_module_findings(&paths(&["a/foo.rs", "a/foo/bar.rs"]));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].file.as_deref(), Some(Path::new("a/foo.rs")));
        assert!(findings[0].message.contains("a/foo"));
    }

    #[test]
    fn the_mod_rs_layout_is_not_a_pair() {
        // `a/foo/mod.rs` owns `a/foo/bar.rs`. That is the accepted layout, and
        // reading `mod` as a stem would report every directory module here.
        let findings = dual_module_findings(&paths(&["a/foo/mod.rs", "a/foo/bar.rs"]));
        assert!(findings.is_empty(), "{findings:?}");
    }

    #[test]
    fn a_leaf_module_beside_unrelated_directories_is_clean() {
        let findings = dual_module_findings(&paths(&["a/foo.rs", "a/other/bar.rs", "b/foo/x.rs"]));
        assert!(findings.is_empty(), "{findings:?}");
    }

    #[test]
    fn a_deeply_nested_child_still_makes_the_pair() {
        // The directory holding source is `a/foo/deep`, not `a/foo`, so a plain
        // equality test against the sibling would miss this one.
        let findings = dual_module_findings(&paths(&["a/foo.rs", "a/foo/deep/bar.rs"]));
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].file.as_deref(), Some(Path::new("a/foo.rs")));
    }

    #[test]
    fn a_prefix_of_a_directory_name_is_not_a_pair() {
        // `a/foobar/` starts with the text `a/foo` but is not `a/foo`'s
        // directory. Comparing path components rather than strings is what
        // keeps this clean.
        let findings = dual_module_findings(&paths(&["a/foo.rs", "a/foobar/bar.rs"]));
        assert!(findings.is_empty(), "{findings:?}");
    }
}
