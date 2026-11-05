//! A string literal that spells a checkout path into a workspace member spells
//! the directory the roster places that member at.
//!
//! WHY: members live under `crates/`, `tools/` and `conform/`, so a package name
//! is not a directory. A literal that joins a package name onto the checkout
//! root keeps compiling after the member moves, and the read it drives finds no
//! tree: a gate scanning a missing directory scans nothing and reports success,
//! and a test reading one reports a defect that is not there. Every literal of
//! the form `<package>/<member entry>` is held to the roster here, at the one
//! choke point every such literal passes through, whichever crate holds it.
//!
//! What this does not catch: a path assembled at run time from a package name
//! (`root.join(package)` or `format!("{package}/src")`), a bare prefix with no
//! member entry after it (`"vyre-libs-"`), and a literal rooted inside another
//! path (`/w/vyre-libs/src`), which names a fixture tree rather than this
//! checkout.

use std::collections::BTreeMap;

use structure_gate::source_scan::string_literals;
use structure_gate::{member_directories, source_file_roster, workspace_root};

/// The entries of a member directory a checkout-relative path names after the
/// member. A package name followed by anything else (`vyre-bench/v1`, a cargo
/// `dependency/feature` reference) is an identifier, not a path.
const MEMBER_ENTRIES: &[&str] = &[
    "src",
    "tests",
    "benches",
    "examples",
    "fuzz",
    "docs",
    "Cargo.toml",
    "README.md",
    "build.rs",
];

/// Every `(package, directory)` pair a literal spells as `<package>/<entry>`
/// for a package whose roster directory is not its name.
fn stale_member_paths<'roster>(
    literal: &str,
    relocated: &'roster BTreeMap<String, String>,
) -> Vec<(&'roster str, &'roster str)> {
    let mut stale = Vec::new();
    for (package, directory) in relocated {
        let needle = format!("{package}/");
        for (at, _) in literal.match_indices(&needle) {
            let rooted = literal[..at].chars().next_back().is_none_or(|before| {
                !(before.is_ascii_alphanumeric() || matches!(before, '_' | '-' | '.' | '/'))
            });
            let entry = literal[at + needle.len()..]
                .split(|c: char| c == '/' || c.is_whitespace() || c == '`' || c == '\'')
                .next()
                .unwrap_or_default();
            if rooted && MEMBER_ENTRIES.contains(&entry) {
                stale.push((package.as_str(), directory.as_str()));
            }
        }
    }
    stale
}

fn relocated(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(package, directory)| ((*package).to_string(), (*directory).to_string()))
        .collect()
}

/// A literal rooted at a relocated member's package name is reported with the
/// directory it should name; the same path through the roster directory is not.
#[test]
fn a_literal_rooted_at_a_package_name_is_stale_and_one_rooted_at_its_directory_is_not() {
    let roster = relocated(&[
        ("acme-libs", "crates/acme-libs"),
        ("acme-tool", "tools/acme-tool"),
    ]);
    for literal in [
        "acme-libs/src/lib.rs",
        "acme-libs/Cargo.toml",
        "see acme-libs/tests/all.rs",
        "`acme-libs/src`",
    ] {
        assert_eq!(
            stale_member_paths(literal, &roster),
            vec![("acme-libs", "crates/acme-libs")],
            "{literal}"
        );
    }
    for literal in [
        "crates/acme-libs/src/lib.rs",
        "/w/acme-libs/src/lib.rs",
        "acme-libs/feature",
        "acme-libs/v1",
        "acme-libs-extra/src/lib.rs",
        "xacme-libs/src/lib.rs",
        "acme-libs",
        "acme-libs/",
        "tools/acme-tool/src/main.rs",
    ] {
        assert!(
            stale_member_paths(literal, &roster).is_empty(),
            "{literal} is not a checkout path into a relocated member"
        );
    }
}

/// A member at its own name is not relocated, so a path through its name is
/// its directory.
#[test]
fn a_member_at_its_own_name_is_never_stale() {
    assert!(stale_member_paths("acme/src/lib.rs", &relocated(&[])).is_empty());
}

/// No literal in any workspace source names a relocated member by its package.
#[test]
fn every_member_path_literal_names_the_roster_directory() {
    let root = workspace_root();
    let directories = member_directories(&root);
    assert!(
        directories.len() >= 40,
        "Fix: the roster resolved {} members; the root manifest parse is reading the wrong file",
        directories.len()
    );
    let relocated: BTreeMap<String, String> = directories
        .into_iter()
        .filter_map(|(package, directory)| {
            let directory = directory
                .strip_prefix(&root)
                .expect("Fix: a roster directory resolves under the workspace root")
                .to_str()?
                .to_string();
            (directory != package).then_some((package, directory))
        })
        .collect();

    let files = source_file_roster(&root);
    assert!(
        files.len() >= 1000,
        "Fix: the source roster holds {} files; the walk is reading the wrong tree",
        files.len()
    );
    let mut stale = Vec::new();
    for relative in &files {
        let text = std::fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("Fix: read {relative}: {error}"));
        for literal in string_literals(&text) {
            for (package, directory) in stale_member_paths(literal, &relocated) {
                stale.push(format!(
                    "{relative}: `{literal}` roots a path at `{package}/`; the member is at `{directory}/`"
                ));
            }
        }
    }
    assert!(
        stale.is_empty(),
        "Fix: spell the roster directory, or resolve it from the package name through \
         `structure_gate::member_directory`:\n{}",
        stale.join("\n")
    );
}
