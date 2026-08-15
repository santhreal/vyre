//! Every workspace member's generated testing guide is classified in `docs/DOCS.toml`.
//!
//! WHY: the `docs-check` gate compares the manifest against the Markdown on
//! disk, so a page is a failure the moment it is written. It still cannot see a
//! guide that was never generated: adding a crate must produce a new
//! `docs/testing/<crate>.md`, and until the generator runs there is no file to
//! be unclassified and no row to be missing. `vyre-registry-link` reached the
//! tree that way: thirty-five members had a row and thirty-six had a guide.
//!
//! The subject here is the MEMBER set, not the file set, so the manifest is
//! judged against the same source of truth the generator uses and a new crate
//! is red on the commit that adds it.
//!
//! What it does not catch: a classified page whose fields are wrong. The
//! manifest gate owns that and validates every row's lifecycle, owner, kind and
//! authority.

use std::collections::BTreeSet;
use std::path::Path;

use super::common::workspace_root;

/// Below this many members the manifest walk has stopped seeing the workspace.
const MINIMUM_MEMBERS: usize = 20;

/// Package names of every workspace member, read from their manifests.
fn workspace_packages(root: &Path) -> BTreeSet<String> {
    let members = structure_gate::workspace_members(root);
    assert!(
        members.len() >= MINIMUM_MEMBERS,
        "Fix: only {} workspace member(s) were derived; the manifest walk is wrong, so this gate would pass by comparing nothing.",
        members.len()
    );
    let mut packages = BTreeSet::new();
    for member in members {
        let manifest = root.join(&member).join("Cargo.toml");
        let text = std::fs::read_to_string(&manifest)
            .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest.display()));
        let table: toml::Table = toml::from_str(&text)
            .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", manifest.display()));
        let name = table
            .get("package")
            .and_then(|package| package.get("name"))
            .and_then(toml::Value::as_str)
            .unwrap_or_else(|| panic!("Fix: {} has no [package].name", manifest.display()));
        packages.insert(name.to_string());
    }
    packages
}

/// Documentation pages `docs/DOCS.toml` classifies under `testing/`.
fn classified_testing_pages(root: &Path) -> BTreeSet<String> {
    let manifest = root.join("docs/DOCS.toml");
    let text = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", manifest.display()));
    let table: toml::Table = toml::from_str(&text)
        .unwrap_or_else(|error| panic!("Fix: cannot parse {}: {error}", manifest.display()));
    table
        .get("page")
        .and_then(toml::Value::as_array)
        .unwrap_or_else(|| panic!("Fix: {} declares no [[page]] rows", manifest.display()))
        .iter()
        .filter_map(|page| page.get("path").and_then(toml::Value::as_str))
        .filter(|path| path.starts_with("testing/"))
        .map(ToString::to_string)
        .collect()
}

#[test]
fn every_workspace_member_has_a_classified_testing_guide() {
    let root = workspace_root();
    let classified = classified_testing_pages(&root);
    assert!(
        classified.len() >= MINIMUM_MEMBERS,
        "Fix: only {} testing page(s) were read from docs/DOCS.toml; the parse is wrong, so this gate would pass by comparing nothing.",
        classified.len()
    );

    let missing: Vec<String> = workspace_packages(&root)
        .into_iter()
        .filter(|package| !classified.contains(&format!("testing/{package}.md")))
        .collect();

    assert!(
        missing.is_empty(),
        "Fix: these workspace members have no `testing/<crate>.md` row in docs/DOCS.toml. Run `python3 scripts/testing_guides.py --write`, add the row beside the other generated testing guides, then run `cargo_full run --bin xtask -- docs-check --write`. Adding a crate must not leave the documentation manifest incomplete:\n  {}",
        missing.join("\n  ")
    );
}

/// The reverse direction: a row must name a member that still exists.
///
/// The `docs-check` gate already fails on a row whose FILE is gone, which covers a
/// deleted guide. It cannot see a guide that outlived its crate, because the
/// file is still there and still classified.
#[test]
fn no_testing_guide_row_names_a_crate_the_workspace_dropped() {
    let root = workspace_root();
    let packages = workspace_packages(&root);

    let orphaned: Vec<String> = classified_testing_pages(&root)
        .into_iter()
        .filter(|page| {
            let crate_name = page
                .trim_start_matches("testing/")
                .trim_end_matches(".md")
                .to_string();
            !packages.contains(&crate_name)
        })
        .collect();

    assert!(
        orphaned.is_empty(),
        "Fix: these docs/DOCS.toml rows classify a testing guide for a crate that is no longer a workspace member. Delete the guide and its row; a current-looking test document for a deleted crate is worse than none:\n  {}",
        orphaned.join("\n  ")
    );
}
