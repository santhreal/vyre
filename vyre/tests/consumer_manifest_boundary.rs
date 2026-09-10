//! What a consumer workspace is allowed to depend on.
//!
//! A consumer under `consumers/` stands in for a caller outside this
//! repository, so it may declare a production dependency only on a published
//! facade or SDK. A dependency on an `internal-engine` or
//! `private-test-support` crate makes the consumer a second view of the
//! compiler's internals, and every seam the consumer exists to prove stops
//! proving anything.
//!
//! The two subjects are each member's own `[package.metadata.vyre]` table and
//! the consumer manifests, all files in this tree, so the check belongs here
//! rather than inside the consumers. Each consumer is a separate workspace
//! with its own build, and a copy of this walk inside one of them audits
//! whichever manifests that copy happens to name.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Publication classes a consumer may not declare a production dependency on.
const FORBIDDEN_PUBLICATION_CLASSES: &[&str] = &["internal-engine", "private-test-support"];

/// The fewest consumer workspaces the walk must find.
///
/// A directory read that returns nothing produces a clean sweep of no
/// manifests, which is indistinguishable from a clean sweep of every manifest.
const CONSUMER_FLOOR: usize = 2;

/// Package name to publication class, for every class a consumer may not take.
///
/// The class is declared beside `publish` in each member's own manifest, which
/// is the one home for it, so the set is derived from the member roster rather
/// than from a second listing a new internal crate would not reach.
fn forbidden_packages(root: &Path) -> BTreeMap<String, String> {
    crate::publication_classes::load_workspace_members(root)
        .into_iter()
        .filter_map(|(name, info)| {
            let class = info.publication_class?;
            FORBIDDEN_PUBLICATION_CLASSES
                .contains(&class.as_str())
                .then_some((name, class))
        })
        .collect()
}

/// Every consumer manifest in the tree, derived from the directory rather than
/// from a list a new consumer would not be added to.
fn consumer_manifests(root: &Path) -> Vec<PathBuf> {
    let consumers = root.join("consumers");
    let entries = std::fs::read_dir(&consumers)
        .unwrap_or_else(|error| panic!("{} must be readable: {error}", consumers.display()));
    let mut manifests: Vec<PathBuf> = entries
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("{} must enumerate: {error}", consumers.display()))
        })
        .map(|entry| entry.path().join("Cargo.toml"))
        .filter(|manifest| manifest.is_file())
        .collect();
    manifests.sort();
    manifests
}

/// Production dependency names a manifest declares.
fn production_dependencies(text: &str, path: &Path) -> BTreeSet<String> {
    let manifest: toml::Value = toml::from_str(text)
        .unwrap_or_else(|error| panic!("{} must be valid TOML: {error}", path.display()));
    manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .map(|table| table.keys().cloned().collect())
        .unwrap_or_default()
}

/// Forbidden dependencies a manifest declares, as `package (class)`.
fn violations(forbidden: &BTreeMap<String, String>, dependencies: &BTreeSet<String>) -> Vec<String> {
    dependencies
        .iter()
        .filter_map(|dependency| {
            forbidden
                .get(dependency)
                .map(|class| format!("{dependency} ({class})"))
        })
        .collect()
}

#[test]
fn no_consumer_declares_a_production_dependency_on_an_internal_crate() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let forbidden = forbidden_packages(&root);
    assert!(
        !forbidden.is_empty(),
        "no workspace member declares a publication class in {FORBIDDEN_PUBLICATION_CLASSES:?}, \
         so this walk admits every dependency"
    );

    let manifests = consumer_manifests(&root);
    assert!(
        manifests.len() >= CONSUMER_FLOOR,
        "the consumers directory yielded {} manifest(s), below the {CONSUMER_FLOOR} floor: the \
         directory walk is broken, not the manifests",
        manifests.len()
    );

    let mut reported = Vec::new();
    for manifest in &manifests {
        let text = std::fs::read_to_string(manifest)
            .unwrap_or_else(|error| panic!("{} must be readable: {error}", manifest.display()));
        for violation in violations(&forbidden, &production_dependencies(&text, manifest)) {
            reported.push(format!("{}: {violation}", manifest.display()));
        }
    }

    assert!(
        reported.is_empty(),
        "a consumer depends only on a published facade or SDK. Fix: drop the dependency or \
         publish what it needs:\n  {}",
        reported.join("\n  ")
    );
}

#[test]
fn a_manifest_that_takes_an_internal_crate_is_reported() {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let forbidden = forbidden_packages(&root);
    let internal = forbidden
        .keys()
        .next()
        .expect("the workspace declares at least one internal crate");

    let injected = format!(
        "[package]\nname = \"injected-consumer\"\nversion = \"0.1.0\"\n\n\
         [dependencies]\nvyre = {{ path = \"../../vyre\" }}\n\
         {internal} = {{ path = \"../../{internal}\" }}\n"
    );
    let path = Path::new("consumers/injected-consumer/Cargo.toml");
    let found = violations(&forbidden, &production_dependencies(&injected, path));

    assert_eq!(
        found.len(),
        1,
        "the walk must report the injected internal dependency and nothing else: {found:?}"
    );
    assert!(
        found[0].starts_with(internal.as_str()),
        "the report names the dependency it refused: {found:?}"
    );
}
