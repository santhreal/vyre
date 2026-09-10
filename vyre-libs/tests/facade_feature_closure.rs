//! Every domain feature this facade declares is reachable from `full`.
//!
//! `full` is the surface a consumer selects when it wants the whole facade. A
//! domain feature that is declared but left out of `full` compiles in a
//! workspace build, because some other member enables it and Cargo unifies
//! features across the graph, and then fails only when the consuming crate is
//! built alone. That is how `graph-dispatch` stayed out of `full` while eight
//! live CUDA parity tests imported `vyre_libs::graph::dispatch`: the workspace
//! check passed and `cargo test -p vyre-driver-cuda` did not.
//!
//! The reachable set is computed from the manifest at run time, so a feature
//! added later is covered without editing this file. Adding one and omitting it
//! from `full` turns this test red.
//!
//! What this does not catch: a feature that is reachable from `full` but whose
//! forwarding entry names the wrong dependency feature. Reachability is a
//! manifest property; that one needs a build.

use std::collections::{BTreeMap, BTreeSet};

/// Features that are deliberately not part of `full`.
///
/// `default` and `full` are selections rather than domains, and
/// `test-fixtures` exposes fixture helpers that are not part of the published
/// surface. Every other declared feature must be reachable.
const NOT_A_DOMAIN: &[&str] = &["default", "full", "test-fixtures"];

/// Parses the `[features]` table of this crate's own manifest.
fn declared_features() -> BTreeMap<String, Vec<String>> {
    // The crate directory is resolved from the working directory through the
    // workspace member roster. A compiled-in manifest path names whichever
    // checkout last built this binary through the shared target directory, so
    // the features read would be that tree's.
    let manifest = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))
        .join("Cargo.toml");
    let text = std::fs::read_to_string(&manifest)
        .expect("Fix: vyre-libs must be able to read its own Cargo.toml.");
    let parsed: toml::Value =
        toml::from_str(&text).expect("Fix: vyre-libs/Cargo.toml must be valid TOML.");
    let table = parsed
        .get("features")
        .and_then(toml::Value::as_table)
        .expect("Fix: vyre-libs/Cargo.toml must declare a [features] table.");

    table
        .iter()
        .map(|(name, entries)| {
            let entries = entries.as_array().unwrap_or_else(|| {
                panic!("Fix: feature `{name}` must be declared as an array of strings.")
            });
            let entries = entries
                .iter()
                .map(|entry| {
                    entry
                        .as_str()
                        .unwrap_or_else(|| {
                            panic!("Fix: every entry of feature `{name}` must be a string.")
                        })
                        .to_owned()
                })
                .collect();
            (name.clone(), entries)
        })
        .collect()
}

/// True when `entry` names a feature of this crate rather than a dependency.
///
/// `dep:x` activates an optional dependency and `x/y` or `x?/y` activates a
/// feature of one. Neither is a local feature name.
fn is_local_feature(entry: &str) -> bool {
    !entry.contains(':') && !entry.contains('/')
}

/// Every local feature reachable from `root`, following local entries only.
fn reachable_from(features: &BTreeMap<String, Vec<String>>, root: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![root.to_owned()];
    while let Some(name) = stack.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(entries) = features.get(&name) else {
            continue;
        };
        stack.extend(entries.iter().filter(|e| is_local_feature(e)).cloned());
    }
    seen
}

#[test]
fn every_declared_domain_feature_is_reachable_from_full() {
    let features = declared_features();
    assert!(
        features.contains_key("full"),
        "Fix: vyre-libs must declare a `full` feature that selects every domain."
    );

    let reachable = reachable_from(&features, "full");
    let missing: Vec<&str> = features
        .keys()
        .map(String::as_str)
        .filter(|name| !NOT_A_DOMAIN.contains(name))
        .filter(|name| !reachable.contains(*name))
        .collect();

    assert!(
        missing.is_empty(),
        "Fix: add {missing:?} to the `full` feature of vyre-libs/Cargo.toml, or, if one of \
         them is not a domain a consumer selects, to NOT_A_DOMAIN in this test with the \
         reason. A domain left out of `full` still builds in a workspace check and fails \
         only when a consumer is built on its own."
    );
}

#[test]
fn full_selects_more_than_default() {
    let features = declared_features();
    // `reachable_from` includes its own root, and the two roots are selections
    // rather than domains, so neither belongs in the compared sets.
    let mut full = reachable_from(&features, "full");
    let mut default = reachable_from(&features, "default");
    for root in ["full", "default"] {
        full.remove(root);
        default.remove(root);
    }

    let default_only: Vec<&String> = default.difference(&full).collect();
    assert!(
        default_only.is_empty(),
        "Fix: `full` must be a superset of `default`; {default_only:?} are reachable from \
         `default` but not from `full`."
    );
    assert!(
        full.len() > default.len(),
        "Fix: `full` must select more domains than `default`."
    );
}

#[test]
fn every_feature_entry_names_something_that_exists() {
    let features = declared_features();
    let dangling: Vec<String> = features
        .iter()
        .flat_map(|(name, entries)| {
            entries
                .iter()
                .filter(|entry| is_local_feature(entry))
                .filter(|entry| !features.contains_key(*entry))
                .map(move |entry| format!("{name} -> {entry}"))
        })
        .collect();

    assert!(
        dangling.is_empty(),
        "Fix: these feature entries name a local feature that is not declared: {dangling:?}."
    );
}
