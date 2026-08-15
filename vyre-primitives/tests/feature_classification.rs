//! WHY: a new domain feature that is not classified as intrinsic or
//! composition is how the "shared builder" third category comes back.
//! Adding a feature to `Cargo.toml` without putting it in
//! `organization.rs` must fail this test.
//!
//! The lists are the charter applied to this crate. `hardware` is the
//! only intrinsic domain. Everything else that is a domain is a
//! composition parked here.

use std::collections::BTreeSet;
use std::path::PathBuf;

use vyre_primitives::organization::{
    COMPOSITION_FEATURES, INTRINSIC_FEATURES, SUPPORT_FEATURES,
};

fn cargo_toml() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

fn feature_names(toml: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    let mut in_features = false;
    for line in toml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_features = trimmed == "[features]";
            continue;
        }
        if !in_features || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some((name, _)) = trimmed.split_once('=') {
            let name = name.trim();
            if name != "default" {
                names.insert(name.to_string());
            }
        }
    }
    names
}

#[test]
fn every_cargo_feature_is_classified_exactly_once() {
    let toml = std::fs::read_to_string(cargo_toml()).expect("Cargo.toml");
    let declared = feature_names(&toml);
    let mut classified = BTreeSet::new();
    for name in INTRINSIC_FEATURES
        .iter()
        .chain(COMPOSITION_FEATURES)
        .chain(SUPPORT_FEATURES)
    {
        assert!(
            classified.insert((*name).to_string()),
            "feature `{name}` is listed in more than one organization class"
        );
    }
    assert_eq!(
        declared, classified,
        "Cargo.toml [features] and src/organization.rs disagree.\n\
         only in Cargo.toml: {:?}\n\
         only in organization.rs: {:?}",
        declared.difference(&classified).collect::<Vec<_>>(),
        classified.difference(&declared).collect::<Vec<_>>()
    );
}

#[test]
fn hardware_is_the_only_intrinsic_domain() {
    assert_eq!(INTRINSIC_FEATURES, &["hardware"]);
    assert!(
        !COMPOSITION_FEATURES.contains(&"hardware"),
        "hardware must not be classified as a parked composition"
    );
}

#[test]
fn parked_compositions_are_named_and_nonempty() {
    assert!(
        !COMPOSITION_FEATURES.is_empty(),
        "the parked-composition list is the defect inventory; an empty list means the move finished and this assertion should change"
    );
    assert!(
        COMPOSITION_FEATURES.contains(&"math") && COMPOSITION_FEATURES.contains(&"graph"),
        "math and graph are the largest parked compositions; losing them from the list is a classification bug, not progress"
    );
}
