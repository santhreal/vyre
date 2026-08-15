//! One classification of every `vyre-primitives` Cargo feature.
//!
//! An operation is admitted here only when it cannot be composed, which means
//! it needs its own arm in a backend emitter and its own arm in the reference
//! interpreter. Marker types are always on; `hardware` is the intrinsic domain.
//! Every other domain feature named below is a composition still resident here,
//! and each one leaving is a move to `vyre-libs`, not a reclassification. Do not
//! add a domain feature without putting it in one of these lists.
//!
//! The lists are crate-private. They describe a parking arrangement that is
//! being dismantled, so an outside caller must not be able to depend on the
//! current contents; the tests at the bottom of this file are their only
//! consumer.

/// Domain feature that belongs in this crate: hardware intrinsics that need a
/// dedicated emitter arm and a dedicated reference-interpreter arm.
pub(crate) const INTRINSIC_FEATURES: &[&str] = &["hardware"];

/// Domain features that are compositions still resident in this crate.
///
/// Reuse count is not an admission criterion. Each of these builds a `Program`
/// from existing IR and belongs in `vyre-libs`. A name leaves this list only by
/// the domain moving; `cat`, `zx`, `dnnf`, `types` and `effects` left that way.
pub(crate) const COMPOSITION_FEATURES: &[&str] = &[
    "bitset",
    "decode",
    "fixpoint",
    "geom",
    "graph",
    "hash",
    "label",
    "matching",
    "math",
    "nfa",
    "nn",
    "opt",
    "parsing",
    "predicate",
    "reduce",
    "text",
    "topology",
    "visual",
];

/// Crate-support features. Not domains.
pub(crate) const SUPPORT_FEATURES: &[&str] = &[
    "all-lego",
    "cpu-parity",
    "gpu",
    "inventory-registry",
    "vyre-foundation",
];

/// WHY: a new domain feature that is not classified as intrinsic or composition
/// is how the "shared builder reused by two dialects" third category comes back.
/// Adding a feature to `Cargo.toml` without putting it in one of the lists above
/// must fail here.
///
/// These are unit tests rather than an integration test because the lists are
/// crate-private: an integration test would force them into the public API, and
/// the public API is the one place a parking arrangement must not appear.
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use super::{COMPOSITION_FEATURES, INTRINSIC_FEATURES, SUPPORT_FEATURES};

    fn cargo_toml() -> PathBuf {
        vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("Cargo.toml")
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
            declared,
            classified,
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
}
