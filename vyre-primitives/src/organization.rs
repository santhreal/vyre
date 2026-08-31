//! One classification of every `vyre-primitives` Cargo feature.
//!
//! An operation is admitted here only when it cannot be composed, which means
//! it needs its own arm in a backend emitter and its own arm in the reference
//! interpreter. Marker types are always on; `hardware` is the intrinsic domain.
//! Every composition domain that was parked here has moved to `vyre-libs`, so
//! the composition list is empty and must stay empty. Do not add a domain
//! feature without putting it in one of these lists.
//!
//! The lists are crate-private. An outside caller must not be able to depend on
//! which domains this crate happens to carry; the tests at the bottom of this
//! file are their only consumer.

/// Domain feature that belongs in this crate: hardware intrinsics that need a
/// dedicated emitter arm and a dedicated reference-interpreter arm.
pub(crate) const INTRINSIC_FEATURES: &[&str] = &["hardware"];

/// Domain features that are compositions still resident in this crate.
///
/// Empty. Reuse count is never an admission criterion: a domain that builds a
/// `Program` from existing IR belongs in `vyre-libs` however many dialects call
/// it. A name may only ever be removed from this list by the domain moving, and
/// nothing may be added.
pub(crate) const COMPOSITION_FEATURES: &[&str] = &[];

/// Crate-support features. Not domains.
pub(crate) const SUPPORT_FEATURES: &[&str] =
    &["cpu-parity", "gpu", "inventory-registry", "vyre-foundation"];

/// WHY: a new domain feature that is not classified as intrinsic or composition
/// is how the "shared builder reused by two dialects" third category comes back,
/// and re-parking a composition here is how the move undoes itself. Adding a
/// feature to `Cargo.toml` without classifying it, or classifying anything as a
/// composition, must fail here.
///
/// These are unit tests rather than an integration test because the lists are
/// crate-private: an integration test would force them into the public API, and
/// the public API is the one place this classification must not appear.
///
/// What this does not catch: a composition smuggled in under the `hardware`
/// feature. That is the `lego-audit` gate's job, which reads the op bodies.
#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use super::{COMPOSITION_FEATURES, INTRINSIC_FEATURES, SUPPORT_FEATURES};

    fn cargo_toml() -> PathBuf {
        vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("Cargo.toml")
    }

    fn features(toml: &str) -> BTreeSet<String> {
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
                let name = name.trim().trim_matches('"');
                if name != "default" {
                    names.insert(name.to_string());
                }
            }
        }
        names
    }

    /// The complete domain selection a consumer names to reach every operation
    /// this crate registers.
    ///
    /// One domain means one feature. There is no aggregate over it: an
    /// aggregate whose single member is the thing it aggregates is a second
    /// name for one fact, and the tree carried one for long enough that three
    /// manifests explained their dependency in terms of composition domains
    /// that had already left this crate.
    fn complete_domain_selection() -> BTreeSet<String> {
        INTRINSIC_FEATURES
            .iter()
            .chain(COMPOSITION_FEATURES)
            .map(|name| (*name).to_string())
            .collect()
    }

    #[test]
    fn every_cargo_feature_is_classified_exactly_once() {
        let toml = std::fs::read_to_string(cargo_toml()).expect("Cargo.toml");
        let declared = features(&toml);
        let mut classified = BTreeSet::new();
        for name in INTRINSIC_FEATURES
            .iter()
            .chain(COMPOSITION_FEATURES)
            .chain(SUPPORT_FEATURES)
        {
            assert!(
                classified.insert((*name).to_string()),
                "Fix: feature `{name}` is listed in more than one organization class"
            );
        }
        assert_eq!(
            declared,
            classified,
            "Fix: Cargo.toml [features] and src/organization.rs disagree.\n\
             only in Cargo.toml: {:?}\n\
             only in organization.rs: {:?}",
            declared.difference(&classified).collect::<Vec<_>>(),
            classified.difference(&declared).collect::<Vec<_>>()
        );
    }

    /// WHY: a classified domain names a Cargo feature that gates a module
    /// directory in this crate. A name that gates nothing is a classification
    /// nobody can act on, and a directory renamed without its class entry
    /// leaves the feature switching on an absent module. Pinning the class to a
    /// literal member set instead would go stale the first time a second
    /// hardware intrinsic domain is admitted, which the lego block rule allows.
    #[test]
    fn every_classified_domain_gates_a_module_in_this_crate() {
        let src =
            vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME")).join("src");
        let domains: Vec<&str> = INTRINSIC_FEATURES
            .iter()
            .chain(COMPOSITION_FEATURES)
            .copied()
            .collect();
        assert!(
            !domains.is_empty(),
            "Fix: this crate carries no classified domain, so no operation in it is reachable through a domain feature"
        );
        for name in domains {
            let module = src.join(name);
            assert!(
                module.join("mod.rs").is_file(),
                "Fix: feature `{name}` is classified as a domain but {} declares no module; classify it as support or add the module",
                module.display()
            );
        }
        assert!(
            !COMPOSITION_FEATURES.contains(&"hardware"),
            "Fix: hardware is a hardware intrinsic, not a composition"
        );
    }

    /// WHY: the parked-composition list was the defect inventory for the move
    /// off this crate. The move is done, so an entry here is a regression: a
    /// composition was admitted into the intrinsic crate again.
    #[test]
    fn no_composition_is_parked_in_the_intrinsic_crate() {
        assert_eq!(
            COMPOSITION_FEATURES.len(),
            0,
            "Fix: `{COMPOSITION_FEATURES:?}` are compositions declared inside vyre-primitives; a domain that builds a Program from existing IR belongs in vyre-libs"
        );
    }

    /// WHY: every domain this crate registers has to be reachable by naming
    /// features it declares, because a consumer that cannot name a domain
    /// cannot link its registrations, and an unlinked registration is missing
    /// from every registry walk while every count still agrees with itself.
    #[test]
    fn every_domain_is_reachable_through_a_declared_feature() {
        let toml = std::fs::read_to_string(cargo_toml()).expect("Cargo.toml");
        let declared = features(&toml);
        let selection = complete_domain_selection();
        assert!(
            !selection.is_empty(),
            "Fix: this crate classifies no domain, so no consumer can reach an operation in it"
        );
        let unreachable: Vec<&String> = selection.difference(&declared).collect();
        assert!(
            unreachable.is_empty(),
            "Fix: classified domain(s) {unreachable:?} name no feature in Cargo.toml, so nothing a consumer can write links them"
        );
    }

    /// WHY: `every_cargo_feature_is_classified_exactly_once` and
    /// `every_domain_is_reachable_through_a_declared_feature` both read the
    /// manifest with a hand-written scan. A scan that silently returns nothing
    /// makes every set comparison trivially pass, so it is proven against text
    /// whose answer is known.
    #[test]
    fn the_manifest_scan_reads_names_out_of_declarations() {
        let toml = "[features]\ndefault = []\nalpha = [\"beta\"]\nhardware = [\n    \"alpha\",\n]\n\"quoted\" = []\n[dependencies]\nnot-a-feature = \"1\"\n";
        assert_eq!(
            features(toml),
            ["alpha", "hardware", "quoted"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        );
    }
}
