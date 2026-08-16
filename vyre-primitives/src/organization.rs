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
pub(crate) const SUPPORT_FEATURES: &[&str] = &[
    "all-lego",
    "cpu-parity",
    "gpu",
    "inventory-registry",
    "vyre-foundation",
];

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

    /// Members of the `all-lego` array, read out of the manifest at run time.
    ///
    /// The array may span lines, so the scan runs from the opening bracket to
    /// the closing one rather than per line.
    fn all_lego_members(toml: &str) -> BTreeSet<String> {
        let start = toml
            .find("all-lego = [")
            .expect("Fix: Cargo.toml must declare the `all-lego` aggregate feature");
        let body = &toml[start..];
        let end = body
            .find(']')
            .expect("Fix: the `all-lego` array must be closed");
        body[..end]
            .split('"')
            .skip(1)
            .step_by(2)
            .map(str::to_string)
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
        let src = vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))
            .join("src");
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
        assert!(
            COMPOSITION_FEATURES.is_empty(),
            "Fix: `{COMPOSITION_FEATURES:?}` are compositions declared inside vyre-primitives; a domain that builds a Program from existing IR belongs in vyre-libs"
        );
    }

    /// WHY: `all-lego` is what a consumer and the registry walker name to reach
    /// every domain. A domain outside the aggregate is invisible to both, and a
    /// support feature inside it makes the aggregate mean something other than
    /// "every domain".
    #[test]
    fn all_lego_equals_the_union_of_the_domain_classes() {
        let toml = std::fs::read_to_string(cargo_toml()).expect("Cargo.toml");
        let members = all_lego_members(&toml);
        let domains: BTreeSet<String> = INTRINSIC_FEATURES
            .iter()
            .chain(COMPOSITION_FEATURES)
            .map(|name| (*name).to_string())
            .collect();
        let missing: Vec<&String> = domains.difference(&members).collect();
        assert!(
            missing.is_empty(),
            "Fix: `all-lego` omits classified domain(s) {missing:?}; a domain outside the aggregate is unreachable through it"
        );
        let extra: Vec<&String> = members.difference(&domains).collect();
        assert!(
            extra.is_empty(),
            "Fix: `all-lego` names {extra:?}, which are not classified domains; the aggregate must aggregate domains and nothing else"
        );
    }

    /// WHY: `all_lego_equals_the_union_of_the_domain_classes` reads the manifest
    /// with a hand-written scan. A scan that silently returns nothing makes both
    /// difference assertions trivially pass, so it is proven against text whose
    /// answer is known.
    #[test]
    fn the_manifest_scans_read_names_out_of_declarations() {
        let toml = "[features]\ndefault = []\nalpha = [\"beta\"]\nall-lego = [\n    \"alpha\",\n    \"beta\",\n]\n\"quoted\" = []\n[dependencies]\nnot-a-feature = \"1\"\n";
        assert_eq!(
            features(toml),
            ["alpha", "all-lego", "quoted"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        );
        assert_eq!(
            all_lego_members(toml),
            ["alpha", "beta"]
                .into_iter()
                .map(str::to_string)
                .collect::<BTreeSet<_>>()
        );
    }
}
