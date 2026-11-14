use super::*;
use std::collections::{BTreeSet, HashSet};

#[test]
fn catalog_entries_have_owner_invariant_and_benchmark_contracts() {
    let catalog = optimization_catalog().expect("Fix: optimizer catalog must build");
    assert!(
        catalog.iter().any(|e| e.name == "const_fold"),
        "Fix: optimizer catalog must include const_fold."
    );
    for entry in &catalog {
        assert!(!entry.name.is_empty(), "catalog pass name must be stable");
        assert!(!entry.owner.is_empty(), "catalog pass owner must be set");
        assert!(
            !entry.invariant.is_empty(),
            "catalog pass invariant must be set"
        );
        assert!(
            !entry.benchmark.is_empty(),
            "catalog pass benchmark contract must be set"
        );
    }
}

#[test]
fn catalog_names_are_unique() {
    let catalog = optimization_catalog().expect("Fix: optimizer catalog must build");
    let mut seen = BTreeSet::new();
    for entry in catalog {
        assert!(
            seen.insert(entry.name),
            "Fix: duplicate optimizer pass name `{}` makes profile reports ambiguous.",
            entry.name
        );
    }
}

#[test]
fn catalog_matches_executable_registered_passes() {
    let catalog = optimization_catalog().expect("Fix: optimizer catalog must build");
    let registered_count = registered_pass_registrations()
        .expect("Fix: registered optimizer passes must schedule")
        .len();

    assert_eq!(
        catalog.len(),
        registered_count,
        "Fix: optimizer catalog entries must correspond 1:1 to runnable registered passes."
    );
    assert!(catalog
        .iter()
        .all(|entry| entry.kind == OptimizationCatalogEntryKind::ExecutablePass));

    let phases = catalog
        .iter()
        .map(|entry| entry.phase)
        .collect::<HashSet<_>>();
    assert!(
        phases.len() >= 4,
        "Fix: optimizer catalog must span multiple scheduler phases."
    );
}

#[test]
fn release_catalog_entries_are_fully_classified() {
    let catalog = optimization_catalog_for_profile(OptimizerProfile::Release)
        .expect("Fix: release optimizer catalog must build");
    assert!(
        !catalog.is_empty(),
        "Fix: release optimizer profile must expose classified passes."
    );
    for entry in catalog {
        assert_ne!(
            entry.phase,
            PassPhase::Unclassified,
            "Fix: release pass `{}` must declare a concrete phase.",
            entry.name
        );
        assert_ne!(
            entry.boundary_class,
            PassBoundaryClass::Unknown,
            "Fix: release pass `{}` must declare an API boundary class.",
            entry.name
        );
        assert_ne!(
            entry.cost_model_family,
            CostModelFamily::Unknown,
            "Fix: release pass `{}` must declare a benchmark/cost model family.",
            entry.name
        );
        assert!(
            entry.preserves_abi,
            "Fix: release pass `{}` must preserve public buffer ABI.",
            entry.name
        );
    }
}
