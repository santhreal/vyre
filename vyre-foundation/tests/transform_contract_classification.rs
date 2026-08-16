//! Tests for foundation transformation contract classifications (Section 188.1).

use std::collections::BTreeSet;
use vyre_foundation::transform::{
    FOUNDATION_TRANSFORM_CLASSIFICATIONS, HOST_REWRITES, TransformContractClass,
};

#[test]
fn all_foundation_transforms_are_classified_uniquely() {
    let mut names = BTreeSet::new();
    for descriptor in FOUNDATION_TRANSFORM_CLASSIFICATIONS {
        assert!(!descriptor.name.is_empty(), "transform name cannot be empty");
        assert!(!descriptor.description.is_empty(), "description cannot be empty");
        assert!(
            names.insert(descriptor.name),
            "duplicate transform name `{}` in classification table",
            descriptor.name
        );
    }
}

#[test]
fn contract_classes_cover_all_categories() {
    let mut classes = BTreeSet::new();
    for descriptor in FOUNDATION_TRANSFORM_CLASSIFICATIONS {
        classes.insert(descriptor.class);
    }

    assert!(classes.contains(&TransformContractClass::RequiredLegalization));
    assert!(classes.contains(&TransformContractClass::CanonicalOptimization));
    assert!(classes.contains(&TransformContractClass::CallerRequestedTransform));
    assert!(classes.contains(&TransformContractClass::SharedStructuralWalk));
    assert!(classes.contains(&TransformContractClass::Analysis));
}

#[test]
fn host_rewrites_are_canonical_optimizations() {
    let classification_map: std::collections::HashMap<&str, TransformContractClass> =
        FOUNDATION_TRANSFORM_CLASSIFICATIONS
            .iter()
            .map(|d| (d.name, d.class))
            .collect();

    for rewrite in HOST_REWRITES {
        let class = classification_map.get(rewrite.name).copied().unwrap_or_else(|| {
            panic!("HOST_REWRITES entry `{}` missing from FOUNDATION_TRANSFORM_CLASSIFICATIONS", rewrite.name)
        });
        assert_eq!(
            class,
            TransformContractClass::CanonicalOptimization,
            "HOST_REWRITES entry `{}` must be a CanonicalOptimization",
            rewrite.name
        );
    }
}
