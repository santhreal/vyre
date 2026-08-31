//! Frozen contracts for extension ids, collectives, and category predicates.
//!
//! These are backend-facing spec surfaces. They must stay stable because
//! conformance certificates and downstream extension crates can depend on
//! `vyre-spec` without linking the full `vyre` workspace.

use vyre_spec::extension::{
    ExtensionAtomicOpId, ExtensionBinOpId, ExtensionDataTypeId, ExtensionRuleConditionId,
    ExtensionTernaryOpId, ExtensionUnOpId,
};
use vyre_spec::{BackendAvailabilityPredicate, Category, CommGroup};

#[test]
fn every_extension_id_family_uses_the_reserved_high_bit_range() {
    let stable_name = "vendor.example.dtype";
    let ids = [
        ExtensionDataTypeId::from_name(stable_name).as_u32(),
        ExtensionBinOpId::from_name(stable_name).as_u32(),
        ExtensionUnOpId::from_name(stable_name).as_u32(),
        ExtensionAtomicOpId::from_name(stable_name).as_u32(),
        ExtensionTernaryOpId::from_name(stable_name).as_u32(),
        ExtensionRuleConditionId::from_name(stable_name).as_u32(),
    ];

    for raw in ids {
        assert_ne!(
            raw & ExtensionDataTypeId::EXTENSION_RANGE_MASK,
            0,
            "Fix: extension ids must set the high bit so wire decoders never confuse them with frozen core tags."
        );
    }
    assert!(
        ids.windows(2).all(|pair| pair[0] == pair[1]),
        "Fix: every extension-id family must share the same deterministic name-to-id contract."
    );
}

#[test]
fn communicator_world_group_and_category_predicates_are_spec_stable() {
    assert_eq!(CommGroup::WORLD.as_u32(), 0);
    assert_eq!(serde_json::to_string(&CommGroup::WORLD).unwrap(), "0");

    let cuda_only = BackendAvailabilityPredicate::new(|backend| backend == "cuda");
    assert!(cuda_only.available("cuda"));
    assert!(!cuda_only.available("wgpu"));
    assert!(!cuda_only.available("cpu"));

    let category = Category::C {
        hardware: "subgroup",
        backend_availability: cuda_only,
    };
    assert_eq!(
        category,
        Category::C {
            hardware: "subgroup",
            backend_availability: BackendAvailabilityPredicate::new(|_| false),
        },
        "Fix: Category equality is hardware-contract equality; predicate identity is intentionally not part of Eq."
    );
    assert!(!Category::A {
        composition_of: vec!["primitive.add"],
    }
    .is_unclassified());
    assert!(Category::unclassified().is_unclassified());
}
