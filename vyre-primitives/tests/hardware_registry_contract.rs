//! Stable intrinsic catalog and deterministic fixture contracts.

use vyre_foundation::ir::Node;
use vyre_foundation::operation::{OperationRegistry, OperationTier};
use vyre_foundation::{ElementPolicy, Uniformity};
use vyre_primitives::hardware::all_entries;
use vyre_primitives::hardware::catalog::intrinsic_facet;

/// WHY: every registry member must retain its signature, deterministic fixture,
/// and typed physical-execution constraints as the intrinsic catalog grows.
#[test]
fn canonical_catalog_is_deterministic_and_fixture_complete() {
    let entries = all_entries().collect::<Vec<_>>();
    assert!(!entries.is_empty());
    assert!(entries.windows(2).all(|pair| pair[0].id < pair[1].id));

    for intrinsic in entries {
        assert_eq!(intrinsic.tier, OperationTier::Intrinsic);
        let canonical = OperationRegistry::global()
            .get(intrinsic.id)
            .expect("canonical intrinsic registration");
        assert_eq!(canonical.id, intrinsic.id);
        assert_eq!(canonical.semantic_version, intrinsic.semantic_version);
        assert_eq!(canonical.signature, intrinsic.signature);
        assert_eq!(canonical.tier, intrinsic.tier);
        let shape = intrinsic_facet(intrinsic.id)
            .unwrap_or_else(|| panic!("missing intrinsic facet for {}", intrinsic.id))
            .shape;
        let signature = intrinsic.signature.as_ref().expect("intrinsic signature");
        assert_eq!(signature.inputs.len(), shape.input_buffers as usize);
        assert_eq!(signature.outputs.len(), shape.output_buffers as usize);
        assert_eq!(
            intrinsic.geometry_requirements.per_invocation_elements,
            ElementPolicy::Scalar,
            "{} must declare its one-element-per-physical-invocation contract",
            intrinsic.id
        );
        let program = intrinsic.program().expect("neutral intrinsic builder");
        match program.entry().first() {
            Some(Node::Region { generator, .. }) => {
                assert_eq!(generator.as_str(), intrinsic.id);
            }
            _ => panic!("{} must build one canonical intrinsic region", intrinsic.id),
        }
        let constraints = intrinsic
            .schedule_constraints()
            .unwrap_or_else(|error| panic!("{} has invalid geometry: {error}", intrinsic.id));
        if program.stats().has_node_barrier() {
            assert_eq!(
                constraints.subgroup_uniformity,
                Uniformity::WorkgroupUniform,
                "{} barrier semantics require workgroup-uniform execution",
                intrinsic.id
            );
            assert!(
                constraints.memory_ordering.is_some(),
                "{} barrier semantics require a typed memory ordering",
                intrinsic.id
            );
        }
        if program.stats().subgroup_ops() {
            assert_eq!(
                constraints.subgroup_uniformity,
                Uniformity::SubgroupUniform,
                "{} subgroup semantics require subgroup-uniform execution",
                intrinsic.id
            );
        }

        let inputs = intrinsic.test_inputs.expect("deterministic inputs");
        let expected = intrinsic.expected_output.expect("deterministic outputs");
        assert_eq!(
            inputs(),
            inputs(),
            "{} input fixtures changed",
            intrinsic.id
        );
        assert_eq!(
            expected(),
            expected(),
            "{} expected fixtures changed",
            intrinsic.id
        );
    }
}
