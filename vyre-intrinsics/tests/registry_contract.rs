//! Stable intrinsic catalog and deterministic fixture contracts.

use vyre_foundation::ir::Node;
use vyre_foundation::operation::{OperationRegistry, OperationTier};
use vyre_intrinsics::operation_catalog::{all_entries, intrinsic_facet};

/// Positive: the linked catalog is deterministic, fully signed, and its
/// fixtures are stable across repeated reads.
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
        let program = intrinsic.program().expect("neutral intrinsic builder");
        match program.entry().first() {
            Some(Node::Region { generator, .. }) => {
                assert_eq!(generator.as_str(), intrinsic.id);
            }
            _ => panic!("{} must build one canonical intrinsic region", intrinsic.id),
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
