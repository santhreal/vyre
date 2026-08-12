//! Semantic operation tier classification contracts.

use vyre_foundation::operation::{classify_operation_id, OperationTier};

#[test]
fn classifies_known_namespaces_without_consumer_coupling() {
    assert_eq!(
        classify_operation_id("vyre-intrinsics::hardware::popcount_u32"),
        OperationTier::Intrinsic
    );
    assert_eq!(
        classify_operation_id("vyre-primitives::graph::toposort"),
        OperationTier::Primitive
    );
    assert_eq!(
        classify_operation_id("vyre-libs::scan::literal_set"),
        OperationTier::Library
    );
    assert_eq!(
        classify_operation_id("core.dispatch"),
        OperationTier::Runtime
    );
}

#[test]
fn classifies_external_crate_namespaces_generically() {
    assert_eq!(
        classify_operation_id("external_frontend::analysis::dataflow"),
        OperationTier::External
    );
    assert_eq!(
        classify_operation_id("community_pack::scan::signature"),
        OperationTier::External
    );
}

#[test]
fn classifies_unqualified_ids_as_unknown() {
    assert_eq!(
        classify_operation_id("not_a_namespace"),
        OperationTier::Unknown
    );
    assert_eq!(classify_operation_id(""), OperationTier::Unknown);
}
