//! Semantic operation tier classification contracts.

use vyre_foundation::operation::{classify_operation_id, OperationTier};

#[test]
fn classifies_known_namespaces_without_consumer_coupling() {
    assert_eq!(
        classify_operation_id("vyre-primitives::hardware::popcount_u32"),
        OperationTier::Intrinsic
    );
    assert_eq!(
        classify_operation_id("vyre-primitives::graph::toposort"),
        OperationTier::Intrinsic
    );
    assert_eq!(
        classify_operation_id("vyre-libs::scan::literal_set"),
        OperationTier::Library
    );
}

/// A host-side runtime capability has no operation namespace.
///
/// Indirect dispatch, NVMe ingest, and zero-copy mapping are reached through
/// the driver and runtime capability surfaces. They used to hold `core.`,
/// `io.` and `mem.` operation ids in the registry, which gave one capability a
/// second identity with no program, no fixtures, and no lowering.
#[test]
fn rejects_host_capability_namespaces() {
    for id in [
        "core.indirect_dispatch",
        "io.dma_from_nvme",
        "io.write_back_to_nvme",
        "mem.zerocopy_map",
        "mem.unmap",
    ] {
        assert_eq!(
            classify_operation_id(id),
            OperationTier::Unknown,
            "`{id}` names a host capability, not an operation; OperationRegistry must refuse it"
        );
    }
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
