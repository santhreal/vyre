//! Operation identity namespace contracts.
//!
//! The namespace is the crate that minted the id. It is frozen at mint time, so
//! it is not a placement fact and no tier is derived from it. These cases pin
//! that separation: a `vyre-primitives::` id whose code lives in `vyre-libs`
//! still reads as a workspace identity, and nothing here returns a tier.

use vyre_foundation::operation::{operation_id_namespace, IdNamespace};

#[test]
fn reads_the_minting_crate_out_of_a_workspace_id() {
    assert_eq!(
        operation_id_namespace("vyre-primitives::hardware::popcount_u32"),
        IdNamespace::Workspace("vyre-primitives")
    );
    assert_eq!(
        operation_id_namespace("vyre-libs::scan::literal_set"),
        IdNamespace::Workspace("vyre-libs")
    );
}

/// A moved operation keeps the namespace of the crate that minted it.
///
/// Eighteen composition domains moved to `vyre-libs` keeping their
/// `vyre-primitives::` ids. Reading the prefix as the owner made 130
/// operations look misplaced and made 154 compositions look like hardware
/// intrinsics, so the prefix answers one question only: who published the id.
#[test]
fn a_moved_operation_keeps_its_minting_namespace() {
    assert_eq!(
        operation_id_namespace("vyre-primitives::graph::toposort"),
        IdNamespace::Workspace("vyre-primitives")
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
            operation_id_namespace(id),
            IdNamespace::Unknown,
            "`{id}` names a host capability, not an operation; OperationRegistry must refuse it"
        );
    }
}

#[test]
fn reads_a_consumer_crate_as_external() {
    assert_eq!(
        operation_id_namespace("external_frontend::analysis::dataflow"),
        IdNamespace::External("external_frontend")
    );
    assert_eq!(
        operation_id_namespace("community_pack::scan::signature"),
        IdNamespace::External("community_pack")
    );
}

#[test]
fn rejects_an_id_that_names_no_crate() {
    assert_eq!(
        operation_id_namespace("not_a_namespace"),
        IdNamespace::Unknown
    );
    assert_eq!(operation_id_namespace(""), IdNamespace::Unknown);
    assert_eq!(operation_id_namespace("::orphan"), IdNamespace::Unknown);
    assert_eq!(
        operation_id_namespace("vyre-libs::"),
        IdNamespace::Unknown,
        "a namespace with no path names no operation"
    );
}
