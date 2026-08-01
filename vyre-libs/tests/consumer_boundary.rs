//! Integration tests for library-layer source boundary invariants.
//!
//! The forbidden-name list and the scan itself live in
//! `vyre_test_support::consumer_boundary`, so adding a downstream product
//! guards every platform crate in one edit. See that module for why the names
//! are spelled as `concat!` pairs.

use vyre_test_support::consumer_boundary::{
    assert_source_does_not_name_downstream_consumers, ConsumerBoundaryScan,
};

/// vyre-libs is a substrate-neutral primitive library: no source file may name a
/// downstream consumer or sibling integration. The former security/dataflow
/// "bridge" exemption for the partner names is gone as of the 0.7.0 de-brand.
/// The dataflow bridge now carries the external engine only as a neutral
/// `external_*` symbol scheme and a runtime producer-id string, so the contract
/// is strict everywhere with no path-based carve-out.
#[test]
fn library_source_does_not_name_downstream_consumers() {
    assert_source_does_not_name_downstream_consumers(
        ConsumerBoundaryScan::for_crate("vyre-libs", env!("CARGO_MANIFEST_DIR"))
            .with_rationale("vyre-libs is a substrate-neutral primitive library"),
    );
}
