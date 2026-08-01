//! Integration tests for platform-driver source boundary invariants.
//!
//! The forbidden-name list and the scan itself live in
//! `vyre_test_support::consumer_boundary`, so adding a downstream product
//! guards every platform crate in one edit. See that module for why the names
//! are spelled as `concat!` pairs.

use vyre_test_support::consumer_boundary::{
    assert_source_does_not_name_downstream_consumers, ConsumerBoundaryScan,
};

#[test]
fn driver_source_does_not_name_downstream_consumers() {
    assert_source_does_not_name_downstream_consumers(
        ConsumerBoundaryScan::for_crate("vyre-driver", env!("CARGO_MANIFEST_DIR"))
            .with_rationale("vyre-driver is a platform crate"),
    );
}
