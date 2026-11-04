//! Integration tests for platform-layer source boundary invariants.
//!
//! The forbidden-name list and the scan itself live in
//! `vyre_test_support::consumer_boundary`, so adding a downstream product
//! guards every platform crate in one edit. See that module for why the names
//! are spelled as `concat!` pairs.

use vyre_test_support::consumer_boundary::{
    assert_source_does_not_name_downstream_consumers, ConsumerBoundaryScan,
};
use vyre_test_support::monorepo::vyre_crate_directory;

#[test]
fn foundation_source_does_not_name_downstream_consumers() {
    assert_source_does_not_name_downstream_consumers(
        ConsumerBoundaryScan::for_crate(
            "vyre-foundation",
            vyre_crate_directory(env!("CARGO_PKG_NAME")),
        )
        .with_rationale("vyre-foundation is a platform crate"),
    );
}
