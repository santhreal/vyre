//! Integration tests for self-substrate source boundary invariants.
//!
//! The forbidden-name list and the scan itself live in
//! `vyre_test_support::consumer_boundary`, so adding a downstream product
//! guards every platform crate in one edit. See that module for why the names
//! are spelled as `concat!` pairs.

use vyre_test_support::consumer_boundary::{
    assert_source_does_not_name_downstream_consumers, ConsumerBoundaryScan,
};
use vyre_test_support::monorepo::vyre_workspace_root;

#[test]
fn pass_engine_source_does_not_name_downstream_consumers() {
    assert_source_does_not_name_downstream_consumers(
        ConsumerBoundaryScan::for_crate(
            "vyre-pass-engine",
            vyre_workspace_root().join("vyre-pass-engine"),
        )
        .skipping_directories(&["archive", "release"])
        .with_rationale("vyre-pass-engine is a platform substrate crate"),
    );
}
