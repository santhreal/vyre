//! Integration test verifying complete coverage of MemoryOrdering variants.

use vyre_test_support::memory_order_variants::{
    assert_covers_every_memory_ordering, memory_ordering_variant_samples,
    MEMORY_ORDERING_VARIANT_FLOOR,
};

#[test]
fn memory_ordering_universe_is_completely_covered() {
    let samples = memory_ordering_variant_samples();
    assert!(samples.len() >= MEMORY_ORDERING_VARIANT_FLOOR);
    assert_covers_every_memory_ordering(&samples);
}
