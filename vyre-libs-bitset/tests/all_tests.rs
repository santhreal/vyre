//! One binary for every integration test in this crate.

/// WHY: a linker that drops unreferenced objects links no operation
/// registration into a binary that never names a partition anchor, and every
/// catalog read in it then walks an empty registry. The Apple linker does drop
/// them, so catalog cases reported the registration missing instead of a wrong
/// value, on that platform only.
///
/// # What it does not catch
///
/// The count proves the partition this binary anchors reached the registry. It
/// does not prove another partition did.
#[cfg(feature = "bitset")]
#[test]
fn the_operation_catalog_is_linked_into_this_binary() {
    vyre_libs_bitset::link_anchor();
    assert!(
        vyre_libs_builder::plumbing::registration::operation_catalog::library_entries().count() > 0,
        "Fix: name this crate's `link_anchor` in the harness root; the registry this binary links \
         is empty, so every catalog case in it asserts over nothing"
    );
}

#[macro_use]
#[path = "gate_fixtures/mod.rs"]
pub mod gate_fixtures;

pub use vyre_test_support::dense_matvec_cases;

#[path = "adversarial_bitset_contains.rs"]
pub mod adversarial_bitset_contains;

#[path = "adversarial_bitset_ops.rs"]
pub mod adversarial_bitset_ops;

#[path = "adversarial_bitset_reduce_matrix.rs"]
pub mod adversarial_bitset_reduce_matrix;

#[path = "adversarial_boolean_packing_four_russians_readiness.rs"]
pub mod adversarial_boolean_packing_four_russians_readiness;

#[path = "bitset_scalar_ir_parity_proptest.rs"]
pub mod bitset_scalar_ir_parity_proptest;

#[path = "bitset_word_contracts.rs"]
pub mod bitset_word_contracts;

#[path = "bitset_words_sizing_contracts.rs"]
pub mod bitset_words_sizing_contracts;

#[path = "four_russians_dense_matvec_generated.rs"]
pub mod four_russians_dense_matvec_generated;

#[path = "frontier_absorb_parity.rs"]
pub mod frontier_absorb_parity;

#[path = "proptest_bitset_words.rs"]
pub mod proptest_bitset_words;

#[path = "proptest_bitset_zero.rs"]
pub mod proptest_bitset_zero;

#[path = "sweep_bitset_oracle_matrix.rs"]
pub mod sweep_bitset_oracle_matrix;

#[path = "sweep_logical_reference_matrix.rs"]
pub mod sweep_logical_reference_matrix;
