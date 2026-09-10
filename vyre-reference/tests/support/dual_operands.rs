//! The byte packing the dual-reference suites feed a facet.
//!
//! The seeded pair generator this module carried is one contract with one owner,
//! `vyre_test_support::scalar_corpora::hostile_pair`. What stays here is the
//! packing, which names this crate's wire layout and so is domain-specific to
//! it. Every target that includes this module calls it.

/// Pack one operand pair into the byte input a dual facet consumes.
pub(crate) fn binary_input(left: u32, right: u32) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(&[left, right])
}
