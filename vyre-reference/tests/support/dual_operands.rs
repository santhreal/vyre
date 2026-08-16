//! The hostile operand pairs and byte packing the dual-reference suites share.
//!
//! WHY: four dual targets carried the same seeded pair generator and the same
//! two-word packing beside it. A copy that changes its multiplier sweeps a
//! different operand space under the same name, so both have one owner here and
//! every consumer calls it.

#![allow(dead_code)]

/// Derive one hostile operand pair from a seed.
pub(crate) fn hostile_pair(seed: u32) -> (u32, u32) {
    let left = seed
        .wrapping_mul(0x85eb_ca6b)
        .rotate_left((seed ^ 0x13) & 31);
    let right = seed
        .wrapping_mul(0xc2b2_ae35)
        .rotate_right((seed ^ 0x29) & 31);
    (left, right)
}

/// Pack one operand pair into the byte input a dual facet consumes.
pub(crate) fn binary_input(left: u32, right: u32) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(&[left, right])
}
