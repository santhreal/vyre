//! Deterministic f32 sampling for the tensor-train decomposition suites.
//!
//! Both decomposition suites need the same uniform draw over `[-2.0, 2.0)`
//! seeded from `vyre_test_support::fixed_point::xorshift32`, and each stated its
//! own copy of the conversion. The range and the mantissa shift are one
//! decision, so they are stated once.

use vyre_test_support::fixed_point::xorshift32;

/// One uniform draw from `[-2.0, 2.0)`, advancing `state`.
pub(crate) fn rand_f32(state: &mut u32) -> f32 {
    let bits = xorshift32(state);
    ((bits >> 8) as f32 / (1u32 << 24) as f32) * 4.0 - 2.0
}
