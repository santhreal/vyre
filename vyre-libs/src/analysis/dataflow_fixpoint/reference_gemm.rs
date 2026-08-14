//! Telemetered host entry point for the semiring matrix product.
//!
//! The product itself, and every fixpoint closure built on it, is owned by
//! `vyre_foundation::pass_substrate::dataflow_fixpoint`. What this crate adds is
//! the call counter, so a parity run can report how many host products it asked
//! for. The closures are re-exported from the owner directly rather than
//! wrapped: a wrapper that only forwards is a second place for the contract to
//! be stated, and the copies here had drifted into carrying the owner's whole
//! test module verbatim.

use super::Semiring;
use crate::telemetry::observability::{bump, dataflow_fixpoint_calls};
use vyre_foundation::pass_substrate::dataflow_fixpoint as foundation_dataflow;

/// Multiply matrices over the selected semiring through the reference oracle.
#[must_use]
pub fn reference_semiring_gemm(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
) -> Vec<u32> {
    let mut c = Vec::new();
    reference_semiring_gemm_into(a, b, m, n, k, semiring, &mut c);
    c
}

/// Multiply matrices over the selected semiring into caller-owned storage.
pub fn reference_semiring_gemm_into(
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
    c: &mut Vec<u32>,
) {
    bump(&dataflow_fixpoint_calls);
    foundation_dataflow::semiring_gemm_cpu_into(a, b, m, n, k, semiring, c);
}

#[cfg(test)]
mod tests {
    use super::{reference_semiring_gemm_into, Semiring};
    use vyre_foundation::pass_substrate::dataflow_fixpoint as foundation_dataflow;

    /// The counter is the only thing this crate adds, so the product must be
    /// the owner's byte for byte, and the caller's allocation must survive the
    /// call: a reallocation here would defeat the reason `_into` exists.
    #[test]
    fn telemetered_gemm_matches_the_owner_and_reuses_the_output() {
        let left = vec![1, 2, 3, 4, 5, 6];
        let right = vec![7, 8, 9, 10, 11, 12];
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();
        reference_semiring_gemm_into(&left, &right, 2, 2, 3, Semiring::Real, &mut out);
        let mut expected = Vec::new();
        foundation_dataflow::semiring_gemm_cpu_into(
            &left,
            &right,
            2,
            2,
            3,
            Semiring::Real,
            &mut expected,
        );
        assert_eq!(out, expected);
        assert_eq!(out.as_ptr(), ptr);
    }
}
