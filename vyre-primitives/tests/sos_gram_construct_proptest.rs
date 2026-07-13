//! Tier 3 - Property: proptest over random monomial-pair tables for `math::sos_certificate::sos_gram_construct`,
//! hardening the HIGH-severity OOB fix (SWEEP-parity-surface-math: `monomial_pairs` is UNVALIDATED
//! data, so a pair index >= coeff_count is an out-of-range gather that the IR must gate to 0 to match
//! the CPU reference's `p_coeffs.get(idx).unwrap_or(0)`; without the gate it OOB-reads p_coeffs on the
//! offending lane, UB on real hardware + divergence from the ref).
//!
//! For each of 6000 random instances the generator DELIBERATELY draws pair indices from a range wider
//! than `coeff_count`, so a large fraction of lanes hit the out-of-range → 0 path (the exact regime the
//! fix protects). It runs the ACTUAL IR through `reference_eval` and asserts bit-for-bit equality with
//! `sos_gram_construct_cpu`; shrinking auto-minimizes any counterexample. Complements the fixed-input
//! `sos_gram_oob_parity.rs` regression with randomized breadth over (m, coeff_count, indices, coeffs).
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::sos_certificate::{sos_gram_construct, sos_gram_construct_cpu};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn run_ir(monomial_pairs: &[u32], p_coeffs: &[u32], m: u32, coeff_count: u32) -> Vec<u32> {
    let program = sos_gram_construct("monomial_pairs", "p_coeffs", "gram", m, coeff_count);
    let cells = (m * m) as usize;
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(monomial_pairs)),
            Value::from(pack(p_coeffs)),
            Value::from(pack(&vec![0u32; cells])),
        ],
    )
    .expect("sos_gram_construct reference evaluation must succeed");
    let index = vyre_reference::output_index(&program, "gram")
        .expect("sos_gram_construct must declare output gram");
    unpack(&outputs[index].to_bytes())[..cells].to_vec()
}

prop_compose! {
    /// Random instance whose pair indices are drawn from `0..(coeff_count + 4)`, so a meaningful
    /// fraction exceed `coeff_count` and exercise the out-of-range → 0 gate (the fix under test).
    fn arb_instance()(m in 1u32..=8, coeff_count in 1u32..=16)
        (m in Just(m),
         coeff_count in Just(coeff_count),
         monomial_pairs in prop::collection::vec(0u32..(coeff_count + 4), (m * m) as usize),
         p_coeffs in prop::collection::vec(0u32..1000, coeff_count as usize))
        -> (u32, u32, Vec<u32>, Vec<u32>) {
        (m, coeff_count, monomial_pairs, p_coeffs)
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6000))]

    #[test]
    fn sos_gram_ir_matches_cpu_with_oob_pair_indices(
        (m, coeff_count, monomial_pairs, p_coeffs) in arb_instance()
    ) {
        let got = run_ir(&monomial_pairs, &p_coeffs, m, coeff_count);
        let want = sos_gram_construct_cpu(&monomial_pairs, &p_coeffs, m);
        prop_assert_eq!(
            &got, &want,
            "m={} coeff_count={} pairs={:?} coeffs={:?}: IR {:?} != cpu {:?}",
            m, coeff_count, monomial_pairs, p_coeffs, got, want
        );
    }

    /// Sanity: at least one lane in the whole sweep must actually hit the OOB → 0 path, else the fix
    /// is not being exercised. Checked per-case against the CPU ref (a pair index >= coeff_count must
    /// yield 0 in the output).
    #[test]
    fn oob_pair_index_yields_zero(
        (m, coeff_count, monomial_pairs, p_coeffs) in arb_instance()
    ) {
        let want = sos_gram_construct_cpu(&monomial_pairs, &p_coeffs, m);
        for (cell, &idx) in monomial_pairs.iter().enumerate() {
            if idx >= coeff_count {
                prop_assert_eq!(want[cell], 0, "OOB pair index {} at cell {} must map to 0", idx, cell);
            }
        }
    }
}
