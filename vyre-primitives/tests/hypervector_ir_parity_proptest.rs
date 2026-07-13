//! Tier 3 - Property: differential proptest driving the ACTUAL hyperdimensional-computing IR of
//! `hash::hypervector_xor_bind` and `hash::hypervector_majority_bundle` through `reference_eval` vs
//! their CPU oracles (`xor_bind_cpu`, `majority_bundle_cpu`). Both had `reference_eval` = 0.
//!
//! - `xor_bind`: per-word `out[t] = a[t] ^ b[t]` — the binding operation of VSA algebra; a lane/word
//!   misalignment diverges.
//! - `majority_bundle`: for each of the 32 bit positions in each word lane, counts set bits across the
//!   `k` stacked hypervectors and sets the output bit iff `count > k/2` (ties round to 0). This is a
//!   NESTED loop (32 bits × k vectors) with a strided `stacked[ii*dim_words + t]` gather — the exact
//!   shape where a wrong stride, an off-by-one on the tie threshold, or a bit-order slip hides.
//!
//! The sweep randomizes `dim_words` (1..=16) and `k` (1..=9, spanning even/odd so the tie rule is
//! exercised both ways), asserting the IR output bit-exact vs the oracle, plus deterministic
//! all-zeros / all-ones / single-vector anchors.
#![cfg(all(feature = "hash", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::hash::hypervector::{
    hypervector_majority_bundle, hypervector_xor_bind, majority_bundle_cpu, xor_bind_cpu,
};

fn pack(d: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(d))
}

fn decode(v: &Value) -> Vec<u32> {
    v.to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn run_xor_bind(a: &[u32], b: &[u32]) -> Vec<u32> {
    let dim_words = a.len() as u32;
    let program = hypervector_xor_bind("a", "b", "out", dim_words);
    // RW buffer `out` is binding 2; a,b are ReadOnly → out is the sole returned buffer.
    let outputs =
        vyre_reference::reference_eval(&program, &[pack(a), pack(b), pack(&vec![0u32; a.len()])])
            .expect("xor_bind reference evaluation must succeed");
    decode(&outputs[0])
}

fn run_majority(hvs: &[Vec<u32>], dim_words: u32) -> Vec<u32> {
    let k = hvs.len() as u32;
    let mut stacked = Vec::with_capacity((k * dim_words) as usize);
    for hv in hvs {
        stacked.extend_from_slice(hv);
    }
    let program = hypervector_majority_bundle("stacked", "out", dim_words, k);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[pack(&stacked), pack(&vec![0u32; dim_words as usize])],
    )
    .expect("majority_bundle reference evaluation must succeed");
    decode(&outputs[0])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn xor_bind_ir_matches_oracle(
        pairs in (1usize..=16).prop_flat_map(|n| (
            prop::collection::vec(any::<u32>(), n),
            prop::collection::vec(any::<u32>(), n),
        ))
    ) {
        let (a, b) = pairs;
        prop_assert_eq!(run_xor_bind(&a, &b), xor_bind_cpu(&a, &b));
    }

    #[test]
    fn majority_bundle_ir_matches_oracle(
        (dim_words, hvs) in (1usize..=16, 1usize..=9).prop_flat_map(|(d, k)| (
            Just(d as u32),
            prop::collection::vec(prop::collection::vec(any::<u32>(), d), k),
        ))
    ) {
        prop_assert_eq!(
            run_majority(&hvs, dim_words), majority_bundle_cpu(&hvs),
            "dim_words={} k={}", dim_words, hvs.len()
        );
    }
}

#[test]
fn hypervector_ir_anchors() {
    // xor_bind: self-inverse and identity.
    let a = vec![0xDEAD_BEEFu32, 0x1234_5678, 0xFFFF_0000];
    let zero = vec![0u32; 3];
    assert_eq!(run_xor_bind(&a, &a), vec![0, 0, 0], "a^a == 0");
    assert_eq!(run_xor_bind(&a, &zero), a, "a^0 == a");
    assert_eq!(run_xor_bind(&a, &a), xor_bind_cpu(&a, &a));

    // majority: unanimous all-ones stays ones; unanimous zeros stays zeros; a single vector is itself.
    let ones = vec![u32::MAX; 4];
    let z = vec![0u32; 4];
    assert_eq!(
        run_majority(&[ones.clone(), ones.clone(), ones.clone()], 4),
        ones
    );
    assert_eq!(run_majority(&[z.clone(), z.clone(), z.clone()], 4), z);
    let single = vec![0xA5A5_5A5Au32, 0x0F0F_F0F0, 1, u32::MAX];
    assert_eq!(
        run_majority(&[single.clone()], 4),
        majority_bundle_cpu(&[single])
    );
}
