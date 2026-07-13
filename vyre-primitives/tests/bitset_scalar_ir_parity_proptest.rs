//! Tier 3 - Property: differential proptest driving the ACTUAL GPU IR of the scalar/whole-buffer
//! bitset mutators — `bitset::zero`, `bitset::set_bit`, `bitset::clear_bit` — through `reference_eval`
//! vs each op's `cpu_ref`.
//!
//! MOTIVATION — real IR gap. These three ops have NO test that runs their Program through
//! `reference_eval` (`ir_parity_tests=0` in the coverage audit); their `sweep_*_volume_oracle_matrix`
//! peers assert `cpu_ref == a second CPU oracle`, and the only randomized-input IR check is the single
//! inventory fixture per op. The kernels are small but exact-index-critical: `set_bit`/`clear_bit`
//! bake `bit_idx` into a COMPILE-TIME `word = bit_idx/32`, `bit = bit_idx%32` with a `word < words`
//! store guard (an out-of-range `bit_idx` must be a silent no-op, matching `slice::get_mut`), and
//! `zero` stores 0 across all `words`. A wrong word/bit split, a `BitNot` mask error on clear, an
//! off-by-one on the store guard, or a missed word in `zero` diverges.
//!
//! NOTE: `bitset::copy` and `bitset::stochastic_and_mul` are deliberately EXCLUDED — they lower through
//! `binary_word.rs`, which is mid-refactor (git-dirty) in the current tree; testing against an
//! in-flight lowering would be flaky. They are recorded for the family sweep once that file settles.
#![cfg(all(feature = "bitset", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_reference::value::Value;

use vyre_primitives::bitset::{clear_bit, set_bit, zero};

fn pack(data: &[u32]) -> Value {
    Value::from(vyre_primitives::wire::pack_u32_slice(data))
}

/// Decode the sole RW `target` buffer (binding 0) from a reference_eval result.
fn decode(outputs: &[Value]) -> Vec<u32> {
    outputs[0]
        .to_bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn bitset_zero_ir_clears_every_word(target in prop::collection::vec(any::<u32>(), 1..=64)) {
        let words = target.len() as u32;
        let program = zero::bitset_zero("target", words);
        let outputs = vyre_reference::reference_eval(&program, &[pack(&target)])
            .expect("bitset_zero reference evaluation must succeed");
        let got = decode(&outputs);

        let mut want = target.clone();
        zero::cpu_ref(&mut want);
        prop_assert_eq!(&got, &want);
        prop_assert!(got.iter().all(|&w| w == 0), "zero must clear all words: {:?}", got);
    }

    #[test]
    fn bitset_set_bit_ir_matches_cpu_ref(
        target in prop::collection::vec(any::<u32>(), 1..=32),
        // In-range bit_idx: [0, words*32). Out-of-range is probed separately below because the
        // reference interpreter statically rejects the constant OOB store index even when guarded.
        raw_bit in 0u32..(32 * 32),
    ) {
        let words = target.len() as u32;
        let bit_idx = raw_bit % (words * 32);
        let program = set_bit::bitset_set_bit("target", bit_idx, words);
        let outputs = vyre_reference::reference_eval(&program, &[pack(&target)])
            .expect("bitset_set_bit reference evaluation must succeed");
        let got = decode(&outputs);

        let mut want = target.clone();
        set_bit::cpu_ref(&mut want, bit_idx);
        prop_assert_eq!(&got, &want, "words={} bit_idx={}", words, bit_idx);
    }

    #[test]
    fn bitset_clear_bit_ir_matches_cpu_ref(
        target in prop::collection::vec(any::<u32>(), 1..=32),
        raw_bit in 0u32..(32 * 32),
    ) {
        let words = target.len() as u32;
        let bit_idx = raw_bit % (words * 32);
        let program = clear_bit::bitset_clear_bit("target", bit_idx, words);
        let outputs = vyre_reference::reference_eval(&program, &[pack(&target)])
            .expect("bitset_clear_bit reference evaluation must succeed");
        let got = decode(&outputs);

        let mut want = target.clone();
        clear_bit::cpu_ref(&mut want, bit_idx);
        prop_assert_eq!(&got, &want, "words={} bit_idx={}", words, bit_idx);
    }
}

/// Deterministic boundary pins: the exact bit at each word seam (31/32/63/64...), all in-range for a
/// 4-word (128-bit) buffer, which the modular random draw is unlikely to hit precisely.
#[test]
fn bitset_scalar_ir_word_seam_boundaries() {
    let words = 4u32;
    let base = vec![0x0F0F_0F0Fu32, 0xFFFF_FFFF, 0x0000_0000, 0xAAAA_5555];
    for bit_idx in [0u32, 1, 31, 32, 33, 63, 64, 95, 96, 127] {
        // set_bit
        let program = set_bit::bitset_set_bit("target", bit_idx, words);
        let got = decode(
            &vyre_reference::reference_eval(&program, &[pack(&base)]).expect("set_bit eval"),
        );
        let mut want = base.clone();
        set_bit::cpu_ref(&mut want, bit_idx);
        assert_eq!(got, want, "set_bit seam bit_idx={bit_idx}");

        // clear_bit
        let program = clear_bit::bitset_clear_bit("target", bit_idx, words);
        let got = decode(
            &vyre_reference::reference_eval(&program, &[pack(&base)]).expect("clear_bit eval"),
        );
        let mut want = base.clone();
        clear_bit::cpu_ref(&mut want, bit_idx);
        assert_eq!(got, want, "clear_bit seam bit_idx={bit_idx}");
    }
}
