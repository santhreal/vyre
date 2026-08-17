//! Shared adversarial bitset/reduce matrix.
//!
//! This replaces generated clone files with one deterministic matrix that
//! drives every scalar reducer, vector unary bitset map, and binary bitset map
//! through the same hostile length/value corpus.


mod wire_words;
use wire_words::{alternating, lcg_u32 as lcg, ramp};

type UnaryScalar = fn(&[u32]) -> u32;
type UnaryVector = fn(&[u32]) -> Vec<u32>;
type UnaryVectorInto = fn(&[u32], &mut Vec<u32>);
type BinaryVector = fn(&[u32], &[u32]) -> Vec<u32>;
type BinaryVectorInto = fn(&[u32], &[u32], &mut Vec<u32>);

#[test]
fn scalar_bitset_and_reduce_ops_cover_adversarial_matrix() {
    assert_unary_scalar("bitset_any", |input| u32::from(input.iter().any(|word| *word != 0)), |input| {
        u32::from(input.iter().any(|word| *word != 0))
    });
    assert_unary_scalar("reduce_all", |input| u32::from(input.iter().all(|value| *value != 0)), |input| {
        u32::from(input.iter().all(|value| *value != 0))
    });
    assert_unary_scalar("reduce_any", |input| u32::from(input.iter().any(|value| *value != 0)), |input| {
        u32::from(input.iter().any(|value| *value != 0))
    });
    assert_unary_scalar("reduce_count", |input| input.iter().map(|word| word.count_ones()).sum(), |input| {
        input.iter().map(|word| word.count_ones()).sum()
    });
    assert_unary_scalar(
        "reduce_count_non_zero",
        |input| input.iter().filter(|value| **value != 0).count() as u32,
        |input| input.iter().filter(|value| **value != 0).count() as u32,
    );
    assert_unary_scalar("reduce_max", |input| input.iter().copied().max().unwrap_or(0), |input| {
        input.iter().copied().max().unwrap_or(0)
    });
    assert_unary_scalar("reduce_min", |input| input.iter().copied().min().unwrap_or(u32::MAX), |input| {
        input.iter().copied().min().unwrap_or(u32::MAX)
    });
    assert_unary_scalar("reduce_sum", |input| input.iter().copied().fold(0u32, u32::wrapping_add), |input| {
        input.iter().copied().fold(0u32, u32::wrapping_add)
    });
}

#[test]
fn unary_bitset_maps_cover_adversarial_matrix_and_reused_outputs() {
    assert_unary_vector(
        "bitset_not",
        vyre_reference::composition_witness::bitset_not_witness,
        vyre_reference::composition_witness::bitset_not_witness_into,
        |input| input.iter().map(|word| !word).collect(),
    );
    assert_unary_vector(
        "bitset_popcount",
        vyre_reference::composition_witness::bitset_popcount_witness,
        vyre_reference::composition_witness::bitset_popcount_witness_into,
        |input| input.iter().map(|word| word.count_ones()).collect(),
    );
}

#[test]
fn binary_bitset_maps_cover_adversarial_matrix_and_reused_outputs() {
    assert_binary_vector(
        "bitset_and",
        vyre_reference::composition_witness::bitset_and_witness,
        vyre_reference::composition_witness::bitset_and_witness_into,
        |lhs, rhs| {
            lhs.iter()
                .zip(rhs)
                .map(|(left, right)| left & right)
                .collect()
        },
    );
    assert_binary_vector(
        "bitset_or",
        vyre_reference::composition_witness::bitset_or_witness,
        vyre_reference::composition_witness::bitset_or_witness_into,
        |lhs, rhs| {
            lhs.iter()
                .zip(rhs)
                .map(|(left, right)| left | right)
                .collect()
        },
    );
    assert_binary_vector(
        "bitset_xor",
        vyre_reference::composition_witness::bitset_xor_witness,
        vyre_reference::composition_witness::bitset_xor_witness_into,
        |lhs, rhs| {
            lhs.iter()
                .zip(rhs)
                .map(|(left, right)| left ^ right)
                .collect()
        },
    );
}

fn assert_unary_scalar(name: &str, actual: UnaryScalar, expected: UnaryScalar) {
    for (case_idx, input) in unary_cases().iter().enumerate() {
        assert_eq!(
            actual(input),
            expected(input),
            "Fix: {name} scalar adversarial case {case_idx} len={} must match the independent oracle.",
            input.len()
        );
    }
}

fn assert_unary_vector(
    name: &str,
    actual: UnaryVector,
    actual_into: UnaryVectorInto,
    expected: UnaryVector,
) {
    for (case_idx, input) in unary_cases().iter().enumerate() {
        let expected = expected(input);
        assert_eq!(
            actual(input),
            expected,
            "Fix: {name} vector adversarial case {case_idx} len={} must match the independent oracle.",
            input.len()
        );

        let mut reused = vec![0xA5A5_A5A5; input.len().saturating_add(9)];
        actual_into(input, &mut reused);
        assert_eq!(
            reused, expected,
            "Fix: {name} cpu_ref_into adversarial case {case_idx} must clear stale output capacity before writing."
        );
    }
}

fn assert_binary_vector(
    name: &str,
    actual: BinaryVector,
    actual_into: BinaryVectorInto,
    expected: BinaryVector,
) {
    for (case_idx, (lhs, rhs)) in binary_cases().iter().enumerate() {
        let expected = expected(lhs, rhs);
        assert_eq!(
            actual(lhs, rhs),
            expected,
            "Fix: {name} binary adversarial case {case_idx} lhs_len={} rhs_len={} must match the independent oracle.",
            lhs.len(),
            rhs.len()
        );

        let mut reused = vec![0x5A5A_5A5A; lhs.len().max(rhs.len()).saturating_add(11)];
        actual_into(lhs, rhs, &mut reused);
        assert_eq!(
            reused, expected,
            "Fix: {name} cpu_ref_into adversarial case {case_idx} must clear stale output capacity before writing."
        );
    }
}

fn unary_cases() -> Vec<Vec<u32>> {
    let mut cases = Vec::new();
    let lengths = [
        0usize, 1, 2, 3, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025,
    ];
    let fills = [
        0u32,
        1,
        u32::MAX,
        0x7FC0_0000,
        0x8000_0000,
        0x5555_5555,
        0xAAAA_AAAA,
        0xDEAD_BEEF,
    ];

    for len in lengths {
        for fill in fills {
            cases.push(vec![fill; len]);
        }
        cases.push(ramp(len, 0));
        cases.push(ramp(len, u32::MAX));
        cases.push(alternating(len, 0, u32::MAX));
        cases.push(alternating(len, 0x5555_5555, 0xAAAA_AAAA));
    }

    for seed in [
        0x0000_0001,
        0xC0FF_EE11,
        0xDEAD_BEEF,
        0xA5A5_5A5A,
        0x8000_0000,
        0xFFFF_FFFE,
    ] {
        for len in lengths {
            cases.push(lcg(seed, len));
        }
    }

    for live_prefix in 0..=256usize {
        let mut input = vec![0u32; 257];
        for (idx, word) in input.iter_mut().take(live_prefix).enumerate() {
            *word = 1u32.rotate_left((idx % 31) as u32);
        }
        cases.push(input);
    }

    cases
}

fn binary_cases() -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut cases = Vec::new();
    let lengths = [0usize, 1, 2, 31, 32, 33, 64, 65, 127, 128, 255, 256, 1024];
    let fills = [0u32, 1, u32::MAX, 0x7FC0_0000, 0x5555_5555, 0xAAAA_AAAA];

    for lhs_len in lengths {
        for rhs_len in lengths {
            cases.push((vec![0; lhs_len], vec![u32::MAX; rhs_len]));
            cases.push((ramp(lhs_len, 0x1357_9BDF), ramp(rhs_len, 0x2468_ACE0)));
            cases.push((
                alternating(lhs_len, 0x5555_5555, 0xAAAA_AAAA),
                alternating(rhs_len, 0xFFFF_0000, 0x0000_FFFF),
            ));
        }
    }

    for lhs_fill in fills {
        for rhs_fill in fills {
            cases.push((vec![lhs_fill; 1024], vec![rhs_fill; 1024]));
            cases.push((vec![lhs_fill; 1], vec![rhs_fill; 1024]));
            cases.push((vec![lhs_fill; 1024], vec![rhs_fill; 1]));
        }
    }

    for seed in [0x1234_5678, 0xA5A5_5A5A, 0xFFFF_FFFE, 0x8000_0000] {
        for len in lengths {
            cases.push((lcg(seed, len), lcg(seed.rotate_left(13), len)));
            cases.push((
                lcg(seed, len),
                lcg(seed.rotate_right(7), len.saturating_add(1)),
            ));
        }
    }

    cases
}
