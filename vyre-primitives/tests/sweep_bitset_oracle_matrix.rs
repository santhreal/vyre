//! Oracle matrix for every packed-bitset word primitive.
//!
//! One generated case list, one assertion body per call shape, and one row per
//! operation. Each row carries an independent oracle written in terms of the
//! specified per-word arithmetic, never in terms of the production body: the
//! word maps recompute the element from `lhs`/`rhs` directly, the in-place ops
//! state their postcondition over the original buffer, and the index queries
//! recompute the addressed bit from its word and offset.
//!
//! `bitset_registry_is_fully_covered` closes the class: it reads the registered
//! bitset operation ids at run time and fails when one is neither in a row
//! above nor in the exemption list with the suite that does own it. A new
//! bitset primitive cannot be added and stay silently unswept.
//!
//! Volume testing.volume - do NOT weaken to shape-only asserts.

#![forbid(unsafe_code)]
#![cfg(all(feature = "bitset", feature = "cpu-parity"))]

use vyre_primitives::bitset::{
    and, and_into, and_not, and_not_into, any, clear_bit, contains, copy, equal, not, or, or_into,
    popcount, set_bit, stochastic_compute, subset_of, test_bit, xor, xor_into, zero,
};

/// Cases produced by each of the two generators below.
const CASES: usize = 16384;

type UnaryVector = fn(&[u32]) -> Vec<u32>;
type UnaryVectorInto = fn(&[u32], &mut Vec<u32>);
type BinaryVector = fn(&[u32], &[u32]) -> Vec<u32>;
type BinaryVectorInto = fn(&[u32], &[u32], &mut Vec<u32>);
type UnaryScalar = fn(&[u32]) -> u32;
type BinaryScalar = fn(&[u32], &[u32]) -> u32;
type IndexedScalar = fn(&[u32], u32) -> u32;
type InPlaceUnary = fn(&mut [u32]);
type InPlaceBinary = fn(&mut [u32], &[u32]);
type InPlaceIndexed = fn(&mut [u32], u32);
type InPlaceUnaryExpect = fn(&[u32]) -> Vec<u32>;
type InPlaceBinaryExpect = fn(&[u32], &[u32]) -> Vec<u32>;
type InPlaceIndexedExpect = fn(&[u32], u32) -> Vec<u32>;

/// `out[w]` from `lhs[w]` and `rhs[w]`, truncated to the shorter input.
const VECTOR_BINARY: &[(&str, BinaryVector, BinaryVectorInto, BinaryVector)] = &[
    (
        "vyre-primitives::bitset::and",
        and::cpu_ref,
        and::cpu_ref_into,
        expect_and,
    ),
    (
        "vyre-primitives::bitset::or",
        or::cpu_ref,
        or::cpu_ref_into,
        expect_or,
    ),
    (
        "vyre-primitives::bitset::xor",
        xor::cpu_ref,
        xor::cpu_ref_into,
        expect_xor,
    ),
    (
        "vyre-primitives::bitset::and_not",
        and_not::cpu_ref,
        and_not::cpu_ref_into,
        expect_and_not,
    ),
    (
        "vyre-primitives::bitset::stochastic_and_mul",
        stochastic_compute::cpu_ref,
        stochastic_compute::cpu_ref_into,
        expect_and,
    ),
];

/// `out[w]` from `input[w]`.
const VECTOR_UNARY: &[(&str, UnaryVector, UnaryVectorInto, UnaryVector)] = &[
    (
        "vyre-primitives::bitset::not",
        not::cpu_ref,
        not::cpu_ref_into,
        expect_not,
    ),
    (
        "vyre-primitives::bitset::popcount",
        popcount::cpu_ref,
        popcount::cpu_ref_into,
        expect_popcount,
    ),
];

/// One scalar answer over the whole bitset.
const SCALAR_UNARY: &[(&str, UnaryScalar, UnaryScalar)] =
    &[("vyre-primitives::bitset::any", any::cpu_ref, expect_any)];

/// One scalar answer over a pair of bitsets, length mismatch included.
const SCALAR_BINARY: &[(&str, BinaryScalar, BinaryScalar)] = &[
    (
        "vyre-primitives::bitset::equal",
        equal::cpu_ref,
        expect_equal,
    ),
    (
        "vyre-primitives::bitset::subset_of",
        subset_of::cpu_ref,
        expect_subset_of,
    ),
];

/// One addressed bit, in range and out of range.
const INDEXED_SCALAR: &[(&str, IndexedScalar, IndexedScalar)] = &[
    (
        "vyre-primitives::bitset::contains",
        contains::cpu_ref,
        expect_bit_at,
    ),
    (
        "vyre-primitives::bitset::test_bit",
        test_bit::cpu_ref,
        expect_bit_at,
    ),
];

/// In-place rewrite of `target` from `target` alone.
const IN_PLACE_UNARY: &[(&str, InPlaceUnary, InPlaceUnaryExpect)] = &[(
    "vyre-primitives::bitset::zero",
    zero::cpu_ref,
    expect_zeroed,
)];

/// In-place rewrite of `target` from `target` and `operand`.
const IN_PLACE_BINARY: &[(&str, InPlaceBinary, InPlaceBinaryExpect)] = &[
    (
        "vyre-primitives::bitset::copy",
        copy::cpu_ref,
        expect_copied,
    ),
    (
        "vyre-primitives::bitset::and_into",
        and_into::cpu_ref,
        expect_and_into,
    ),
    (
        "vyre-primitives::bitset::or_into",
        or_into::cpu_ref,
        expect_or_into,
    ),
    (
        "vyre-primitives::bitset::xor_into",
        xor_into::cpu_ref,
        expect_xor_into,
    ),
    (
        "vyre-primitives::bitset::and_not_into",
        and_not_into::cpu_ref,
        expect_and_not_into,
    ),
];

/// In-place rewrite of one addressed bit of `target`.
const IN_PLACE_INDEXED: &[(&str, InPlaceIndexed, InPlaceIndexedExpect)] = &[
    (
        "vyre-primitives::bitset::set_bit",
        set_bit::cpu_ref,
        expect_bit_set,
    ),
    (
        "vyre-primitives::bitset::clear_bit",
        clear_bit::cpu_ref,
        expect_bit_cleared,
    ),
];

/// Registered bitset ids this matrix deliberately does not sweep, and the
/// suite that owns each. A registered id absent from both lists fails.
#[cfg(feature = "inventory-registry")]
const EXEMPT: &[(&str, &str)] = &[
    (
        "vyre-primitives::bitset::select1_query",
        "vyre-libs/tests/succinct_rank_select_adversarial_contracts.rs",
    ),
    (
        "vyre-primitives::bitset::four_russians_apply_byte_lut",
        "vyre-primitives/tests/adversarial_boolean_packing_four_russians_readiness.rs",
    ),
    (
        "vyre-primitives::bitset::four_russians_dense_matvec_byte_lut",
        "vyre-primitives/tests/four_russians_dense_matvec_generated.rs",
    ),
];

#[test]
fn vector_binary_bitset_maps_match_independent_oracles() {
    for (name, actual, actual_into, expected) in VECTOR_BINARY {
        for (case_idx, (lhs, rhs)) in word_pairs().enumerate() {
            let expected_out = expected(&lhs, &rhs);
            assert_eq!(
                actual(&lhs, &rhs),
                expected_out,
                "Fix: {name} case {case_idx} lhs_len={} rhs_len={} must match the per-word oracle.",
                lhs.len(),
                rhs.len()
            );

            let mut reused = vec![0x5A5A_5A5A; lhs.len().max(rhs.len()).saturating_add(11)];
            actual_into(&lhs, &rhs, &mut reused);
            assert_eq!(
                reused, expected_out,
                "Fix: {name} cpu_ref_into case {case_idx} must clear stale output capacity before writing."
            );
        }
    }
}

#[test]
fn vector_unary_bitset_maps_match_independent_oracles() {
    for (name, actual, actual_into, expected) in VECTOR_UNARY {
        for (case_idx, input) in word_pairs().map(|(lhs, _)| lhs).enumerate() {
            let expected_out = expected(&input);
            assert_eq!(
                actual(&input),
                expected_out,
                "Fix: {name} case {case_idx} len={} must match the per-word oracle.",
                input.len()
            );

            let mut reused = vec![0xA5A5_A5A5; input.len().saturating_add(9)];
            actual_into(&input, &mut reused);
            assert_eq!(
                reused, expected_out,
                "Fix: {name} cpu_ref_into case {case_idx} must clear stale output capacity before writing."
            );
        }
    }
}

#[test]
fn scalar_unary_bitset_queries_match_independent_oracles() {
    for (name, actual, expected) in SCALAR_UNARY {
        for (case_idx, input) in word_pairs().map(|(lhs, _)| lhs).enumerate() {
            assert_eq!(
                actual(&input),
                expected(&input),
                "Fix: {name} case {case_idx} len={} must match the independent oracle.",
                input.len()
            );
        }
    }
}

#[test]
fn scalar_binary_bitset_queries_match_independent_oracles() {
    for (name, actual, expected) in SCALAR_BINARY {
        for (case_idx, (lhs, rhs)) in word_pairs().enumerate() {
            assert_eq!(
                actual(&lhs, &rhs),
                expected(&lhs, &rhs),
                "Fix: {name} case {case_idx} lhs_len={} rhs_len={} must match the independent oracle.",
                lhs.len(),
                rhs.len()
            );
        }
    }
}

#[test]
fn indexed_bitset_queries_match_independent_oracles() {
    for (name, actual, expected) in INDEXED_SCALAR {
        for (case_idx, (buf, _)) in word_pairs().enumerate() {
            for index in probe_indices(case_idx, buf.len()) {
                assert_eq!(
                    actual(&buf, index),
                    expected(&buf, index),
                    "Fix: {name} case {case_idx} index={index} len={} must match the addressed-bit oracle.",
                    buf.len()
                );
            }
        }
    }
}

#[test]
fn in_place_unary_bitset_updates_match_independent_oracles() {
    for (name, actual, expected) in IN_PLACE_UNARY {
        for (case_idx, target) in word_pairs().map(|(lhs, _)| lhs).enumerate() {
            let expected_out = expected(&target);
            let mut updated = target.clone();
            actual(&mut updated);
            assert_eq!(
                updated, expected_out,
                "Fix: {name} case {case_idx} len={} must leave the specified buffer state.",
                target.len()
            );
        }
    }
}

#[test]
fn in_place_binary_bitset_updates_match_independent_oracles() {
    for (name, actual, expected) in IN_PLACE_BINARY {
        for (case_idx, (target, operand)) in word_pairs().enumerate() {
            let expected_out = expected(&target, &operand);
            let mut updated = target.clone();
            actual(&mut updated, &operand);
            assert_eq!(
                updated, expected_out,
                "Fix: {name} case {case_idx} target_len={} operand_len={} must leave the specified buffer state.",
                target.len(),
                operand.len()
            );
        }
    }
}

#[test]
fn in_place_indexed_bitset_updates_match_independent_oracles() {
    for (name, actual, expected) in IN_PLACE_INDEXED {
        for (case_idx, (target, _)) in word_pairs().enumerate() {
            for index in probe_indices(case_idx, target.len()) {
                let expected_out = expected(&target, index);
                let mut updated = target.clone();
                actual(&mut updated, index);
                assert_eq!(
                    updated, expected_out,
                    "Fix: {name} case {case_idx} index={index} len={} must leave the specified buffer state.",
                    target.len()
                );
            }
        }
    }
}

#[cfg(feature = "inventory-registry")]
#[test]
fn bitset_registry_is_fully_covered() {
    let covered = swept_ids();
    for operation in vyre_foundation::operation::OperationRegistry::global().iter() {
        if !operation.id.starts_with("vyre-primitives::bitset::") {
            continue;
        }
        let exempt = EXEMPT.iter().find(|(id, _)| *id == operation.id);
        assert!(
            covered.contains(&operation.id) || exempt.is_some(),
            "Fix: registered bitset operation {} is swept by no row of this matrix and is not listed in EXEMPT. Add a row for it, or exempt it and name the suite that proves it.",
            operation.id
        );
    }
    for (id, owner) in EXEMPT {
        assert!(
            !covered.contains(id),
            "Fix: bitset operation {id} is both swept here and exempted to {owner}. Drop the exemption."
        );
        assert!(
            vyre_foundation::operation::OperationRegistry::global()
                .get(id)
                .is_some(),
            "Fix: exempted bitset operation {id} is no longer registered. Drop the exemption, or restore the registration {owner} proves."
        );
        let owner_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join(owner);
        assert!(
            owner_path.is_file(),
            "Fix: bitset operation {id} is exempted to {owner}, which is not a file in this workspace. Name the suite that actually proves it."
        );
    }
}

#[cfg(feature = "inventory-registry")]
fn swept_ids() -> Vec<&'static str> {
    let mut ids: Vec<&'static str> = Vec::new();
    ids.extend(VECTOR_BINARY.iter().map(|(id, ..)| *id));
    ids.extend(VECTOR_UNARY.iter().map(|(id, ..)| *id));
    ids.extend(SCALAR_UNARY.iter().map(|(id, ..)| *id));
    ids.extend(SCALAR_BINARY.iter().map(|(id, ..)| *id));
    ids.extend(INDEXED_SCALAR.iter().map(|(id, ..)| *id));
    ids.extend(IN_PLACE_UNARY.iter().map(|(id, ..)| *id));
    ids.extend(IN_PLACE_BINARY.iter().map(|(id, ..)| *id));
    ids.extend(IN_PLACE_INDEXED.iter().map(|(id, ..)| *id));
    ids
}

fn expect_and(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    zip_words(lhs, rhs, |left, right| left & right)
}

fn expect_or(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    zip_words(lhs, rhs, |left, right| left | right)
}

fn expect_xor(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    zip_words(lhs, rhs, |left, right| left ^ right)
}

fn expect_and_not(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    zip_words(lhs, rhs, |left, right| left & !right)
}

fn expect_not(input: &[u32]) -> Vec<u32> {
    input.iter().map(|word| !word).collect()
}

fn expect_popcount(input: &[u32]) -> Vec<u32> {
    input.iter().map(|word| word.count_ones()).collect()
}

fn expect_any(input: &[u32]) -> u32 {
    u32::from(input.iter().any(|word| *word != 0))
}

fn expect_equal(lhs: &[u32], rhs: &[u32]) -> u32 {
    u32::from(lhs.len() == rhs.len() && lhs.iter().zip(rhs).all(|(left, right)| left == right))
}

fn expect_subset_of(lhs: &[u32], rhs: &[u32]) -> u32 {
    let shared = lhs.len().min(rhs.len());
    let overlap_clean = lhs
        .iter()
        .zip(rhs)
        .all(|(left, right)| (left & !right) == 0);
    let tail_empty = lhs[shared..].iter().all(|word| *word == 0);
    u32::from(overlap_clean && tail_empty)
}

fn expect_bit_at(buf: &[u32], index: u32) -> u32 {
    let word = (index / 32) as usize;
    let offset = index % 32;
    buf.get(word)
        .map_or(0, |value| (value >> offset) & 1)
}

fn expect_zeroed(target: &[u32]) -> Vec<u32> {
    vec![0; target.len()]
}

fn expect_copied(target: &[u32], operand: &[u32]) -> Vec<u32> {
    overwrite_words(target, operand, |_, replacement| replacement)
}

fn expect_and_into(target: &[u32], operand: &[u32]) -> Vec<u32> {
    overwrite_words(target, operand, |original, operand| original & operand)
}

fn expect_or_into(target: &[u32], operand: &[u32]) -> Vec<u32> {
    overwrite_words(target, operand, |original, operand| original | operand)
}

fn expect_xor_into(target: &[u32], operand: &[u32]) -> Vec<u32> {
    overwrite_words(target, operand, |original, operand| original ^ operand)
}

fn expect_and_not_into(target: &[u32], operand: &[u32]) -> Vec<u32> {
    overwrite_words(target, operand, |original, operand| original & !operand)
}

fn expect_bit_set(target: &[u32], index: u32) -> Vec<u32> {
    update_addressed_word(target, index, |word, mask| word | mask)
}

fn expect_bit_cleared(target: &[u32], index: u32) -> Vec<u32> {
    update_addressed_word(target, index, |word, mask| word & !mask)
}

/// Per-word map over the shared prefix of two bitsets.
fn zip_words(lhs: &[u32], rhs: &[u32], combine: impl Fn(u32, u32) -> u32) -> Vec<u32> {
    let shared = lhs.len().min(rhs.len());
    (0..shared).map(|word| combine(lhs[word], rhs[word])).collect()
}

/// `target` after an in-place update over its shared prefix with `operand`.
///
/// Words past the shared prefix keep their original value: an in-place bitset
/// op is specified to touch only the words both buffers address.
fn overwrite_words(
    target: &[u32],
    operand: &[u32],
    combine: impl Fn(u32, u32) -> u32,
) -> Vec<u32> {
    let shared = target.len().min(operand.len());
    target
        .iter()
        .enumerate()
        .map(|(word, original)| {
            if word < shared {
                combine(*original, operand[word])
            } else {
                *original
            }
        })
        .collect()
}

/// `target` after a single-bit update, a no-op when the word is out of range.
fn update_addressed_word(
    target: &[u32],
    index: u32,
    combine: impl Fn(u32, u32) -> u32,
) -> Vec<u32> {
    let addressed = (index / 32) as usize;
    let mask = 1u32 << (index % 32);
    target
        .iter()
        .enumerate()
        .map(|(word, original)| {
            if word == addressed {
                combine(*original, mask)
            } else {
                *original
            }
        })
        .collect()
}

/// In-range, boundary, and far out-of-range bit indices for one case.
fn probe_indices(case_idx: usize, words: usize) -> [u32; 3] {
    let span = words as u32 * 32 + 17;
    [
        (case_idx as u32).wrapping_mul(0x9E37_79B9) % span,
        (case_idx as u32) % (words as u32 * 32 + 11),
        u32::MAX,
    ]
}

/// Every generated pair, from both generators.
///
/// The two generators produce different word content over the same length
/// distribution, so every op is swept over both populations rather than
/// whichever one its file happened to use.
fn word_pairs() -> impl Iterator<Item = (Vec<u32>, Vec<u32>)> {
    let wide = (0..CASES).map(|case| {
        let seed = case as u64 ^ 0xB175_E7B1_7500_0000;
        (
            lcg_u64(seed, lhs_len(seed)),
            lcg_u64(seed.rotate_left(17) ^ 0xD00D_F00D, rhs_len(seed)),
        )
    });
    let narrow = (0..CASES).map(|case| {
        let seed = case as u64 ^ 0xB175_E7B1_7500_0000;
        (
            lcg_u32(seed as u32, lhs_len(seed)),
            lcg_u32(seed.rotate_left(17) as u32 ^ 0xD00D_F00D, rhs_len(seed)),
        )
    });
    wide.chain(narrow)
}

fn lhs_len(seed: u64) -> usize {
    1 + ((seed >> 3) as usize % 129)
}

fn rhs_len(seed: u64) -> usize {
    1 + ((seed >> 11) as usize % 129)
}

fn lcg_u64(seed: u64, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state as u32)
                .rotate_left((idx % 31) as u32)
                .wrapping_mul(0x9E37_79B9)
        })
        .collect()
}

fn lcg_u32(seed: u32, len: usize) -> Vec<u32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223)
                .rotate_left((idx % 31) as u32);
            state ^ (idx as u32).wrapping_mul(0x85EB_CA6B)
        })
        .collect()
}
