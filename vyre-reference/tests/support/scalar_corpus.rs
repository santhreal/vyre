//! The scalar value corpora the reference sweeps draw their cases from.
//!
//! WHY: four sweep harnesses carried a copy of the same anchor list and their
//! own generator walk beside it. The anchors are the interesting bit patterns of
//! a width, so a copy that gains one and a copy that does not describe different
//! coverage under the same name, and nothing said which copy was authoritative.
//! Each width's anchors and each generator walk now have one owner here, and a
//! consumer composes them.
//!
//! Every walk is a fixed shift or multiply sequence seeded by a constant, so a
//! failing case is reproducible from its index alone.

#![allow(dead_code)]

/// The bit patterns a `u32` sweep must always contain.
///
/// Powers of two and their neighbours, the 16-bit boundary, the signed
/// boundaries reinterpreted as unsigned, and the top of the range.
pub(crate) fn u32_anchors() -> Vec<u32> {
    vec![
        0,
        1,
        2,
        3,
        7,
        8,
        15,
        16,
        31,
        32,
        63,
        64,
        127,
        128,
        255,
        256,
        1023,
        1024,
        u32::from(u16::MAX),
        u32::from(u16::MAX) + 1,
        i32::MAX as u32,
        i32::MIN as u32,
        u32::MAX - 1,
        u32::MAX,
    ]
}

/// The bit patterns a `u64` sweep must always contain.
pub(crate) fn u64_anchors() -> Vec<u64> {
    vec![
        0,
        1,
        2,
        3,
        7,
        8,
        15,
        16,
        31,
        32,
        63,
        64,
        127,
        128,
        255,
        256,
        1023,
        1024,
        u64::from(u16::MAX),
        u64::from(u16::MAX) + 1,
        u64::from(u32::MAX),
        u64::from(u32::MAX) + 1,
        i64::MAX as u64,
        i64::MIN as u64,
        u64::MAX - 1,
        u64::MAX,
    ]
}

/// The bit patterns an `i32` sweep must always contain.
///
/// Both signs of every magnitude boundary, since a signed operation can be
/// correct on one sign and wrong on the other.
pub(crate) fn i32_anchors() -> Vec<i32> {
    vec![
        i32::MIN,
        i32::MIN + 1,
        -1_000_000,
        -65_536,
        -257,
        -256,
        -129,
        -128,
        -2,
        -1,
        0,
        1,
        2,
        3,
        31,
        32,
        127,
        128,
        255,
        256,
        65_535,
        65_536,
        i32::MAX - 1,
        i32::MAX,
    ]
}

/// The bit patterns an `f32` sweep must always contain.
///
/// Both zeros, both infinities, the smallest normal and smallest subnormal of
/// each sign, a quiet NaN and a payload-bearing NaN.
pub(crate) fn f32_anchors() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        0.5,
        -0.5,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0x7FC0_1234),
    ]
}

/// The `u32` storage-graph corpus: anchors plus a rotated xorshift walk.
pub(crate) fn u32_corpus() -> Vec<u32> {
    let mut values = u32_anchors();
    let mut state = 0x9e37_79b9u32;
    for index in 0..512u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push(state.rotate_left(index & 31));
    }
    values.sort_unstable();
    values.dedup();
    values
}

/// The `u64` storage-graph corpus: anchors plus a rotated xorshift walk.
pub(crate) fn u64_corpus() -> Vec<u64> {
    let mut values = u64_anchors();
    let mut state = 0x243f_6a88_85a3_08d3u64;
    for index in 0..1024u32 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(state.rotate_left(index & 63));
    }
    values.sort_unstable();
    values.dedup();
    values
}

/// The `i32` storage-graph corpus: anchors plus a rotated linear-congruential
/// walk reinterpreted as signed, so negative operands are well represented.
pub(crate) fn i32_corpus() -> Vec<i32> {
    let mut values = i32_anchors();
    let mut state = 0x6a09_e667u32;
    for index in 0..512u32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(i32::from_ne_bytes(
            state.rotate_left(index & 31).to_ne_bytes(),
        ));
    }
    values.sort_unstable();
    values.dedup();
    values
}

/// The `f32` storage-graph corpus: anchors plus a xorshift bit-pattern walk.
///
/// Unsorted and undeduplicated, unlike the integer corpora: NaN has no total
/// order, so sorting a corpus that carries one is not defined.
pub(crate) fn f32_corpus() -> Vec<f32> {
    let mut values = f32_anchors();
    let mut state = 0x3c6e_f372u32;
    for index in 0..256u32 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push(f32::from_bits(state.rotate_left(index & 31)));
    }
    values
}

/// The `u32` corpus the dual evaluator sweep pairs exhaustively.
///
/// Shorter walk than [`u32_corpus`], and a multiplicative one rather than a
/// xorshift: that sweep evaluates every ordered pair, so its cost is quadratic
/// in the corpus length and the length is part of its contract.
pub(crate) fn u32_evaluator_corpus() -> Vec<u32> {
    let mut values = u32_anchors();
    let mut state = 0x9e37_79b9u32;
    for _ in 0..232 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        values.push(state);
    }
    values.sort_unstable();
    values.dedup();
    values
}
