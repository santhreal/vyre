//! Contract tests for succinct bitvector rank metadata.
//!
//! These tests exercise the public Cat-A builders through the reference
//! interpreter so the rank/select substrate has an executable oracle before
//! parser and graph code depend on it.

#![cfg(feature = "math-succinct")]
#![allow(deprecated)]
mod succinct_words;
mod wire_words;
use vyre_reference::value::Value;
use wire_words::u32_bytes;

#[test]
fn rank_superblocks_store_zero_prefix_and_total_sentinel() {
    let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0];
    let outputs = succinct_words::superblocks(&bits, bits.len() as u32, 2);

    assert_eq!(
        outputs,
        vec![0, 4, 20],
        "superblocks must be prefix counts plus a total-popcount sentinel"
    );
}

#[test]
fn rank_queries_count_bits_strictly_before_each_offset() {
    let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0];
    let superblocks = [0u32, 4, 20];
    let queries = [0u32, 1, 4, 63, 64, 80, 112, 127];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 2);

    assert_eq!(
        got,
        vec![0, 1, 3, 3, 4, 4, 20, 20],
        "rank is exclusive of the queried bit offset"
    );
}

#[test]
fn rank_builders_reject_zero_word_superblocks() {
    let err = vyre_libs::math::succinct::try_rank1_superblocks("bits", "superblocks", 1, 0)
        .expect_err("zero-sized superblocks must be rejected");
    assert_eq!(
        err.to_string(),
        "Fix: rank superblock size must be at least one u32 word"
    );

    let err = vyre_libs::math::succinct::try_rank1_query(
        "bits",
        "superblocks",
        "queries",
        "out",
        1,
        1,
        0,
    )
    .expect_err("zero-sized query superblocks must be rejected");
    assert_eq!(
        err.to_string(),
        "Fix: rank superblock size must be at least one u32 word"
    );
}

#[test]
fn rank_query_traps_out_of_bounds_offsets() {
    let program =
        vyre_libs::math::succinct::rank1_query("bits", "superblocks", "queries", "out", 1, 1, 1);
    let result = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(&[0u32])),
            Value::from(u32_bytes(&[0u32, 0])),
            Value::from(u32_bytes(&[32u32])),
        ],
    );

    let err =
        result.expect_err("rank1_query must fail loudly when a query addresses a missing word");
    assert!(
        err.to_string().contains("rank-query-out-of-bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn rank_superblocks_carry_across_more_blocks_than_lanes() {
    // 300 one-word superblocks over a 256-lane workgroup: the scan runs twice
    // and the second pass has to open at the first pass's total. A dropped
    // carry leaves every superblock past 255 short by the count of the first
    // 256 words, which the totals of a single-pass bitvector cannot show.
    let bits: Vec<u32> = (0..300u32)
        .map(|word| word.wrapping_mul(2_654_435_761))
        .collect();
    let outputs = succinct_words::superblocks(&bits, bits.len() as u32, 1);

    let mut expected = Vec::with_capacity(bits.len() + 1);
    let mut prefix = 0u32;
    for word in &bits {
        expected.push(prefix);
        prefix += word.count_ones();
    }
    expected.push(prefix);
    assert_eq!(outputs, expected);
}
