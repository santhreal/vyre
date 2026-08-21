//! Adversarial contract tests for succinct rank/select metadata.
//!
//! Coverage: rank1_superblocks builder, rank1_query builder, boundary
//! traps, monotonicity, all-zeros / all-ones vectors, partial final
//! blocks, and cross-word query offsets.
//!
#![cfg(feature = "math-succinct")]
#![allow(deprecated)]
mod succinct_words;
mod wire_words;
use vyre_reference::value::Value;
use wire_words::u32_bytes;

// ---------------------------------------------------------------------------
// Rank1 superblocks  -  specific value contracts
// ---------------------------------------------------------------------------

#[test]
fn rank_superblocks_all_zeros_yields_zero_prefixes() {
    let bits = [0u32; 8];
    let outputs = succinct_words::superblocks(&bits, 8, 2);

    assert_eq!(
        outputs,
        vec![0, 0, 0, 0, 0],
        "all-zeros bitvector must produce all-zero superblocks"
    );
}

#[test]
fn rank_superblocks_all_ones_yields_linear_prefixes() {
    let bits = [0xFFFF_FFFFu32; 4];
    let outputs = succinct_words::superblocks(&bits, 4, 2);

    // Each word has 32 bits. Block 0 covers words 0..1 = 64 bits set.
    // Block 1 covers words 2..3 = another 64 bits set.
    // Superblock[0] = 0, [1] = 64, [2] = 128 (sentinel = total)
    assert_eq!(outputs, vec![0, 64, 128]);
}

#[test]
fn rank_superblocks_single_word() {
    let bits = [0b1011u32];
    let outputs = succinct_words::superblocks(&bits, 1, 1);

    // popcount(0b1011) = 3
    assert_eq!(outputs, vec![0, 3]);
}

#[test]
fn rank_superblocks_partial_final_block() {
    // 5 words, block size 2 words -> blocks at words 0,2,4 plus sentinel
    // = superblocks[0..=3] (0, count(words 0-1), count(words 0-3), total)
    let bits = [0b1111u32, 0b1111, 0b1111, 0b1111, 0b1111];
    let outputs = succinct_words::superblocks(&bits, 5, 2);

    // Each word popcount = 4.
    // sb[0] = 0
    // sb[1] = popcount(words 0..1) = 8
    // sb[2] = popcount(words 0..3) = 16
    // sb[3] = popcount(words 0..4) = 20 (sentinel = total)
    assert_eq!(outputs, vec![0, 8, 16, 20]);
}

#[test]
fn rank_superblocks_block_size_equal_word_count() {
    let bits = [0xFFFF_0000u32, 0x0000_FFFF, 0x1234_5678];
    let outputs = succinct_words::superblocks(&bits, 3, 3);

    let total = 16 + 16 + bits[2].count_ones();
    assert_eq!(outputs, vec![0, total]);
}

#[test]
fn rank_superblocks_block_size_one() {
    let bits = [0b1u32, 0b11, 0b111, 0b1111];
    let outputs = succinct_words::superblocks(&bits, 4, 1);

    // popcounts: 1, 2, 3, 4
    assert_eq!(outputs, vec![0, 1, 3, 6, 10]);
}

// ---------------------------------------------------------------------------
// Rank1 query  -  specific value contracts
// ---------------------------------------------------------------------------

#[test]
fn rank_query_at_bit_zero() {
    let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0u32];
    let superblocks = [0u32, 4, 20]; // from the harness fixture
    let queries = [0u32];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 2);

    // rank before bit 0 is always 0
    assert_eq!(got, vec![0]);
}

#[test]
fn rank_query_at_bit_one() {
    let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0u32];
    let superblocks = [0u32, 4, 20];
    let queries = [1u32];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 2);

    // bit0=1, so rank before bit1 = 1
    assert_eq!(got, vec![1]);
}

#[test]
fn rank_query_cross_word_boundary_31_32() {
    let bits = [0xFFFF_FFFFu32, 0x0000_0001u32];
    let superblocks = [0u32, 32, 33];
    let queries = [31u32, 32u32];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 1);

    // rank before bit 31 = 31 (bits 0..30 are all set)
    // rank before bit 32 = 32 (all bits in word 0 are set)
    assert_eq!(got, vec![31, 32]);
}

#[test]
fn rank_query_at_last_bit() {
    let bits = [0xFFFF_FFFFu32; 2];
    let superblocks = [0u32, 32, 64];
    let queries = [63u32];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 1);

    assert_eq!(got, vec![63]);
}

#[test]
fn rank_query_all_ones_monotonic() {
    let bits = [0xFFFF_FFFFu32; 4];
    let superblocks = [0u32, 64, 128];
    let queries: Vec<u32> = (0..=127).step_by(4).collect();
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 2);

    for (i, &q) in queries.iter().enumerate() {
        assert_eq!(
            got[i], q,
            "rank before bit {q} in all-ones vector must equal {q}"
        );
    }
}

#[test]
fn rank_query_all_zeros_always_zero() {
    let bits = [0u32; 8];
    // Need to build superblocks first
    let superblocks = succinct_words::superblocks(&bits, 8, 2);

    let queries = [0u32, 1, 31, 32, 63, 64, 127, 128, 255];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 2);

    assert!(
        got.iter().all(|&v| v == 0),
        "rank on all-zeros must be 0 everywhere"
    );
}

#[test]
fn rank_query_sparse_bits() {
    let bits = [0b1u32, 0, 0, 0x8000_0000]; // bits 0 and 127 set
    let superblocks = succinct_words::superblocks(&bits, 4, 2);

    let queries = [0u32, 1, 64, 127];
    let got = succinct_words::rank_query(&bits, &superblocks, &queries, 2);

    // rank(0)=0, rank(1)=1, rank(64)=1, rank(127)=1 (strictly before 127)
    assert_eq!(got, vec![0, 1, 1, 1]);
}

// ---------------------------------------------------------------------------
// Error / boundary contracts
// ---------------------------------------------------------------------------

#[test]
fn rank_builders_reject_zero_block_words() {
    let err = vyre_libs::math::succinct::try_rank1_superblocks("bits", "sb", 1, 0)
        .expect_err("zero block size must be rejected");
    assert_eq!(
        err.to_string(),
        "Fix: rank superblock size must be at least one u32 word"
    );

    let err = vyre_libs::math::succinct::try_rank1_query("bits", "sb", "q", "out", 1, 1, 0)
        .expect_err("zero block size in query must be rejected");
    assert_eq!(
        err.to_string(),
        "Fix: rank superblock size must be at least one u32 word"
    );
}

#[test]
fn rank_query_traps_out_of_bounds() {
    let program = vyre_libs::math::succinct::rank1_query("bits", "sb", "q", "out", 1, 1, 1);
    let result = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(&[0u32])),
            Value::from(u32_bytes(&[0u32, 0])),
            Value::from(u32_bytes(&[32u32])),
        ],
    );

    let err = result
        .expect_err("rank1_query must fail loudly when query offset addresses a missing word");
    assert!(
        err.to_string().contains("rank-query-out-of-bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn rank_query_traps_far_out_of_bounds() {
    let program = vyre_libs::math::succinct::rank1_query("bits", "sb", "q", "out", 2, 1, 1);
    let result = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(u32_bytes(&[0u32, 0])),
            Value::from(u32_bytes(&[0u32, 0, 0])),
            Value::from(u32_bytes(&[100u32])),
        ],
    );

    let err = result.expect_err("rank1_query must fail loudly for far-out-of-bounds bit indices");
    assert!(
        err.to_string().contains("rank-query-out-of-bounds"),
        "unexpected error: {err}"
    );
}

// ---------------------------------------------------------------------------
// Select1 query contracts
// ---------------------------------------------------------------------------

#[test]
fn select_query_specific_sparse_positions() {
    let bits = [0b1011u32, 0x8000_0000, 0xFFFF_0000, 0u32];
    let queries = [1u32, 2, 3, 4, 5, 20];
    let outputs = succinct_words::select_query(&bits, &queries);

    assert_eq!(
        outputs,
        vec![0, 1, 3, 63, 80, 95],
        "select1 must return zero-based bit positions for one-based ranks"
    );
}

#[test]
fn select_query_all_ones_is_rank_minus_one() {
    let bits = [0xFFFF_FFFFu32; 2];
    let queries = [1u32, 2, 31, 32, 33, 64];
    let outputs = succinct_words::select_query(&bits, &queries);

    assert_eq!(
        outputs,
        queries.iter().map(|rank| rank - 1).collect::<Vec<_>>()
    );
}

#[test]
fn select_query_traps_zero_rank() {
    let result = succinct_words::try_select_query(&[1u32], &[0u32]);

    let err = result.expect_err("select1_query must reject k == 0");
    assert!(
        err.to_string().contains("select-query-zero-rank"),
        "unexpected error: {err}"
    );
}

#[test]
fn select_query_traps_rank_past_total_popcount() {
    let result = succinct_words::try_select_query(&[0b1011u32], &[4u32]);

    let err = result.expect_err("select1_query must reject ranks beyond total popcount");
    assert!(
        err.to_string().contains("select-query-rank-out-of-bounds"),
        "unexpected error: {err}"
    );
}

#[test]
fn rank_metadata_remains_monotone_with_select1_available() {
    let bits = [0b1010_1010u32, 0x5555_5555, 0x0F0F_0F0F, 0xFF00_FF00];
    let sb = succinct_words::superblocks(&bits, 4, 1);
    let total: u32 = bits.iter().map(|w| w.count_ones()).sum();
    assert_eq!(
        sb[sb.len() - 1],
        total,
        "final sentinel must equal total popcount"
    );

    // Monotonicity: each superblock must be >= the previous
    for window in sb.windows(2) {
        assert!(
            window[1] >= window[0],
            "superblocks must be monotonically non-decreasing"
        );
    }
}
