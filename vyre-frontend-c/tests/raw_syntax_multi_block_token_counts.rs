//! The raw-byte syntax parser must count every token across every block.
//!
//! Sparse token compaction runs in two GPU stages. A block-total stage scans one
//! `BLOCK_LANES`-lane workgroup per block and writes that block's token count to
//! `block_totals[block]`; a compact stage then ranks each token by the scanned
//! prefix of those totals. The block-total stage's only sized buffer is
//! `block_totals` (one word per block) and its input arrives as a resident device
//! blob, so dispatch grid inference saw nothing bigger than `num_blocks / 1024`
//! and launched a single workgroup. Block 0 computed its total, every later block
//! kept the zero it was allocated with, and the reported token count silently
//! collapsed to `block_totals[0] + <tokens in the final block>`.
//!
//! That is a wrong answer, not a failure: a 66560-token translation unit reported
//! 2048 tokens and parsing carried on. The tests below pin the exact count at and
//! around each block boundary, because the old behavior was correct for one and
//! two blocks and only diverged from the third block on. Each source is `n`
//! semicolons, which the C lexer tokenizes one-for-one, so the expected count is
//! `n` with no estimation involved.
#![forbid(unsafe_code)]

use std::sync::{Mutex, MutexGuard};

use vyre_driver_cuda as _;
use vyre_driver_wgpu as _;
use vyre_frontend_c::api::parse_syntax_bytes;

/// One GPU parse at a time. The frontend shares a resident dispatch backend and
/// per-thread scratch, and a panic under a poisoned guard would cascade into
/// every sibling test.
fn parse_guard() -> MutexGuard<'static, ()> {
    static GUARD: Mutex<()> = Mutex::new(());
    GUARD
        .lock()
        .expect("raw-syntax multi-block token-count mutex poisoned")
}

/// Parse `n` semicolons and return the token count the parser reports.
fn semicolon_token_count(n: usize) -> u32 {
    let source = vec![b';'; n];
    parse_syntax_bytes(&source)
        .unwrap_or_else(|error| {
            panic!("raw-byte GPU syntax parse of {n} semicolons failed: {error}")
        })
        .token_count
}

/// Lanes per block in the sparse compaction stages. Mirrors
/// `vyre_primitives::reduce::multi_block_prefix_scan::BLOCK_LANES`; asserted
/// equal below so this file cannot drift from the primitive.
const BLOCK_LANES: usize = 1024;

/// The block size these tests are built around must be the one the pipeline uses.
///
/// Every boundary below is derived from `BLOCK_LANES`. If the primitive changed
/// its block size and this file did not, the boundary cases would land in the
/// middle of a block and stop testing the thing they were written for.
#[test]
fn block_lane_count_matches_the_prefix_scan_primitive() {
    assert_eq!(
        BLOCK_LANES as u32,
        vyre_primitives::reduce::multi_block_prefix_scan::BLOCK_LANES,
        "Fix: re-derive this file's boundary cases from the primitive's BLOCK_LANES."
    );
}

/// One partial block: the single-workgroup case that always worked.
#[test]
fn a_partial_first_block_counts_every_token() {
    let _guard = parse_guard();
    assert_eq!(semicolon_token_count(512), 512);
}

/// Exactly one full block.
#[test]
fn one_full_block_counts_every_token() {
    let _guard = parse_guard();
    assert_eq!(semicolon_token_count(BLOCK_LANES), BLOCK_LANES as u32);
}

/// The first token past one block, where a second block enters the scan.
#[test]
fn the_first_token_of_the_second_block_is_counted() {
    let _guard = parse_guard();
    assert_eq!(
        semicolon_token_count(BLOCK_LANES + 1),
        BLOCK_LANES as u32 + 1
    );
}

/// Two full blocks. The last size the one-workgroup bug still answered
/// correctly, so it is the control for the divergence cases below.
#[test]
fn two_full_blocks_count_every_token() {
    let _guard = parse_guard();
    assert_eq!(
        semicolon_token_count(2 * BLOCK_LANES),
        2 * BLOCK_LANES as u32
    );
}

/// The first size the one-workgroup bug got wrong: three blocks, of which the
/// middle one was never scanned. This reported 1025 instead of 2049.
#[test]
fn a_third_block_is_not_dropped_from_the_middle_of_the_scan() {
    let _guard = parse_guard();
    assert_eq!(
        semicolon_token_count(2 * BLOCK_LANES + 1),
        2 * BLOCK_LANES as u32 + 1
    );
}

/// Three full blocks. Reported 2048 instead of 3072.
#[test]
fn three_full_blocks_count_every_token() {
    let _guard = parse_guard();
    assert_eq!(
        semicolon_token_count(3 * BLOCK_LANES),
        3 * BLOCK_LANES as u32
    );
}

/// Eight blocks: the count must grow with the input rather than saturating.
///
/// Under the bug every input from three blocks up reported the same 2048, so a
/// test that only checked "more than two blocks" against a fixed number could
/// have passed by coincidence. This one pins a count no truncation produces.
#[test]
fn eight_full_blocks_count_every_token() {
    let _guard = parse_guard();
    assert_eq!(
        semicolon_token_count(8 * BLOCK_LANES),
        8 * BLOCK_LANES as u32
    );
}

/// Past the 65536-token AST window, the size the frontend's own cap-lift
/// regression exercises. Reported 2048 instead of 66560.
#[test]
fn a_source_beyond_one_ast_window_counts_every_token() {
    let _guard = parse_guard();
    let n = 65_536 + BLOCK_LANES;
    assert_eq!(semicolon_token_count(n), n as u32);
}

/// Token counts must be strictly increasing in input size across block
/// boundaries.
///
/// The bug's signature was a count that stopped tracking the input: 4096, 8192
/// and 66560 semicolons all reported 2048. Monotonicity catches that shape even
/// if a future truncation lands on different numbers than the ones pinned above.
#[test]
fn token_counts_increase_with_input_size_across_block_boundaries() {
    let _guard = parse_guard();
    let sizes = [
        BLOCK_LANES,
        2 * BLOCK_LANES,
        3 * BLOCK_LANES,
        5 * BLOCK_LANES,
        9 * BLOCK_LANES,
    ];
    let mut previous = 0;
    for size in sizes {
        let count = semicolon_token_count(size);
        assert_eq!(
            count, size as u32,
            "Fix: {size} semicolons must report {size} tokens, got {count}."
        );
        assert!(
            count > previous,
            "Fix: token count must grow with input size, {count} did not exceed {previous}."
        );
        previous = count;
    }
}
