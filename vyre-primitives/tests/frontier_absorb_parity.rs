//! GPU-IR parity for the `bitset::frontier` fused frontier-absorption builders.
//!
//! `frontier_absorb_new_bits_program` and its `no_counts` / `for_node_count`
//! variants fuse one BFS frontier-closure step but had no parity test (found by
//! the registry-coverage closure gate). Per word `w` (lane `t < words`):
//!   domain_mask = (w == last_word) ? final_word_mask : 0xFFFF_FFFF
//!   new_bits    = (neighbors[w] & domain_mask) & !visited[w]
//!   next_wave[w] = new_bits
//!   visited[w]  |= new_bits
//!   added_counts[w] = popcount(new_bits)   // only the counted variant
//! The tail mask drops out-of-domain bits in the final word. This pins all three
//! against a hand-computed reference via `reference_eval`, asserting the exact
//! visited / next_wave / added_counts words (Testing-Contract: real values).
#![cfg(feature = "bitset")]

use vyre_foundation::ir::Program;
use vyre_primitives::bitset::frontier::{
    frontier_absorb_new_bits_for_node_count_program, frontier_absorb_new_bits_no_counts_program,
    frontier_absorb_new_bits_program,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Read a returned output buffer BY NAME rather than by fixed position. These builders
/// return several writable buffers (visited/next_wave/added_counts); locating each via
/// the interpreter's own `output_index` keeps the asserts correct even if the buffer
/// binding order changes or a fused intermediate is later inserted ahead of them 
/// the drift class that bit the multi-block prefix-scan harness.
fn out_by_name(program: &Program, outputs: &[Value], name: &str) -> Vec<u32> {
    let index = vyre_reference::output_index(program, name)
        .unwrap_or_else(|| panic!("Fix: frontier absorb program must declare output `{name}`"));
    unpack(&outputs[index].to_bytes())
}

// Shared fixture: 2 words, final word keeps only its low 4 bits (mask 0x0F).
const VISITED_INIT: [u32; 2] = [0b0011, 0b0000];
const NEIGHBORS: [u32; 2] = [0b1101, 0xFFFF_FFFF];
// w0: mask 0xFFFFFFFF; in_domain=0b1101; new=0b1101 & !0b0011 = 0b1100 (bits 2,3).
// w1: mask 0x0000000F; in_domain=0xFF & 0x0F = 0b1111; new=0b1111 & !0 = 0b1111.
const EXP_VISITED: [u32; 2] = [0b1111, 0b1111]; // old | new
const EXP_NEXT_WAVE: [u32; 2] = [0b1100, 0b1111]; // new_bits
const EXP_ADDED_COUNTS: [u32; 2] = [2, 4]; // popcount(new_bits)
const FINAL_WORD_MASK: u32 = 0x0000_000F;

#[test]
fn absorb_new_bits_counted_matches_reference() {
    let program = frontier_absorb_new_bits_program(
        "visited",
        "neighbors",
        "next_wave",
        "added_counts",
        2,
        FINAL_WORD_MASK,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&VISITED_INIT)),
            Value::from(pack(&NEIGHBORS)),
            Value::from(pack(&[0u32; 2])),
            Value::from(pack(&[0u32; 2])),
        ],
    )
    .expect("frontier absorb reference evaluation must succeed");
    assert_eq!(
        out_by_name(&program, &outputs, "visited"),
        EXP_VISITED,
        "visited"
    );
    assert_eq!(
        out_by_name(&program, &outputs, "next_wave"),
        EXP_NEXT_WAVE,
        "next_wave"
    );
    assert_eq!(
        out_by_name(&program, &outputs, "added_counts"),
        EXP_ADDED_COUNTS,
        "added_counts (popcount of new bits)"
    );
}

#[test]
fn absorb_new_bits_no_counts_matches_reference() {
    let program = frontier_absorb_new_bits_no_counts_program(
        "visited",
        "neighbors",
        "next_wave",
        2,
        FINAL_WORD_MASK,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&VISITED_INIT)),
            Value::from(pack(&NEIGHBORS)),
            Value::from(pack(&[0u32; 2])),
        ],
    )
    .expect("frontier absorb (no counts) reference evaluation must succeed");
    assert_eq!(
        out_by_name(&program, &outputs, "visited"),
        EXP_VISITED,
        "visited"
    );
    assert_eq!(
        out_by_name(&program, &outputs, "next_wave"),
        EXP_NEXT_WAVE,
        "next_wave"
    );
}

#[test]
fn absorb_for_node_count_derives_words_and_tail_mask() {
    // node_count = 36 → bitset_words = 2, tail_mask = (1<<4)-1 = 0x0F: exactly the
    // shared fixture's shape. This locks the node_count → (words, final_word_mask)
    // derivation (bitset_words + frontier_tail_mask) against the direct builder.
    let program = frontier_absorb_new_bits_for_node_count_program(
        "visited",
        "neighbors",
        "next_wave",
        "added_counts",
        36,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(&VISITED_INIT)),
            Value::from(pack(&NEIGHBORS)),
            Value::from(pack(&[0u32; 2])),
            Value::from(pack(&[0u32; 2])),
        ],
    )
    .expect("frontier absorb (for node count) reference evaluation must succeed");
    assert_eq!(
        out_by_name(&program, &outputs, "visited"),
        EXP_VISITED,
        "visited"
    );
    assert_eq!(
        out_by_name(&program, &outputs, "next_wave"),
        EXP_NEXT_WAVE,
        "next_wave"
    );
    assert_eq!(
        out_by_name(&program, &outputs, "added_counts"),
        EXP_ADDED_COUNTS,
        "added_counts"
    );
}
