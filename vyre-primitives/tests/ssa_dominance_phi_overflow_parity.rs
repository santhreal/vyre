//! `ssa_dominance_scan` phi-buffer overflow parity guard.
//!
//! Regression for a latent GPU/CPU parity OOB WRITE. `phi_idx` is a SHARED
//! running allocation offset (`atomic_add` of 4 per match across all lanes). When
//! total matches exceed `phi_words / 4`, the later offsets index PAST
//! `out_phi_nodes`. The reference interpreter SILENTLY DROPS those out-of-bounds
//! stores (masking the hazard), but a real GPU (CUDA does no bounds-checking)
//! would corrupt adjacent memory. The fix gates the three phi-record stores
//! against `buf_len(out_phi_nodes)` while STILL incrementing `out_phi_count` via
//! the atomic, the canonical GPU append-buffer overflow protocol, where the
//! count rising past capacity is the caller's LOUD overflow signal (not a silent
//! fallback, Law 10). This test drives an overflow and asserts zero OOB accesses.
#![cfg(feature = "parsing")]

use vyre_primitives::parsing::ast_ops::AST_ASSIGN;
use vyre_primitives::parsing::ssa_dominance_scan::ssa_dominance_scan_program;
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

#[test]
fn phi_stores_past_capacity_are_gated_not_oob_and_count_signals_overflow() {
    // Three assignments, all to variable 7, each in a DISTINCT block. Every earlier
    // assignment therefore has a later rival, so lanes t=0 and t=1 each allocate a
    // phi record (t=2 has no later node within num_nodes, so it allocates nothing).
    // `out_phi_nodes` holds only phi_words = 4 u32s = ONE 4-slot record, so the
    // SECOND allocation (phi_idx = 4) runs PAST the buffer end. Without the capacity
    // gate that is three OOB stores at indices 4,5,6.
    let num_nodes = 3u32;
    let phi_words = 4u32;
    let program = ssa_dominance_scan_program(num_nodes, phi_words);

    // `ast_rights` are all identical so the written record is `[7, 100, 100]`
    // regardless of WHICH lane wins the first slot, the reference's cross-lane
    // atomic order can then not affect the asserted bytes, keeping the assertion
    // exact (Law 6) and order-robust at once.
    let inputs = vec![
        Value::from(pack_u32_slice(&[AST_ASSIGN, AST_ASSIGN, AST_ASSIGN])), // ast_opcodes
        Value::from(pack_u32_slice(&[100, 100, 100])),                      // ast_rights
        Value::from(pack_u32_slice(&[7, 7, 7])),                            // ast_vals
        Value::from(pack_u32_slice(&[1, 2, 3])),                            // block_headers
        Value::from(vec![0u8; phi_words as usize * 4]),                     // out_phi_nodes
        Value::from(vec![0u8; 4]),                                          // out_phi_count
    ];

    let (outputs, report) = vyre_reference::reference_eval_oob_report(&program, &inputs)
        .expect("ssa_dominance_scan must reference-evaluate the overflow fixture");

    // THE parity assertion: zero out-of-bounds accesses. The interpreter would
    // silently drop an ungated OOB store, so before the fix `report.oob_stores`
    // would be 3 here (the second allocation's three writes at 4,5,6).
    assert_eq!(
        report.total(),
        0,
        "Fix: phi stores past out_phi_nodes capacity must be GATED (got {} OOB store(s), \
         {} OOB load(s)); an ungated store corrupts memory on a real GPU",
        report.oob_stores,
        report.oob_loads
    );

    let phi_out = vyre_reference::output_index(&program, "out_phi_nodes")
        .expect("out_phi_nodes is a reference output");
    let count_out = vyre_reference::output_index(&program, "out_phi_count")
        .expect("out_phi_count is a reference output");
    let phi_nodes = decode_u32_le_bytes_all(&outputs[phi_out].to_bytes());
    let count = decode_u32_le_bytes_all(&outputs[count_out].to_bytes());

    // The single in-capacity record: var_id = 7, rights[t] = 100, rights[lookahead]
    // = 100. Slot 3 is the 4th word of the allocation stride and is never written.
    assert_eq!(
        phi_nodes,
        vec![7, 100, 100, 0],
        "the one in-bounds phi record must be written exactly; the overflow record is dropped"
    );
    // Overflow is LOUD, not silent (Law 10): the atomic counted BOTH allocations
    // (2 × 4 = 8) even though only one fit, so count > phi_words tells the caller to
    // re-run with a larger buffer.
    assert_eq!(
        count,
        vec![8],
        "out_phi_count must count both allocations (8) so count > phi_words={phi_words} signals overflow"
    );
}
