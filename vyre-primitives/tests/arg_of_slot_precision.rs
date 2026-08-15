//! Slot precision of `arg_of_slot`, against its CPU reference.
//!
//! WHY this exists: `arg_of` masks on the generic `CALL_ARG` bit and every
//! caller-visible test used it, so the slot-precise builder shipped with no
//! proof that slot `N` selects only slot-`N` edges. That is the exact defect the
//! per-slot subkind bits were added to fix: a slot mask that quietly widened
//! back to `CALL_ARG` would restore the over-match and still pass a test that
//! only checks slot 0.
//!
//! Every slot in `0..=CALL_ARG_MAX_SLOT` is walked, plus the first slot past the
//! bit budget, whose documented behaviour is the recall-safe fallback to the
//! generic bit.
#![cfg(all(feature = "cpu-parity", feature = "predicate"))]

use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_primitives::predicate::arg_of::{arg_of_slot, cpu_ref_slot};
use vyre_primitives::predicate::edge_kind;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Four nodes; three call-argument edges carrying the generic bit and one slot
/// bit each: `0 -> 1` in slot 0, `0 -> 2` in slot 1, `3 -> 1` in slot 1.
const NODE_COUNT: u32 = 4;
const EDGE_OFFSETS: [u32; 5] = [0, 2, 2, 2, 3];
const EDGE_TARGETS: [u32; 3] = [1, 2, 1];
/// Destinations 1 and 2, the arguments both calls point at.
const FRONTIER_IN: [u32; 1] = [0b0110];

fn edge_kinds() -> [u32; 3] {
    [
        edge_kind::CALL_ARG | edge_kind::CALL_ARG_0,
        edge_kind::CALL_ARG | edge_kind::CALL_ARG_1,
        edge_kind::CALL_ARG | edge_kind::CALL_ARG_1,
    ]
}

/// Evaluate `arg_of_slot` on the fixture graph and return the output bitset.
fn gpu_frontier(slot: u32) -> Vec<u32> {
    let program = arg_of_slot(
        ProgramGraphShape::new(NODE_COUNT, EDGE_TARGETS.len() as u32),
        "fin",
        "fout",
        slot,
    );
    let inputs = [
        Value::from(pack(&[0u32; NODE_COUNT as usize])),
        Value::from(pack(&EDGE_OFFSETS)),
        Value::from(pack(&EDGE_TARGETS)),
        Value::from(pack(&edge_kinds())),
        Value::from(pack(&[0u32; NODE_COUNT as usize])),
        Value::from(pack(&FRONTIER_IN)),
        Value::from(pack(&[0u32])),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .unwrap_or_else(|error| panic!("arg_of_slot({slot}) must evaluate: {error}"));
    unpack(&outputs[0].to_bytes())
}

fn cpu_frontier(slot: u32) -> Vec<u32> {
    cpu_ref_slot(
        NODE_COUNT,
        &EDGE_OFFSETS,
        &EDGE_TARGETS,
        &edge_kinds(),
        &FRONTIER_IN,
        slot,
    )
}

#[test]
fn every_slot_matches_its_cpu_reference() {
    for slot in 0..=edge_kind::CALL_ARG_MAX_SLOT + 1 {
        assert_eq!(
            gpu_frontier(slot),
            cpu_frontier(slot),
            "arg_of_slot({slot}) must agree with cpu_ref_slot({slot})"
        );
    }
}

#[test]
fn a_slot_selects_only_the_edges_stamped_with_it() {
    // Slot 0 is carried by `0 -> 1` alone, so only node 0 lights up.
    assert_eq!(gpu_frontier(0), vec![0b0001]);
    // Slot 1 is carried by `0 -> 2` and `3 -> 1`, so both sources light up.
    assert_eq!(gpu_frontier(1), vec![0b1001]);
    // No edge carries slot 2, so the frontier is empty. A mask that widened
    // back to the generic bit would report every call instead.
    assert_eq!(gpu_frontier(2), vec![0]);
}

#[test]
fn a_slot_past_the_bit_budget_falls_back_to_every_call_argument_edge() {
    let beyond = edge_kind::CALL_ARG_MAX_SLOT + 1;
    assert_eq!(
        edge_kind::call_arg_slot(beyond),
        edge_kind::CALL_ARG,
        "the documented fallback for a slot past the budget is the generic bit"
    );
    // Recall-safe: every source of a call-argument edge into the frontier.
    assert_eq!(gpu_frontier(beyond), vec![0b1001]);
}
