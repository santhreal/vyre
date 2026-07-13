//! reference_eval parity for the three self-substrate `Program` builders the registry-closure
//! gate flagged as uncovered (no test named them): `build_dce_bfs_program` +
//! `build_persistent_bfs_program` (optimizer/dce_program.rs) and `dispatch_softmax`
//! (math/dataflow_compaction_pipeline.rs). Each is a thin wrapper over a private impl, so we
//! pin the OBSERVABLE behavior through the CPU reference interpreter, asserting exact bytes
//! (never `!is_empty`: Testing Contract).
//!
//! Drains the vyre-self-substrate slice of BACKLOG.md WIRING-tautology-closure-25crates.
#![forbid(unsafe_code)]

use vyre_foundation::ir::Program;
use vyre_primitives::graph::program_graph::ProgramGraphShape;
use vyre_reference::value::Value;
use vyre_self_substrate::math::dataflow_compaction_pipeline::dispatch_softmax;
use vyre_self_substrate::optimizer::dce_program::{
    build_dce_bfs_program, build_persistent_bfs_program,
};

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ---- CSR BFS (build_dce_bfs_program / build_persistent_bfs_program) ----

/// A single kind bit carried on both edges of the test graph.
const EDGE_KIND: u32 = 1;

/// Execute a BFS program over the 3-node line graph
///   0 --(kind=1)--> 1 --(kind=1)--> 2
/// seeded at frontier {0}. Returns `(frontier_out word, changed words)`.
///
/// Buffer/binding order (see `build_persistent_bfs_program_internal`): the five read-only
/// ProgramGraph buffers [nodes, edge_offsets, edge_targets, edge_kind_mask, node_tags], then
/// `frontier_in` (ReadOnly), then the two writable outputs `frontier_out` and `changed`.
fn run_line_bfs(program: &Program, changed_words: usize) -> (u32, Vec<u32>) {
    let pg_nodes = [0u32, 0, 0];
    // CSR row offsets: node0 owns edge [0,1); node1 owns [1,2); node2 owns [2,2).
    let pg_edge_offsets = [0u32, 1, 2, 2];
    let pg_edge_targets = [1u32, 2];
    let pg_edge_kind_mask = [EDGE_KIND, EDGE_KIND];
    let pg_node_tags = [0u32, 0, 0];
    let fin = [0b001u32]; // seed {0}
    let fout = [0u32]; // BFS seeds frontier_out from frontier_in in the entry
    let changed = vec![0u32; changed_words];

    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(pack(&pg_nodes)),
            Value::from(pack(&pg_edge_offsets)),
            Value::from(pack(&pg_edge_targets)),
            Value::from(pack(&pg_edge_kind_mask)),
            Value::from(pack(&pg_node_tags)),
            Value::from(pack(&fin)),
            Value::from(pack(&fout)),
            Value::from(pack(&changed)),
        ],
    )
    .expect("BFS program must execute under reference_eval");

    let frontier = unpack(&outputs[0].to_bytes())[0];
    let changed_out = unpack(&outputs[1].to_bytes());
    (frontier, changed_out)
}

#[test]
fn dce_bfs_reaches_the_full_line_graph() {
    // build_dce_bfs_program uses allow_mask = u32::MAX (every edge kind allowed) and a
    // non-sticky `changed` (count 1). Two hops from {0} must reach {0,1,2}.
    let program = build_dce_bfs_program(ProgramGraphShape::new(3, 2), 8);
    let (frontier, changed) = run_line_bfs(&program, 1);
    assert_eq!(
        frontier, 0b111,
        "DCE BFS from {{0}} over 0->1->2 must reach {{0,1,2}} (0b111), got {frontier:#05b}"
    );
    assert_eq!(changed.len(), 1, "non-sticky DCE `changed` buffer is a single word");
}

#[test]
fn persistent_bfs_honors_allow_mask_and_latches_sticky_changed() {
    // Matching allow_mask -> full reach; sticky `changed` slot 1 latches 1.
    let reachable = build_persistent_bfs_program(ProgramGraphShape::new(3, 2), 8, EDGE_KIND);
    let (reached, changed) = run_line_bfs(&reachable, 2);
    assert_eq!(
        reached, 0b111,
        "persistent BFS with allow_mask matching the edge kind must reach {{0,1,2}}"
    );
    assert_eq!(
        changed[1], 1,
        "sticky changed (slot 1) must latch 1 once any node is newly added across iterations"
    );

    // An allow_mask DISJOINT from the edge kind blocks every traversal: the frontier stays {0}.
    // This proves `allow_mask` is threaded into the emitted IR, not silently ignored.
    let blocked = build_persistent_bfs_program(ProgramGraphShape::new(3, 2), 8, EDGE_KIND << 1);
    let (blocked_frontier, _c) = run_line_bfs(&blocked, 2);
    assert_eq!(
        blocked_frontier, 0b001,
        "an allow_mask ({}) disjoint from the edge kind ({EDGE_KIND}) must block traversal, \
         leaving only the seed {{0}}",
        EDGE_KIND << 1
    );
}

// ---- fixed-point softmax (dispatch_softmax) ----

#[test]
fn dispatch_softmax_normalizes_precomputed_exponentials_in_16_16() {
    // dispatch_softmax delegates to the primitive `softmax_step`, which computes
    //   sum = Σ pre_exp[i];  out[i] = (pre_exp[i] << 16) / max(sum, 1)
    // For pre_exp = [1,2,3,4], sum = 10, so out[i] = pre_exp[i] * 65536 / 10 (integer div).
    let pre_exp = [1u32, 2, 3, 4];
    let out_init = [0u32; 4];
    let outputs = vyre_reference::reference_eval(
        &dispatch_softmax("pre_exp", "out", 4),
        &[Value::from(pack(&pre_exp)), Value::from(pack(&out_init))],
    )
    .expect("softmax program must execute under reference_eval");
    let out = unpack(&outputs[0].to_bytes());
    assert_eq!(
        &out[..4],
        &[6553u32, 13107, 19660, 26214],
        "16.16 fixed-point softmax over [1,2,3,4] (sum=10): pre_exp[i]*65536/10"
    );
}
