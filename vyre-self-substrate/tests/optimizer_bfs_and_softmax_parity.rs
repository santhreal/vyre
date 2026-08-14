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
use vyre_libs::solvers::dataflow_compaction_pipeline::dispatch_softmax;
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
/// `frontier_in` (ReadOnly), then the three writable outputs `frontier_out`, `changed`, and
/// `converged`.
fn run_line_bfs(program: &Program, changed_words: usize) -> (u32, Vec<u32>, u32) {
    let pg_nodes = [0u32, 0, 0];
    // CSR row offsets: node0 owns edge [0,1); node1 owns [1,2); node2 owns [2,2).
    let pg_edge_offsets = [0u32, 1, 2, 2];
    let pg_edge_targets = [1u32, 2];
    let pg_edge_kind_mask = [EDGE_KIND, EDGE_KIND];
    let pg_node_tags = [0u32, 0, 0];
    let fin = [0b001u32]; // seed {0}
    let fout = [0u32]; // BFS seeds frontier_out from frontier_in in the entry
    let changed = vec![0u32; changed_words];
    let converged = [0u32];

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
            Value::from(pack(&converged)),
        ],
    )
    .expect("BFS program must execute under reference_eval");

    let frontier = unpack(&outputs[0].to_bytes())[0];
    let changed_out = unpack(&outputs[1].to_bytes());
    let converged_out = unpack(&outputs[2].to_bytes())[0];
    (frontier, changed_out, converged_out)
}

/// Execute a BFS program over an arbitrary CSR graph in a chosen workgroup order.
/// Returns `(frontier_out words, changed words, converged)`.
fn run_bfs_graph(
    program: &Program,
    node_count: u32,
    edge_targets: &[u32],
    edge_offsets: &[u32],
    seed_word: u32,
    reversed: bool,
) -> (Vec<u32>, Vec<u32>, u32) {
    let words = ((node_count + 31) / 32) as usize;
    let changed_words = program
        .buffers()
        .iter()
        .find(|buffer| buffer.name() == "changed")
        .expect("the BFS program must declare its convergence-flag buffer")
        .count() as usize;
    let mut fin = vec![0u32; words];
    fin[0] = seed_word;
    let inputs = [
        Value::from(pack(&vec![0u32; node_count as usize])),
        Value::from(pack(edge_offsets)),
        Value::from(pack(edge_targets)),
        Value::from(pack(&vec![EDGE_KIND; edge_targets.len()])),
        Value::from(pack(&vec![0u32; node_count as usize])),
        Value::from(pack(&fin)),
        Value::from(pack(&vec![0u32; words])),
        Value::from(pack(&vec![0u32; changed_words])),
        Value::from(pack(&[0u32])),
    ];
    let outputs = if reversed {
        vyre_reference::reference_eval_lane_reversed(program, &inputs)
    } else {
        vyre_reference::reference_eval(program, &inputs)
    }
    .expect("BFS program must execute under the reference interpreter");
    (
        unpack(&outputs[0].to_bytes()),
        unpack(&outputs[1].to_bytes()),
        unpack(&outputs[2].to_bytes())[0],
    )
}

/// A two-level star that forces the SECOND level's discovery onto a source index
/// only a second workgroup's lanes reach at stride 0: `0 -> 1500`, then
/// `1500 -> every other node`. 2000 nodes, 1999 edges.
fn two_level_star_2000() -> (u32, Vec<u32>, Vec<u32>) {
    let node_count = 2000_u32;
    let hub = 1500_u32;
    let mut targets = vec![hub];
    targets.extend((1..node_count).filter(|&j| j != hub));
    // Node 0 owns edge [0,1); the hub owns [1, 1999); everyone else owns nothing.
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    for node in 0..=node_count {
        offsets.push(match node {
            0 => 0,
            n if n <= hub => 1,
            _ => targets.len() as u32,
        });
    }
    (node_count, targets, offsets)
}

/// PROBE, NOT A SOUNDNESS PROOF. Read the second paragraph before citing this test.
///
/// At 2000 nodes `build_dce_bfs_program` spans TWO workgroups of 1024, and its
/// persistent loop shares ONE global `changed` word across them: a plain store
/// clears it, gated on global `gid_x == 0`, so exactly one lane in the whole grid
/// clears, and every group `atomic_or`s the same word. The only fences in the loop
/// are workgroup-scoped, so nothing orders one group's set against another group's
/// early-exit read. The graph here puts the entire second-level discovery on hub
/// source 1500, which lane 1500 (workgroup 1) reaches at stride 0, and discovery is
/// attributed EXCLUSIVELY to whichever group's `atomic_or` on `frontier_out`
/// actually flips the bit, so the hub's discovery can belong to workgroup 1 alone.
/// On real hardware workgroup 0 can then read a flag workgroup 1 has not yet set,
/// take the early exit, store `converged = 1` and report a truncated closure as a
/// true fixpoint.
///
/// This test asserts both workgroup orders produce the FULL closure, and it passes.
/// That agreement is NOT evidence the site is sound. The reference interpreter runs
/// each workgroup's ENTIRE persistent loop to completion before starting the next,
/// so the groups never interleave WITHIN an iteration, and the race lives strictly
/// there. Forward, workgroup 0's stride-1 pass covers the hub itself and wins every
/// flip; reversed, workgroup 1 runs before the entry seeds `frontier_out` (the seed
/// is gated `t < 63`, all in workgroup 0), so it finds an empty frontier, exits and
/// contributes nothing. Both schedules hide the defect by construction, and no
/// workgroup order available to this interpreter can expose it.
///
/// So what this pins is the REACH of the CPU-reference evidence, not the safety of
/// the site. Do not read a green run here as a soundness argument for the shared
/// cleared flag; that argument was made from redundant node coverage, and redundant
/// coverage does not imply redundant discovery attribution.
///
/// AND THE REVERSED-ORDER RESULT IS ITSELF A SECOND DEFECT, not a scheduling
/// artifact. Workgroup 1 contributing nothing is not evidence that workgroup 1
/// cannot matter; it is evidence that the entry seed of `frontier_out` is ordered
/// against other groups' reads by a WORKGROUP-scoped barrier only, so a group that
/// runs before the seeding group reads an unseeded frontier. That bug is what hides
/// the flag race in this order. Two defects at one site, the second masking the
/// first, is why "group 1 did nothing, so the shared flag is survivable here" reads
/// as a safety argument and is not one.
#[test]
fn dce_bfs_multi_workgroup_agreement_is_a_backend_limit_not_a_soundness_proof() {
    let (node_count, targets, offsets) = two_level_star_2000();
    let program =
        build_dce_bfs_program(ProgramGraphShape::new(node_count, targets.len() as u32), 8);

    let span = program
        .buffers()
        .iter()
        .map(|buffer| buffer.count())
        .max()
        .expect("the BFS program must declare buffers");
    assert_eq!(
        span.div_ceil(program.workgroup_size()[0]),
        2,
        "this fixture must actually span two workgroups, or it probes nothing"
    );

    // Every node is reachable: 0 -> 1500 -> all the rest. 2000 bits set.
    let mut expected = vec![u32::MAX; 62];
    expected.push(0x0000_FFFF);

    for reversed in [false, true] {
        let (frontier, _changed, converged) =
            run_bfs_graph(&program, node_count, &targets, &offsets, 0b1, reversed);
        assert_eq!(
            frontier, expected,
            "the closure must contain all 2000 nodes (reversed={reversed})"
        );
        assert_eq!(
            converged, 1,
            "the loop must reach a real fixpoint within 8 iterations (reversed={reversed})"
        );
    }
}

#[test]
fn dce_bfs_reaches_the_full_line_graph() {
    // build_dce_bfs_program uses allow_mask = u32::MAX (every edge kind allowed) and a
    // non-sticky `changed` (count 1). Two hops from {0} must reach {0,1,2}.
    let program = build_dce_bfs_program(ProgramGraphShape::new(3, 2), 8);
    let (frontier, changed, converged) = run_line_bfs(&program, 1);
    assert_eq!(
        frontier, 0b111,
        "DCE BFS from {{0}} over 0->1->2 must reach {{0,1,2}} (0b111), got {frontier:#05b}"
    );
    assert_eq!(
        changed.len(),
        1,
        "non-sticky DCE `changed` buffer is a single word"
    );
    assert_eq!(
        converged, 1,
        "a 3-node line graph closes in 2 hops with an 8-iteration budget, so the kernel must \
         take the early-exit branch and report a real fixpoint"
    );
}

/// A budget too small to close the graph must report `converged == 0` while still
/// returning the partial frontier it did reach. This is the signal every consumer
/// keys off: without it a truncated closure is indistinguishable from a fixpoint,
/// and DCE would delete code that is reachable but not yet discovered.
#[test]
fn dce_bfs_reports_a_truncated_closure_when_the_iteration_budget_runs_out() {
    let program = build_dce_bfs_program(ProgramGraphShape::new(3, 2), 1);
    let (frontier, _changed, converged) = run_line_bfs(&program, 1);
    assert_eq!(
        frontier, 0b011,
        "one iteration from {{0}} over 0->1->2 reaches only {{0,1}}, got {frontier:#05b}"
    );
    assert_eq!(
        converged, 0,
        "the loop used its whole 1-iteration budget while still growing, so this is a partial \
         closure and must not be reported as converged"
    );
}

#[test]
fn persistent_bfs_honors_allow_mask_and_latches_sticky_changed() {
    // Matching allow_mask -> full reach; sticky `changed` slot 1 latches 1.
    let reachable = build_persistent_bfs_program(ProgramGraphShape::new(3, 2), 8, EDGE_KIND);
    let (reached, changed, converged) = run_line_bfs(&reachable, 2);
    assert_eq!(
        reached, 0b111,
        "persistent BFS with allow_mask matching the edge kind must reach {{0,1,2}}"
    );
    assert_eq!(
        changed[1], 1,
        "sticky changed (slot 1) must latch 1 once any node is newly added across iterations"
    );
    assert_eq!(
        converged, 1,
        "the traversal closes well inside its 8-iteration budget, so the sticky variant must \
         report a fixpoint too"
    );

    // An allow_mask DISJOINT from the edge kind blocks every traversal: the frontier stays {0}.
    // This proves `allow_mask` is threaded into the emitted IR, not silently ignored.
    let blocked = build_persistent_bfs_program(ProgramGraphShape::new(3, 2), 8, EDGE_KIND << 1);
    let (blocked_frontier, _c, blocked_converged) = run_line_bfs(&blocked, 2);
    assert_eq!(
        blocked_frontier,
        0b001,
        "an allow_mask ({}) disjoint from the edge kind ({EDGE_KIND}) must block traversal, \
         leaving only the seed {{0}}",
        EDGE_KIND << 1
    );
    assert_eq!(
        blocked_converged, 1,
        "a blocked traversal reaches its fixpoint immediately (nothing can grow), so it is \
         converged rather than truncated"
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

/// Largest declared non-shared binding, which is the extent `vyre-driver` launches
/// over. `wg_scratch` is this program's only shared binding and is declared at 256,
/// below every span this test exercises, so a plain maximum is exact here.
fn dispatch_span(program: &Program) -> u32 {
    program
        .buffers()
        .iter()
        .map(|buffer| buffer.count())
        .max()
        .expect("the BFS program must declare buffers")
}

fn live_node_count(frontier: &[u32], node_count: u32) -> u32 {
    (0..node_count)
        .filter(|id| (frontier[(id / 32) as usize] >> (id % 32)) & 1 == 1)
        .count() as u32
}

/// `chain_len` edges forming `0 -> 1 -> ... -> chain_len`, padded with isolated
/// nodes out to `node_count` so the DECLARED bindings exceed one workgroup.
fn chain_in(node_count: u32, chain_len: u32) -> (Vec<u32>, Vec<u32>) {
    let targets = (1..=chain_len).collect();
    let offsets = (0..=node_count).map(|node| node.min(chain_len)).collect();
    (offsets, targets)
}

/// Run the DCE analysis exactly as `dce_via_encoded` runs it: ONE dispatch, nine
/// slots, root node 0 seeded. Returns the final frontier and the converged word.
fn drive_one_dispatch(
    program: &Program,
    node_count: u32,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    reversed: bool,
) -> (Vec<u32>, u32) {
    let words = ((node_count + 31) / 32) as usize;
    let mut seed = vec![0u32; words];
    seed[0] = 1;
    let inputs = [
        Value::from(pack(&vec![0u32; node_count as usize])),
        Value::from(pack(edge_offsets)),
        Value::from(pack(edge_targets)),
        Value::from(pack(&vec![EDGE_KIND; edge_targets.len()])),
        Value::from(pack(&vec![0u32; node_count as usize])),
        Value::from(pack(&seed)),
        Value::from(pack(&vec![0u32; words])),
        Value::from(pack(&[0u32])),
        Value::from(pack(&[0u32])),
    ];
    let outputs = if reversed {
        vyre_reference::reference_eval_lane_reversed(program, &inputs)
    } else {
        vyre_reference::reference_eval(program, &inputs)
    }
    .expect("the DCE analysis program must execute under the reference interpreter");
    (
        unpack(&outputs[0].to_bytes()),
        unpack(&outputs[2].to_bytes())[0],
    )
}

/// A graph wider than one workgroup reaches its COMPLETE closure in one dispatch,
/// in either lane order.
///
/// This is the contract that a host-repetition scheme was once added to provide and
/// that the persistent loop already provides: the loop relaxes to a fixpoint inside
/// the kernel, so a chain 40 deep is fully reached even though the declared bindings
/// span two workgroups. The exact count is load-bearing in both directions. 41 is
/// the whole chain, so a form that stopped relaxing after the first wave would
/// report 2, and one that lost the tail would report fewer than 41; the 989 padded
/// isolated nodes must stay dark, so a form that seeded or expanded them would
/// report 1030.
///
/// `converged` must be 1. `dce_via_encoded` REFUSES a graph whose analysis reports
/// 0, on the grounds that running DCE against a truncated liveness set deletes live
/// code, so a form that reached the closure without observing the fixpoint would
/// turn a healthy compile into a hard error.
///
/// Lane reversal is what makes this more than a smoke test: the traversal accumulates
/// with `atomic_or` and must be order-independent, so any dependence on lane order
/// shows up as a divergence between the two runs.
#[test]
fn dce_analysis_reaches_the_full_closure_above_one_workgroup_in_one_dispatch() {
    let node_count = 1030;
    let chain_len = 40;
    let (offsets, targets) = chain_in(node_count, chain_len);
    let program = build_dce_bfs_program(ProgramGraphShape::new(node_count, chain_len), node_count);

    // Without this the test silently stops exercising the multi-workgroup case if
    // the width ever changes, which is the failure mode that makes a green suite
    // meaningless rather than merely narrower.
    assert!(
        dispatch_span(&program).div_ceil(program.workgroup_size()[0]) > 1,
        "this test only means something while 1030 nodes span more than one workgroup; \
         span {} over width {} does not",
        dispatch_span(&program),
        program.workgroup_size()[0]
    );

    for reversed in [false, true] {
        let (frontier, converged) =
            drive_one_dispatch(&program, node_count, &offsets, &targets, reversed);
        assert_eq!(
            live_node_count(&frontier, node_count),
            chain_len + 1,
            "the whole 41-node chain and nothing else must be live (reversed={reversed})"
        );
        assert_eq!(
            converged, 1,
            "the analysis must OBSERVE the fixpoint, or dce_via_encoded refuses the graph \
             (reversed={reversed})"
        );
    }
}
