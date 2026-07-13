//! IR-validity gate for graph Program-builders that are NOT in the inventory registry
//! (so the registry's `every_registered_primitive_program_is_ir_valid` net does not reach
//! them) and whose only tests exercise a CPU oracle (never the IR PROGRAM).
//!
//! This is the exact gap the `union_find` shadow bug fell through: `union_find_program`
//! emitted IR that failed validation (a duplicate-binding shadow the no-shadowing validator
//! AND the CUDA backend both reject), yet its only tests string-matched the IR dump, so it
//! shipped IR-invalid. Order-dependent builders (queue compaction) can't join the parity
//! registry (their output slot order is nondeterministic), but they MUST still emit valid
//! IR. Validation runs BEFORE input binding in the interpreter, so evaluating with empty
//! inputs surfaces "failed IR validation" for a broken program while a valid one only
//! reports a benign "missing input" (a fixture-free way to assert IR validity).
#![cfg(feature = "graph")]

use vyre_foundation::ir::Program;
use vyre_primitives::graph::program_graph::ProgramGraphShape;

/// Assert a built program passes IR validation. Empty inputs are intentional: the
/// interpreter validates first, so a valid program fails later with "missing input"
/// (accepted here) while an IR-invalid one fails with "failed IR validation" (rejected).
fn assert_ir_valid(name: &str, program: &Program) {
    match vyre_reference::reference_eval(program, &[]) {
        Ok(_) => {} // validated and (somehow) ran, still valid IR
        Err(err) => {
            let msg = format!("{err}");
            assert!(
                !msg.contains("failed IR validation"),
                "Fix: `{name}` builds IR that FAILS validation (the no-shadowing validator + the \
                 CUDA backend both reject it, the op cannot run on the reference OR the GPU). \
                 Only-CPU-oracle tests do not catch this; see the union_find shadow bug. Error: {msg}"
            );
        }
    }
}

#[test]
fn csr_queue_delta_enqueue_emits_valid_ir() {
    let program = vyre_primitives::graph::csr_queue_delta::csr_queue_delta_enqueue(
        "active_queue",
        "active_len",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "accumulator",
        "next_queue",
        "next_len",
        4,           // node_count
        4,           // edge_count
        8,           // active_queue_capacity
        16,          // next_queue_capacity
        0xFFFF_FFFF, // allow_mask
    );
    assert_ir_valid("csr_queue_delta_enqueue", &program);
}

#[test]
fn csr_queue_split_low_forward_traverse_emits_valid_ir() {
    let program = vyre_primitives::graph::csr_queue_split::csr_queue_split_low_forward_traverse(
        "active_queue",
        "queue_len",
        "edge_offsets",
        "edge_targets",
        "edge_kind_mask",
        "frontier_out",
        "high_queue",
        "high_len",
        4,           // node_count
        4,           // edge_count
        8,           // queue_capacity
        8,           // high_queue_capacity
        2,           // high_degree_threshold
        0xFFFF_FFFF, // allow_mask
    );
    assert_ir_valid("csr_queue_split_low_forward_traverse", &program);
}

#[test]
fn matroid_exchange_bfs_step_emits_valid_ir() {
    let program = vyre_primitives::graph::matroid::matroid_exchange_bfs_step(
        "frontier_in",
        "exchange_adj",
        "visited",
        "frontier_out",
        "any_change",
        4,
    );
    assert_ir_valid("matroid_exchange_bfs_step", &program);
}

#[test]
fn do_intervention_delete_incoming_emits_valid_ir() {
    let program = vyre_primitives::graph::do_calculus::do_intervention_delete_incoming(
        "adjacency",
        "intervention_mask",
        "out_adjacency",
        4,
    );
    assert_ir_valid("do_intervention_delete_incoming", &program);
}

#[test]
fn do_rule2_reverse_incoming_emits_valid_ir() {
    // Sibling of do_intervention_delete_incoming (above) in do_calculus.rs: it has a
    // `do_rule2_reverse_incoming_cpu` oracle but its IR PROGRAM was never run through
    // reference_eval, so its IR was never validated, the exact gap the union_find
    // find-walk/IR-shadow bugs fell through (CPU-oracle-covered, IR-unchecked).
    let program = vyre_primitives::graph::do_calculus::do_rule2_reverse_incoming(
        "adjacency",
        "treatment_mask",
        "out_adjacency",
        4,
    );
    assert_ir_valid("do_rule2_reverse_incoming", &program);
}

#[test]
fn backdoor_descendants_check_emits_valid_ir() {
    let program = vyre_primitives::graph::adjustment_set::backdoor_descendants_check(
        "candidate_z",
        "descendants_of_x",
        "out_violation",
        4,
    );
    assert_ir_valid("backdoor_descendants_check", &program);
}

#[test]
fn persistent_bfs_emits_valid_ir() {
    // Full composed persistent-BFS Program (not just the registered per-iteration step): its
    // only tests exercise the `cpu_ref` oracle, never the IR. node_count=4 stays under the
    // workgroup bound so it builds the single-dispatch composed kernel (the grid-sync variant
    // is a distinct GridSync program with its own coverage).
    let program = vyre_primitives::graph::persistent_bfs::persistent_bfs(
        ProgramGraphShape::new(4, 4),
        "frontier_in",
        "frontier_out",
        0xFFFF_FFFF, // edge_kind_mask
        2,           // max_iters
    );
    assert_ir_valid("persistent_bfs", &program);
}

#[test]
fn persistent_bfs_batch_emits_valid_ir() {
    // Batched multi-query persistent-BFS Program (same oracle-only coverage gap as above).
    let program = vyre_primitives::graph::persistent_bfs::persistent_bfs_batch(
        ProgramGraphShape::new(4, 4),
        "frontier_in",
        "frontier_out",
        "changed",
        1,           // query_count
        0xFFFF_FFFF, // edge_kind_mask
        2,           // max_iters
    );
    assert_ir_valid("persistent_bfs_batch", &program);
}

#[test]
fn toposort_program_emits_valid_ir() {
    // Lane0-serialized queue toposort (order-dependent output slots → can't join the race-net
    // registry), and its only test exercises the CPU oracle, never the IR, so the actual
    // Program was never validated. Simple buffer-string signature makes it a clean gate add.
    let program = vyre_primitives::graph::toposort::toposort_program(
        4, // node_count
        "offsets",
        "targets",
        "indeg_scratch",
        "queue_scratch",
        "order_out",
    );
    assert_ir_valid("toposort_program", &program);
}

#[test]
fn chebyshev_filter_emits_valid_ir() {
    let program = vyre_primitives::graph::chebyshev_filter::chebyshev_filter(
        "laplacian",
        "signal",
        "coeffs",
        "output",
        "scratch",
        4,
        3,
    );
    assert_ir_valid("chebyshev_filter", &program);
}
