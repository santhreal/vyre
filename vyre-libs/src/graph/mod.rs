//! Graph traversal, dominance, and AST buffer compositions.
//!
//! Host-side packed layout lives in [`vyre_foundation::vast`]. The programs
//! here are minimal GPU-facing slices of that contract.
//!
//! The path IS the interface. Callers address primitive program builders
//! and layout validators by explicit module path; no wildcard re-exports.

/// AST walk orders over the packed VAST buffer layout.
///
/// Behind `graph` because the walk bodies come from `graph::vast_tree_walk`.
#[cfg(feature = "graph")]
pub(crate) mod ast_walk;

/// Graph traversal, dominance, and dispatch-pipeline compositions.
#[cfg(feature = "graph-dispatch")]
pub mod dispatch;

#[cfg(test)]
pub use ast_walk::pack_spine_fixture;
#[cfg(feature = "graph")]
pub use ast_walk::{
    ast_walk, ast_walk_postorder, ast_walk_postorder_nodes, ast_walk_preorder,
    pack_branching_fixture,
};

/// Kahn's-algorithm topological sort.
#[cfg(feature = "graph")]
pub mod toposort;

/// GPU-resident depth-wave dispatcher for bottom-up callee-before-caller
/// computations. Composes `Node::Loop` + `Node::Barrier` with a per-lane depth
/// predicate; no new sub-op.
#[cfg(feature = "graph")]
pub mod level_wave;

/// Reachability scan  -  given a source set + edge list, which nodes are
/// transitively reachable?
#[cfg(feature = "graph")]
pub mod reachable;

/// Canonical 5-buffer ProgramGraph ABI (CSR wire format, shared by every graph
/// primitive).
#[cfg(feature = "graph")]
pub mod program_graph;

#[cfg(feature = "graph")]
pub(crate) fn checked_csr_offset_count(node_count: u32, op_name: &str) -> Result<u32, String> {
    node_count.checked_add(1).ok_or_else(|| {
        format!(
            "Fix: {op_name} node_count + 1 overflows u32 for node_count={node_count}. Shard the CSR graph before GPU dispatch."
        )
    })
}

#[cfg(feature = "graph")]
pub(crate) fn u32_slice_fingerprint(values: &[u32]) -> u64 {
    padded_u32_slice_fingerprint(values, values.len())
}

#[cfg(feature = "graph")]
pub(crate) fn padded_u32_slice_fingerprint(values: &[u32], padded_words: usize) -> u64 {
    use crate::hash::fnv1a::{fnv1a64_initial_state, fnv1a64_update_byte};

    let mut hash = fnv1a64_initial_state();
    for byte in (padded_words as u64).to_le_bytes() {
        hash = fnv1a64_update_byte(hash, byte);
    }
    for index in 0..padded_words {
        let value = values.get(index).copied().unwrap_or(0);
        for byte in value.to_le_bytes() {
            hash = fnv1a64_update_byte(hash, byte);
        }
    }
    hash
}

#[cfg(feature = "graph")]
#[inline]
pub(crate) const fn mix32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

#[cfg(test)]
pub(crate) fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        message.to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        format!("{payload:?}")
    }
}

/// The ONE named-field input bundle every CSR closure entry point takes, so a
/// run of same-typed CSR slices cannot transpose silently at a call site.
#[cfg(feature = "graph")]
pub mod csr_closure_inputs;
/// One BFS step that accumulates into frontier_out and reports changes.
#[cfg(feature = "graph")]
pub mod csr_forward_or_changed;
/// One BFS frontier step over ProgramGraph CSR.
#[cfg(feature = "graph")]
pub mod csr_forward_traverse;
/// The ONE canonical CSR neighbor-expansion edge-scan, shared by every
/// `csr_forward_or_changed` variant and the persistent-BFS batch step. Lives at
/// `graph/` level because it is the common parent of both consumer subsystems;
/// burying it inside one of them would force the other to reach across a sibling.
#[cfg(feature = "graph")]
pub(crate) mod edge_scan;
/// The ONE packed-bitset addressing skeleton: word index, bit mask, the
/// bit-is-set probe, and the atomic set with 0-to-1 flip detection. Peer of
/// `edge_scan` for the same reason: every consumer is a sibling.
#[cfg(feature = "graph")]
pub(crate) mod frontier_bits;
/// One persistent-BFS workgroup step with coalesced change detection.
#[cfg(feature = "graph")]
pub mod persistent_bfs_step;

/// Reverse-direction in-place frontier step that reports changes.
#[cfg(feature = "graph")]
pub mod csr_backward_or_changed;
/// Reverse-direction BFS frontier step.
#[cfg(feature = "graph")]
pub mod csr_backward_traverse;

/// Total outgoing-edge count over the active frontier. Building block for
/// load-balanced one-thread-per-edge expansion that beats naive
/// one-thread-per-node on power-law graphs.
#[cfg(feature = "graph")]
pub mod csr_frontier_degree_sum;
/// Device-side active-frontier queue materialization and queue-driven CSR
/// expansion for sparse dataflow waves.
#[cfg(feature = "graph")]
pub mod csr_frontier_queue;
/// The one CSR frontier step: the Program builder for either edge direction and
/// the host reference that walks the same two directions.
#[cfg(feature = "graph")]
pub(crate) mod csr_frontier_step;
/// Queue-to-queue sparse CSR delta expansion for GPU-resident fixpoint waves.
#[cfg(feature = "graph")]
pub mod csr_queue_delta;
/// Mixed queue traversal that keeps low-degree rows scalar and sends only hubs
/// to row-strided teams.
#[cfg(feature = "graph")]
pub mod csr_queue_split;
/// Row-strided queue-driven CSR expansion for high-degree active rows.
#[cfg(feature = "graph")]
pub mod csr_queue_strided;

/// One BFS step over BOTH forward + backward edges.
#[cfg(feature = "graph")]
pub mod csr_bidirectional;

/// Dominance-frontier query for SSA phi placement.
#[cfg(feature = "graph")]
pub mod dominator_frontier;

/// Exact immediate-dominator tree (Lengauer-Tarjan CPU reference +
/// Cooper-Harvey-Kennedy serial GPU kernel).
#[cfg(feature = "graph")]
pub mod dominator_tree;

/// Walk parent-pointer array from a target back to the root; emit the
/// materialized path into a u32 buffer.
#[cfg(feature = "graph")]
pub mod path_reconstruct;

/// Motif witness helpers over ProgramGraph edge constraints.
#[cfg(feature = "graph")]
pub mod motif;

/// Forward-Backward strongly-connected components decomposition over
/// ProgramGraph CSR.
#[cfg(feature = "graph")]
pub mod scc_decompose;

/// Exploded-supergraph builder  -  (CFG x fact) pairs as graph vertices so
/// IFDS/IDE reduces to `csr_forward_traverse`.
#[cfg(feature = "graph")]
pub mod exploded;

/// Adaptive CSR / dense bitmatrix traversal  -  picks representation per tile
/// based on frontier density.
#[cfg(feature = "graph")]
pub mod adaptive_traverse;
/// Shared state/index frontier headers for graph and automata worklists.
#[cfg(feature = "graph")]
pub mod state_index_frontier;

/// Persistent BFS  -  multi-step frontier expansion in a single dispatch.
#[cfg(feature = "graph")]
pub mod persistent_bfs;

/// IR Extension interface registering Alias-solving opcodes to the compiler
/// front-end.
#[cfg(all(test, feature = "graph"))]
pub mod alias_registry;

/// Lock-free Union-Find for subset alias resolving constraint grids.
#[cfg(feature = "graph")]
pub mod union_find;
/// Packed-AST tree walk over ProgramGraph CSR.
#[cfg(feature = "graph")]
pub mod vast_tree_walk;

/// 3D sub-warp dataflow tensors.
#[cfg(feature = "graph")]
pub mod tensor_flow_forward;
#[cfg(all(test, feature = "graph"))]
mod tensor_flow_forward_tests;

/// K-step Chebyshev polynomial filter on a graph Laplacian. Composes from
/// `math::semiring_gemm` (each step is one `n x n . n x 1` Real-semiring
/// matvec).
#[cfg(feature = "graph")]
pub mod chebyshev_filter;

/// Sum-product circuit (probabilistic circuit) per-node evaluator. Composes
/// with `level_wave` for bottom-up evaluation.
#[cfg(feature = "graph")]
pub mod sum_product_circuit;

/// Pearl do-calculus graph surgery  -  incoming-edge deletion for
/// `do(X = x)` interventions.
#[cfg(feature = "graph")]
pub mod do_calculus;

/// Back-door / front-door adjustment set predicates for causal inference.
/// Composes with do-calculus for full ID-algorithm pipelines.
#[cfg(feature = "graph")]
pub mod adjustment_set;

/// Matroid intersection  -  exchange-graph BFS step for combinatorial
/// scheduling and bipartite matching.
#[cfg(feature = "graph")]
pub mod matroid;

/// Sheaf neural network diagonal-form diffusion step.
#[cfg(feature = "graph")]
pub mod sheaf;

/// Probabilistic knowledge compilation d-DNNF evaluator. Composes with
/// sum_product_circuit for probability-weighted variants.
#[cfg(feature = "graph")]
pub mod knowledge_compile;

/// Functorial data migration. Schema-functor application as graph
/// rewrite.
#[cfg(feature = "graph")]
pub mod functorial;

/// Monoidal-category sequential composition. String-diagram compilation
/// primitive.
#[cfg(feature = "graph")]
pub mod string_diagram;
