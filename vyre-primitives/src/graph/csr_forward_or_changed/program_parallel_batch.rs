use vyre_foundation::ir::{Expr, Program};

use super::program_parallel_batch_global::csr_forward_or_changed_parallel_batch_global_indexed;
use crate::graph::program_graph::ProgramGraphShape;

/// Parallel in-place expansion for several frontier accumulators at once.
///
/// Invocation axis 0 is the source node and axis 1 is the query/frontier index.
/// `frontier_out` is laid out as `query_count` consecutive bitsets, each
/// containing `bitset_words(shape.node_count)` u32 words. `changed` contains
/// one u32 flag per query.
#[must_use]
pub fn csr_forward_or_changed_parallel_batch(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
) -> Program {
    // Fail fast on an invalid flat-frontier shape rather than silently degrading
    // to an inert empty kernel (silent recall loss). Use
    // `try_csr_forward_or_changed_parallel_batch` for structured handling.
    try_csr_forward_or_changed_parallel_batch(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
    )
    .unwrap_or_else(|error| panic!("{error}"))
}

/// Parallel in-place expansion for several frontier accumulators with checked
/// flat-frontier sizing.
///
/// The per-query batch expansion IS the global-slot batch expansion
/// ([`csr_forward_or_changed_parallel_batch_global_indexed`]) with the convergence
/// index set to the query lane (invocation axis 1) and exactly one `changed` slot
/// per query. Routing through that ONE canonical CSR edge-scan builder, instead of
/// re-emitting the neighbor-expansion inner loop a second time, keeps the loop in a
/// single home so the two batch variants cannot drift (ONE-PLACE). The emitted IR is
/// byte-identical to the previous hand-written body (locked by the batch parity
/// tests + the graph oracle/fixpoint matrices).
pub fn try_csr_forward_or_changed_parallel_batch(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
    query_count: u32,
) -> Result<Program, String> {
    if query_count == 0 {
        return Err(
            "Fix: csr_forward_or_changed_parallel_batch requires at least one query frontier."
                .to_string(),
        );
    }
    // changed_index = query (axis 1); changed_slots = query_count (one flag per
    // query); no prologue or extra buffers ⇒ identical program to the hand-written
    // batch body.
    csr_forward_or_changed_parallel_batch_global_indexed(
        shape,
        frontier_out,
        changed,
        edge_kind_mask,
        query_count,
        Expr::InvocationId { axis: 1 },
        query_count,
        Vec::new(),
        Vec::new(),
    )
}
