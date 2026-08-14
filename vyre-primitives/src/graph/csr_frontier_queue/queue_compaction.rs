//! Queue-length initialization and node-per-lane frontier compaction.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::MemoryOrdering;

use super::{
    FRONTIER_QUEUE_LEN_INIT_OP_ID, FRONTIER_TO_QUEUE_OP_ID, FRONTIER_TO_QUEUE_PARALLEL_OP_ID,
    FRONTIER_TO_QUEUE_WORKGROUP_LANES,
};
use crate::bitset::bitset_words;
use crate::graph::frontier_bits::{when_bit_set, BitAccess};

/// Build a GPU program that initializes the active queue length scalar.
///
/// This replaces a per-wave host-to-device zero upload in resident sparse
/// traversal pipelines. Keeping initialization as a separate single-lane
/// device step avoids the global-synchronization race that would occur if the
/// multi-workgroup compaction kernel tried to clear and atomically increment
/// the same scalar.
#[must_use]
pub fn frontier_queue_len_init(queue_len: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(queue_len, 0, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(FRONTIER_QUEUE_LEN_INIT_OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::store(queue_len, Expr::u32(0), Expr::u32(0))]),
        }],
    )
}

/// Build a GPU program that appends every active frontier node to a queue.
///
/// This is a single-workgroup cooperative scan: lane 0 clears `queue_len`, a
/// workgroup barrier orders that clear, then the lanes of that one workgroup
/// walk `node_count` in [`FRONTIER_TO_QUEUE_WORKGROUP_LANES`]-wide strides.
/// Sparse queue traversal is selected only for low-density frontiers, so
/// avoiding a separate queue-length init launch is more valuable than spreading
/// this scan across every SM. Use [`frontier_to_queue_parallel`] when the
/// frontier is large enough to want every SM.
///
/// Single-workgroup is enforced STRUCTURALLY, not assumed. Every lane whose
/// global id is at or above the workgroup width retires without touching
/// memory, so the program computes the same queue for any dispatch span the
/// driver picks. That gate is load-bearing twice over, and both failures are
/// silent wrong answers rather than crashes:
///
/// 1. Duplicate coverage. `q_src` is `q_iter * WIDTH + q_lane` over a GLOBAL
///    `q_lane`, so at `G` workgroups the lanes of group `g` re-derive `q_src`
///    values group 0 already covered at a higher `q_iter`. Every active node at
///    or above the workgroup width would be appended once per covering group and
///    `queue_len` inflated by the same factor.
/// 2. Lost clear. Lane 0's clear of `queue_len` is a PLAIN store ordered only by
///    a WORKGROUP-scope barrier. Nothing orders it against another group's
///    `atomic_add`, so a second group's increment could land before the clear
///    and be erased.
///
/// The span is not the caller's to choose: this program contains an atomic, and
/// once a program contains any atomic the driver widens the dispatch to the
/// largest non-shared binding, which here is `active_queue` at `queue_capacity`.
/// A capacity that merely matches the node count therefore already produces a
/// multi-workgroup grid.
#[must_use]
pub fn frontier_to_queue(
    frontier_in: &str,
    active_queue: &str,
    queue_len: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    if node_count == 0 || queue_capacity == 0 {
        return crate::invalid_output_program(FRONTIER_TO_QUEUE_OP_ID,
        queue_len,
        DataType::U32,
        format!(
            "Fix: frontier_to_queue requires node_count > 0 and queue_capacity > 0, got node_count={node_count} queue_capacity={queue_capacity}."
        ),);
    }
    let lane = Expr::InvocationId { axis: 0 };
    let words = bitset_words(node_count);
    let lanes = FRONTIER_TO_QUEUE_WORKGROUP_LANES;
    let scan_iters = node_count.div_ceil(lanes).max(1);
    let body = vec![
        Node::let_bind("q_lane", lane.clone()),
        Node::if_then(
            Expr::eq(Expr::var("q_lane"), Expr::u32(0)),
            vec![Node::store(queue_len, Expr::u32(0), Expr::u32(0))],
        ),
        Node::barrier_with_ordering(MemoryOrdering::SeqCst),
        // Only the lanes of the FIRST workgroup scan. Beyond that width the
        // strided walk below would re-cover source nodes group 0 already
        // covered, double-appending them and inflating `queue_len`.
        Node::if_then(
            Expr::lt(Expr::var("q_lane"), Expr::u32(lanes)),
            vec![Node::loop_for(
                "q_iter",
                Expr::u32(0),
                Expr::u32(scan_iters),
                vec![
                    Node::let_bind(
                        "q_src",
                        Expr::add(
                            Expr::mul(Expr::var("q_iter"), Expr::u32(lanes)),
                            Expr::var("q_lane"),
                        ),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("q_src"), Expr::u32(node_count)),
                        when_bit_set(
                            frontier_in,
                            &Expr::var("q_src"),
                            BitAccess {
                                word: "q_word_idx",
                                mask: "q_bit_mask",
                                value: "q_src_word",
                            },
                            |word| word,
                            vec![
                                Node::let_bind(
                                    "q_slot",
                                    Expr::atomic_add(queue_len, Expr::u32(0), Expr::u32(1)),
                                ),
                                Node::if_then(
                                    Expr::lt(Expr::var("q_slot"), Expr::u32(queue_capacity)),
                                    vec![Node::store(
                                        active_queue,
                                        Expr::var("q_slot"),
                                        Expr::var("q_src"),
                                    )],
                                ),
                            ],
                        ),
                    ),
                ],
            )],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(active_queue, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(queue_capacity),
            BufferDecl::storage(queue_len, 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [lanes, 1, 1],
        vec![Node::Region {
            generator: Ident::from(FRONTIER_TO_QUEUE_OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

/// Build a multi-workgroup GPU program that appends active frontier nodes to a queue.
///
/// The caller must clear `queue_len` before dispatch, for example with
/// `frontier_queue_len_init` or a fused resident reset step. Unlike
/// `frontier_to_queue`, this variant maps one lane to one source node and is
/// the right materializer for large packed frontiers.
#[must_use]
pub fn frontier_to_queue_parallel(
    frontier_in: &str,
    active_queue: &str,
    queue_len: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    if node_count == 0 || queue_capacity == 0 {
        return crate::invalid_output_program(FRONTIER_TO_QUEUE_PARALLEL_OP_ID,
        queue_len,
        DataType::U32,
        format!(
            "Fix: frontier_to_queue_parallel requires node_count > 0 and queue_capacity > 0, got node_count={node_count} queue_capacity={queue_capacity}."
        ),);
    }
    let lane = Expr::InvocationId { axis: 0 };
    let words = bitset_words(node_count);
    let body = vec![
        Node::let_bind("qp_src", lane),
        Node::if_then(
            Expr::lt(Expr::var("qp_src"), Expr::u32(node_count)),
            when_bit_set(
                frontier_in,
                &Expr::var("qp_src"),
                BitAccess {
                    word: "qp_word_idx",
                    mask: "qp_bit_mask",
                    value: "qp_src_word",
                },
                |word| word,
                vec![
                    Node::let_bind(
                        "qp_slot",
                        Expr::atomic_add(queue_len, Expr::u32(0), Expr::u32(1)),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("qp_slot"), Expr::u32(queue_capacity)),
                        vec![Node::store(
                            active_queue,
                            Expr::var("qp_slot"),
                            Expr::var("qp_src"),
                        )],
                    ),
                ],
            ),
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(active_queue, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(queue_capacity),
            BufferDecl::storage(queue_len, 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from(FRONTIER_TO_QUEUE_PARALLEL_OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}
