//! Row-strided queue-to-queue sparse CSR expansion.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::bitset::bitset_words;

use super::CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE;

/// Canonical op id for row-strided queue-to-queue delta CSR expansion.
pub const CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID: &str =
    "vyre-primitives::graph::csr_queue_delta_strided_enqueue";

/// Fixed lane team assigned to each queued source row in the strided delta path.
pub const CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE: u32 =
    crate::graph::csr_queue_strided::CSR_QUEUE_STRIDED_FORWARD_LANES_PER_SOURCE;

/// Dispatch grid for row-strided queue-to-queue delta expansion.
#[must_use]
pub const fn csr_queue_delta_strided_dispatch_grid(active_queue_capacity: u32) -> [u32; 3] {
    let total_lanes =
        active_queue_capacity.saturating_mul(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE);
    let blocks = total_lanes.div_ceil(CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE[0]);
    [if blocks == 0 { 1 } else { blocks }, 1, 1]
}

/// Build a row-strided delta enqueue program for skewed CSR source rows.
///
/// This uses the same resident buffer ABI as
/// [`super::csr_queue_delta_enqueue`], but assigns a fixed lane team to each
/// queued source and stripes that source row across the team. It keeps
/// high-degree IFDS hubs from serializing all edge work behind a single
/// invocation.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn csr_queue_delta_strided_enqueue(
    active_queue: &str,
    active_len: &str,
    edge_offsets: &str,
    edge_targets: &str,
    edge_kind_mask: &str,
    accumulator: &str,
    next_queue: &str,
    next_len: &str,
    node_count: u32,
    edge_count: u32,
    active_queue_capacity: u32,
    next_queue_capacity: u32,
    allow_mask: u32,
) -> Program {
    if node_count == 0 || active_queue_capacity == 0 || next_queue_capacity == 0 {
        return crate::invalid_output_program(
            CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID,
            next_len,
            DataType::U32,
            format!(
                "Fix: csr_queue_delta_strided_enqueue requires node_count > 0 and non-zero queue capacities, got node_count={node_count} active_queue_capacity={active_queue_capacity} next_queue_capacity={next_queue_capacity}."
            ),
        );
    }

    let lane = Expr::InvocationId { axis: 0 };
    let words = bitset_words(node_count);
    let physical_edge_count = edge_count.max(1);
    let lanes_per_source = Expr::u32(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE);
    let body = vec![
        Node::let_bind("qds_lane", lane),
        Node::let_bind(
            "qds_queue_idx",
            Expr::div(Expr::var("qds_lane"), lanes_per_source.clone()),
        ),
        Node::let_bind(
            "qds_edge_lane",
            Expr::rem(Expr::var("qds_lane"), lanes_per_source.clone()),
        ),
        Node::if_then(
            Expr::lt(Expr::var("qds_queue_idx"), Expr::u32(active_queue_capacity)),
            vec![Node::if_then(
                Expr::lt(
                    Expr::var("qds_queue_idx"),
                    Expr::load(active_len, Expr::u32(0)),
                ),
                vec![
                    Node::let_bind(
                        "qds_src",
                        Expr::load(active_queue, Expr::var("qds_queue_idx")),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("qds_src"), Expr::u32(node_count)),
                        vec![
                            Node::let_bind(
                                "qds_edge_start",
                                Expr::load(edge_offsets, Expr::var("qds_src")),
                            ),
                            Node::let_bind(
                                "qds_edge_end",
                                Expr::load(
                                    edge_offsets,
                                    Expr::add(Expr::var("qds_src"), Expr::u32(1)),
                                ),
                            ),
                            Node::let_bind(
                                "qds_degree",
                                Expr::sub(Expr::var("qds_edge_end"), Expr::var("qds_edge_start")),
                            ),
                            Node::let_bind(
                                "qds_full_iters",
                                Expr::div(Expr::var("qds_degree"), lanes_per_source),
                            ),
                            Node::let_bind(
                                "qds_tail_iter",
                                Expr::select(
                                    Expr::ne(
                                        Expr::rem(
                                            Expr::var("qds_degree"),
                                            Expr::u32(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE),
                                        ),
                                        Expr::u32(0),
                                    ),
                                    Expr::u32(1),
                                    Expr::u32(0),
                                ),
                            ),
                            Node::let_bind(
                                "qds_iters",
                                Expr::add(Expr::var("qds_full_iters"), Expr::var("qds_tail_iter")),
                            ),
                            Node::loop_for(
                                "qds_iter",
                                Expr::u32(0),
                                Expr::var("qds_iters"),
                                vec![
                                    Node::let_bind(
                                        "qds_edge_offset",
                                        Expr::add(
                                            Expr::var("qds_edge_lane"),
                                            Expr::mul(
                                                Expr::var("qds_iter"),
                                                Expr::u32(CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE),
                                            ),
                                        ),
                                    ),
                                    Node::if_then(
                                        Expr::lt(
                                            Expr::var("qds_edge_offset"),
                                            Expr::var("qds_degree"),
                                        ),
                                        vec![
                                            Node::let_bind(
                                                "qds_e",
                                                Expr::add(
                                                    Expr::var("qds_edge_start"),
                                                    Expr::var("qds_edge_offset"),
                                                ),
                                            ),
                                            Node::if_then(
                                                Expr::lt(Expr::var("qds_e"), Expr::u32(edge_count)),
                                                vec![
                                                    Node::let_bind(
                                                        "qds_kind",
                                                        Expr::load(
                                                            edge_kind_mask,
                                                            Expr::var("qds_e"),
                                                        ),
                                                    ),
                                                    Node::if_then(
                                                        Expr::ne(
                                                            Expr::bitand(
                                                                Expr::var("qds_kind"),
                                                                Expr::u32(allow_mask),
                                                            ),
                                                            Expr::u32(0),
                                                        ),
                                                        vec![
                                                            Node::let_bind(
                                                                "qds_dst",
                                                                Expr::load(
                                                                    edge_targets,
                                                                    Expr::var("qds_e"),
                                                                ),
                                                            ),
                                                            Node::if_then(
                                                                Expr::lt(
                                                                    Expr::var("qds_dst"),
                                                                    Expr::u32(node_count),
                                                                ),
                                                                vec![
                                                                    Node::let_bind(
                                                                        "qds_dst_word",
                                                                        Expr::shr(
                                                                            Expr::var("qds_dst"),
                                                                            Expr::u32(5),
                                                                        ),
                                                                    ),
                                                                    Node::let_bind(
                                                                        "qds_dst_bit",
                                                                        Expr::shl(
                                                                            Expr::u32(1),
                                                                            Expr::bitand(
                                                                                Expr::var(
                                                                                    "qds_dst",
                                                                                ),
                                                                                Expr::u32(31),
                                                                            ),
                                                                        ),
                                                                    ),
                                                                    Node::let_bind(
                                                                        "qds_old",
                                                                        Expr::atomic_or(
                                                                            accumulator,
                                                                            Expr::var(
                                                                                "qds_dst_word",
                                                                            ),
                                                                            Expr::var(
                                                                                "qds_dst_bit",
                                                                            ),
                                                                        ),
                                                                    ),
                                                                    Node::if_then(
                                                                        Expr::eq(
                                                                            Expr::bitand(
                                                                                Expr::var(
                                                                                    "qds_old",
                                                                                ),
                                                                                Expr::var(
                                                                                    "qds_dst_bit",
                                                                                ),
                                                                            ),
                                                                            Expr::u32(0),
                                                                        ),
                                                                        vec![
                                                                            Node::let_bind(
                                                                                "qds_slot",
                                                                                Expr::atomic_add(
                                                                                    next_len,
                                                                                    Expr::u32(0),
                                                                                    Expr::u32(1),
                                                                                ),
                                                                            ),
                                                                            Node::if_then(
                                                                                Expr::lt(
                                                                                    Expr::var(
                                                                                        "qds_slot",
                                                                                    ),
                                                                                    Expr::u32(
                                                                                        next_queue_capacity,
                                                                                    ),
                                                                                ),
                                                                                vec![Node::store(
                                                                                    next_queue,
                                                                                    Expr::var(
                                                                                        "qds_slot",
                                                                                    ),
                                                                                    Expr::var(
                                                                                        "qds_dst",
                                                                                    ),
                                                                                )],
                                                                            ),
                                                                        ],
                                                                    ),
                                                                ],
                                                            ),
                                                        ],
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(active_queue, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(active_queue_capacity),
            BufferDecl::storage(active_len, 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage(edge_offsets, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(node_count + 1),
            BufferDecl::storage(edge_targets, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(physical_edge_count),
            BufferDecl::storage(edge_kind_mask, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(physical_edge_count),
            BufferDecl::storage(accumulator, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
            BufferDecl::storage(next_queue, 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(next_queue_capacity),
            BufferDecl::storage(next_len, 7, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE,
        vec![Node::Region {
            generator: Ident::from(CSR_QUEUE_DELTA_STRIDED_ENQUEUE_OP_ID),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_strided_program_keeps_delta_queue_abi_and_expands_grid() {
        let program = csr_queue_delta_strided_enqueue(
            "active_queue",
            "active_len",
            "edge_offsets",
            "edge_targets",
            "edge_kind_mask",
            "accumulator",
            "next_queue",
            "next_len",
            64,
            7,
            8,
            16,
            1,
        );

        assert_eq!(
            program.workgroup_size,
            CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE
        );
        assert_eq!(program.buffers.len(), 8);
        assert_eq!(program.buffers[0].name.as_ref(), "active_queue");
        assert_eq!(program.buffers[0].count, 8);
        assert_eq!(program.buffers[6].name.as_ref(), "next_queue");
        assert_eq!(program.buffers[6].count, 16);
        assert_eq!(
            csr_queue_delta_strided_dispatch_grid(8),
            [
                (8 * CSR_QUEUE_DELTA_STRIDED_LANES_PER_SOURCE)
                    .div_ceil(CSR_QUEUE_DELTA_ENQUEUE_WORKGROUP_SIZE[0]),
                1,
                1,
            ]
        );
    }
}
