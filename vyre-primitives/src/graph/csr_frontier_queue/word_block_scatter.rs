//! Deterministic packed-frontier scatter: source-ordered queue materialization
//! from per-word partials plus either a local block prefix or precomputed offsets.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::sizing_diagnostics::{
    checked_frontier_u32_product, invalid_frontier_queue_sizing_program,
};
use super::{
    FRONTIER_WORD_BLOCK_OFFSETS_TO_QUEUE_PARALLEL_OP_ID,
    FRONTIER_WORD_BLOCK_PREFIX_TO_QUEUE_PARALLEL_OP_ID, FRONTIER_WORD_SCAN_BLOCK_LANES,
};
use crate::bitset::bitset_words;

/// Build the deterministic scatter pass for packed-frontier queue materialization.
///
/// `word_partials` must come from [`frontier_word_counts_scan_pass_a`](super::frontier_word_counts_scan_pass_a), and
/// `block_totals` must be the block-total output from that same pass. The
/// scatter computes the tiny block prefix locally, preserving source-node order
/// without an additional block-scan dispatch. It writes `queue_len` as the full
/// in-range active-node count even when the bounded queue truncates the
/// materialized entries.
#[must_use]
pub fn frontier_word_block_prefix_to_queue_parallel(
    frontier_in: &str,
    word_partials: &str,
    block_totals: &str,
    active_queue: &str,
    queue_len: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    frontier_word_queue_scatter_program(
        FRONTIER_WORD_BLOCK_PREFIX_TO_QUEUE_PARALLEL_OP_ID,
        FrontierWordBlockOffsetSource::SumPreviousTotals { block_totals },
        frontier_in,
        word_partials,
        active_queue,
        queue_len,
        node_count,
        queue_capacity,
    )
}

/// Build the deterministic scatter pass using precomputed per-block offsets.
///
/// `block_offsets` must be the in-place output of
/// [`frontier_word_block_offsets_in_place`](super::frontier_word_block_offsets_in_place). This keeps scatter work O(words)
/// for multi-block frontiers by replacing the per-word previous-block loop with
/// one block-offset load.
#[must_use]
pub fn frontier_word_block_offsets_to_queue_parallel(
    frontier_in: &str,
    word_partials: &str,
    block_offsets: &str,
    active_queue: &str,
    queue_len: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    frontier_word_queue_scatter_program(
        FRONTIER_WORD_BLOCK_OFFSETS_TO_QUEUE_PARALLEL_OP_ID,
        FrontierWordBlockOffsetSource::PrecomputedOffsets { block_offsets },
        frontier_in,
        word_partials,
        active_queue,
        queue_len,
        node_count,
        queue_capacity,
    )
}

#[derive(Clone, Copy)]
enum FrontierWordBlockOffsetSource<'a> {
    SumPreviousTotals { block_totals: &'a str },
    PrecomputedOffsets { block_offsets: &'a str },
}

impl FrontierWordBlockOffsetSource<'_> {
    fn buffer_name(&self) -> &str {
        match self {
            FrontierWordBlockOffsetSource::SumPreviousTotals { block_totals } => block_totals,
            FrontierWordBlockOffsetSource::PrecomputedOffsets { block_offsets } => block_offsets,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn frontier_word_queue_scatter_program(
    op_id: &'static str,
    block_offset_source: FrontierWordBlockOffsetSource<'_>,
    frontier_in: &str,
    word_partials: &str,
    active_queue: &str,
    queue_len: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    if node_count == 0 || queue_capacity == 0 {
        return crate::invalid_output_program(op_id,
        queue_len,
        DataType::U32,
        format!(
            "Fix: {op_id} requires node_count > 0 and queue_capacity > 0, got node_count={node_count} queue_capacity={queue_capacity}."
        ),);
    }
    let words = bitset_words(node_count);
    let num_blocks = words.div_ceil(FRONTIER_WORD_SCAN_BLOCK_LANES).max(1);
    let total_partials =
        match checked_frontier_u32_product(num_blocks, FRONTIER_WORD_SCAN_BLOCK_LANES, op_id) {
            Ok(total_partials) => total_partials,
            Err(error) => return invalid_frontier_queue_sizing_program(op_id, queue_len, error),
        };
    let tail_bits = node_count & 31;
    let tail_mask = if tail_bits == 0 {
        u32::MAX
    } else {
        (1_u32 << tail_bits) - 1
    };
    let lane = Expr::InvocationId { axis: 0 };
    let mut block_offset_body = Vec::new();
    match block_offset_source {
        FrontierWordBlockOffsetSource::SumPreviousTotals { block_totals } => {
            block_offset_body.push(Node::let_bind("fwq_block_offset", Expr::u32(0)));
            block_offset_body.push(Node::loop_for(
                "fwq_prev_block",
                Expr::u32(0),
                Expr::var("fwq_block"),
                vec![Node::assign(
                    "fwq_block_offset",
                    Expr::add(
                        Expr::var("fwq_block_offset"),
                        Expr::load(block_totals, Expr::var("fwq_prev_block")),
                    ),
                )],
            ));
        }
        FrontierWordBlockOffsetSource::PrecomputedOffsets { block_offsets } => {
            block_offset_body.push(Node::let_bind(
                "fwq_block_offset",
                Expr::load(block_offsets, Expr::var("fwq_block")),
            ));
        }
    }
    let mut word_body = vec![
        Node::let_bind(
            "fwq_src_base",
            Expr::mul(Expr::var("fwq_word_idx"), Expr::u32(32)),
        ),
        Node::let_bind(
            "fwq_block",
            Expr::div(
                Expr::var("fwq_word_idx"),
                Expr::u32(FRONTIER_WORD_SCAN_BLOCK_LANES),
            ),
        ),
        Node::let_bind(
            "fwq_word",
            Expr::load(frontier_in, Expr::var("fwq_word_idx")),
        ),
    ];
    if tail_bits != 0 {
        word_body.push(Node::if_then(
            Expr::eq(Expr::var("fwq_word_idx"), Expr::u32(words - 1)),
            vec![Node::assign(
                "fwq_word",
                Expr::bitand(Expr::var("fwq_word"), Expr::u32(tail_mask)),
            )],
        ));
    }
    word_body.extend(block_offset_body);
    word_body.extend([
        Node::let_bind("fwq_active_bits", Expr::popcount(Expr::var("fwq_word"))),
        Node::let_bind(
            "fwq_end",
            Expr::add(
                Expr::load(word_partials, Expr::var("fwq_word_idx")),
                Expr::var("fwq_block_offset"),
            ),
        ),
        Node::let_bind(
            "fwq_start",
            Expr::sub(Expr::var("fwq_end"), Expr::var("fwq_active_bits")),
        ),
        Node::let_bind("fwq_remaining", Expr::var("fwq_word")),
        Node::loop_for(
            "fwq_rank",
            Expr::u32(0),
            Expr::var("fwq_active_bits"),
            vec![
                Node::let_bind("fwq_bit", Expr::ctz(Expr::var("fwq_remaining"))),
                Node::let_bind(
                    "fwq_src",
                    Expr::add(Expr::var("fwq_src_base"), Expr::var("fwq_bit")),
                ),
                Node::let_bind(
                    "fwq_slot",
                    Expr::add(Expr::var("fwq_start"), Expr::var("fwq_rank")),
                ),
                Node::if_then(
                    Expr::and(
                        Expr::lt(Expr::var("fwq_slot"), Expr::u32(queue_capacity)),
                        Expr::lt(Expr::var("fwq_src"), Expr::u32(node_count)),
                    ),
                    vec![Node::store(
                        active_queue,
                        Expr::var("fwq_slot"),
                        Expr::var("fwq_src"),
                    )],
                ),
                Node::assign(
                    "fwq_remaining",
                    Expr::bitand(
                        Expr::var("fwq_remaining"),
                        Expr::sub(Expr::var("fwq_remaining"), Expr::u32(1)),
                    ),
                ),
            ],
        ),
        Node::if_then(
            Expr::eq(Expr::var("fwq_word_idx"), Expr::u32(words - 1)),
            vec![Node::store(queue_len, Expr::u32(0), Expr::var("fwq_end"))],
        ),
    ]);

    let body = vec![
        Node::let_bind("fwq_word_idx", lane),
        Node::if_then(
            Expr::lt(Expr::var("fwq_word_idx"), Expr::u32(words)),
            word_body,
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(word_partials, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(total_partials),
            BufferDecl::storage(
                block_offset_source.buffer_name(),
                2,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(num_blocks),
            BufferDecl::storage(active_queue, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(queue_capacity),
            BufferDecl::storage(queue_len, 4, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [256, 1, 1],
        vec![Node::Region {
            generator: Ident::from(op_id),
            source_region: None,
            body: Arc::new(body),
        }],
    )
}
