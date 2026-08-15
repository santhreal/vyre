//! Packed-word frontier compaction: one lane per u32 frontier word, one atomic
//! queue reservation per nonzero word.

use vyre_foundation::algebra::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::{
    FRONTIER_WORDS_TO_QUEUE_CLEAR_OUT_PARALLEL_OP_ID, FRONTIER_WORDS_TO_QUEUE_PARALLEL_OP_ID,
};
use crate::bitset::bitset_words;

/// Build a multi-workgroup GPU program that appends active frontier nodes to a
/// queue by scanning packed frontier words.
///
/// The caller must clear `queue_len` before dispatch. This variant maps one
/// lane to one packed u32 frontier word and performs one atomic queue
/// reservation per nonzero word, so sparse packed frontiers launch 32x fewer
/// lanes than `frontier_to_queue_parallel` and avoid per-active-bit atomics.
#[must_use]
pub fn frontier_words_to_queue_parallel(
    frontier_in: &str,
    active_queue: &str,
    queue_len: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    frontier_words_to_queue_parallel_program(
        FRONTIER_WORDS_TO_QUEUE_PARALLEL_OP_ID,
        frontier_in,
        active_queue,
        queue_len,
        None,
        node_count,
        queue_capacity,
    )
}

/// Build a packed-frontier queue materializer that also clears `frontier_out`.
///
/// The caller must still clear `queue_len` before dispatch. Folding the output
/// clear into this packed-word scan removes a separate full-frontier reset pass
/// from resident sparse traversal sequences without changing the queue ABI.
#[must_use]
pub fn frontier_words_to_queue_clear_out_parallel(
    frontier_in: &str,
    active_queue: &str,
    queue_len: &str,
    frontier_out: &str,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    frontier_words_to_queue_parallel_program(
        FRONTIER_WORDS_TO_QUEUE_CLEAR_OUT_PARALLEL_OP_ID,
        frontier_in,
        active_queue,
        queue_len,
        Some(frontier_out),
        node_count,
        queue_capacity,
    )
}

fn frontier_words_to_queue_parallel_program(
    op_id: &'static str,
    frontier_in: &str,
    active_queue: &str,
    queue_len: &str,
    frontier_out_to_clear: Option<&str>,
    node_count: u32,
    queue_capacity: u32,
) -> Program {
    if node_count == 0 || queue_capacity == 0 {
        return trap_program(op_id, Some((queue_len, DataType::U32)), format!(
            "Fix: {op_id} requires node_count > 0 and queue_capacity > 0, got node_count={node_count} queue_capacity={queue_capacity}."
        ));
    }
    let lane = Expr::InvocationId { axis: 0 };
    let words = bitset_words(node_count);
    let tail_bits = node_count & 31;
    let tail_mask = if tail_bits == 0 {
        u32::MAX
    } else {
        (1_u32 << tail_bits) - 1
    };
    let mut word_body = vec![
        Node::let_bind(
            "qw_src_base",
            Expr::mul(Expr::var("qw_word_idx"), Expr::u32(32)),
        ),
        Node::let_bind(
            "qw_remaining",
            Expr::load(frontier_in, Expr::var("qw_word_idx")),
        ),
    ];
    if tail_bits != 0 {
        word_body.push(Node::if_then(
            Expr::eq(Expr::var("qw_word_idx"), Expr::u32(words - 1)),
            vec![Node::assign(
                "qw_remaining",
                Expr::bitand(Expr::var("qw_remaining"), Expr::u32(tail_mask)),
            )],
        ));
    }
    word_body.push(Node::if_then(
        Expr::ne(Expr::var("qw_remaining"), Expr::u32(0)),
        vec![
            Node::let_bind("qw_active_bits", Expr::popcount(Expr::var("qw_remaining"))),
            Node::let_bind(
                "qw_base_slot",
                Expr::atomic_add(queue_len, Expr::u32(0), Expr::var("qw_active_bits")),
            ),
            Node::loop_for(
                "qw_rank",
                Expr::u32(0),
                Expr::var("qw_active_bits"),
                vec![
                    Node::let_bind("qw_bit", Expr::ctz(Expr::var("qw_remaining"))),
                    Node::let_bind(
                        "qw_src",
                        Expr::add(Expr::var("qw_src_base"), Expr::var("qw_bit")),
                    ),
                    Node::let_bind(
                        "qw_slot",
                        Expr::add(Expr::var("qw_base_slot"), Expr::var("qw_rank")),
                    ),
                    Node::if_then(
                        Expr::lt(Expr::var("qw_slot"), Expr::u32(queue_capacity)),
                        vec![Node::store(
                            active_queue,
                            Expr::var("qw_slot"),
                            Expr::var("qw_src"),
                        )],
                    ),
                    Node::assign(
                        "qw_remaining",
                        Expr::bitand(
                            Expr::var("qw_remaining"),
                            Expr::sub(Expr::var("qw_remaining"), Expr::u32(1)),
                        ),
                    ),
                ],
            ),
        ],
    ));
    if let Some(frontier_out) = frontier_out_to_clear {
        word_body.insert(
            0,
            Node::store(frontier_out, Expr::var("qw_word_idx"), Expr::u32(0)),
        );
    }

    let body = vec![
        Node::let_bind("qw_word_idx", lane),
        Node::if_then(
            Expr::lt(Expr::var("qw_word_idx"), Expr::u32(words)),
            word_body,
        ),
    ];
    let mut buffers = vec![
        BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(words),
        BufferDecl::storage(active_queue, 1, BufferAccess::ReadWrite, DataType::U32)
            .with_count(queue_capacity),
        BufferDecl::storage(queue_len, 2, BufferAccess::ReadWrite, DataType::U32).with_count(1),
    ];
    if let Some(frontier_out) = frontier_out_to_clear {
        buffers.push(
            BufferDecl::storage(frontier_out, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
        );
    }
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
}
