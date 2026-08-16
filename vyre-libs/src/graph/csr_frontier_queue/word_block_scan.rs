//! Deterministic packed-frontier prefix scan: per-word popcounts, per-block
//! totals, and the in-place conversion of block totals into queue offsets.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::sizing_diagnostics::{
    checked_frontier_u32_product, invalid_frontier_queue_sizing_program, try_u32_byte_range,
};
use super::{FRONTIER_WORD_BLOCK_OFFSETS_IN_PLACE_OP_ID, FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID};
use crate::bitset::bitset_words;

/// Build Pass A for deterministic packed-frontier queue materialization.
///
/// Each workgroup scans one block of packed frontier words. Lane `L` in block
/// `B` computes the in-range popcount for word `B*1024 + L`, then participates
/// in a local inclusive Hillis-Steele scan. The program writes one per-word
/// inclusive count into `word_partials` and one per-block total into
/// `block_totals`.
#[must_use]
pub fn frontier_word_counts_scan_pass_a(
    frontier_in: &str,
    word_partials: &str,
    block_totals: &str,
    node_count: u32,
) -> Program {
    if node_count == 0 {
        return trap_program(
            FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID,
            Some((word_partials, DataType::U32)),
            "Fix: frontier_word_counts_scan_pass_a requires node_count > 0.".to_string(),
        );
    }
    let words = bitset_words(node_count);
    let block_lanes = 1024_u32;
    let num_blocks = words.div_ceil(block_lanes).max(1);
    let total_partials = match checked_frontier_u32_product(
        num_blocks,
        block_lanes,
        "frontier_word_counts_scan_pass_a partial word count",
    ) {
        Ok(total_partials) => total_partials,
        Err(error) => {
            return invalid_frontier_queue_sizing_program(
                FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID,
                word_partials,
                error,
            );
        }
    };
    let partial_bytes =
        match try_u32_byte_range(total_partials, "frontier_word_counts_scan_pass_a partials") {
            Ok(partial_bytes) => partial_bytes,
            Err(error) => {
                return invalid_frontier_queue_sizing_program(
                    FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID,
                    word_partials,
                    error,
                );
            }
        };
    let block_total_bytes =
        match try_u32_byte_range(num_blocks, "frontier_word_counts_scan_pass_a block totals") {
            Ok(block_total_bytes) => block_total_bytes,
            Err(error) => {
                return invalid_frontier_queue_sizing_program(
                    FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID,
                    block_totals,
                    error,
                );
            }
        };
    let tail_bits = node_count & 31;
    let tail_mask = if tail_bits == 0 {
        u32::MAX
    } else {
        (1_u32 << tail_bits) - 1
    };

    let lane = Expr::var("fwcs_lane");
    let block = Expr::var("fwcs_block");
    let global = Expr::var("fwcs_global");
    let scratch_a = format!("__{word_partials}_fwcs_scratch_a");
    let scratch_b = format!("__{word_partials}_fwcs_scratch_b");

    let mut body = Vec::new();
    body.push(Node::let_bind("fwcs_lane", Expr::LocalId { axis: 0 }));
    body.push(Node::let_bind("fwcs_block", Expr::WorkgroupId { axis: 0 }));
    body.push(Node::let_bind(
        "fwcs_global",
        Expr::add(
            Expr::mul(block.clone(), Expr::u32(block_lanes)),
            lane.clone(),
        ),
    ));
    body.push(Node::store(&scratch_a, lane.clone(), Expr::u32(0)));
    let mut load_word = vec![Node::let_bind(
        "fwcs_word",
        Expr::load(frontier_in, global.clone()),
    )];
    if tail_bits != 0 {
        load_word.push(Node::if_then(
            Expr::eq(global.clone(), Expr::u32(words - 1)),
            vec![Node::assign(
                "fwcs_word",
                Expr::bitand(Expr::var("fwcs_word"), Expr::u32(tail_mask)),
            )],
        ));
    }
    load_word.push(Node::store(
        &scratch_a,
        lane.clone(),
        Expr::popcount(Expr::var("fwcs_word")),
    ));
    body.push(Node::if_then(
        Expr::lt(global.clone(), Expr::u32(words)),
        load_word,
    ));
    body.push(Node::Barrier {
        ordering: MemoryOrdering::SeqCst,
    });

    body.extend(crate::reduce::workgroup_tree::blelloch_inclusive_sum_nodes(
        &scratch_a,
        &scratch_b,
        &lane,
        block_lanes,
    ));

    body.push(Node::if_then(
        Expr::lt(global.clone(), Expr::u32(words)),
        vec![Node::store(
            word_partials,
            global.clone(),
            Expr::load(&scratch_a, lane.clone()),
        )],
    ));
    body.push(Node::if_then(
        Expr::eq(lane.clone(), Expr::u32(block_lanes - 1)),
        vec![Node::store(
            block_totals,
            block.clone(),
            Expr::load(&scratch_a, lane.clone()),
        )],
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::output(word_partials, 1, DataType::U32)
                .with_count(total_partials)
                .with_output_byte_range(0..partial_bytes),
            BufferDecl::storage(block_totals, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_blocks)
                .with_pipeline_live_out(true)
                .with_output_byte_range(0..block_total_bytes),
            BufferDecl::workgroup(&scratch_a, block_lanes, DataType::U32),
            BufferDecl::workgroup(&scratch_b, block_lanes, DataType::U32),
        ],
        [block_lanes, 1, 1],
        vec![wrap_anonymous_region(
            FRONTIER_WORD_COUNTS_SCAN_PASS_A_OP_ID,
            body,
        )],
    )
}

/// Convert per-block active counts into exclusive per-block queue offsets.
///
/// The conversion is in-place: after this program runs, `block_totals[B]`
/// contains the number of active nodes in all prior blocks. For up to 1024
/// blocks this uses one guarded workgroup scan; beyond that it falls back to a
/// single-lane linear scan over block metadata, which is still O(blocks)
/// instead of the old O(words * blocks) scatter-side prefix work.
#[must_use]
pub fn frontier_word_block_offsets_in_place(block_totals: &str, node_count: u32) -> Program {
    if node_count == 0 {
        return trap_program(
            FRONTIER_WORD_BLOCK_OFFSETS_IN_PLACE_OP_ID,
            Some((block_totals, DataType::U32)),
            "Fix: frontier_word_block_offsets_in_place requires node_count > 0.".to_string(),
        );
    }
    let words = bitset_words(node_count);
    let block_lanes = 1024_u32;
    let num_blocks = words.div_ceil(block_lanes).max(1);
    let block_total_bytes = match try_u32_byte_range(
        num_blocks,
        "frontier_word_block_offsets_in_place block totals",
    ) {
        Ok(block_total_bytes) => block_total_bytes,
        Err(error) => {
            return invalid_frontier_queue_sizing_program(
                FRONTIER_WORD_BLOCK_OFFSETS_IN_PLACE_OP_ID,
                block_totals,
                error,
            );
        }
    };
    if num_blocks <= block_lanes {
        return frontier_word_block_offsets_single_workgroup(
            block_totals,
            num_blocks,
            block_total_bytes,
        );
    }
    frontier_word_block_offsets_single_lane(block_totals, num_blocks, block_total_bytes)
}

fn frontier_word_block_offsets_single_workgroup(
    block_totals: &str,
    num_blocks: u32,
    block_total_bytes: usize,
) -> Program {
    let lane = Expr::var("fwbo_lane");
    let scratch_a = format!("__{block_totals}_fwbo_scratch_a");
    let scratch_b = format!("__{block_totals}_fwbo_scratch_b");
    let mut body = Vec::new();
    body.push(Node::let_bind("fwbo_lane", Expr::LocalId { axis: 0 }));
    body.push(Node::store(&scratch_a, lane.clone(), Expr::u32(0)));
    body.push(Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(num_blocks)),
        vec![Node::store(
            &scratch_a,
            lane.clone(),
            Expr::load(block_totals, lane.clone()),
        )],
    ));
    body.push(Node::Barrier {
        ordering: MemoryOrdering::SeqCst,
    });

    body.extend(crate::reduce::workgroup_tree::blelloch_inclusive_sum_nodes(
        &scratch_a, &scratch_b, &lane, 1024,
    ));

    body.push(Node::if_then(
        Expr::lt(lane.clone(), Expr::u32(num_blocks)),
        vec![
            Node::if_then(
                Expr::eq(lane.clone(), Expr::u32(0)),
                vec![Node::store(block_totals, lane.clone(), Expr::u32(0))],
            ),
            Node::if_then(
                Expr::ne(lane.clone(), Expr::u32(0)),
                vec![Node::store(
                    block_totals,
                    lane.clone(),
                    Expr::load(&scratch_a, Expr::sub(lane.clone(), Expr::u32(1))),
                )],
            ),
        ],
    ));

    Program::wrapped(
        vec![
            BufferDecl::storage(block_totals, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_blocks)
                .with_pipeline_live_out(true)
                .with_output_byte_range(0..block_total_bytes),
            BufferDecl::workgroup(&scratch_a, 1024, DataType::U32),
            BufferDecl::workgroup(&scratch_b, 1024, DataType::U32),
        ],
        [1024, 1, 1],
        vec![wrap_anonymous_region(
            FRONTIER_WORD_BLOCK_OFFSETS_IN_PLACE_OP_ID,
            body,
        )],
    )
}

fn frontier_word_block_offsets_single_lane(
    block_totals: &str,
    num_blocks: u32,
    block_total_bytes: usize,
) -> Program {
    let body = vec![
        Node::let_bind("fwbo_running", Expr::u32(0)),
        Node::loop_for(
            "fwbo_block",
            Expr::u32(0),
            Expr::u32(num_blocks),
            vec![
                Node::let_bind(
                    "fwbo_total",
                    Expr::load(block_totals, Expr::var("fwbo_block")),
                ),
                Node::store(
                    block_totals,
                    Expr::var("fwbo_block"),
                    Expr::var("fwbo_running"),
                ),
                Node::assign(
                    "fwbo_running",
                    Expr::add(Expr::var("fwbo_running"), Expr::var("fwbo_total")),
                ),
            ],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage(block_totals, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(num_blocks)
                .with_pipeline_live_out(true)
                .with_output_byte_range(0..block_total_bytes),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            FRONTIER_WORD_BLOCK_OFFSETS_IN_PLACE_OP_ID,
            body,
        )],
    )
}
