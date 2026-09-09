//! Emitted-program shape contracts for the frontier queue op family.
//!
//! These assert the workgroup size, buffer layout, and atomic placement every
//! builder in this module emits, so a lowering change that silently duplicates
//! coverage or moves an atomic into a per-bit loop fails here.

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::visit::{walk_exprs, walk_nodes};

use super::{
    csr_queue_forward_traverse, frontier_queue_len_init, frontier_to_queue,
    frontier_to_queue_parallel, frontier_word_block_offsets_in_place,
    frontier_word_block_offsets_to_queue_parallel, frontier_word_block_prefix_to_queue_parallel,
    frontier_word_counts_scan_pass_a, frontier_words_to_queue_clear_out_parallel,
    frontier_words_to_queue_parallel,
};

#[test]
fn packed_word_queue_reserves_once_per_nonzero_word() {
    let program = frontier_words_to_queue_parallel("frontier", "queue", "len", 35, 16);

    assert_eq!(
        count_atomic_exprs(&program),
        1,
        "packed-word queue materialization should have one static queue reservation site"
    );
    assert_eq!(
        loop_atomic_count(&program, "qw_rank"),
        Some(0),
        "per-active-bit scatter loop must not issue atomics after the word-level reservation"
    );
    assert!(
        assignment_contains_u32(&program, "qw_remaining", 0b111),
        "the packed-word materializer must mask tail bits before popcounting the final frontier word"
        );
}

#[test]
fn emitted_programs_have_stable_shapes() {
    let queue_len_init = frontier_queue_len_init("len");
    assert_eq!(queue_len_init.workgroup_size, [1, 1, 1]);
    assert_eq!(queue_len_init.buffers.len(), 1);
    let queue = frontier_to_queue("frontier", "queue", "len", 64, 8);
    assert_eq!(queue.workgroup_size, [256, 1, 1]);
    assert_eq!(queue.buffers.len(), 3);
    let parallel_queue = frontier_to_queue_parallel("frontier", "queue", "len", 64, 8);
    assert_eq!(parallel_queue.workgroup_size, [256, 1, 1]);
    assert_eq!(parallel_queue.buffers.len(), 3);
    let word_queue = frontier_words_to_queue_parallel("frontier", "queue", "len", 64, 8);
    assert_eq!(word_queue.workgroup_size, [256, 1, 1]);
    assert_eq!(word_queue.buffers.len(), 3);
    assert_eq!(word_queue.buffers[0].count, 2);
    let word_queue_clear =
        frontier_words_to_queue_clear_out_parallel("frontier", "queue", "len", "out", 64, 8);
    assert_eq!(word_queue_clear.workgroup_size, [256, 1, 1]);
    assert_eq!(word_queue_clear.buffers.len(), 4);
    assert_eq!(word_queue_clear.buffers[0].count, 2);
    assert_eq!(word_queue_clear.buffers[3].name.as_ref(), "out");
    assert_eq!(word_queue_clear.buffers[3].count, 2);
    let word_scan = frontier_word_counts_scan_pass_a("frontier", "partials", "block_totals", 64);
    assert_eq!(word_scan.workgroup_size, [1024, 1, 1]);
    assert_eq!(word_scan.buffers.len(), 5);
    assert_eq!(word_scan.buffers[0].count, 2);
    assert_eq!(word_scan.buffers[1].count, 1024);
    assert_eq!(word_scan.buffers[2].count, 1);
    let block_offsets = frontier_word_block_offsets_in_place("block_totals", 32_897);
    assert_eq!(block_offsets.workgroup_size, [1024, 1, 1]);
    assert_eq!(block_offsets.buffers.len(), 3);
    assert_eq!(block_offsets.buffers[0].count, 2);
    let huge_block_offsets = frontier_word_block_offsets_in_place("block_totals", 33_554_433);
    assert_eq!(huge_block_offsets.workgroup_size, [1, 1, 1]);
    assert_eq!(huge_block_offsets.buffers.len(), 1);
    assert_eq!(huge_block_offsets.buffers[0].count, 1025);
    let prefix_queue = frontier_word_block_prefix_to_queue_parallel(
        "frontier",
        "partials",
        "block_totals",
        "queue",
        "len",
        64,
        8,
    );
    assert_eq!(prefix_queue.workgroup_size, [256, 1, 1]);
    assert_eq!(prefix_queue.buffers.len(), 5);
    assert_eq!(prefix_queue.buffers[0].count, 2);
    assert_eq!(prefix_queue.buffers[1].count, 1024);
    assert_eq!(prefix_queue.buffers[2].count, 1);
    let offset_queue = frontier_word_block_offsets_to_queue_parallel(
        "frontier",
        "partials",
        "block_offsets",
        "queue",
        "len",
        32_897,
        8,
    );
    assert_eq!(offset_queue.workgroup_size, [256, 1, 1]);
    assert_eq!(offset_queue.buffers.len(), 5);
    assert_eq!(offset_queue.buffers[0].count, 1029);
    assert_eq!(offset_queue.buffers[1].count, 2048);
    assert_eq!(offset_queue.buffers[2].count, 2);
    assert!(
        !format!("{:?}", offset_queue.entry()).contains("fwq_prev_block"),
        "precomputed-offset scatter must not retain the per-word previous-block loop"
    );
    let traverse = csr_queue_forward_traverse(
        "queue", "len", "offsets", "targets", "kinds", "out", 64, 7, 8, 1,
    );
    assert_eq!(traverse.workgroup_size, [256, 1, 1]);
    assert_eq!(traverse.buffers.len(), 6);
}

fn count_atomic_exprs(program: &Program) -> usize {
    let mut count = 0;
    walk_exprs(program, |expr| {
        if matches!(expr, Expr::Atomic { .. }) {
            count += 1;
        }
    });
    count
}

fn loop_atomic_count(program: &Program, loop_var: &str) -> Option<usize> {
    let mut count = None;
    walk_nodes(program, |node| {
        if count.is_some() {
            return;
        }
        if let Node::Loop { var, body, .. } = node {
            if var.as_ref() == loop_var {
                let loop_program = Program::wrapped(Vec::new(), [1, 1, 1], body.clone());
                count = Some(count_atomic_exprs(&loop_program));
            }
        }
    });
    count
}

fn assignment_contains_u32(program: &Program, target: &str, value: u32) -> bool {
    let mut found = false;
    walk_nodes(program, |node| {
        if found {
            return;
        }
        if let Node::Assign { name, value: expr } = node {
            if name.as_ref() == target && expr_contains_u32(expr, value) {
                found = true;
            }
        }
    });
    found
}

fn expr_contains_u32(expr: &Expr, value: u32) -> bool {
    let mut found = false;
    let expr_program = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::let_bind("__expr_probe", expr.clone())],
    );
    walk_exprs(&expr_program, |expr| {
        if matches!(expr, Expr::LitU32(found_value) if *found_value == value) {
            found = true;
        }
    });
    found
}
