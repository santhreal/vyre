//! Dense-bitmatrix traversal step: one destination node per lane, ORing its
//! reverse-adjacency bitrow against the frontier.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region};

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::frontier_plan::ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE;
use super::OP_ID;
use crate::bitset::bitset_words;
use crate::graph::frontier_bits::{set_bit, BitAccess};

/// Build the GPU Program for one dense step. Invocation `d`
/// computes `frontier_out[d] = any bit of (adj_rows[d] &
/// frontier_in) is set`.
#[must_use]
pub(crate) fn validate_dense_adj_counts(
    op_id: &str,
    frontier_out: &str,
    node_count: u32,
    op_name: &str,
) -> Result<(u32, u32), Program> {
    if node_count == 0 {
        return Err(trap_program(
            op_id,
            Some((frontier_out, DataType::U32)),
            format!("Fix: {op_name} requires node_count > 0, got 0."),
        ));
    }
    let words = bitset_words(node_count);
    let Some(adj_count) = u64::from(node_count).checked_mul(u64::from(words)) else {
        return Err(trap_program(
            op_id,
            Some((frontier_out, DataType::U32)),
            format!("Fix: {op_name} dense buffer size overflows u64 ({node_count} nodes x {words} words)."),
        ));
    };
    if adj_count > u64::from(u32::MAX) {
        return Err(trap_program(
            op_id,
            Some((frontier_out, DataType::U32)),
            format!("Fix: {op_name} dense buffer size {adj_count} exceeds u32::MAX ({node_count} nodes x {words} words). Partition the graph."),
        ));
    }
    Ok((words, adj_count as u32))
}

pub(crate) fn dense_reverse_scan_body(
    frontier_in: &str,
    frontier_out: &str,
    adj_rows_dense: &str,
    lane: &Expr,
    words: u32,
    prefix: &str,
) -> Vec<Node> {
    let row_start_name = format!("{prefix}row_start");
    let hit_name = format!("{prefix}hit");
    let w_name = format!("{prefix}w");
    let word_idx_name = format!("{prefix}word_idx");
    let bit_mask_name = format!("{prefix}bit_mask");
    let prev_name = format!("{prefix}prev");

    vec![
        Node::let_bind(
            row_start_name.clone(),
            Expr::mul(lane.clone(), Expr::u32(words)),
        ),
        Node::let_bind(hit_name.clone(), Expr::u32(0)),
        Node::loop_for(
            w_name.clone(),
            Expr::u32(0),
            Expr::u32(words),
            vec![Node::assign(
                hit_name.clone(),
                Expr::bitor(
                    Expr::var(hit_name.clone()),
                    Expr::bitand(
                        Expr::load(
                            adj_rows_dense,
                            Expr::add(Expr::var(row_start_name), Expr::var(w_name.clone())),
                        ),
                        Expr::load(frontier_in, Expr::var(w_name)),
                    ),
                ),
            )],
        ),
        Node::if_then(
            Expr::ne(Expr::var(hit_name), Expr::u32(0)),
            set_bit(
                frontier_out,
                lane,
                BitAccess {
                    word: &word_idx_name,
                    mask: &bit_mask_name,
                    value: &prev_name,
                },
                |word| word,
                Vec::new(),
            ),
        ),
    ]
}

/// Build a GPU Program for one dense reverse-row scan step.
#[must_use]
pub fn adaptive_dense_step(
    frontier_in: &str,
    frontier_out: &str,
    adj_rows_dense: &str,
    node_count: u32,
) -> Program {
    let (words, adj_count_u32) =
        match validate_dense_adj_counts(OP_ID, frontier_out, node_count, "adaptive_dense_step") {
            Ok(counts) => counts,
            Err(trap) => return trap,
        };
    let d = Expr::InvocationId { axis: 0 };
    let body = dense_reverse_scan_body(frontier_in, frontier_out, adj_rows_dense, &d, words, "");

    Program::wrapped(
        vec![
            BufferDecl::storage(frontier_in, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(words),
            BufferDecl::storage(frontier_out, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(words),
            BufferDecl::storage(adj_rows_dense, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(adj_count_u32),
        ],
        ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE,
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::lt(d.clone(), Expr::u32(node_count)),
                body,
            )],
        )],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitted_program_has_expected_shape() {
        let p = adaptive_dense_step("fin", "fout", "adj", 64);
        assert_eq!(p.workgroup_size, ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE);
        let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
        assert_eq!(names, vec!["fin", "fout", "adj"]);
        let find = |name: &str| p.buffers.iter().find(|b| b.name() == name).unwrap().count;
        let words = bitset_words(64);
        assert_eq!(find("fin"), words);
        assert_eq!(find("fout"), words);
        assert_eq!(find("adj"), 64 * words);
    }
}
