//! Dense-bitmatrix traversal step: one destination node per lane, ORing its
//! reverse-adjacency bitrow against the frontier.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::frontier_plan::ADAPTIVE_TRAVERSAL_LINEAR_WORKGROUP_SIZE;
use super::OP_ID;
use crate::bitset::bitset_words;

/// Build the GPU Program for one dense step. Invocation `d`
/// computes `frontier_out[d] = any bit of (adj_rows[d] &
/// frontier_in) is set`.
#[must_use]
pub fn adaptive_dense_step(
    frontier_in: &str,
    frontier_out: &str,
    adj_rows_dense: &str,
    node_count: u32,
) -> Program {
    if node_count == 0 {
        return crate::invalid_output_program(
            OP_ID,
            frontier_out,
            DataType::U32,
            "Fix: adaptive_dense_step requires node_count > 0, got 0.".to_string(),
        );
    }
    let words = bitset_words(node_count);
    // PHASE7_GRAPH C1: the adjacency buffer size is `node_count *
    // words`. A u32 × u32 multiply wraps silently for non-trivial
    // inputs (e.g. node_count ≈ 400k, words ≈ 12.5k wraps past
    // u32::MAX), producing a tiny buffer and catastrophic OOB
    // reads/writes. Check in u64 first and refuse programs we
    // cannot represent faithfully.
    let Some(adj_count) = u64::from(node_count).checked_mul(u64::from(words)) else {
        return crate::invalid_output_program(OP_ID,
        frontier_out,
        DataType::U32,
        format!("Fix: adaptive_dense_step buffer size overflows u64 ({node_count} nodes x {words} words)."),);
    };
    if adj_count > u64::from(u32::MAX) {
        return crate::invalid_output_program(OP_ID,
        frontier_out,
        DataType::U32,
        format!("Fix: adaptive_dense_step buffer size {adj_count} exceeds u32::MAX ({node_count} nodes x {words} words). Partition the graph or use csr_forward_traverse."),);
    }
    let adj_count_u32 = adj_count as u32;
    let d = Expr::InvocationId { axis: 0 };

    let body: Vec<Node> = vec![
        Node::let_bind("row_start", Expr::mul(d.clone(), Expr::u32(words))),
        Node::let_bind("hit", Expr::u32(0)),
        Node::loop_for(
            "w",
            Expr::u32(0),
            Expr::u32(words),
            vec![Node::assign(
                "hit",
                Expr::bitor(
                    Expr::var("hit"),
                    Expr::bitand(
                        Expr::load(
                            adj_rows_dense,
                            Expr::add(Expr::var("row_start"), Expr::var("w")),
                        ),
                        Expr::load(frontier_in, Expr::var("w")),
                    ),
                ),
            )],
        ),
        Node::if_then(
            Expr::ne(Expr::var("hit"), Expr::u32(0)),
            vec![
                Node::let_bind("word_idx", Expr::shr(d.clone(), Expr::u32(5))),
                Node::let_bind(
                    "bit_mask",
                    Expr::shl(Expr::u32(1), Expr::bitand(d.clone(), Expr::u32(31))),
                ),
                Node::let_bind(
                    "_",
                    Expr::atomic_or(frontier_out, Expr::var("word_idx"), Expr::var("bit_mask")),
                ),
            ],
        ),
    ];

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
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::lt(d.clone(), Expr::u32(node_count)),
                body,
            )]),
        }],
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
