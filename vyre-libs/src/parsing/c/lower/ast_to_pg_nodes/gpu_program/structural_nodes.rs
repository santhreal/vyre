use super::*;

/// Lower structural VAST rows (`kind`, `span`, `parent`, `payload`) into
/// packed Program-Graph rows:
/// `(kind, span_start, span_end, parent_idx, first_child_idx, next_sibling_idx)`.
///
/// `num_nodes` controls both dispatch bounds and buffer sizing so this stays
/// composable with one-thread-per-node invocation. Inputs outside the declared
/// `num_nodes` range are masked by the dispatch bound.
#[must_use]
pub fn c_lower_ast_to_pg_nodes(vast_nodes: &str, num_nodes: Expr, out_pg_nodes: &str) -> Program {
    let t = Expr::InvocationId { axis: 0 };

    let row = VastRow {
        vast_nodes,
        base: Expr::mul(t.clone(), Expr::u32(VAST_NODE_STRIDE_U32)),
    };
    let pg_base = Expr::mul(t.clone(), Expr::u32(PG_NODE_STRIDE_U32));

    let mut loop_body = row.structural_bindings();
    loop_body.extend(store_pg_node_row(out_pg_nodes, &pg_base));

    let in_words = infer_node_count_words(&num_nodes)
        .saturating_mul(VAST_NODE_STRIDE_U32)
        .max(1);
    let out_words = infer_node_count_words(&num_nodes)
        .saturating_mul(PG_NODE_STRIDE_U32)
        .max(1);

    Program::wrapped(
        vec![
            BufferDecl::storage(vast_nodes, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(in_words),
            BufferDecl::output(out_pg_nodes, 1, DataType::U32).with_count(out_words),
        ],
        [256, 1, 1],
        vec![crate::region::wrap_anonymous(
            OP_ID,
            vec![Node::if_then(
                Expr::lt(t.clone(), num_nodes.clone()),
                loop_body,
            )],
        )],
    )
    .with_entry_op_id(OP_ID)
}
