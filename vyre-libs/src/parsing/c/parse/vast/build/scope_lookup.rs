use super::*;

/// The reverse scope walk for one row, as an operation of its own.
pub(crate) const SCOPE_OPEN_FOR_ROW_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_scope_open_for_row";

/// Declare `out_name` and fill it with the scope walk, as a block of
/// `parent_op_id`.
pub(crate) fn emit_scope_open_for_index(
    parent_op_id: &str,
    vast_nodes: &str,
    idx: Expr,
    out_name: &str,
    prefix: &str,
) -> Vec<Node> {
    vec![
        Node::let_bind(out_name, Expr::u32(SENTINEL)),
        emit_scope_open_scan_phase(parent_op_id, vast_nodes, idx, out_name, prefix),
    ]
}

/// The scope walk as a block of `parent_op_id`, for a caller that declared
/// `out_name` itself.
pub(crate) fn emit_scope_open_scan_phase(
    parent_op_id: &str,
    vast_nodes: &str,
    idx: Expr,
    out_name: &str,
    prefix: &str,
) -> Node {
    child_phase(
        parent_op_id,
        SCOPE_OPEN_FOR_ROW_OP_ID,
        emit_scope_open_scan_assign_for_index(vast_nodes, idx, out_name, prefix),
    )
}

/// Walk backwards from `idx` to the innermost enclosing `{` that is still open
/// and assign its row index to `out_name`, leaving the caller's value in place
/// when the row is at file scope.
///
/// This runs for EVERY row, not only identifiers: the CPU oracle writes
/// `scope_open_before(node_idx)` to the scope field unconditionally, so gating
/// it on `raw_kind == TOK_IDENTIFIER` leaves the carrier at `SENTINEL` on every
/// brace, paren and semicolon and diverges on all of them.
pub(crate) fn emit_scope_open_scan_assign_for_index(
    vast_nodes: &str,
    idx: Expr,
    out_name: &str,
    prefix: &str,
) -> Vec<Node> {
    let depth = format!("{prefix}_depth");
    let scan = format!("{prefix}_scan");
    let rev = format!("{prefix}_idx");
    let kind = format!("{prefix}_kind");

    vec![
        Node::let_bind(&depth, Expr::u32(0)),
        Node::loop_for(
            &scan,
            Expr::u32(0),
            idx.clone(),
            vec![
                Node::let_bind(
                    &rev,
                    Expr::sub(Expr::sub(idx, Expr::u32(1)), Expr::var(&scan)),
                ),
                Node::let_bind(
                    &kind,
                    Expr::load(
                        vast_nodes,
                        Expr::mul(Expr::var(&rev), Expr::u32(VAST_NODE_STRIDE_U32)),
                    ),
                ),
                Node::if_then(
                    Expr::eq(Expr::var(&kind), Expr::u32(TOK_RBRACE)),
                    vec![Node::assign(
                        &depth,
                        Expr::add(Expr::var(&depth), Expr::u32(1)),
                    )],
                ),
                Node::if_then(
                    Expr::eq(Expr::var(out_name), Expr::u32(SENTINEL)),
                    vec![Node::if_then(
                        Expr::eq(Expr::var(&kind), Expr::u32(TOK_LBRACE)),
                        vec![Node::if_then_else(
                            Expr::eq(Expr::var(&depth), Expr::u32(0)),
                            vec![Node::assign(out_name, Expr::var(&rev))],
                            vec![Node::assign(
                                &depth,
                                Expr::sub(Expr::var(&depth), Expr::u32(1)),
                            )],
                        )],
                    )],
                ),
            ],
        ),
    ]
}

/// The registered operation: the scope walk for one row, on its own.
pub(in crate::parsing::c::parse::vast) fn c11_typedef_scope_open_for_row() -> Program {
    const SCOPE_OPEN: &str = "phase_scope_open";

    let mut body = vec![Node::let_bind(SCOPE_OPEN, Expr::u32(SENTINEL))];
    body.extend(emit_scope_open_scan_assign_for_index(
        phase_program::NODES,
        phase_row(),
        SCOPE_OPEN,
        "phase_scope",
    ));
    phase_program(
        SCOPE_OPEN_FOR_ROW_OP_ID,
        PhaseInputs::Row,
        SCOPE_OPEN,
        body,
    )
}
