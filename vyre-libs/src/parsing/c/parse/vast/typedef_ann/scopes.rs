use super::*;

pub fn c11_precompute_vast_scopes(
    vast_nodes: &str,
    num_nodes: Expr,
    out_scoped_vast_nodes: &str,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let mut scan_body = Vec::new();
    scan_body.extend([
        Node::let_bind(
            "scope_row",
            Expr::mul(Expr::var("scope_i"), Expr::u32(VAST_NODE_STRIDE_U32)),
        ),
        Node::let_bind("scope_kind", Expr::load(vast_nodes, Expr::var("scope_row"))),
    ]);
    scan_body.extend(store_row_with_overrides(
        out_scoped_vast_nodes,
        vast_nodes,
        &Expr::var("scope_row"),
        &[(VAST_TYPEDEF_SCOPE_FIELD, "scope_current")],
    ));
    scan_body.extend([
        Node::if_then(
            Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_LBRACE)),
            vec![
                Node::store(
                    "__vast_scope_stack",
                    Expr::var("scope_stack_depth"),
                    Expr::var("scope_i"),
                ),
                Node::assign(
                    "scope_stack_depth",
                    Expr::add(Expr::var("scope_stack_depth"), Expr::u32(1)),
                ),
                Node::assign("scope_current", Expr::var("scope_i")),
            ],
        ),
        Node::if_then(
            Expr::and(
                Expr::eq(Expr::var("scope_kind"), Expr::u32(TOK_RBRACE)),
                Expr::gt(Expr::var("scope_stack_depth"), Expr::u32(0)),
            ),
            vec![
                Node::assign(
                    "scope_stack_depth",
                    Expr::sub(Expr::var("scope_stack_depth"), Expr::u32(1)),
                ),
                Node::let_bind(
                    "scope_safe_top",
                    Expr::select(
                        Expr::gt(Expr::var("scope_stack_depth"), Expr::u32(0)),
                        Expr::sub(Expr::var("scope_stack_depth"), Expr::u32(1)),
                        Expr::u32(0),
                    ),
                ),
                Node::assign(
                    "scope_current",
                    Expr::select(
                        Expr::gt(Expr::var("scope_stack_depth"), Expr::u32(0)),
                        Expr::load("__vast_scope_stack", Expr::var("scope_safe_top")),
                        Expr::u32(SENTINEL),
                    ),
                ),
            ],
        ),
    ]);

    let body = vec![Node::if_then(
        Expr::eq(t.clone(), Expr::u32(0)),
        vec![
            Node::let_bind("scope_stack_depth", Expr::u32(0)),
            Node::let_bind("scope_current", Expr::u32(SENTINEL)),
            Node::loop_for("scope_i", Expr::u32(0), num_nodes.clone(), scan_body),
        ],
    )];

    let rows = declared_rows(&num_nodes);
    let scope_stack_decl = if c11_precompute_vast_scopes_uses_global_stack(rows) {
        BufferDecl::storage(
            "__vast_scope_stack",
            2,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(rows)
        .with_output_byte_range(0..0)
    } else {
        BufferDecl::workgroup("__vast_scope_stack", rows, DataType::U32)
    };
    Program::wrapped(
        vec![
            vast_nodes_input(vast_nodes, 0, rows),
            vast_nodes_scratch(out_scoped_vast_nodes, 1, rows),
            scope_stack_decl,
        ],
        [1, 1, 1],
        vec![wrap_anonymous(PRECOMPUTE_VAST_SCOPES_OP_ID, body)],
    )
    .with_entry_op_id(PRECOMPUTE_VAST_SCOPES_OP_ID)
    .with_non_composable_with_self(true)
}

pub(super) const VAST_SCOPE_STACK_WORKGROUP_MAX_U32: u32 = 0;

#[must_use]
pub fn c11_precompute_vast_scopes_uses_global_stack(num_nodes: u32) -> bool {
    num_nodes.max(1) > VAST_SCOPE_STACK_WORKGROUP_MAX_U32
}
