use super::*;

pub(in crate::parsing::c::parse::vast) const DECL_KIND_FOR_ROW_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_decl_kind_for_row";
pub(in crate::parsing::c::parse::vast) const DECL_KIND_FOR_ROW_PACKED_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_decl_kind_for_row_packed_haystack";

pub(crate) fn emit_declaration_kind_for_index(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: &Expr,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    packed_haystack: bool,
    decl_contexts: Option<&str>,
) -> Vec<Node> {
    emit_declaration_kind_for_index_inner(
        vast_nodes,
        decl_contexts,
        idx,
        out_name,
        prefix,
        Some((haystack, haystack_len, packed_haystack)),
    )
}

pub(crate) fn emit_builtin_declaration_kind_for_index(
    vast_nodes: &str,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    decl_contexts: Option<&str>,
) -> Vec<Node> {
    emit_declaration_kind_for_index_inner(vast_nodes, decl_contexts, idx, out_name, prefix, None)
}

fn decl_kind_phase_program(op_id: &str, packed_haystack: bool) -> Program {
    const NODES: &str = "phase_vast_nodes";
    const HAYSTACK: &str = "phase_haystack";
    const ROW: &str = "phase_row";
    const HAYSTACK_LEN: &str = "phase_haystack_len";
    const NUM_NODES: &str = "phase_num_nodes";
    const RESULT: &str = "phase_result";

    let row = Expr::load(ROW, Expr::u32(0));
    let haystack_len = Expr::load(HAYSTACK_LEN, Expr::u32(0));
    let mut body = vec![Node::let_bind(
        "annot_num_nodes",
        Expr::load(NUM_NODES, Expr::u32(0)),
    )];
    body.extend(emit_declaration_kind_for_index(
        NODES,
        HAYSTACK,
        &haystack_len,
        row,
        RESULT,
        "phase_decl",
        packed_haystack,
        None,
    ));
    body.push(Node::store(RESULT, Expr::u32(0), Expr::var(RESULT)));

    let buffers = vec![
        BufferDecl::storage(NODES, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(VAST_NODE_STRIDE_U32),
        BufferDecl::storage(HAYSTACK, 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        BufferDecl::storage(ROW, 2, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        BufferDecl::storage(HAYSTACK_LEN, 3, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        BufferDecl::storage(NUM_NODES, 4, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        BufferDecl::output(RESULT, 5, DataType::U32).with_count(1),
    ];
    let implementation = child_phase(op_id, &format!("{op_id}::declaration_scan"), body);
    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous(op_id, vec![implementation])],
    )
    .with_entry_op_id(op_id)
}

pub(in crate::parsing::c::parse::vast) fn c11_typedef_decl_kind_for_row() -> Program {
    decl_kind_phase_program(DECL_KIND_FOR_ROW_OP_ID, false)
}

pub(in crate::parsing::c::parse::vast) fn c11_typedef_decl_kind_for_row_packed_haystack() -> Program
{
    decl_kind_phase_program(DECL_KIND_FOR_ROW_PACKED_OP_ID, true)
}

pub(crate) fn emit_declaration_kind_result_assignment(
    out_name: &str,
    is_identifier: Expr,
    declarator_follower: Expr,
    previous_token_allows_declarator: Expr,
    next_token_allows_declarator: Expr,
    contextual_declarator_allowed: Expr,
    has_typedef: Expr,
    has_type: Expr,
) -> Node {
    Node::if_then(
        Expr::and(
            is_identifier,
            Expr::and(
                declarator_follower,
                Expr::and(
                    previous_token_allows_declarator,
                    Expr::and(
                        next_token_allows_declarator,
                        Expr::and(
                            contextual_declarator_allowed,
                            Expr::or(has_typedef.clone(), has_type),
                        ),
                    ),
                ),
            ),
        ),
        vec![Node::assign(
            out_name,
            Expr::select(has_typedef, Expr::u32(1), Expr::u32(2)),
        )],
    )
}
