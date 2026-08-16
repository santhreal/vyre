use super::*;

pub(in crate::parsing::c::parse::vast) const DECL_KIND_FOR_ROW_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_decl_kind_for_row";
pub(in crate::parsing::c::parse::vast) const DECL_KIND_FOR_ROW_PACKED_OP_ID: &str =
    "vyre-libs::parsing::c11_typedef_decl_kind_for_row_packed_haystack";

/// The builtin declaration kind of one row, as an operation of its own.
pub(crate) const BUILTIN_DECL_KIND_FOR_ROW_OP_ID: &str =
    "vyre-libs::parsing::c11_builtin_declaration_kind_for_row";

/// The declaration kind of row `idx`, including the typedef-name lookup the
/// prefix scan needs the source text for.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_declaration_kind_for_index(
    parent_op_id: &str,
    vast_nodes: &str,
    haystack: &str,
    haystack_len: &Expr,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    packed_haystack: bool,
    decl_contexts: Option<&str>,
) -> Vec<Node> {
    let mut nodes = vec![Node::let_bind(out_name, Expr::u32(0))];
    nodes.extend(emit_declaration_kind_for_index_inner(
        parent_op_id,
        vast_nodes,
        decl_contexts,
        idx,
        out_name,
        prefix,
        Some((haystack, haystack_len, packed_haystack)),
    ));
    nodes
}

/// The declaration kind of row `idx` from structure alone.
///
/// Without a precomputed context table this is the registered
/// [`BUILTIN_DECL_KIND_FOR_ROW_OP_ID`] operation, so the emission names it as a
/// block of `parent_op_id`. With one, the scan reads a table the operation does
/// not declare, so it stays an inline part of its caller.
pub(crate) fn emit_builtin_declaration_kind_for_index(
    parent_op_id: &str,
    vast_nodes: &str,
    idx: Expr,
    out_name: &str,
    prefix: &str,
    decl_contexts: Option<&str>,
) -> Vec<Node> {
    let owner = if decl_contexts.is_some() {
        parent_op_id
    } else {
        BUILTIN_DECL_KIND_FOR_ROW_OP_ID
    };
    let scan = emit_declaration_kind_for_index_inner(
        owner,
        vast_nodes,
        decl_contexts,
        idx,
        out_name,
        prefix,
        None,
    );
    let mut nodes = vec![Node::let_bind(out_name, Expr::u32(0))];
    if decl_contexts.is_some() {
        nodes.extend(scan);
    } else {
        nodes.push(child_phase(
            parent_op_id,
            BUILTIN_DECL_KIND_FOR_ROW_OP_ID,
            scan,
        ));
    }
    nodes
}

/// The registered operation: the builtin declaration kind of one row.
pub(in crate::parsing::c::parse::vast) fn c11_builtin_declaration_kind_for_row() -> Program {
    const KIND: &str = "phase_declaration_kind";

    let mut body = vec![Node::let_bind(KIND, Expr::u32(0))];
    body.extend(emit_declaration_kind_for_index_inner(
        BUILTIN_DECL_KIND_FOR_ROW_OP_ID,
        phase_program::NODES,
        None,
        phase_row(),
        KIND,
        "phase_builtin_decl",
        None,
    ));
    phase_program(
        BUILTIN_DECL_KIND_FOR_ROW_OP_ID,
        PhaseInputs::RowAndNumNodes,
        KIND,
        body,
    )
}

fn decl_kind_phase_program(op_id: &str, packed_haystack: bool) -> Program {
    const KIND: &str = "phase_declaration_kind";

    let haystack_len = phase_haystack_len();
    phase_program(
        op_id,
        PhaseInputs::RowWithHaystack { packed_haystack },
        KIND,
        emit_declaration_kind_for_index(
            op_id,
            phase_program::NODES,
            phase_program::HAYSTACK,
            &haystack_len,
            phase_row(),
            KIND,
            "phase_decl",
            packed_haystack,
            None,
        ),
    )
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
