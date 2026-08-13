use super::*;

pub fn c11_prehash_vast_identifiers(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_hashed_vast_nodes: &str,
) -> Program {
    c11_prehash_vast_identifiers_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_hashed_vast_nodes,
        false,
    )
}

pub fn c11_prehash_vast_identifiers_packed_haystack(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_hashed_vast_nodes: &str,
) -> Program {
    c11_prehash_vast_identifiers_impl(
        vast_nodes,
        haystack,
        haystack_len,
        num_nodes,
        out_hashed_vast_nodes,
        true,
    )
}

pub(super) fn c11_prehash_vast_identifiers_impl(
    vast_nodes: &str,
    haystack: &str,
    haystack_len: Expr,
    num_nodes: Expr,
    out_hashed_vast_nodes: &str,
    packed_haystack: bool,
) -> Program {
    let t = Expr::InvocationId { axis: 0 };
    let base = Expr::mul(t.clone(), Expr::u32(VAST_NODE_STRIDE_U32));

    let row = IdentifierRowHash {
        vast_nodes,
        haystack,
        haystack_len: &haystack_len,
        row_base: base.clone(),
        packed_haystack,
        names: IdentifierRowHashNames {
            start: "tok_start",
            len: "tok_len",
            hash: "name_hash",
            cursor: "hash_i",
            byte: "hash_byte",
        },
    };

    let mut loop_body = vec![Node::let_bind("raw_kind", Expr::load(vast_nodes, base.clone()))];
    loop_body.extend(row.nodes(Expr::eq(
        Expr::var("raw_kind"),
        Expr::u32(TOK_IDENTIFIER),
    )));

    for field in 0..VAST_NODE_STRIDE_U32 {
        let value = if field == VAST_TYPEDEF_SYMBOL_FIELD {
            Expr::var("name_hash")
        } else {
            Expr::load(vast_nodes, Expr::add(base.clone(), Expr::u32(field)))
        };
        loop_body.push(Node::store(
            out_hashed_vast_nodes,
            Expr::add(base.clone(), Expr::u32(field)),
            value,
        ));
    }

    let n = node_count(&num_nodes).max(1);
    Program::wrapped(
        vec![
            BufferDecl::storage(vast_nodes, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(n.saturating_mul(VAST_NODE_STRIDE_U32)),
            BufferDecl::storage(haystack, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_word_count(&haystack_len, packed_haystack)),
            BufferDecl::storage(
                out_hashed_vast_nodes,
                2,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(n.saturating_mul(VAST_NODE_STRIDE_U32)),
        ],
        [256, 1, 1],
        vec![wrap_anonymous(
            PREHASH_VAST_IDENTIFIERS_OP_ID,
            vec![Node::if_then(Expr::lt(t, num_nodes), loop_body)],
        )],
    )
    .with_entry_op_id(PREHASH_VAST_IDENTIFIERS_OP_ID)
    .with_non_composable_with_self(true)
}
