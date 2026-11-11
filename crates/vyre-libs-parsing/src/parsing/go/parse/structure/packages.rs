//! `package` declarations and imported string spans.

use crate::parsing::go::parse::token_predicates::{
    emit_keyword_span_record_nodes, emit_span_record_nodes, token_is_ident, token_is_keyword,
    token_type_eq,
};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_libs_builder::builder::trip_count::clamped_by_extents;
use vyre_spec::go_token::{TOK_LPAREN, TOK_RPAREN, TOK_STRING};

/// Extract `package` declarations and imported string spans.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_extract_packages_and_imports(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_packages: &str,
    out_package_counts: &str,
    out_imports: &str,
    out_import_counts: &str,
) -> Program {
    // `num_tokens` is the caller's token count and reaches three forward scans
    // as their trip count, so it is clamped once here to the token array it
    // indexes.
    let num_tokens = clamped_by_extents(num_tokens, tok_types, [tok_starts, tok_lens]);
    let t = Expr::logical_index(0);
    let body = vec![
        emit_keyword_span_record_nodes(
            haystack,
            tok_types,
            tok_starts,
            tok_lens,
            t.clone(),
            num_tokens.clone(),
            b"package",
            token_is_ident(tok_types, Expr::add(t.clone(), Expr::u32(1))),
            out_packages,
            out_package_counts,
            "pkg_idx",
        ),
        Node::if_then(
            Expr::lt(Expr::add(t.clone(), Expr::u32(1)), num_tokens.clone()),
            vec![Node::if_then(
                token_is_keyword(
                    haystack,
                    tok_types,
                    tok_starts,
                    tok_lens,
                    t.clone(),
                    b"import",
                ),
                vec![
                    Node::if_then(
                        token_type_eq(tok_types, Expr::add(t.clone(), Expr::u32(1)), TOK_STRING),
                        emit_span_record_nodes(
                            tok_starts,
                            tok_lens,
                            out_imports,
                            out_import_counts,
                            "import_idx",
                            Expr::add(t.clone(), Expr::u32(1)),
                        ),
                    ),
                    Node::if_then(
                        token_type_eq(tok_types, Expr::add(t.clone(), Expr::u32(1)), TOK_LPAREN),
                        vec![
                            Node::let_bind("import_done", Expr::u32(0)),
                            Node::loop_for(
                                "scan",
                                Expr::add(t.clone(), Expr::u32(2)),
                                num_tokens.clone(),
                                vec![Node::if_then(
                                    Expr::eq(Expr::var("import_done"), Expr::u32(0)),
                                    vec![
                                        Node::if_then(
                                            token_type_eq(tok_types, Expr::var("scan"), TOK_STRING),
                                            emit_span_record_nodes(
                                                tok_starts,
                                                tok_lens,
                                                out_imports,
                                                out_import_counts,
                                                "import_idx",
                                                Expr::var("scan"),
                                            ),
                                        ),
                                        Node::if_then(
                                            token_type_eq(tok_types, Expr::var("scan"), TOK_RPAREN),
                                            vec![Node::assign("import_done", Expr::u32(1))],
                                        ),
                                    ],
                                )],
                            ),
                        ],
                    ),
                ],
            )],
        ),
    ];

    let mut buffers = super::super::token_stream_decls(tok_types, tok_starts, tok_lens, haystack);
    buffers.extend([
        BufferDecl::storage(out_packages, 4, BufferAccess::ReadWrite, DataType::U32),
        BufferDecl::storage(
            out_package_counts,
            5,
            BufferAccess::ReadWrite,
            DataType::U32,
        )
        .with_count(1),
        BufferDecl::storage(out_imports, 6, BufferAccess::ReadWrite, DataType::U32),
        BufferDecl::storage(out_import_counts, 7, BufferAccess::ReadWrite, DataType::U32)
            .with_count(1),
    ]);

    Program::wrapped(
        buffers,
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_packages_and_imports",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_packages_and_imports")
    .with_non_composable_with_self(true)
}
