use crate::parsing::go::parse::token_predicates::{
    emit_keyword_span_record_nodes, emit_span_record_nodes, token_is_ident, token_is_keyword,
    token_len, token_start, token_type_eq,
};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::go_token::{TOK_LBRACE, TOK_LPAREN, TOK_RBRACE, TOK_RPAREN, TOK_STRING};

/// Words per emitted Go declaration record.
pub const GO_DECL_RECORD_WORDS: u32 = 5;
/// Words per emitted Go span record.
pub const GO_SPAN_RECORD_WORDS: u32 = 2;
/// Function declaration kind.
pub const GO_DECL_FUNC: u32 = 1;
/// Method declaration kind.
pub const GO_DECL_METHOD: u32 = 2;
/// Interface declaration kind.
pub const GO_DECL_INTERFACE: u32 = 3;

/// The one brace-balanced body span plus declaration record Go declaration
/// extraction emits.
///
/// Function, method, and interface declarations differ only in where the body
/// scan starts, which kind word is recorded, and which token carries the name.
struct GoDeclSpan<'a> {
    tok_types: &'a str,
    tok_starts: &'a str,
    tok_lens: &'a str,
    out_decls: &'a str,
    out_decl_counts: &'a str,
    num_tokens: &'a Expr,
}

impl GoDeclSpan<'_> {
    /// Scans forward from `scan_from` for the outermost balanced `{`..`}` pair,
    /// then appends the five-word record for `name_tok`.
    fn nodes(&self, scan_from: Expr, kind: Expr, name_tok: Expr) -> Vec<Node> {
        vec![
            Node::let_bind("body_start", Expr::u32(0)),
            Node::let_bind("body_end", Expr::u32(0)),
            Node::let_bind("brace_depth", Expr::u32(0)),
            Node::let_bind("brace_done", Expr::u32(0)),
            Node::loop_for(
                "scan",
                scan_from,
                self.num_tokens.clone(),
                vec![Node::if_then(
                    Expr::eq(Expr::var("brace_done"), Expr::u32(0)),
                    vec![
                        Node::if_then(
                            token_type_eq(self.tok_types, Expr::var("scan"), TOK_LBRACE),
                            vec![
                                Node::if_then(
                                    Expr::eq(Expr::var("brace_depth"), Expr::u32(0)),
                                    vec![Node::assign(
                                        "body_start",
                                        token_start(self.tok_starts, Expr::var("scan")),
                                    )],
                                ),
                                Node::assign(
                                    "brace_depth",
                                    Expr::add(Expr::var("brace_depth"), Expr::u32(1)),
                                ),
                            ],
                        ),
                        Node::if_then(
                            token_type_eq(self.tok_types, Expr::var("scan"), TOK_RBRACE),
                            vec![
                                Node::assign(
                                    "brace_depth",
                                    Expr::sub(Expr::var("brace_depth"), Expr::u32(1)),
                                ),
                                Node::if_then(
                                    Expr::eq(Expr::var("brace_depth"), Expr::u32(0)),
                                    vec![
                                        Node::assign(
                                            "body_end",
                                            Expr::add(
                                                token_start(self.tok_starts, Expr::var("scan")),
                                                token_len(self.tok_lens, Expr::var("scan")),
                                            ),
                                        ),
                                        Node::assign("brace_done", Expr::u32(1)),
                                    ],
                                ),
                            ],
                        ),
                    ],
                )],
            ),
            Node::let_bind(
                "decl_idx",
                Expr::atomic_add(
                    self.out_decl_counts,
                    Expr::u32(0),
                    Expr::u32(GO_DECL_RECORD_WORDS),
                ),
            ),
            Node::store(self.out_decls, Expr::var("decl_idx"), kind),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(1)),
                token_start(self.tok_starts, name_tok.clone()),
            ),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(2)),
                token_len(self.tok_lens, name_tok),
            ),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(3)),
                Expr::var("body_start"),
            ),
            Node::store(
                self.out_decls,
                Expr::add(Expr::var("decl_idx"), Expr::u32(4)),
                Expr::var("body_end"),
            ),
        ]
    }
}

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
    let t = Expr::gid_x();
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

    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
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
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_packages_and_imports",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_packages_and_imports")
    .with_non_composable_with_self(true)
}

/// Extract function, method, and interface declarations.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_extract_declarations(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
    num_tokens: Expr,
    out_decls: &str,
    out_decl_counts: &str,
) -> Program {
    let t = Expr::gid_x();
    let decl_span = GoDeclSpan {
        tok_types,
        tok_starts,
        tok_lens,
        out_decls,
        out_decl_counts,
        num_tokens: &num_tokens,
    };
    let body = vec![
        Node::if_then(
            token_is_keyword(
                haystack,
                tok_types,
                tok_starts,
                tok_lens,
                t.clone(),
                b"func",
            ),
            vec![
                Node::let_bind("decl_kind", Expr::u32(GO_DECL_FUNC)),
                Node::let_bind("name_tok", Expr::add(t.clone(), Expr::u32(1))),
                Node::if_then(
                    token_type_eq(tok_types, Expr::var("name_tok"), TOK_LPAREN),
                    vec![
                        Node::assign("decl_kind", Expr::u32(GO_DECL_METHOD)),
                        Node::let_bind("recv_depth", Expr::u32(0)),
                        Node::let_bind("recv_done", Expr::u32(0)),
                        Node::let_bind("recv_end", Expr::var("name_tok")),
                        Node::loop_for(
                            "scan",
                            Expr::var("name_tok"),
                            num_tokens.clone(),
                            vec![Node::if_then(
                                Expr::eq(Expr::var("recv_done"), Expr::u32(0)),
                                vec![
                                    Node::if_then(
                                        token_type_eq(tok_types, Expr::var("scan"), TOK_LPAREN),
                                        vec![Node::assign(
                                            "recv_depth",
                                            Expr::add(Expr::var("recv_depth"), Expr::u32(1)),
                                        )],
                                    ),
                                    Node::if_then(
                                        token_type_eq(tok_types, Expr::var("scan"), TOK_RPAREN),
                                        vec![
                                            Node::assign(
                                                "recv_depth",
                                                Expr::sub(Expr::var("recv_depth"), Expr::u32(1)),
                                            ),
                                            Node::if_then(
                                                Expr::eq(Expr::var("recv_depth"), Expr::u32(0)),
                                                vec![
                                                    Node::assign("recv_done", Expr::u32(1)),
                                                    Node::assign("recv_end", Expr::var("scan")),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )],
                        ),
                        Node::assign("name_tok", Expr::add(Expr::var("recv_end"), Expr::u32(1))),
                    ],
                ),
                Node::if_then(
                    token_is_ident(tok_types, Expr::var("name_tok")),
                    decl_span.nodes(
                        Expr::add(Expr::var("name_tok"), Expr::u32(1)),
                        Expr::var("decl_kind"),
                        Expr::var("name_tok"),
                    ),
                ),
            ],
        ),
        Node::if_then(
            Expr::lt(Expr::add(t.clone(), Expr::u32(2)), num_tokens.clone()),
            vec![Node::if_then(
                Expr::and(
                    token_is_keyword(
                        haystack,
                        tok_types,
                        tok_starts,
                        tok_lens,
                        t.clone(),
                        b"type",
                    ),
                    Expr::and(
                        token_is_ident(tok_types, Expr::add(t.clone(), Expr::u32(1))),
                        token_is_keyword(
                            haystack,
                            tok_types,
                            tok_starts,
                            tok_lens,
                            Expr::add(t.clone(), Expr::u32(2)),
                            b"interface",
                        ),
                    ),
                ),
                decl_span.nodes(
                    Expr::add(t.clone(), Expr::u32(3)),
                    Expr::u32(GO_DECL_INTERFACE),
                    Expr::add(t.clone(), Expr::u32(1)),
                ),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage(out_decls, 4, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage(out_decl_counts, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_extract_declarations",
            vec![Node::if_then(Expr::lt(t, num_tokens), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_extract_declarations")
    .with_non_composable_with_self(true)
}
