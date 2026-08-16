use super::*;
use crate::parsing::c::lex::lexer::classify::stages::{
    block_comment_scan, block_comment_start, char_start, classify_prologue, float_start,
    identifier_scan, identifier_start, integer_start, line_comment_scan, line_comment_start,
    number_scan, preproc_start, quoted_literal_scan, string_start, token_start_expr, ClassifyCtx,
    ScanNames, TokenStartOpts,
};
use crate::parsing::c::lex::lexer::sections;

/// One per-invocation sparse lexer. Every entry point in this module is a
/// preset of these knobs; the classification stages themselves live in
/// `classify::stages` and are shared with the serial lexers.
pub(super) struct SparseLexerSpec<'a> {
    pub(super) haystack: &'a str,
    pub(super) out_tok_types: &'a str,
    pub(super) out_tok_starts: &'a str,
    pub(super) out_tok_lens: &'a str,
    pub(super) out_counts: &'a str,
    pub(super) haystack_len: u32,
    /// Drop the start and length columns from the output contract; the caller
    /// only needs token types.
    pub(super) suppress_span_readback: bool,
    /// Write a per-position emit flag into `out_counts` instead of a count.
    pub(super) emit_flags: bool,
    pub(super) layout: SparseHaystackLayout,
    /// Replay the line state before `t` so a mid-line `#` is not a directive.
    pub(super) track_preproc_lines: bool,
    /// Replay string, character, and comment state before `t` so a token start
    /// inside a literal is suppressed.
    pub(super) track_literals: bool,
    /// Emit per-workgroup emit-count totals into this buffer.
    pub(super) block_totals: Option<&'a str>,
}

/// Token-start rule for the sparse scanners: NUL-padded packed haystacks, an
/// ellipsis whose second dot continues the token, and the declared length as
/// the authoritative bound.
const SPARSE_TOKEN_START: TokenStartOpts = TokenStartOpts {
    dot_pair_is_tail: true,
    nul_is_space: true,
    bound_by_declared_len: true,
};

pub(super) fn c11_lexer_regular_sparse_impl(spec: &SparseLexerSpec<'_>) -> Program {
    let SparseLexerSpec {
        haystack,
        out_tok_types,
        out_tok_starts,
        out_tok_lens,
        out_counts,
        haystack_len,
        suppress_span_readback,
        emit_flags,
        layout,
        track_preproc_lines,
        track_literals,
        block_totals,
    } = *spec;

    let workgroup_lanes = if block_totals.is_some() {
        crate::reduce::multi_block_prefix_scan::BLOCK_LANES
    } else {
        256
    };
    let t = Expr::InvocationId { axis: 0 };
    let ctx = ClassifyCtx::sparse(haystack, haystack_len, layout, MAX_SPARSE_TOKEN_SCAN);
    let byte_at = |index: Expr| ctx.byte_at(index);

    let preliminary_start = Expr::and(
        Expr::lt(t.clone(), Expr::u32(haystack_len)),
        token_start_expr(&ctx, t.clone(), &SPARSE_TOKEN_START),
    );
    let mut string_state_prefix = vec![
        Node::let_bind("sparse_preliminary_start", preliminary_start),
        Node::let_bind("sparse_inside_string", Expr::u32(0)),
        Node::let_bind("sparse_inside_char", Expr::u32(0)),
        Node::let_bind("sparse_inside_line_comment", Expr::u32(0)),
        Node::let_bind("sparse_inside_block_comment", Expr::u32(0)),
        Node::let_bind("sparse_literal_escape", Expr::u32(0)),
    ];
    if track_literals {
        string_state_prefix.push(Node::if_then(
            Expr::var("sparse_preliminary_start"),
            vec![Node::loop_for(
                "sparse_literal_backscan",
                Expr::saturating_sub(t.clone(), Expr::u32(MAX_SPARSE_TOKEN_SCAN)),
                t.clone(),
                vec![
                    Node::let_bind(
                        "sparse_literal_scan_byte",
                        byte_at(Expr::var("sparse_literal_backscan")),
                    ),
                    Node::let_bind(
                        "sparse_literal_scan_prev",
                        Expr::select(
                            Expr::gt(Expr::var("sparse_literal_backscan"), Expr::u32(0)),
                            byte_at(Expr::saturating_sub(
                                Expr::var("sparse_literal_backscan"),
                                Expr::u32(1),
                            )),
                            Expr::u32(0),
                        ),
                    ),
                    Node::let_bind(
                        "sparse_literal_scan_next",
                        byte_at(Expr::add(
                            Expr::var("sparse_literal_backscan"),
                            Expr::u32(1),
                        )),
                    ),
                    Node::if_then_else(
                        Expr::eq(Expr::var("sparse_inside_line_comment"), Expr::u32(1)),
                        vec![Node::if_then(
                            Expr::or(
                                byte_eq(Expr::var("sparse_literal_scan_byte"), b'\n'),
                                byte_eq(Expr::var("sparse_literal_scan_byte"), b'\r'),
                            ),
                            vec![Node::assign("sparse_inside_line_comment", Expr::u32(0))],
                        )],
                        vec![Node::if_then_else(
                            Expr::eq(Expr::var("sparse_inside_block_comment"), Expr::u32(1)),
                            vec![Node::if_then(
                                Expr::and(
                                    byte_eq(Expr::var("sparse_literal_scan_prev"), b'*'),
                                    byte_eq(Expr::var("sparse_literal_scan_byte"), b'/'),
                                ),
                                vec![Node::assign("sparse_inside_block_comment", Expr::u32(0))],
                            )],
                            vec![Node::if_then_else(
                                Expr::eq(Expr::var("sparse_literal_escape"), Expr::u32(1)),
                                vec![Node::assign("sparse_literal_escape", Expr::u32(0))],
                                vec![Node::if_then_else(
                                    Expr::and(
                                        byte_eq(Expr::var("sparse_literal_scan_byte"), b'\\'),
                                        Expr::or(
                                            Expr::eq(
                                                Expr::var("sparse_inside_string"),
                                                Expr::u32(1),
                                            ),
                                            Expr::eq(Expr::var("sparse_inside_char"), Expr::u32(1)),
                                        ),
                                    ),
                                    vec![Node::assign("sparse_literal_escape", Expr::u32(1))],
                                    vec![
                                        Node::if_then(
                                            Expr::and(
                                                Expr::and(
                                                    byte_eq(
                                                        Expr::var("sparse_literal_scan_byte"),
                                                        b'/',
                                                    ),
                                                    byte_eq(
                                                        Expr::var("sparse_literal_scan_next"),
                                                        b'/',
                                                    ),
                                                ),
                                                Expr::and(
                                                    Expr::eq(
                                                        Expr::var("sparse_inside_string"),
                                                        Expr::u32(0),
                                                    ),
                                                    Expr::eq(
                                                        Expr::var("sparse_inside_char"),
                                                        Expr::u32(0),
                                                    ),
                                                ),
                                            ),
                                            vec![Node::assign(
                                                "sparse_inside_line_comment",
                                                Expr::u32(1),
                                            )],
                                        ),
                                        Node::if_then(
                                            Expr::and(
                                                Expr::and(
                                                    byte_eq(
                                                        Expr::var("sparse_literal_scan_byte"),
                                                        b'/',
                                                    ),
                                                    byte_eq(
                                                        Expr::var("sparse_literal_scan_next"),
                                                        b'*',
                                                    ),
                                                ),
                                                Expr::and(
                                                    Expr::eq(
                                                        Expr::var("sparse_inside_string"),
                                                        Expr::u32(0),
                                                    ),
                                                    Expr::eq(
                                                        Expr::var("sparse_inside_char"),
                                                        Expr::u32(0),
                                                    ),
                                                ),
                                            ),
                                            vec![Node::assign(
                                                "sparse_inside_block_comment",
                                                Expr::u32(1),
                                            )],
                                        ),
                                        Node::if_then(
                                            Expr::and(
                                                byte_eq(
                                                    Expr::var("sparse_literal_scan_byte"),
                                                    b'"',
                                                ),
                                                Expr::eq(
                                                    Expr::var("sparse_inside_char"),
                                                    Expr::u32(0),
                                                ),
                                            ),
                                            vec![Node::assign(
                                                "sparse_inside_string",
                                                Expr::select(
                                                    Expr::eq(
                                                        Expr::var("sparse_inside_string"),
                                                        Expr::u32(0),
                                                    ),
                                                    Expr::u32(1),
                                                    Expr::u32(0),
                                                ),
                                            )],
                                        ),
                                        Node::if_then(
                                            Expr::and(
                                                byte_eq(
                                                    Expr::var("sparse_literal_scan_byte"),
                                                    b'\'',
                                                ),
                                                Expr::eq(
                                                    Expr::var("sparse_inside_string"),
                                                    Expr::u32(0),
                                                ),
                                            ),
                                            vec![Node::assign(
                                                "sparse_inside_char",
                                                Expr::select(
                                                    Expr::eq(
                                                        Expr::var("sparse_inside_char"),
                                                        Expr::u32(0),
                                                    ),
                                                    Expr::u32(1),
                                                    Expr::u32(0),
                                                ),
                                            )],
                                        ),
                                    ],
                                )],
                            )],
                        )],
                    ),
                ],
            )],
        ));
    }
    if track_preproc_lines {
        string_state_prefix.extend([
            Node::let_bind("sparse_line_allows_directive", Expr::u32(1)),
            Node::let_bind("sparse_inside_preproc_line", Expr::u32(0)),
            Node::loop_for(
                "sparse_preproc_state_scan",
                Expr::u32(0),
                t.clone(),
                vec![
                    Node::let_bind(
                        "sparse_preproc_state_byte",
                        byte_at(Expr::var("sparse_preproc_state_scan")),
                    ),
                    Node::if_then_else(
                        Expr::or(
                            byte_eq(Expr::var("sparse_preproc_state_byte"), b'\n'),
                            byte_eq(Expr::var("sparse_preproc_state_byte"), b'\r'),
                        ),
                        vec![
                            Node::assign("sparse_inside_preproc_line", Expr::u32(0)),
                            Node::assign("sparse_line_allows_directive", Expr::u32(1)),
                        ],
                        vec![Node::if_then(
                            Expr::eq(Expr::var("sparse_inside_preproc_line"), Expr::u32(0)),
                            vec![Node::if_then(
                                Expr::eq(Expr::var("sparse_line_allows_directive"), Expr::u32(1)),
                                vec![Node::if_then_else(
                                    Expr::or(
                                        byte_eq(Expr::var("sparse_preproc_state_byte"), b' '),
                                        byte_eq(Expr::var("sparse_preproc_state_byte"), b'\t'),
                                    ),
                                    Vec::new(),
                                    vec![Node::if_then_else(
                                        byte_eq(Expr::var("sparse_preproc_state_byte"), b'#'),
                                        vec![Node::assign(
                                            "sparse_inside_preproc_line",
                                            Expr::u32(1),
                                        )],
                                        vec![Node::assign(
                                            "sparse_line_allows_directive",
                                            Expr::u32(0),
                                        )],
                                    )],
                                )],
                            )],
                        )],
                    ),
                ],
            ),
        ]);
    } else {
        string_state_prefix.extend([
            Node::let_bind("sparse_line_allows_directive", Expr::u32(0)),
            Node::let_bind("sparse_inside_preproc_line", Expr::u32(0)),
        ]);
    }

    let mut classify_at_pos = classify_prologue(&ctx, &t, true);
    classify_at_pos.push(identifier_start());
    if track_literals {
        classify_at_pos.push(string_start());
        classify_at_pos.push(char_start(true));
        classify_at_pos.push(line_comment_start());
        if !suppress_span_readback {
            classify_at_pos.push(line_comment_scan(
                &ctx,
                &ScanNames {
                    done: "sparse_comment_done",
                    scan: "sparse_scan_line_comment",
                },
            ));
        }
        classify_at_pos.push(block_comment_start());
        if !suppress_span_readback {
            classify_at_pos.push(block_comment_scan(
                &ctx,
                &ScanNames {
                    done: "sparse_block_comment_done",
                    scan: "sparse_scan_block_comment",
                },
            ));
            classify_at_pos.push(quoted_literal_scan(
                &ctx,
                TOK_STRING,
                &ScanNames {
                    done: "sparse_string_done",
                    scan: "sparse_scan_string",
                },
                "sparse_string_literal_escape",
                b'"',
                None,
            ));
            classify_at_pos.push(quoted_literal_scan(
                &ctx,
                TOK_CHAR,
                &ScanNames {
                    done: "sparse_char_done",
                    scan: "sparse_scan_char",
                },
                "sparse_char_literal_escape",
                b'\'',
                Some(TOK_ERR_UNTERMINATED_CHAR),
            ));
        }
    }
    if !suppress_span_readback {
        classify_at_pos.push(identifier_scan(
            &ctx,
            &ScanNames {
                done: "sparse_ident_done",
                scan: "sparse_scan_ident",
            },
            true,
        ));
    }
    classify_at_pos.push(integer_start());
    classify_at_pos.push(float_start());
    if !suppress_span_readback {
        classify_at_pos.push(number_scan(
            &ctx,
            &ScanNames {
                done: "sparse_number_done",
                scan: "sparse_scan_number",
            },
            "sparse_number_is_float",
        ));
    }
    classify_at_pos.push(preproc_start("sparse_line_allows_directive"));
    if !suppress_span_readback {
        classify_at_pos.push(preproc_row_scan(&ctx));
    }
    classify_at_pos.extend(sections::operator_punct_pushes());

    let mut emit_stores = vec![Node::store(out_tok_types, t.clone(), Expr::var("tok_type"))];
    if !suppress_span_readback {
        emit_stores.push(Node::store(out_tok_starts, t.clone(), Expr::var("pos")));
        emit_stores.push(Node::store(out_tok_lens, t.clone(), Expr::var("tok_len")));
    }
    if emit_flags {
        emit_stores.push(Node::store(
            out_counts,
            t.clone(),
            Expr::var("sparse_visible_emit"),
        ));
    }
    classify_at_pos.push(Node::let_bind(
        "sparse_visible_emit",
        Expr::select(
            Expr::and(
                Expr::and(
                    Expr::eq(Expr::var("emit"), Expr::u32(1)),
                    Expr::ne(Expr::var("tok_type"), Expr::u32(0)),
                ),
                Expr::ne(Expr::var("tok_type"), Expr::u32(TOK_COMMENT)),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
    classify_at_pos.push(Node::if_then(
        Expr::eq(Expr::var("sparse_visible_emit"), Expr::u32(1)),
        emit_stores,
    ));
    if block_totals.is_some() {
        classify_at_pos.push(Node::store(
            "__sparse_lexer_block_count",
            Expr::LocalId { axis: 0 },
            Expr::var("sparse_visible_emit"),
        ));
    }

    let mut out_tok_starts_decl =
        BufferDecl::storage(out_tok_starts, 2, BufferAccess::ReadWrite, DataType::U32).with_count(
            if suppress_span_readback {
                1
            } else {
                haystack_len
            },
        );
    let mut out_tok_lens_decl =
        BufferDecl::storage(out_tok_lens, 3, BufferAccess::ReadWrite, DataType::U32).with_count(
            if suppress_span_readback {
                1
            } else {
                haystack_len
            },
        );
    if suppress_span_readback {
        out_tok_starts_decl = out_tok_starts_decl.with_output_byte_range(0..0);
        out_tok_lens_decl = out_tok_lens_decl.with_output_byte_range(0..0);
    }
    let mut out_counts_decl =
        BufferDecl::storage(out_counts, 4, BufferAccess::ReadWrite, DataType::U32)
            .with_count(if emit_flags { haystack_len } else { 1 });
    if !emit_flags {
        out_counts_decl = out_counts_decl.with_output_byte_range(0..0);
    }
    let is_start = Expr::and(
        Expr::var("sparse_preliminary_start"),
        Expr::and(
            Expr::eq(Expr::var("sparse_inside_string"), Expr::u32(0)),
            Expr::and(
                Expr::eq(Expr::var("sparse_inside_char"), Expr::u32(0)),
                Expr::and(
                    Expr::eq(Expr::var("sparse_inside_line_comment"), Expr::u32(0)),
                    Expr::and(
                        Expr::eq(Expr::var("sparse_inside_block_comment"), Expr::u32(0)),
                        Expr::eq(Expr::var("sparse_inside_preproc_line"), Expr::u32(0)),
                    ),
                ),
            ),
        ),
    );
    let region_body = if let Some(block_totals) = block_totals {
        let lane = Expr::var("lane");
        let block = Expr::var("block");
        let scratch_a = "__sparse_lexer_block_count";
        let scratch_b = "__sparse_lexer_block_count_reduce";
        let mut body = string_state_prefix.clone();
        body.push(Node::store(out_tok_types, t.clone(), Expr::u32(0)));
        if emit_flags {
            body.push(Node::store(out_counts, t.clone(), Expr::u32(0)));
        }
        body.extend([
            Node::let_bind("lane", Expr::LocalId { axis: 0 }),
            Node::let_bind("block", Expr::WorkgroupId { axis: 0 }),
            Node::store(scratch_a, lane.clone(), Expr::u32(0)),
            Node::if_then(is_start, classify_at_pos),
            Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
            },
        ]);
        let mut stride = 1_u32;
        while stride < workgroup_lanes {
            body.push(Node::store(
                scratch_b,
                lane.clone(),
                Expr::load(scratch_a, lane.clone()),
            ));
            let previous_lane = Expr::add(lane.clone(), Expr::u32(0u32.wrapping_sub(stride)));
            body.push(Node::if_then(
                Expr::lt(Expr::u32(stride.saturating_sub(1)), lane.clone()),
                vec![Node::store(
                    scratch_b,
                    lane.clone(),
                    Expr::add(
                        Expr::load(scratch_a, lane.clone()),
                        Expr::load(scratch_a, previous_lane),
                    ),
                )],
            ));
            body.push(Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
            });
            body.push(Node::store(
                scratch_a,
                lane.clone(),
                Expr::load(scratch_b, lane.clone()),
            ));
            body.push(Node::Barrier {
                ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
            });
            stride *= 2;
        }
        body.push(Node::if_then(
            Expr::eq(lane, Expr::u32(workgroup_lanes - 1)),
            vec![Node::store(
                block_totals,
                block,
                Expr::load(scratch_a, Expr::u32(workgroup_lanes - 1)),
            )],
        ));
        body
    } else {
        let mut body = string_state_prefix;
        body.push(Node::store(out_tok_types, t.clone(), Expr::u32(0)));
        if emit_flags {
            body.push(Node::store(out_counts, t.clone(), Expr::u32(0)));
        }
        body.push(Node::if_then(is_start, classify_at_pos));
        body
    };

    let sparse_types_decl = if block_totals.is_some() {
        BufferDecl::output(out_tok_types, 1, DataType::U32).with_count(haystack_len)
    } else {
        BufferDecl::storage(out_tok_types, 1, BufferAccess::ReadWrite, DataType::U32)
            .with_count(haystack_len)
    };
    if block_totals.is_some() {
        out_tok_starts_decl = BufferDecl::output(out_tok_starts, 2, DataType::U32).with_count(
            if suppress_span_readback {
                1
            } else {
                haystack_len
            },
        );
        out_tok_lens_decl = BufferDecl::output(out_tok_lens, 3, DataType::U32).with_count(
            if suppress_span_readback {
                1
            } else {
                haystack_len
            },
        );
    }
    let haystack_element = match layout {
        SparseHaystackLayout::RawU8 => DataType::U8,
        SparseHaystackLayout::Contiguous
        | SparseHaystackLayout::PackedU32
        | SparseHaystackLayout::ExpandedU32 => DataType::U32,
    };
    let haystack_count = match layout {
        SparseHaystackLayout::PackedU32 => haystack_len.max(1).div_ceil(4).max(1),
        SparseHaystackLayout::Contiguous | SparseHaystackLayout::ExpandedU32 => haystack_len.max(1),
        SparseHaystackLayout::RawU8 => 0,
    };
    let mut buffers = vec![
        BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, haystack_element)
            .with_count(haystack_count),
        sparse_types_decl,
        out_tok_starts_decl,
        out_tok_lens_decl,
    ];
    if block_totals.is_none() || emit_flags {
        buffers.push(out_counts_decl);
    }
    if let Some(block_totals) = block_totals {
        buffers.push(
            BufferDecl::output(block_totals, if emit_flags { 5 } else { 4 }, DataType::U32)
                .with_count(haystack_len.div_ceil(workgroup_lanes).max(1)),
        );
        buffers.push(BufferDecl::workgroup(
            "__sparse_lexer_block_count",
            workgroup_lanes,
            DataType::U32,
        ));
        buffers.push(BufferDecl::workgroup(
            "__sparse_lexer_block_count_reduce",
            workgroup_lanes,
            DataType::U32,
        ));
    }
    Program::wrapped(
        buffers,
        [workgroup_lanes, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::c_lexer_regular_sparse",
            region_body,
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::c_lexer_regular_sparse")
    .with_non_composable_with_self(true)
}

/// Extend a directive row to the next physical line end. The sparse scanners
/// run after preprocessing, where line splices are already resolved, so this
/// stops at the first newline rather than following a backslash continuation.
fn preproc_row_scan(ctx: &ClassifyCtx<'_>) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(1));
    Node::if_then(
        Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_PREPROC)),
        vec![
            Node::let_bind("sparse_preproc_done", Expr::u32(0)),
            Node::loop_for(
                "sparse_scan_preproc",
                start.clone(),
                ctx.scan_bound(start, MAX_SPARSE_TOKEN_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var("sparse_preproc_done"), Expr::u32(0)),
                    vec![
                        Node::let_bind(
                            "sparse_preproc_scan_byte",
                            ctx.byte_at(Expr::var("sparse_scan_preproc")),
                        ),
                        Node::if_then_else(
                            Expr::or(
                                byte_eq(Expr::var("sparse_preproc_scan_byte"), b'\n'),
                                byte_eq(Expr::var("sparse_preproc_scan_byte"), b'\r'),
                            ),
                            vec![Node::assign("sparse_preproc_done", Expr::u32(1))],
                            vec![Node::assign(
                                "tok_len",
                                Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                            )],
                        ),
                    ],
                )],
            ),
        ],
    )
}
