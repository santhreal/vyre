//! The single C11 token-classification walk.
//!
//! Every C11 lexer builder in this crate  -  the dense serial `c11_lexer`, the
//! reduced serial `c11_lexer_regular`, the ranked and sparse per-invocation
//! variants, and the packed / expanded / raw-u8 sparse family  -  composes the
//! stages below. None of them owns a private copy of a classifier.
//!
//! A stage is parameterized on three axes:
//!  - [`SparseHaystackLayout`]: how one haystack byte is addressed. The dense
//!    serial lexer is the [`SparseHaystackLayout::Contiguous`] case, where the
//!    caller's loop bound or an explicit guard already establishes
//!    in-boundsness so the load carries no per-access check.
//!  - the scan-bound policy on [`ClassifyCtx`]: per-stage caps against
//!    `buf_len(haystack)`, or one uniform cap against the declared length.
//!  - the IR variable names a stage binds, passed as [`ScanNames`], because the
//!    serial and sparse walks bind distinct names in the same scope tree.
//!
//! Composition order is the caller's: `set_token` fires only while `emit == 0`,
//! so the sequence in which a builder pushes these stages is that builder's
//! precedence rule and is deliberately not centralized here.

use super::*;

/// Emit `token` of length `len` at the current position when `condition`
/// holds and no earlier stage has already claimed the position.
///
/// The `emit == 0` guard is what makes stage order the precedence rule.
pub(crate) fn set_token(condition: Expr, token: u32, len: Expr) -> Node {
    Node::if_then(
        Expr::and(Expr::eq(Expr::var("emit"), Expr::u32(0)), condition),
        vec![
            Node::assign("emit", Expr::u32(1)),
            Node::assign("tok_type", Expr::u32(token)),
            Node::assign("tok_len", len),
        ],
    )
}

/// How a haystack byte is addressed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SparseHaystackLayout {
    /// One byte per `u32` word, loaded without a bounds check. The serial
    /// lexers use this: their scan loops are already bounded by
    /// `buf_len(haystack)` and their lookaheads carry an explicit guard.
    Contiguous,
    /// Four bytes per `u32` word, little-endian within the word.
    PackedU32,
    /// One byte per `u32` word, zero outside the declared length.
    ExpandedU32,
    /// Native `u8` elements with a runtime-sized buffer.
    RawU8,
}

/// Byte source plus scan-bound policy for one classification walk.
pub(crate) struct ClassifyCtx<'a> {
    haystack: &'a str,
    haystack_len: u32,
    layout: SparseHaystackLayout,
    /// `Some(cap)` bounds every scan loop at `min(pos + cap, haystack_len)`;
    /// `None` bounds each loop at `min(start + <stage cap>, buf_len(haystack))`.
    uniform_scan_cap: Option<u32>,
}

/// The two loop-scoped IR variables a scan stage binds: its completion flag and
/// its induction variable.
pub(crate) struct ScanNames<'a> {
    pub(crate) done: &'a str,
    pub(crate) scan: &'a str,
}

/// Which byte sequences a per-invocation scanner treats as a token start.
pub(crate) struct TokenStartOpts {
    /// The second `.` of an ellipsis continues the preceding token.
    pub(crate) dot_pair_is_tail: bool,
    /// A zero byte counts as whitespace, as in a NUL-padded packed haystack.
    pub(crate) nul_is_space: bool,
    /// Bound token starts by the declared length rather than `buf_len`.
    pub(crate) bound_by_declared_len: bool,
}

impl<'a> ClassifyCtx<'a> {
    /// Serial walk over a contiguous one-byte-per-word haystack.
    pub(crate) fn contiguous(haystack: &'a str, haystack_len: u32) -> Self {
        Self {
            haystack,
            haystack_len,
            layout: SparseHaystackLayout::Contiguous,
            uniform_scan_cap: None,
        }
    }

    /// Per-invocation walk over a one-byte-per-word haystack, bounds-checked
    /// against the declared length on every access.
    pub(crate) fn expanded(haystack: &'a str, haystack_len: u32) -> Self {
        Self {
            haystack,
            haystack_len,
            layout: SparseHaystackLayout::ExpandedU32,
            uniform_scan_cap: None,
        }
    }

    /// Per-invocation walk with one uniform scan cap across every stage.
    pub(crate) fn sparse(
        haystack: &'a str,
        haystack_len: u32,
        layout: SparseHaystackLayout,
        scan_cap: u32,
    ) -> Self {
        Self {
            haystack,
            haystack_len,
            layout,
            uniform_scan_cap: Some(scan_cap),
        }
    }

    pub(crate) fn haystack(&self) -> &'a str {
        self.haystack
    }

    pub(crate) fn haystack_len(&self) -> u32 {
        self.haystack_len
    }

    /// Load the byte at `index` under this walk's layout.
    pub(crate) fn byte_at(&self, index: Expr) -> Expr {
        match self.layout {
            SparseHaystackLayout::Contiguous => byte_load(self.haystack, index),
            SparseHaystackLayout::PackedU32 => {
                let word = Expr::load(self.haystack, Expr::shr(index.clone(), Expr::u32(2)));
                let shift = Expr::shl(Expr::bitand(index.clone(), Expr::u32(3)), Expr::u32(3));
                Expr::select(
                    Expr::lt(index, Expr::u32(self.haystack_len)),
                    Expr::bitand(Expr::shr(word, shift), Expr::u32(0xFF)),
                    Expr::u32(0),
                )
            }
            SparseHaystackLayout::ExpandedU32 => {
                byte_at_or_zero(self.haystack, index, self.haystack_len)
            }
            SparseHaystackLayout::RawU8 => {
                let runtime_len =
                    Expr::min(Expr::buf_len(self.haystack), Expr::u32(self.haystack_len));
                let in_bounds = Expr::lt(index.clone(), runtime_len.clone());
                let safe_index = Expr::select(
                    in_bounds.clone(),
                    index,
                    Expr::saturating_sub(runtime_len, Expr::u32(1)),
                );
                let byte = Expr::bitand(
                    Expr::cast(DataType::U32, Expr::load(self.haystack, safe_index)),
                    Expr::u32(0xFF),
                );
                Expr::select(in_bounds, byte, Expr::u32(0))
            }
        }
    }

    /// Load the byte `offset` positions past `base`. The contiguous layout has
    /// no per-access check, so the lookahead carries its own `buf_len` guard.
    pub(crate) fn lookahead(&self, base: &Expr, offset: u32) -> Expr {
        let index = Expr::add(base.clone(), Expr::u32(offset));
        match self.layout {
            SparseHaystackLayout::Contiguous => Expr::select(
                Expr::lt(index.clone(), Expr::buf_len(self.haystack)),
                self.byte_at(index),
                Expr::u32(0),
            ),
            _ => self.byte_at(index),
        }
    }

    /// Upper bound for a scan loop starting at `start`, where `stage_cap` is
    /// the per-stage cap used when this walk has no uniform cap.
    pub(crate) fn scan_bound(&self, start: Expr, stage_cap: u32) -> Expr {
        match self.uniform_scan_cap {
            Some(cap) => Expr::min(
                Expr::add(Expr::var("pos"), Expr::u32(cap)),
                Expr::u32(self.haystack_len),
            ),
            None => scan_upper_bound_with_cap(self.haystack, start, stage_cap),
        }
    }
}

/// Workgroup size shared by every C11 lexer entry point.
pub(crate) const LEXER_WORKGROUP_SIZE: u32 = 256;

/// Buffer table shared by every lexer entry point that writes three token
/// columns plus a single count word.
pub(crate) fn token_column_buffers(
    haystack: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32)
            .with_count(haystack_len.max(1)),
        BufferDecl::storage(out_tok_types, 1, BufferAccess::ReadWrite, DataType::U32)
            .with_count(haystack_len.max(1)),
        BufferDecl::storage(out_tok_starts, 2, BufferAccess::ReadWrite, DataType::U32)
            .with_count(haystack_len.max(1)),
        BufferDecl::storage(out_tok_lens, 3, BufferAccess::ReadWrite, DataType::U32)
            .with_count(haystack_len.max(1)),
        BufferDecl::storage(out_counts, 4, BufferAccess::ReadWrite, DataType::U32).with_count(1),
    ]
}

/// The eight bindings every classifier opens with: the byte window around
/// `pos` and the mutable `emit` / `tok_type` / `tok_len` accumulators.
///
/// `bind_pos` binds `pos` from `pos_expr`; serial callers bind it themselves in
/// the enclosing cursor loop and pass `false`.
pub(crate) fn classify_prologue(ctx: &ClassifyCtx<'_>, pos: &Expr, bind_pos: bool) -> Vec<Node> {
    let mut nodes = Vec::with_capacity(9);
    if bind_pos {
        nodes.push(Node::let_bind("pos", pos.clone()));
    }
    nodes.push(Node::let_bind("byte", ctx.byte_at(pos.clone())));
    nodes.push(Node::let_bind(
        "prev_byte",
        Expr::select(
            Expr::gt(pos.clone(), Expr::u32(0)),
            ctx.byte_at(Expr::saturating_sub(pos.clone(), Expr::u32(1))),
            Expr::u32(0),
        ),
    ));
    nodes.push(Node::let_bind("next_byte", ctx.lookahead(pos, 1)));
    nodes.push(Node::let_bind("next2_byte", ctx.lookahead(pos, 2)));
    nodes.push(Node::let_bind("emit", Expr::u32(0)));
    nodes.push(Node::let_bind("tok_type", Expr::u32(TOK_WHITESPACE)));
    nodes.push(Node::let_bind("tok_len", Expr::u32(1)));
    nodes
}

/// An identifier opens where an ident-start byte does not continue a previous
/// identifier.
pub(crate) fn identifier_start() -> Node {
    set_token(
        Expr::and(
            is_ident_start(Expr::var("byte")),
            Expr::not(is_ident_continue(Expr::var("prev_byte"))),
        ),
        TOK_IDENTIFIER,
        Expr::u32(1),
    )
}

/// Extend an identifier over its continuation bytes. `allow_quote` additionally
/// absorbs `'`, which the sparse scanners use to keep a digit separator inside
/// the identifier run.
pub(crate) fn identifier_scan(
    ctx: &ClassifyCtx<'_>,
    names: &ScanNames<'_>,
    allow_quote: bool,
) -> Node {
    let continues = if allow_quote {
        Expr::or(
            is_ident_continue(Expr::var("scan_byte")),
            byte_eq(Expr::var("scan_byte"), b'\''),
        )
    } else {
        is_ident_continue(Expr::var("scan_byte"))
    };
    let start = Expr::add(Expr::var("pos"), Expr::u32(1));
    Node::if_then(
        Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_IDENTIFIER)),
        vec![
            Node::let_bind(names.done, Expr::u32(0)),
            Node::loop_for(
                names.scan,
                start.clone(),
                ctx.scan_bound(start, MAX_IDENT_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var(names.done), Expr::u32(0)),
                    vec![
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var(names.scan))),
                        Node::if_then_else(
                            continues,
                            vec![Node::assign(
                                "tok_len",
                                Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                            )],
                            vec![Node::assign(names.done, Expr::u32(1))],
                        ),
                    ],
                )],
            ),
        ],
    )
}

/// An integer opens on a digit that does not continue a previous identifier.
pub(crate) fn integer_start() -> Node {
    set_token(
        Expr::and(
            is_digit(Expr::var("byte")),
            Expr::not(is_ident_continue(Expr::var("prev_byte"))),
        ),
        TOK_INTEGER,
        Expr::u32(1),
    )
}

/// A float opens on `.` immediately followed by a digit.
pub(crate) fn float_start() -> Node {
    set_token(
        Expr::and(
            byte_eq(Expr::var("byte"), b'.'),
            is_digit(Expr::var("next_byte")),
        ),
        TOK_FLOAT,
        Expr::u32(1),
    )
}

/// Extend an integer over a plain digit run. Used by the reduced lexers, which
/// have no float, exponent, or suffix grammar.
pub(crate) fn digit_run_scan(ctx: &ClassifyCtx<'_>, names: &ScanNames<'_>) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(1));
    Node::if_then(
        Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_INTEGER)),
        vec![
            Node::let_bind(names.done, Expr::u32(0)),
            Node::loop_for(
                names.scan,
                start.clone(),
                ctx.scan_bound(start, MAX_NUMBER_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var(names.done), Expr::u32(0)),
                    vec![
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var(names.scan))),
                        Node::if_then_else(
                            is_digit(Expr::var("scan_byte")),
                            vec![Node::assign(
                                "tok_len",
                                Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                            )],
                            vec![Node::assign(names.done, Expr::u32(1))],
                        ),
                    ],
                )],
            ),
        ],
    )
}

/// Extend an integer or float over the full C numeric-literal tail: suffix
/// bytes, a fractional dot, and a signed `e`/`p` exponent. Promotes the token
/// to `TOK_FLOAT` when a dot or exponent was consumed. `is_float` names the
/// walk-local flag that carries that promotion.
pub(crate) fn number_scan(ctx: &ClassifyCtx<'_>, names: &ScanNames<'_>, is_float: &str) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(1));
    Node::if_then(
        Expr::or(
            Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_INTEGER)),
            Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_FLOAT)),
        ),
        vec![
            Node::let_bind(names.done, Expr::u32(0)),
            Node::let_bind(
                is_float,
                Expr::select(
                    Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_FLOAT)),
                    Expr::u32(1),
                    Expr::u32(0),
                ),
            ),
            Node::loop_for(
                names.scan,
                start.clone(),
                ctx.scan_bound(start, MAX_NUMBER_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var(names.done), Expr::u32(0)),
                    vec![
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var(names.scan))),
                        Node::let_bind(
                            "scan_prev",
                            ctx.byte_at(Expr::saturating_sub(
                                Expr::var(names.scan),
                                Expr::u32(1),
                            )),
                        ),
                        Node::let_bind("scan_next", ctx.lookahead(&Expr::var(names.scan), 1)),
                        Node::let_bind(
                            "scan_can_start_exponent",
                            Expr::and(
                                Expr::or(
                                    byte_eq(Expr::var("scan_byte"), b'e'),
                                    Expr::or(
                                        byte_eq(Expr::var("scan_byte"), b'E'),
                                        Expr::or(
                                            byte_eq(Expr::var("scan_byte"), b'p'),
                                            byte_eq(Expr::var("scan_byte"), b'P'),
                                        ),
                                    ),
                                ),
                                Expr::or(
                                    is_digit(Expr::var("scan_next")),
                                    Expr::or(
                                        byte_eq(Expr::var("scan_next"), b'+'),
                                        byte_eq(Expr::var("scan_next"), b'-'),
                                    ),
                                ),
                            ),
                        ),
                        Node::let_bind(
                            "scan_is_exponent_sign",
                            Expr::and(
                                Expr::or(
                                    byte_eq(Expr::var("scan_byte"), b'+'),
                                    byte_eq(Expr::var("scan_byte"), b'-'),
                                ),
                                Expr::or(
                                    byte_eq(Expr::var("scan_prev"), b'e'),
                                    Expr::or(
                                        byte_eq(Expr::var("scan_prev"), b'E'),
                                        Expr::or(
                                            byte_eq(Expr::var("scan_prev"), b'p'),
                                            byte_eq(Expr::var("scan_prev"), b'P'),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        Node::let_bind("scan_is_float_dot", byte_eq(Expr::var("scan_byte"), b'.')),
                        Node::let_bind(
                            "scan_is_number_tail",
                            Expr::or(
                                is_ident_continue(Expr::var("scan_byte")),
                                Expr::or(
                                    Expr::var("scan_is_float_dot"),
                                    Expr::var("scan_is_exponent_sign"),
                                ),
                            ),
                        ),
                        Node::if_then_else(
                            Expr::var("scan_is_number_tail"),
                            vec![
                                Node::assign(
                                    "tok_len",
                                    Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                                ),
                                Node::if_then(
                                    Expr::or(
                                        Expr::var("scan_is_float_dot"),
                                        Expr::var("scan_can_start_exponent"),
                                    ),
                                    vec![Node::assign(is_float, Expr::u32(1))],
                                ),
                            ],
                            vec![Node::assign(names.done, Expr::u32(1))],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var(is_float), Expr::u32(1)),
                vec![Node::assign("tok_type", Expr::u32(TOK_FLOAT))],
            ),
        ],
    )
}

/// A line comment opens on `//`.
pub(crate) fn line_comment_start() -> Node {
    set_token(
        Expr::and(
            byte_eq(Expr::var("byte"), b'/'),
            byte_eq(Expr::var("next_byte"), b'/'),
        ),
        TOK_COMMENT,
        Expr::u32(2),
    )
}

/// Extend a line comment to the next newline.
pub(crate) fn line_comment_scan(ctx: &ClassifyCtx<'_>, names: &ScanNames<'_>) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(2));
    Node::if_then(
        Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_COMMENT)),
        vec![
            Node::let_bind(names.done, Expr::u32(0)),
            Node::loop_for(
                names.scan,
                start.clone(),
                ctx.scan_bound(start, MAX_COMMENT_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var(names.done), Expr::u32(0)),
                    vec![
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var(names.scan))),
                        Node::if_then_else(
                            byte_eq(Expr::var("scan_byte"), b'\n'),
                            vec![Node::assign(names.done, Expr::u32(1))],
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

/// A block comment opens on `/*`.
pub(crate) fn block_comment_start() -> Node {
    set_token(
        Expr::and(
            byte_eq(Expr::var("byte"), b'/'),
            byte_eq(Expr::var("next_byte"), b'*'),
        ),
        TOK_COMMENT,
        Expr::u32(2),
    )
}

/// Extend a block comment to its `*/`, or mark it unterminated.
pub(crate) fn block_comment_scan(ctx: &ClassifyCtx<'_>, names: &ScanNames<'_>) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(2));
    Node::if_then(
        Expr::and(
            Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_COMMENT)),
            byte_eq(Expr::var("next_byte"), b'*'),
        ),
        vec![
            Node::let_bind(names.done, Expr::u32(0)),
            Node::loop_for(
                names.scan,
                start.clone(),
                ctx.scan_bound(start, MAX_BLOCK_COMMENT_SCAN),
                vec![Node::if_then(
                    Expr::eq(Expr::var(names.done), Expr::u32(0)),
                    vec![
                        Node::assign("tok_len", Expr::add(Expr::var("tok_len"), Expr::u32(1))),
                        Node::let_bind("scan_byte", ctx.byte_at(Expr::var(names.scan))),
                        Node::let_bind("scan_next", ctx.lookahead(&Expr::var(names.scan), 1)),
                        Node::if_then(
                            Expr::and(
                                byte_eq(Expr::var("scan_byte"), b'*'),
                                byte_eq(Expr::var("scan_next"), b'/'),
                            ),
                            vec![
                                Node::assign(
                                    "tok_len",
                                    Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                                ),
                                Node::assign(names.done, Expr::u32(1)),
                            ],
                        ),
                    ],
                )],
            ),
            Node::if_then(
                Expr::eq(Expr::var(names.done), Expr::u32(0)),
                vec![Node::assign(
                    "tok_type",
                    Expr::u32(TOK_ERR_UNTERMINATED_COMMENT),
                )],
            ),
        ],
    )
}

/// A string literal opens on `"`.
pub(crate) fn string_start() -> Node {
    set_token(byte_eq(Expr::var("byte"), b'"'), TOK_STRING, Expr::u32(1))
}

/// A character literal opens on `'`. `require_ident_boundary` additionally
/// rejects a `'` that continues an identifier, which is how the sparse
/// scanners keep a digit separator out of the literal grammar.
pub(crate) fn char_start(require_ident_boundary: bool) -> Node {
    let condition = if require_ident_boundary {
        Expr::and(
            byte_eq(Expr::var("byte"), b'\''),
            Expr::not(is_ident_continue(Expr::var("prev_byte"))),
        )
    } else {
        byte_eq(Expr::var("byte"), b'\'')
    };
    set_token(condition, TOK_CHAR, Expr::u32(1))
}

/// A preprocessor row opens on `#` where `directive_var` records that the line
/// so far permits a directive.
pub(crate) fn preproc_start(directive_var: &str) -> Node {
    set_token(
        Expr::and(
            byte_eq(Expr::var("byte"), b'#'),
            Expr::eq(Expr::var(directive_var), Expr::u32(1)),
        ),
        TOK_PREPROC,
        Expr::u32(1),
    )
}

/// Extend a quoted literal of `tok_type` to its closing `quote`, honouring
/// backslash escapes through the `escape` flag. `unterminated` names the token
/// type to fall back to when the closing quote is never reached; `None` leaves
/// the token type alone.
pub(crate) fn quoted_literal_scan(
    ctx: &ClassifyCtx<'_>,
    tok_type: u32,
    names: &ScanNames<'_>,
    escape: &str,
    quote: u8,
    unterminated: Option<u32>,
) -> Node {
    let start = Expr::add(Expr::var("pos"), Expr::u32(1));
    let mut body = vec![
        Node::let_bind(names.done, Expr::u32(0)),
        Node::let_bind(escape, Expr::u32(0)),
        Node::loop_for(
            names.scan,
            start.clone(),
            ctx.scan_bound(start, MAX_LITERAL_SCAN),
            vec![Node::if_then(
                Expr::eq(Expr::var(names.done), Expr::u32(0)),
                vec![
                    Node::let_bind("scan_byte", ctx.byte_at(Expr::var(names.scan))),
                    Node::assign("tok_len", Expr::add(Expr::var("tok_len"), Expr::u32(1))),
                    Node::if_then_else(
                        Expr::eq(Expr::var(escape), Expr::u32(1)),
                        vec![Node::assign(escape, Expr::u32(0))],
                        vec![Node::if_then_else(
                            byte_eq(Expr::var("scan_byte"), b'\\'),
                            vec![Node::assign(escape, Expr::u32(1))],
                            vec![Node::if_then(
                                byte_eq(Expr::var("scan_byte"), quote),
                                vec![Node::assign(names.done, Expr::u32(1))],
                            )],
                        )],
                    ),
                ],
            )],
        ),
    ];
    if let Some(unterminated) = unterminated {
        body.push(Node::if_then(
            Expr::eq(Expr::var(names.done), Expr::u32(0)),
            vec![Node::assign("tok_type", Expr::u32(unterminated))],
        ));
    }
    Node::if_then(
        Expr::eq(Expr::var("tok_type"), Expr::u32(tok_type)),
        body,
    )
}

/// Whitespace under a per-invocation scanner.
fn space_expr(value: Expr, nul_is_space: bool) -> Expr {
    if nul_is_space {
        Expr::or(
            byte_eq(value.clone(), 0),
            crate::parsing::core::ascii_whitespace_expr(value),
        )
    } else {
        crate::parsing::core::ascii_whitespace_expr(value)
    }
}

/// True where `index` lands on the second or third byte of a multi-byte
/// operator, so a per-invocation scanner must not open a token there.
fn operator_tail_expr(ctx: &ClassifyCtx<'_>, index: Expr, dot_pair_is_tail: bool) -> Expr {
    let b = ctx.byte_at(index.clone());
    let prev = Expr::select(
        Expr::gt(index.clone(), Expr::u32(0)),
        ctx.byte_at(Expr::saturating_sub(index.clone(), Expr::u32(1))),
        Expr::u32(0),
    );
    let prev2 = Expr::select(
        Expr::gt(index.clone(), Expr::u32(1)),
        ctx.byte_at(Expr::saturating_sub(index, Expr::u32(2))),
        Expr::u32(0),
    );

    // `<<=` / `>>=`: the `=` trails a doubled shift.
    let mut doubled_chain = Expr::and(
        byte_eq(b.clone(), b'='),
        Expr::or(
            Expr::and(
                byte_eq(prev.clone(), b'<'),
                byte_eq(prev2.clone(), b'<'),
            ),
            Expr::and(byte_eq(prev.clone(), b'>'), byte_eq(prev2, b'>')),
        ),
    );
    let mut doubled: Vec<u8> = vec![b'+', b'-', b'&', b'|', b'<', b'>'];
    if dot_pair_is_tail {
        doubled.insert(0, b'.');
    }
    for byte in doubled.iter().rev() {
        doubled_chain = Expr::or(
            Expr::and(byte_eq(b.clone(), *byte), byte_eq(prev.clone(), *byte)),
            doubled_chain,
        );
    }

    // `op=`: the `=` trails a compound-assignment or comparison operator.
    let compound_prevs = [
        b'+', b'-', b'*', b'/', b'%', b'&', b'|', b'^', b'=', b'!', b'<', b'>',
    ];
    let mut compound_prev = byte_eq(prev.clone(), b'>');
    for byte in compound_prevs.iter().rev().skip(1) {
        compound_prev = Expr::or(byte_eq(prev.clone(), *byte), compound_prev);
    }
    let compound = Expr::and(byte_eq(b.clone(), b'='), compound_prev);

    Expr::or(
        Expr::and(byte_eq(b, b'>'), byte_eq(prev, b'-')),
        Expr::or(compound, doubled_chain),
    )
}

/// True where a per-invocation scanner should open a token at `index`.
pub(crate) fn token_start_expr(
    ctx: &ClassifyCtx<'_>,
    index: Expr,
    opts: &TokenStartOpts,
) -> Expr {
    let b = ctx.byte_at(index.clone());
    let prev = Expr::select(
        Expr::gt(index.clone(), Expr::u32(0)),
        ctx.byte_at(Expr::saturating_sub(index.clone(), Expr::u32(1))),
        Expr::u32(0),
    );
    let bound = if opts.bound_by_declared_len {
        Expr::u32(ctx.haystack_len())
    } else {
        Expr::buf_len(ctx.haystack())
    };
    Expr::and(
        Expr::lt(index.clone(), bound),
        Expr::and(
            Expr::not(space_expr(b.clone(), opts.nul_is_space)),
            Expr::and(
                Expr::not(Expr::and(is_ident_continue(b), is_ident_continue(prev))),
                Expr::not(operator_tail_expr(ctx, index, opts.dot_pair_is_tail)),
            ),
        ),
    )
}

/// The serial lexer shell: a single invocation walks `cursor` over the haystack
/// and runs `classify_at_pos` as a child phase at every token start.
pub(crate) struct SerialLexer<'a> {
    pub(crate) op_id: &'a str,
    pub(crate) haystack: &'a str,
    pub(crate) out_tok_types: &'a str,
    pub(crate) out_tok_starts: &'a str,
    pub(crate) out_tok_lens: &'a str,
    pub(crate) out_counts: &'a str,
    pub(crate) haystack_len: u32,
}

impl SerialLexer<'_> {
    pub(crate) fn build(&self, classify_at_pos: Vec<Node>) -> Program {
        let t = Expr::InvocationId { axis: 0 };
        Program::wrapped(
            token_column_buffers(
                self.haystack,
                self.out_tok_types,
                self.out_tok_starts,
                self.out_tok_lens,
                self.out_counts,
                self.haystack_len,
            ),
            [LEXER_WORKGROUP_SIZE, 1, 1],
            {
                let entry_body = vec![Node::if_then(
                    Expr::eq(t, Expr::u32(0)),
                    vec![
                        Node::let_bind("cursor", Expr::u32(0)),
                        Node::let_bind("line_allows_directive", Expr::u32(1)),
                        Node::let_bind("tok_idx", Expr::u32(0)),
                        Node::loop_for(
                            "token_iter",
                            Expr::u32(0),
                            Expr::buf_len(self.haystack),
                            vec![Node::if_then(
                                Expr::lt(Expr::var("cursor"), Expr::buf_len(self.haystack)),
                                {
                                    let mut body =
                                        vec![Node::let_bind("pos", Expr::var("cursor"))];
                                    body.push(child_phase(
                                        self.op_id,
                                        &format!("{}::classify_at_pos", self.op_id),
                                        classify_at_pos,
                                    ));
                                    body
                                },
                            )],
                        ),
                        Node::store(self.out_counts, Expr::u32(0), Expr::var("tok_idx")),
                    ],
                )];
                vec![wrap_anonymous(self.op_id, entry_body)]
            },
        )
        .with_entry_op_id(self.op_id)
        .with_non_composable_with_self(true)
    }
}
