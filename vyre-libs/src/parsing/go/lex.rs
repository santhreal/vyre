use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

// `vyre_spec::go_token` owns the numbering of these ids. They are the wire
// contract between the GPU lexer program below and every host matcher that
// reads its sparse token rows, so a caller that reads a token kind names that
// module rather than this one.
use vyre_spec::go_token::*;

fn byte_load(buffer: &str, index: Expr) -> Expr {
    crate::builder::state_machine::TableStateMachineComposer::masked_byte_load(buffer, index)
}

fn byte_eq(expr: Expr, byte: u8) -> Expr {
    Expr::eq(expr, Expr::u32(u32::from(byte)))
}

fn byte_between(expr: Expr, low: u8, high: u8) -> Expr {
    Expr::and(
        Expr::ge(expr.clone(), Expr::u32(u32::from(low))),
        Expr::le(expr, Expr::u32(u32::from(high))),
    )
}

fn ident_start(expr: Expr) -> Expr {
    Expr::or(
        Expr::or(
            byte_between(expr.clone(), b'a', b'z'),
            byte_between(expr.clone(), b'A', b'Z'),
        ),
        byte_eq(expr, b'_'),
    )
}

fn ident_continue(expr: Expr) -> Expr {
    Expr::or(ident_start(expr.clone()), byte_between(expr, b'0', b'9'))
}

fn punctuation_token(byte: Expr, token: u32, chr: u8) -> Vec<Node> {
    vec![Node::if_then(
        byte_eq(byte, chr),
        vec![
            Node::assign("emit", Expr::u32(1)),
            Node::assign("tok_type", Expr::u32(token)),
            Node::assign("tok_len", Expr::u32(1)),
        ],
    )]
}

/// Byte-oriented Go lexer over a `u32`-encoded byte stream.
///
/// Each invocation owns one source byte and emits at most one token. Identifiers
/// and string literals are maximally munched by a forward scan from the start
/// byte. Punctuation is emitted as fixed-width one-byte tokens, with `<-`
/// treated as a dedicated channel operator token.
///
/// # Output is SPARSE and source-indexed
///
/// This stage writes token `i` into slot `i`, the index of its first source
/// byte, and sets `out_emit_flags[i]` to 1 there. Positions that begin no
/// token keep a zero flag. The result is therefore one slot per source byte,
/// mostly empty, in source order by construction.
///
/// It does NOT produce the dense stream the extractors read. Run
/// [`go_compact_tokens`] to get that, with an exclusive prefix scan of the
/// emit flags in between:
///
/// ```text
/// go_lexer                              -> sparse types/starts/lens + emit flags
/// multi_block_prefix_scan_sum_exclusive -> emit_offsets
/// go_compact_tokens                     -> dense types/starts/lens + count
/// ```
///
/// The reason for the extra pass is correctness, not tidiness. This stage used
/// to allocate each token's slot with `atomicAdd(out_counts, 1)`, which packs
/// the stream densely in ATOMIC ARRIVAL order. Arrival order is arbitrary on a
/// GPU and not reproducible between runs, while every Go extractor reads
/// ADJACENT tokens and so requires source order: `go_extract_channel_sends`,
/// for instance, matches `tok_types[t] == IDENTIFIER && tok_types[t + 1] ==
/// ARROW`. The two contracts never met. Measured on the fixture corpus, the
/// extractors found 0 packages, 0 imports and 0 sends in files that plainly
/// contain them, because the tokens they were pattern-matching over had been
/// shuffled. A prefix scan gives every emitting position a slot ordered by
/// position rather than by whichever lane got to the counter first.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_lexer(
    haystack: &str,
    quote_ranks: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_emit_flags: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::gid_x();

    let mut body = vec![
        Node::let_bind("byte", byte_load(haystack, t.clone())),
        Node::let_bind(
            "prev_byte",
            Expr::select(
                Expr::gt(t.clone(), Expr::u32(0)),
                byte_load(haystack, Expr::sub(t.clone(), Expr::u32(1))),
                Expr::u32(0),
            ),
        ),
        Node::let_bind(
            "next_byte",
            Expr::select(
                Expr::lt(Expr::add(t.clone(), Expr::u32(1)), Expr::u32(haystack_len)),
                byte_load(haystack, Expr::add(t.clone(), Expr::u32(1))),
                Expr::u32(0),
            ),
        ),
        Node::let_bind("emit", Expr::u32(0)),
        Node::let_bind("tok_type", Expr::u32(TOK_NONE)),
        Node::let_bind("tok_len", Expr::u32(0)),
    ];

    body.push(Node::if_then(
        Expr::and(
            ident_start(Expr::var("byte")),
            Expr::not(ident_continue(Expr::var("prev_byte"))),
        ),
        vec![
            Node::assign("emit", Expr::u32(1)),
            Node::assign("tok_type", Expr::u32(TOK_IDENTIFIER)),
            Node::assign("tok_len", Expr::u32(1)),
            Node::let_bind("still_ident", Expr::u32(1)),
            Node::loop_for(
                "scan",
                Expr::add(t.clone(), Expr::u32(1)),
                Expr::u32(haystack_len),
                vec![Node::if_then(
                    Expr::eq(Expr::var("still_ident"), Expr::u32(1)),
                    vec![
                        Node::let_bind("scan_byte", byte_load(haystack, Expr::var("scan"))),
                        Node::if_then_else(
                            ident_continue(Expr::var("scan_byte")),
                            vec![Node::assign(
                                "tok_len",
                                Expr::add(Expr::var("tok_len"), Expr::u32(1)),
                            )],
                            vec![Node::assign("still_ident", Expr::u32(0))],
                        ),
                    ],
                )],
            ),
        ],
    ));

    // A quote OPENS a string only when an even number of quotes precede it.
    //
    // Without the parity test the lane sitting on a string's CLOSING quote
    // decides it is an opening quote too, scans forward to the next quote in
    // the file, and emits a second, bogus string token spanning the gap
    // between two literals. Every grouped import was therefore counted twice:
    // `import ( "fmt" )` produced `"fmt"` and a phantom literal starting at
    // that string's own closing quote.
    //
    // Parity cannot be decided from one byte, so it arrives as a scanned
    // input. `quote_ranks[i]` is the number of quotes strictly before `i`,
    // produced by the same exclusive prefix scan the token compaction uses.
    body.push(Node::if_then(
        Expr::and(
            byte_eq(Expr::var("byte"), b'"'),
            Expr::eq(
                Expr::rem(Expr::load(quote_ranks, t.clone()), Expr::u32(2)),
                Expr::u32(0),
            ),
        ),
        vec![
            Node::assign("emit", Expr::u32(1)),
            Node::assign("tok_type", Expr::u32(TOK_STRING)),
            Node::assign("tok_len", Expr::u32(1)),
            Node::let_bind("string_done", Expr::u32(0)),
            Node::loop_for(
                "scan",
                Expr::add(t.clone(), Expr::u32(1)),
                Expr::u32(haystack_len),
                vec![Node::if_then(
                    Expr::eq(Expr::var("string_done"), Expr::u32(0)),
                    vec![
                        Node::assign("tok_len", Expr::add(Expr::var("tok_len"), Expr::u32(1))),
                        // Stop at the next DELIMITING quote, not at the next
                        // quote byte: an escaped `\"` inside the literal is
                        // content and would otherwise cut the token short. A
                        // position holds a delimiter exactly when the quote
                        // rank increases across it, which is the same flag
                        // stream the parity test above reads.
                        Node::if_then(
                            Expr::and(
                                Expr::lt(
                                    Expr::add(Expr::var("scan"), Expr::u32(1)),
                                    Expr::u32(haystack_len),
                                ),
                                Expr::gt(
                                    Expr::load(
                                        quote_ranks,
                                        Expr::add(Expr::var("scan"), Expr::u32(1)),
                                    ),
                                    Expr::load(quote_ranks, Expr::var("scan")),
                                ),
                            ),
                            vec![Node::assign("string_done", Expr::u32(1))],
                        ),
                    ],
                )],
            ),
        ],
    ));

    body.push(Node::if_then(
        Expr::and(
            byte_eq(Expr::var("byte"), b'<'),
            byte_eq(Expr::var("next_byte"), b'-'),
        ),
        vec![
            Node::assign("emit", Expr::u32(1)),
            Node::assign("tok_type", Expr::u32(TOK_ARROW)),
            Node::assign("tok_len", Expr::u32(2)),
        ],
    ));

    body.extend(punctuation_token(Expr::var("byte"), TOK_LPAREN, b'('));
    body.extend(punctuation_token(Expr::var("byte"), TOK_RPAREN, b')'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_LBRACE, b'{'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_RBRACE, b'}'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_LBRACKET, b'['));
    body.extend(punctuation_token(Expr::var("byte"), TOK_RBRACKET, b']'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_COMMA, b','));
    body.extend(punctuation_token(Expr::var("byte"), TOK_DOT, b'.'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_SEMICOLON, b';'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_COLON, b':'));
    body.extend(punctuation_token(Expr::var("byte"), TOK_ASSIGN, b'='));
    body.extend(punctuation_token(Expr::var("byte"), TOK_STAR, b'*'));
    // Go's implicit semicolon. See TOK_NEWLINE for why the statement boundary
    // has to be in the token stream at all.
    body.extend(punctuation_token(Expr::var("byte"), TOK_NEWLINE, b'\n'));

    // Nothing inside a string literal is a token.
    //
    // Every rule above looks at one byte in isolation, so the `x` in `"x"` was
    // emitted as an identifier and the `.` in `"a.b"` as a dot. That inflates
    // the stream with tokens that never appear in the source as code, and the
    // extractors read adjacency, so a stray identifier between two real tokens
    // silently breaks every `tok_types[t + 1]` match around it.
    //
    // A byte is inside a literal exactly when an odd number of delimiting
    // quotes precede it. The opening quote itself sits at an even rank, so the
    // string token survives; its contents and its closing quote do not.
    body.push(Node::if_then(
        Expr::eq(
            Expr::rem(Expr::load(quote_ranks, t.clone()), Expr::u32(2)),
            Expr::u32(1),
        ),
        vec![
            Node::assign("emit", Expr::u32(0)),
            Node::assign("tok_type", Expr::u32(TOK_NONE)),
            Node::assign("tok_len", Expr::u32(0)),
        ],
    ));

    // Write at the SOURCE index, never at an atomically allocated slot. Every
    // position writes all four words so a stale buffer cannot leave a token
    // behind: a non-emitting position stores a zero flag and zeroed fields.
    body.push(Node::store(out_emit_flags, t.clone(), Expr::var("emit")));
    body.push(Node::store(out_tok_types, t.clone(), Expr::var("tok_type")));
    body.push(Node::store(
        out_tok_starts,
        t.clone(),
        Expr::select(
            Expr::eq(Expr::var("emit"), Expr::u32(1)),
            t.clone(),
            Expr::u32(0),
        ),
    ));
    body.push(Node::store(out_tok_lens, t.clone(), Expr::var("tok_len")));

    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(quote_ranks, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_types, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_starts, 3, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_lens, 4, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_emit_flags, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_lexer",
            vec![Node::if_then(Expr::lt(t, Expr::u32(haystack_len)), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_lexer")
    .with_non_composable_with_self(true)
}

/// Mark every string-delimiting double-quote byte in the source with a 1.
///
/// First stage of the string-literal parity chain. Scanning these flags gives
/// each quote its ordinal, and a quote opens a literal exactly when that
/// ordinal is even. See the quote-parity comment in [`go_lexer`] for the
/// phantom-literal bug this prevents.
///
/// An ESCAPED quote is not a delimiter and must not be flagged. `"say \"hi\""`
/// contains six quote bytes but only two delimiters, and flagging all six
/// inverts the parity of every literal in the rest of the file: each subsequent
/// opening quote then looks like a closing one, so the lexer stops recognising
/// strings and starts lexing their contents as code. A quote is escaped when an
/// ODD number of backslashes immediately precedes it, which is why the run has
/// to be counted rather than just testing the previous byte: in `"a\\"` the
/// final quote is a real delimiter because the two backslashes escape each
/// other.
#[must_use]
pub fn go_quote_flags(haystack: &str, out_quote_flags: &str, haystack_len: u32) -> Program {
    let t = Expr::gid_x();
    // Count the run of backslashes ending at `t - 1`. The lane walks the whole
    // prefix because the IR has no backward loop; the string scan in `go_lexer`
    // already has the same per-lane cost profile.
    let backslash_run = vec![
        Node::let_bind("run", Expr::u32(0)),
        Node::loop_for(
            "back",
            Expr::u32(0),
            t.clone(),
            vec![Node::if_then_else(
                byte_eq(byte_load(haystack, Expr::var("back")), b'\\'),
                vec![Node::assign(
                    "run",
                    Expr::add(Expr::var("run"), Expr::u32(1)),
                )],
                vec![Node::assign("run", Expr::u32(0))],
            )],
        ),
    ];
    let mut body = backslash_run;
    body.push(Node::store(
        out_quote_flags,
        t.clone(),
        Expr::select(
            Expr::and(
                byte_eq(byte_load(haystack, t.clone()), b'"'),
                Expr::eq(Expr::rem(Expr::var("run"), Expr::u32(2)), Expr::u32(0)),
            ),
            Expr::u32(1),
            Expr::u32(0),
        ),
    ));
    Program::wrapped(
        vec![
            BufferDecl::storage(haystack, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_quote_flags, 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_quote_flags",
            vec![Node::if_then(
                Expr::lt(t.clone(), Expr::u32(haystack_len)),
                body,
            )],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_quote_flags")
    .with_non_composable_with_self(true)
}

/// Scan the quote flags into per-byte quote ordinals for [`go_lexer`].
///
/// Same shared scan primitive as [`go_scan_emit_flags`]; named separately only
/// so each stage of the Go pipeline reads as what it is.
#[must_use]
pub fn go_scan_quote_flags(quote_flags: &str, quote_ranks: &str, haystack_len: u32) -> Program {
    crate::reduce::multi_block_prefix_scan::multi_block_prefix_scan_sum_exclusive_u32(
        quote_flags,
        quote_ranks,
        haystack_len,
    )
}

/// Build the exclusive prefix scan that turns emit flags into dense slots.
///
/// Wraps the shared `multi_block_prefix_scan` primitive rather than restating
/// a scan, so the Go frontend and every other consumer share one scan
/// implementation. `emit_offsets[i]` ends up holding the number of tokens that
/// start strictly before source position `i`, which is exactly the dense index
/// position `i`'s token belongs at.
#[must_use]
pub fn go_scan_emit_flags(emit_flags: &str, emit_offsets: &str, haystack_len: u32) -> Program {
    crate::reduce::multi_block_prefix_scan::multi_block_prefix_scan_sum_exclusive_u32(
        emit_flags,
        emit_offsets,
        haystack_len,
    )
}

/// Compact the sparse per-byte token stream into a dense, source-ordered one.
///
/// Reads the sparse arrays [`go_lexer`] wrote and the offsets
/// [`go_scan_emit_flags`] produced, and copies each emitting position's token
/// to `dense[emit_offsets[i]]`. Because the offset is a prefix count over
/// source positions, the dense stream is in source order: position `i` lands
/// before position `j` whenever `i < j`. The total token count is the
/// exclusive offset of the last position plus its own flag, which the final
/// lane writes to `out_counts[0]`.
///
/// This is the stage that makes `tok_types[t + 1]` mean "the next token in the
/// source", which every Go extractor depends on.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn go_compact_tokens(
    sparse_types: &str,
    sparse_starts: &str,
    sparse_lens: &str,
    emit_flags: &str,
    emit_offsets: &str,
    out_tok_types: &str,
    out_tok_starts: &str,
    out_tok_lens: &str,
    out_counts: &str,
    haystack_len: u32,
) -> Program {
    let t = Expr::gid_x();
    let last = haystack_len.saturating_sub(1);

    let body = vec![
        Node::let_bind("flag", Expr::load(emit_flags, t.clone())),
        Node::let_bind("slot", Expr::load(emit_offsets, t.clone())),
        Node::if_then(
            Expr::eq(Expr::var("flag"), Expr::u32(1)),
            vec![
                Node::store(
                    out_tok_types,
                    Expr::var("slot"),
                    Expr::load(sparse_types, t.clone()),
                ),
                Node::store(
                    out_tok_starts,
                    Expr::var("slot"),
                    Expr::load(sparse_starts, t.clone()),
                ),
                Node::store(
                    out_tok_lens,
                    Expr::var("slot"),
                    Expr::load(sparse_lens, t.clone()),
                ),
            ],
        ),
        // The last source position knows the total: every token starts at or
        // before it, so its exclusive offset plus its own flag is the count.
        // One lane writes it, so no atomic and no ordering question.
        Node::if_then(
            Expr::eq(t.clone(), Expr::u32(last)),
            vec![Node::store(
                out_counts,
                Expr::u32(0),
                Expr::add(Expr::var("slot"), Expr::var("flag")),
            )],
        ),
    ];

    Program::wrapped(
        vec![
            BufferDecl::storage(sparse_types, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(sparse_starts, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(sparse_lens, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(emit_flags, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(emit_offsets, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_types, 5, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_starts, 6, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_tok_lens, 7, BufferAccess::ReadWrite, DataType::U32)
                .with_count(haystack_len),
            BufferDecl::storage(out_counts, 8, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::parsing::go_compact_tokens",
            vec![Node::if_then(Expr::lt(t, Expr::u32(haystack_len)), body)],
        )],
    )
    .with_entry_op_id("vyre-libs::parsing::go_compact_tokens")
    .with_non_composable_with_self(true)
}
