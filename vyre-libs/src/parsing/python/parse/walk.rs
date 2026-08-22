//! The one Python token-stream AST walk.
//!
//! Every Python extractor dispatches one invocation per token, resolves a
//! dotted name from that token, and writes a fixed-width record. This module
//! owns the two pieces all of them share: the bounded dotted-name segment walk
//! and the GPU program envelope around it.

use super::{search_next_token_into, token_word_at};
use crate::parsing::composition::child_phase;
use crate::parsing::python::{INVALID_POS, MAX_DOTTED_SEGMENTS};
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_spec::python_token::{TOK_DOT, TOK_IDENTIFIER};

/// The `a.b.c` walk shared by every Python extractor that resolves a dotted
/// name: `import` targets, `with` managers, call heads, and decorators.
pub(crate) struct DottedName<'a> {
    /// Token-type buffer every scan reads.
    pub tok_types: &'a str,
    /// Token count bounding every forward scan.
    pub haystack_len: u32,
    /// Position of the leading identifier. Seeds both the accumulator and the
    /// walk cursor, and anchors the resolved span.
    pub head: Expr,
    /// Variable that ends up holding the last identifier of the chain.
    pub accumulator: &'a str,
}

impl DottedName<'_> {
    /// Carriers the walk assigns across loop iterations.
    ///
    /// These are `let_bind`s rather than assignments inside the loop because
    /// the validator scopes a binding to its block, and re-binding per
    /// iteration is a re-declaration (V008). Emit them in a scope that
    /// outlives [`Self::walk`].
    pub(crate) fn carriers(&self) -> Vec<Node> {
        vec![
            Node::let_bind(self.accumulator, self.head.clone()),
            Node::let_bind("cursor", self.head.clone()),
            Node::let_bind("dot_pos", Expr::u32(INVALID_POS)),
            Node::let_bind("after_dot", Expr::u32(INVALID_POS)),
        ]
    }

    /// Bounded walk that extends the accumulator over `.ident` segments.
    ///
    /// The `cursor != INVALID_POS` guard is load-bearing. `cursor` holds
    /// `u32::MAX` once the chain ends, `cursor + 1` wraps to 0, and an
    /// unguarded rescan therefore restarts at token 0 and can pull an
    /// unrelated `.ident` pair from the head of the unit into the accumulator,
    /// which underflows the emitted span length.
    pub(crate) fn walk(&self) -> Node {
        Node::loop_for(
            "seg",
            Expr::u32(0),
            Expr::u32(MAX_DOTTED_SEGMENTS),
            vec![
                Node::assign("dot_pos", Expr::u32(INVALID_POS)),
                Node::assign("after_dot", Expr::u32(INVALID_POS)),
                Node::if_then(
                    Expr::ne(Expr::var("cursor"), Expr::u32(INVALID_POS)),
                    search_next_token_into(
                        "dot_pos",
                        Expr::add(Expr::var("cursor"), Expr::u32(1)),
                        self.tok_types,
                        self.haystack_len,
                    ),
                ),
                Node::if_then(
                    Expr::eq(
                        token_word_at(self.tok_types, Expr::var("dot_pos"), self.haystack_len),
                        Expr::u32(TOK_DOT),
                    ),
                    search_next_token_into(
                        "after_dot",
                        Expr::add(Expr::var("dot_pos"), Expr::u32(1)),
                        self.tok_types,
                        self.haystack_len,
                    ),
                ),
                Node::if_then(
                    Expr::eq(
                        token_word_at(self.tok_types, Expr::var("after_dot"), self.haystack_len),
                        Expr::u32(TOK_IDENTIFIER),
                    ),
                    vec![
                        Node::assign(self.accumulator, Expr::var("after_dot")),
                        Node::assign("cursor", Expr::var("after_dot")),
                    ],
                ),
                Node::if_then(
                    Expr::ne(
                        token_word_at(self.tok_types, Expr::var("after_dot"), self.haystack_len),
                        Expr::u32(TOK_IDENTIFIER),
                    ),
                    vec![Node::assign("cursor", Expr::u32(INVALID_POS))],
                ),
            ],
        )
    }

    /// `(source_offset, source_length)` of the resolved chain.
    pub(crate) fn span(&self, tok_starts: &str, tok_lens: &str) -> [Expr; 2] {
        [
            token_word_at(tok_starts, self.head.clone(), self.haystack_len),
            Expr::add(
                Expr::sub(
                    token_word_at(tok_starts, Expr::var(self.accumulator), self.haystack_len),
                    token_word_at(tok_starts, self.head.clone(), self.haystack_len),
                ),
                token_word_at(tok_lens, Expr::var(self.accumulator), self.haystack_len),
            ),
        ]
    }
}

/// The GPU program envelope every Python token-stream extractor shares:
/// one invocation per token, bounds-masked, attributed to one child phase.
pub(crate) struct TokenPass<'a> {
    /// Registered op id, also the wrapping region generator.
    pub op_id: &'a str,
    /// Primitive op id the single child phase attributes the work to.
    pub child_op_id: &'a str,
    /// Token-type buffer at binding 0.
    pub tok_types: &'a str,
    /// Token start-offset buffer at binding 1.
    pub tok_starts: &'a str,
    /// Token length buffer at binding 2.
    pub tok_lens: &'a str,
    /// Token count; bounds the dispatch and sizes every token buffer.
    pub haystack_len: u32,
}

impl<'a> TokenPass<'a> {
    /// Token pass linked against the canonical line index child phase.
    pub(crate) fn with_line_index(
        op_id: &'a str,
        tok_types: &'a str,
        tok_starts: &'a str,
        tok_lens: &'a str,
        haystack_len: u32,
    ) -> Self {
        Self {
            op_id,
            child_op_id: crate::text::LINE_INDEX_OP_ID,
            tok_types,
            tok_starts,
            tok_lens,
            haystack_len,
        }
    }

    /// Build a standard single-output-record program.
    pub(crate) fn build_record_program(
        &self,
        out_records: &str,
        out_counts: &str,
        record_words: u32,
        body: Vec<Node>,
    ) -> Program {
        let mut buffers = self.token_buffers();
        buffers.extend(self.record_buffers(out_records, out_counts, 3, record_words));
        self.program(buffers, body)
    }
    /// The three read-only token buffers, at bindings 0 through 2.
    pub(crate) fn token_buffers(&self) -> Vec<BufferDecl> {
        [self.tok_types, self.tok_starts, self.tok_lens]
            .into_iter()
            .enumerate()
            .map(|(binding, name)| {
                BufferDecl::storage(name, binding as u32, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(self.haystack_len)
            })
            .collect()
    }

    /// A `(records, counts)` output pair at `binding` and `binding + 1`.
    pub(crate) fn record_buffers(
        &self,
        records: &str,
        counts: &str,
        binding: u32,
        record_words: u32,
    ) -> Vec<BufferDecl> {
        vec![
            BufferDecl::storage(records, binding, BufferAccess::ReadWrite, DataType::U32)
                .with_count(self.haystack_len.saturating_mul(record_words)),
            BufferDecl::storage(counts, binding + 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ]
    }

    /// Wrap a per-token `body` into the dispatchable program.
    pub(crate) fn program(&self, buffers: Vec<BufferDecl>, body: Vec<Node>) -> Program {
        Program::wrapped(
            buffers,
            [256, 1, 1],
            vec![wrap_anonymous_region(
                self.op_id,
                vec![child_phase(
                    self.op_id,
                    self.child_op_id,
                    vec![Node::if_then(
                        Expr::lt(Expr::InvocationId { axis: 0 }, Expr::u32(self.haystack_len)),
                        body,
                    )],
                )],
            )],
        )
        .with_entry_op_id(self.op_id)
        .with_non_composable_with_self(true)
    }
}

/// Pack a sparse `(position, token, length)` list into the three token buffers
/// a `TOKENS`-wide extractor fixture dispatches over.
pub(crate) fn pack_sparse_tokens(
    tokens: &[(usize, u32, u32)],
    slots: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut tok_types = vec![0u8; slots * 4];
    let mut tok_starts = vec![0u8; slots * 4];
    let mut tok_lens = vec![0u8; slots * 4];
    for &(pos, tok, len) in tokens {
        let base = pos * 4;
        tok_types[base..base + 4].copy_from_slice(&tok.to_le_bytes());
        tok_starts[base..base + 4].copy_from_slice(&(pos as u32).to_le_bytes());
        tok_lens[base..base + 4].copy_from_slice(&len.to_le_bytes());
    }
    (tok_types, tok_starts, tok_lens)
}
/// Pack an array of u32 words into a constant little-endian byte array padded with zeros.
pub(crate) const fn pack_words_padded_bytes<const W: usize, const OUT: usize>(
    words: [u32; W],
) -> [u8; OUT] {
    let mut out = [0u8; OUT];
    let mut i = 0;
    while i < W {
        let b = words[i].to_le_bytes();
        out[i * 4] = b[0];
        out[i * 4 + 1] = b[1];
        out[i * 4 + 2] = b[2];
        out[i * 4 + 3] = b[3];
        i += 1;
    }
    out
}
