//! The token stream every registered typedef row-phase fixture runs on.
//!
//! One witness serves all five phase ops. The declared buffer extents in
//! `typedef_ann::row_phases`, `build::declaration_kind` and
//! `build::typedef_visibility` are sized from the constants here, and the
//! fixtures encode the same witness into buffer bytes, so an extent and the
//! payload it must hold cannot drift apart.

#![allow(missing_docs)] // Internal fixture witness, documented at the owning module boundary.
use crate::parsing::c::lex::tokens::*;
use crate::parsing::c::source_bytes::source_haystack_words;

/// `typedef int T ; void f ( void ) { T v ; }`, one space between tokens.
///
/// Row 9 opens the only block, row 10 uses the typedef name `T` inside it and
/// row 11 is the declarator `v`. This is the smallest stream on which the
/// scope walk, the declaration scan and the visibility scan each return
/// something other than the sentinel they start from.
const PHASE_WITNESS_TOKENS: &[(u32, &str)] = &[
    (TOK_TYPEDEF, "typedef"),
    (TOK_INT, "int"),
    (TOK_IDENTIFIER, "T"),
    (TOK_SEMICOLON, ";"),
    (TOK_VOID, "void"),
    (TOK_IDENTIFIER, "f"),
    (TOK_LPAREN, "("),
    (TOK_VOID, "void"),
    (TOK_RPAREN, ")"),
    (TOK_LBRACE, "{"),
    (TOK_IDENTIFIER, "T"),
    (TOK_IDENTIFIER, "v"),
    (TOK_SEMICOLON, ";"),
    (TOK_RBRACE, "}"),
];

/// VAST rows the witness builds.
pub(in crate::parsing::c::parse::vast) const PHASE_WITNESS_ROWS: u32 =
    PHASE_WITNESS_TOKENS.len() as u32;

/// Source bytes the witness token spans index into.
pub(in crate::parsing::c::parse::vast) const PHASE_WITNESS_SOURCE_LEN: u32 = witness_source_len();

const fn witness_source_len() -> u32 {
    let mut total = 0usize;
    let mut index = 0usize;
    while index < PHASE_WITNESS_TOKENS.len() {
        if index > 0 {
            total += 1;
        }
        total += PHASE_WITNESS_TOKENS[index].1.len();
        index += 1;
    }
    total as u32
}

/// The witness source and the VAST rows the CPU builder oracle derives from it.
#[cfg(any(test, feature = "cpu-parity"))]
pub(in crate::parsing::c::parse::vast) struct PhaseWitness {
    /// Raw source bytes the row spans index into.
    pub(in crate::parsing::c::parse::vast) source: Vec<u8>,
    /// `reference_c11_build_vast_nodes` output, little-endian.
    pub(in crate::parsing::c::parse::vast) node_bytes: Vec<u8>,
    /// The same rows as words, for the CPU reference oracles.
    pub(in crate::parsing::c::parse::vast) node_words: Vec<u32>,
}

#[cfg(any(test, feature = "cpu-parity"))]
impl PhaseWitness {
    /// Assemble the source text and token table together so the spans cannot
    /// disagree with the bytes they index.
    pub(in crate::parsing::c::parse::vast) fn build() -> Self {
        let mut source = Vec::with_capacity(PHASE_WITNESS_SOURCE_LEN as usize);
        let mut tok_types = Vec::with_capacity(PHASE_WITNESS_TOKENS.len());
        let mut tok_starts = Vec::with_capacity(PHASE_WITNESS_TOKENS.len());
        let mut tok_lens = Vec::with_capacity(PHASE_WITNESS_TOKENS.len());
        for (kind, text) in PHASE_WITNESS_TOKENS {
            if !source.is_empty() {
                source.push(b' ');
            }
            tok_types.push(*kind);
            tok_starts.push(source.len() as u32);
            tok_lens.push(text.len() as u32);
            source.extend_from_slice(text.as_bytes());
        }
        let node_bytes = super::reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
        let node_words = node_bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Self {
            source,
            node_bytes,
            node_words,
        }
    }

    /// The identifier text of `row`, read back through the row's own span so
    /// the oracle argument comes from the witness rather than a literal.
    pub(in crate::parsing::c::parse::vast) fn lexeme(&self, row: u32) -> &[u8] {
        let base = row as usize * super::VAST_NODE_STRIDE_U32 as usize;
        let start = self.node_words[base + 5] as usize;
        let len = self.node_words[base + 6] as usize;
        &self.source[start..start + len]
    }

    /// The haystack buffer in the layout `load_source_byte` reads: one byte per
    /// `u32` word when resident-expanded, four little-endian bytes per word
    /// when packed.
    pub(in crate::parsing::c::parse::vast) fn haystack_bytes(
        &self,
        packed_haystack: bool,
    ) -> Vec<u8> {
        let words = source_haystack_words(PHASE_WITNESS_SOURCE_LEN, packed_haystack) as usize;
        let mut bytes = vec![0u8; words * 4];
        if packed_haystack {
            bytes[..self.source.len()].copy_from_slice(&self.source);
        } else {
            for (word, byte) in bytes.chunks_exact_mut(4).zip(&self.source) {
                word[0] = *byte;
            }
        }
        bytes
    }
}
