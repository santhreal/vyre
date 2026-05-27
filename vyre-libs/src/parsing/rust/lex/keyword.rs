//! Keyword promotion: identifier → keyword token id.
//!
//! Run after lexing as a sparse table lookup.  The nano-subset only
//! recognises a small closed set; everything else stays `RUST_TOK_IDENT`.

use vyre_primitives::text::haystack::{Haystack, PackedHaystack};

use super::tokens::*;

/// Keywords in the nano-subset, ordered by frequency (hot → cold) so the
/// linear scan in the CPU fast-path exits early on common hits.
static NANO_KEYWORDS: &[(&str, u16)] = &[
    ("let", RUST_TOK_LET),
    ("fn", RUST_TOK_FN),
    ("if", RUST_TOK_IF),
    ("else", RUST_TOK_ELSE),
    ("return", RUST_TOK_RETURN),
    ("mut", RUST_TOK_MUT),
    ("while", RUST_TOK_WHILE),
    ("true", RUST_TOK_LITERAL_BOOL),
    ("false", RUST_TOK_LITERAL_BOOL),
    ("i32", RUST_TOK_I32),
    ("bool", RUST_TOK_BOOL),
];

/// Packed keyword haystack for GPU dispatch.
pub static KEYWORD_HAYSTACK: PackedHaystack = PackedHaystack::from_const(NANO_KEYWORDS);

/// Promote an identifier string to its keyword token id, if any.
pub fn promote(ident: &str) -> Option<u16> {
    // CPU fast-path: tiny table, linear scan.
    for (kw, tok) in NANO_KEYWORDS {
        if *kw == ident {
            return Some(*tok);
        }
    }
    None
}

/// GPU kernel entry: given a flat buffer of (offset, len) identifier spans,
/// promote each to a keyword or leave as `RUST_TOK_IDENT`.
///
/// This is a vyre::Program builder, not the kernel itself; the actual
/// dispatch lives in `lexer::core`.
pub fn keyword_promote_plan() -> KeywordPromotePlan {
    KeywordPromotePlan {
        haystack: &KEYWORD_HAYSTACK,
    }
}

/// Plan struct for the keyword promotion stage.  Consumed by the lexer
/// pipeline to build the GPU dispatch graph.
pub struct KeywordPromotePlan {
    haystack: &'static PackedHaystack,
}

impl KeywordPromotePlan {
    /// Build the promotion vyre::Program.
    pub fn build(&self) -> vyre::ir::Program {
        // TODO(v0.0.1): implement as a vyre::Program over the identifier
        // span buffer.  For now this is a CPU fallback stub.
        vyre::ir::Program::default()
    }
}
