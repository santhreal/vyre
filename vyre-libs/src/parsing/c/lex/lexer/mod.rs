//! C11 lexer program builder.
//!
//! `c11_lexer` constructs a single `Vec<Node>` by appending classifier
//! sub-builders:
//!  - `byte_exprs.rs`: IR expressions over one haystack byte
//!  - `classify/`: the token-classification stages and the serial, ranked, and
//!    sparse lexers composed from them
//!  - `core_sparse.rs`: the packed / expanded / raw-u8 sparse entry points
//!  - `sections.rs`: operator-table and epilogue builders too large for the
//!    composition files
//!  - `digraphs.rs`: digraph + line-splice resolution pass

mod byte_exprs;
mod classify;
mod core_sparse;
mod digraphs;
mod sections;
mod single_pass;
mod sparse_compact;

pub use classify::{
    c11_lexer, c11_lexer_regular, c11_lexer_regular_ranked, c11_lexer_regular_sparse,
};
pub use core_sparse::{
    c11_lexer_regular_sparse_no_directives_no_backscan,
    c11_lexer_regular_sparse_packed_haystack_with_block_totals,
    c11_lexer_regular_sparse_packed_haystack_with_flags,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan,
    c11_lexer_regular_sparse_u8_haystack_with_flags,
};
pub use digraphs::c11_lex_digraphs;
pub use single_pass::{c11_lex_regular_single_pass, c11_lex_single_pass};
pub use sparse_compact::{c11_compact_sparse_tokens, c11_compact_sparse_tokens_output};

fn identifier_fixture_inputs(capacity: usize) -> Vec<Vec<Vec<u8>>> {
    let bytes = capacity * std::mem::size_of::<u32>();
    vec![vec![
        vec![b'a'; bytes],
        vec![0u8; bytes],
        vec![0u8; bytes],
        vec![0u8; bytes],
        vec![0u8; std::mem::size_of::<u32>()],
    ]]
}

// Sibling re-exports keep each lexer submodule on one explicit helper surface.
// If a helper stops being shared by multiple active lexer builders, move it
// into the single module that owns it.
