//! C11 pipeline modules  -  lex / preprocess / parse / pipeline.
pub(crate) mod atomic_collect;

/// DFA lexer pipeline (lexer, tokens, keywords).
pub mod lex;
/// Lowering from structural parse to packed graph (PG) nodes.
pub mod lower;
/// Structural parser.
pub mod parse;
/// End-to-end example Programs for the C11 pipeline.
pub mod pipeline;
/// Preprocessor expansion.
pub mod preprocess;
/// Semantic analysis of C structures and declarations.
pub mod sema;
/// Source byte addressing helpers shared by expanded and packed GPU haystacks.
pub mod source_bytes;
