//! Go structural extraction passes.

/// Go-specific AST-shaped operations: goroutines, channels, defer.
pub mod ast_ops;
/// Declaration/package/import extraction.
pub mod structure;
mod token_predicates;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType};

/// The four read-only buffers every Go extraction kernel reads, at slots 0
/// to 3.
///
/// The token kind, start offset and byte length arrays, and the packed source
/// bytes. Every kernel takes the same stream at the same slots, and one that
/// numbered them differently would bind another kernel's buffer.
fn token_stream_decls(
    tok_types: &str,
    tok_starts: &str,
    tok_lens: &str,
    haystack: &str,
) -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage(tok_types, 0, BufferAccess::ReadOnly, DataType::U32),
        BufferDecl::storage(tok_starts, 1, BufferAccess::ReadOnly, DataType::U32),
        BufferDecl::storage(tok_lens, 2, BufferAccess::ReadOnly, DataType::U32),
        BufferDecl::storage(haystack, 3, BufferAccess::ReadOnly, DataType::U32),
    ]
}
