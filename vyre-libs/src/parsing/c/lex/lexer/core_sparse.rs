#![allow(missing_docs)]
mod block_totals;

use crate::parsing::c::lex::tokens::*;
use crate::region::wrap_anonymous;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub use block_totals::c11_lexer_regular_sparse_packed_haystack_with_block_totals;

use super::core::helpers::SparseHaystackLayout;
use super::helpers::byte_eq;

mod bounds;
mod entrypoints;
mod sparse_impl;

pub use entrypoints::{
    c11_lexer_regular_sparse_no_directives_no_backscan,
    c11_lexer_regular_sparse_packed_haystack_with_flags,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan,
    c11_lexer_regular_sparse_u8_haystack_with_flags,
};

use bounds::*;
use sparse_impl::{c11_lexer_regular_sparse_impl, SparseLexerSpec};
