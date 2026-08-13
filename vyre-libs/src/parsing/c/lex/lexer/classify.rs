//! Serial and per-invocation C11 lexer builders.
//!
//! `stages` owns the one token-classification walk every builder in this
//! module and in `core_sparse` composes; the files beside it are thin
//! compositions that pick stages, name their loop variables, and choose an
//! output shell. `dense.rs` is the full C11 grammar over a contiguous haystack,
//! `regular.rs` the reduced identifier/integer grammar, `ranked.rs` and
//! `sparse.rs` the per-invocation layouts.

#![allow(missing_docs)] // Internal lexer-builder helpers are documented at the owning module boundary.
use crate::parsing::c::lex::tokens::*;
use crate::parsing::composition::child_phase;
use crate::region::wrap_anonymous;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::byte_exprs::{
    ascii, byte_at_or_zero, byte_eq, byte_load, is_digit, is_ident_continue, is_ident_start,
    is_valid_escape_byte,
};

mod dense;
mod parallel_common;
mod ranked;
mod regular;
mod scan_bounds;
mod sparse;

pub(super) mod stages;

pub use dense::c11_lexer;
pub use ranked::c11_lexer_regular_ranked;
pub use regular::c11_lexer_regular;
pub use sparse::c11_lexer_regular_sparse;

use scan_bounds::*;
