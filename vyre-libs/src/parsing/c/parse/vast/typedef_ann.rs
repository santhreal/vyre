//! GPU typedef-name annotation and symbol-linking builders.

#![allow(missing_docs)] // Internal VAST-builder helpers are documented at the owning module boundary.
use crate::parsing::c::lex::tokens::*;
use crate::parsing::c::source_bytes::source_haystack_words;
use crate::region::wrap_anonymous;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::build::*;
use super::token_grammar::*;
use super::*;

mod annotate;
mod decl_contexts;
mod decl_prefix_starts;
mod global_fast;
mod haystack_words;
mod precomputed_visible_type;
mod prehash;
mod row_io;
mod row_phases;
mod scopes;
mod symbol_bucket;
mod symbol_links;

pub use annotate::{
    c11_annotate_typedef_names, c11_annotate_typedef_names_packed_haystack,
    c11_annotate_typedef_names_precomputed_context,
    c11_annotate_typedef_names_precomputed_context_packed_haystack,
    c11_annotate_typedef_names_precomputed_scope,
    c11_annotate_typedef_names_precomputed_scope_packed_haystack,
};
pub use decl_contexts::c11_precompute_vast_decl_contexts;
pub use decl_prefix_starts::c11_precompute_vast_decl_prefix_starts;
pub use global_fast::c11_annotate_global_typedef_names_fast;
pub use precomputed_visible_type::{
    c11_precompute_vast_visible_type, c11_precompute_vast_visible_type_packed_haystack,
};
pub use prehash::{c11_prehash_vast_identifiers, c11_prehash_vast_identifiers_packed_haystack};
pub use scopes::{c11_precompute_vast_scopes, c11_precompute_vast_scopes_uses_global_stack};
pub use symbol_links::c11_link_vast_typedef_symbols;

use haystack_words::*;
use row_io::*;
use symbol_bucket::*;
