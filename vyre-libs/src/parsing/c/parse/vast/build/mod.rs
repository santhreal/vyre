//! GPU VAST structural-node construction builders.

#![allow(missing_docs)] // Internal VAST-builder helpers are documented at the owning module boundary.
use crate::parsing::c::lex::tokens::*;
use crate::parsing::c::source_bytes::load_source_byte;
use crate::parsing::composition::child_phase;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::build_declaration_kind_inner::emit_declaration_kind_for_index_inner;
use super::phase_program::{self, phase_haystack_len, phase_program, phase_row, PhaseInputs};
use super::token_grammar::*;
use super::*;

mod declaration_kind;
mod enclosing_function;
mod identifier_hash;
mod scope_lookup;
mod structural_builder;
mod typedef_visibility;
mod vast_row_fields;

pub(super) use declaration_kind::{
    c11_builtin_declaration_kind_for_row, c11_typedef_decl_kind_for_row,
    c11_typedef_decl_kind_for_row_packed_haystack, emit_builtin_declaration_kind_for_index,
    emit_declaration_kind_result_assignment, BUILTIN_DECL_KIND_FOR_ROW_OP_ID,
    DECL_KIND_FOR_ROW_OP_ID, DECL_KIND_FOR_ROW_PACKED_OP_ID,
};
pub(super) use enclosing_function::{
    c11_enclosing_function_lparen_for_row, emit_enclosing_function_lparen_for_index,
    ENCLOSING_FUNCTION_LPAREN_FOR_ROW_OP_ID,
};
pub(super) use identifier_hash::{
    c11_identifier_row_hash, c11_identifier_row_hash_packed_haystack,
    emit_identifier_hash_for_row, emit_identifier_source_hash_for_index, IdentifierRowHash,
    IdentifierRowHashNames, IDENTIFIER_ROW_HASH_OP_ID, IDENTIFIER_ROW_HASH_PACKED_OP_ID,
};
pub(super) use scope_lookup::{
    c11_typedef_scope_open_for_row, emit_scope_open_for_index, emit_scope_open_scan_phase,
    SCOPE_OPEN_FOR_ROW_OP_ID,
};
pub use structural_builder::{c11_build_vast_nodes, c11_build_vast_nodes_uses_global_last_child};
pub(super) use typedef_visibility::{
    c11_typedef_visible_name_for_row, c11_typedef_visible_name_for_row_packed_haystack,
    emit_precomputed_declaration_kind_for_index, emit_typedef_visibility_scan_precomputed_context,
    emit_visible_typedef_name_for_index, VISIBLE_NAME_FOR_ROW_OP_ID,
    VISIBLE_NAME_FOR_ROW_PACKED_OP_ID,
};
pub(super) use vast_row_fields::{
    vast_bounded_row_kind_expr, vast_next_row_kind_expr, vast_prior_row_kind_expr,
    vast_row_base_expr, vast_row_field_expr, vast_row_kind_expr, vast_row_kind_from_base_expr,
    vast_row_parent_from_base_expr,
};
