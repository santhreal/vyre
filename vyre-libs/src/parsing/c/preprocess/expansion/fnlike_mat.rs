//! Materialized function-like macro expansion builder.

use vyre_foundation::ir::{Expr, Node};

use super::fnlike_replacement::*;
use super::paste_branch::*;
use super::regular_branch::*;
use super::string_branch::*;
use super::MacroByteLayout;

/// Buffer and bound inputs of the materialized function-like replacement.
pub(super) struct MaterializedFunctionLikeSpec<'a> {
    /// Input token-type stream.
    pub(super) in_tok_types: &'a str,
    /// Input token start offsets.
    pub(super) in_tok_starts: &'a str,
    /// Input token byte lengths.
    pub(super) in_tok_lens: &'a str,
    /// Input source byte arena.
    pub(super) source_words: &'a str,
    /// Element packing of `source_words`.
    pub(super) source_layout: MacroByteLayout,
    /// Replacement token-type table.
    pub(super) macro_vals: &'a str,
    /// Replacement parameter-index table.
    pub(super) macro_replacement_params: &'a str,
    /// Replacement token start offsets.
    pub(super) macro_replacement_starts: &'a str,
    /// Replacement token byte lengths.
    pub(super) macro_replacement_lens: &'a str,
    /// Replacement source byte arena.
    pub(super) macro_replacement_words: &'a str,
    /// Element packing of `macro_replacement_words`.
    pub(super) macro_replacement_layout: MacroByteLayout,
    /// Output token-type stream.
    pub(super) out_tok_types: &'a str,
    /// Output token start offsets.
    pub(super) out_tok_starts: &'a str,
    /// Output token byte lengths.
    pub(super) out_tok_lens: &'a str,
    /// Output source byte arena.
    pub(super) out_source_words: &'a str,
    /// Workgroup argument start bounds.
    pub(super) macro_arg_starts: &'a str,
    /// Workgroup argument end bounds.
    pub(super) macro_arg_ends: &'a str,
    /// Input token count.
    pub(super) num_tokens: Expr,
    /// Length of the input source arena.
    pub(super) source_len: Expr,
    /// Length of the replacement source arena.
    pub(super) macro_replacement_source_len: Expr,
    /// Output token capacity.
    pub(super) max_out_tokens: u32,
    /// Output byte capacity.
    pub(super) max_out_source_bytes: u32,
}

pub(super) fn emit_materialized_function_like_replacement(
    spec: MaterializedFunctionLikeSpec<'_>,
) -> Vec<Node> {
    let stringify = emit_materialized_stringification_branch(
        spec.macro_replacement_params,
        spec.macro_replacement_starts,
        spec.macro_replacement_lens,
        spec.macro_replacement_words,
        spec.macro_replacement_layout,
        spec.macro_replacement_source_len.clone(),
        spec.macro_arg_starts,
        spec.macro_arg_ends,
        spec.in_tok_starts,
        spec.in_tok_lens,
        spec.source_words,
        spec.source_layout,
        spec.source_len.clone(),
        spec.out_tok_types,
        spec.out_tok_starts,
        spec.out_tok_lens,
        spec.out_source_words,
        spec.max_out_tokens,
        spec.max_out_source_bytes,
        spec.num_tokens.clone(),
    );
    let paste = emit_materialized_function_paste_branch(MaterializedPasteBranchSpec {
        in_tok_types: spec.in_tok_types,
        in_tok_starts: spec.in_tok_starts,
        in_tok_lens: spec.in_tok_lens,
        source_words: spec.source_words,
        source_layout: spec.source_layout,
        macro_vals: spec.macro_vals,
        macro_replacement_params: spec.macro_replacement_params,
        macro_replacement_starts: spec.macro_replacement_starts,
        macro_replacement_lens: spec.macro_replacement_lens,
        macro_replacement_words: spec.macro_replacement_words,
        macro_replacement_layout: spec.macro_replacement_layout,
        out_tok_types: spec.out_tok_types,
        out_tok_starts: spec.out_tok_starts,
        out_tok_lens: spec.out_tok_lens,
        out_source_words: spec.out_source_words,
        macro_arg_starts: spec.macro_arg_starts,
        macro_arg_ends: spec.macro_arg_ends,
        num_tokens: spec.num_tokens.clone(),
        source_len: spec.source_len.clone(),
        macro_replacement_source_len: spec.macro_replacement_source_len.clone(),
        max_out_tokens: spec.max_out_tokens,
        max_out_source_bytes: spec.max_out_source_bytes,
    });
    let regular = emit_materialized_regular_replacement_branch(
        spec.in_tok_types,
        spec.in_tok_starts,
        spec.in_tok_lens,
        spec.source_words,
        spec.source_layout,
        spec.macro_replacement_starts,
        spec.macro_replacement_lens,
        spec.macro_replacement_words,
        spec.macro_replacement_layout,
        spec.out_tok_types,
        spec.out_tok_starts,
        spec.out_tok_lens,
        spec.out_source_words,
        spec.macro_arg_starts,
        spec.macro_arg_ends,
        spec.num_tokens.clone(),
        spec.source_len.clone(),
        spec.macro_replacement_source_len.clone(),
        spec.max_out_tokens,
        spec.max_out_source_bytes,
    );

    emit_function_like_replacement_walk(FunctionLikeReplacementSpec {
        in_tok_types: spec.in_tok_types,
        macro_vals: spec.macro_vals,
        macro_replacement_params: spec.macro_replacement_params,
        macro_arg_starts: spec.macro_arg_starts,
        macro_arg_ends: spec.macro_arg_ends,
        num_tokens: spec.num_tokens,
        stringify,
        paste,
        regular,
    })
}
