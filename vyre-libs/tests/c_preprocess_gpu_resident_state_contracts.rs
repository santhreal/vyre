//! GPU-resident preprocessor state contracts for the C parser megakernel.
//!
//! Table-driven assertions covering:
//! - macro table arena (slot geometry, probe discipline, empty-sentinel)
//! - function-like macro arg arena (bound tracking, parameter substitution)
//! - conditional stack (depth mask, active/taken bits, nesting overflow)
//! - directive metadata (kind token IDs, payload evaluation, phase-2 splicing)
//! - expansion queue (warp-base accumulation, source-ordered emission)
//! - overflow diagnostics (every trap path must fail loudly, not panic)
//! - collision-safe macro names (FNV-1a + byte-exact verification)
#![cfg(feature = "c-parser")]
#![allow(deprecated)]

#[path = "../../tests/support/c_frontend/mod.rs"]
mod c_frontend;
mod wire_words;
mod harness;
use c_frontend::macro_expansion::{
    hash_token, run_dynamic_macro_expansion as run_dynamic, MacroFixture as DynamicFixture,
};
use wire_words::{decode_u32_words, u32_bytes};
use std::panic::{catch_unwind, AssertUnwindSafe};
use harness::c_macro_table::{
    fnv1a32, macro_slot, run_named_macro_expansion, NamedMacroFixture, TokenStream, EMPTY_SLOT,
    NAME_POOL_BYTES, TABLE_MASK, TABLE_SLOTS,
};
use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::preprocess::expansion::{
    opt_conditional_mask, opt_conditional_mask_with_directives, C_MACRO_KIND_FUNCTION_LIKE,
    C_MACRO_KIND_OBJECT_LIKE, C_MACRO_REPLACEMENT_LITERAL,
};
use vyre_libs::parsing::c::preprocess::{
    c_translation_phase_line_splice, reference_c_preprocessor_directive_metadata,
    CPreprocessorDirectiveKind,
};
use vyre_reference::value::Value;
// ---------------------------------------------------------------------------
// Constants mirroring the megakernel state layout
// ---------------------------------------------------------------------------
#[allow(dead_code)]
const MAX_FN_ARGS: u32 = 16;
// ---------------------------------------------------------------------------
// Runners
// ---------------------------------------------------------------------------
fn run_conditional_mask(tok_types: &[u32]) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = opt_conditional_mask("tok_types", "out_mask", Expr::u32(tok_types.len() as u32));
    let input_bytes = if tok_types.is_empty() {
        vec![0u8; 4]
    } else {
        u32_bytes(tok_types)
    };
    let values = [
        Value::from(input_bytes),
        Value::from(vec![0u8; tok_types.len().max(1) * 4]),
    ];
    vyre_reference::reference_eval(&program, &values)
}
fn run_conditional_mask_with_directives(
    tok_types: &[u32],
    directive_kinds: &[u32],
    directive_values: &[u32],
) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    let program = opt_conditional_mask_with_directives(
        "tok_types",
        "directive_kinds",
        "directive_values",
        "out_mask",
        Expr::u32(tok_types.len() as u32),
    );
    let values = [
        Value::from(u32_bytes(tok_types)),
        Value::from(u32_bytes(directive_kinds)),
        Value::from(u32_bytes(directive_values)),
        Value::from(vec![0u8; tok_types.len() * 4]),
    ];
    vyre_reference::reference_eval(&program, &values)
}
// ---------------------------------------------------------------------------
// 1. Macro table arena contracts
// ---------------------------------------------------------------------------
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__collision_safe_macro_name_hash_matches_fnv1a32.rs"]
mod c_preprocess_gpu_resident_state_contracts_collision_safe_macro_name_hash_matches_fnv1a32;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__collision_safe_macro_name_long_name_exceeds_pool_bounds_fails_loudly.rs"]
mod c_preprocess_gpu_resident_state_contracts_collision_safe_macro_name_long_name_exceeds_pool_bounds_fails_loudly;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__conditional_stack_else_selects_when_nothing_taken.rs"]
mod c_preprocess_gpu_resident_state_contracts_conditional_stack_else_selects_when_nothing_taken;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__conditional_stack_endif_without_open_fails_loudly.rs"]
mod c_preprocess_gpu_resident_state_contracts_conditional_stack_endif_without_open_fails_loudly;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__conditional_stack_open_increments_depth_and_sets_active_bit.rs"]
mod c_preprocess_gpu_resident_state_contracts_conditional_stack_open_increments_depth_and_sets_active_bit;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__conditional_stack_unclosed_open_fails_loudly.rs"]
mod c_preprocess_gpu_resident_state_contracts_conditional_stack_unclosed_open_fails_loudly;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__directive_metadata_evaluates_ifdef_truth.rs"]
mod c_preprocess_gpu_resident_state_contracts_directive_metadata_evaluates_ifdef_truth;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__directive_metadata_rejects_span_not_covering_logical_row.rs"]
mod c_preprocess_gpu_resident_state_contracts_directive_metadata_rejects_span_not_covering_logical_row;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__dynamic_macro_table_probe_skips_empty_slots_then_matches.rs"]
mod c_preprocess_gpu_resident_state_contracts_dynamic_macro_table_probe_skips_empty_slots_then_matches;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__expansion_queue_accumulates_warp_base_per_token.rs"]
mod c_preprocess_gpu_resident_state_contracts_expansion_queue_accumulates_warp_base_per_token;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__function_like_macro_arg_arena_preserves_nested_parens.rs"]
mod c_preprocess_gpu_resident_state_contracts_function_like_macro_arg_arena_preserves_nested_parens;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__function_like_macro_parameter_count_overflow_fails_loudly.rs"]
mod c_preprocess_gpu_resident_state_contracts_function_like_macro_parameter_count_overflow_fails_loudly;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__macro_replacement_range_must_be_inside_table_bounds.rs"]
mod c_preprocess_gpu_resident_state_contracts_macro_replacement_range_must_be_inside_table_bounds;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__macro_table_has_exactly_4096_slots.rs"]
mod c_preprocess_gpu_resident_state_contracts_macro_table_has_exactly_4096_slots;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__overflow_function_like_macro_missing_rparen.rs"]
mod c_preprocess_gpu_resident_state_contracts_overflow_function_like_macro_missing_rparen;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__overflow_named_macro_expansion_output_capacity.rs"]
mod c_preprocess_gpu_resident_state_contracts_overflow_named_macro_expansion_output_capacity;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__table_conditional_mask_all_directive_shapes.rs"]
mod c_preprocess_gpu_resident_state_contracts_table_conditional_mask_all_directive_shapes;
#[path = "contract_cases/c_preprocess_gpu_resident_state_contracts__table_line_splice_offset_map_is_monotonic.rs"]
mod c_preprocess_gpu_resident_state_contracts_table_line_splice_offset_map_is_monotonic;
