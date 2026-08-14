//! Named macro-expansion dispatch builder.

#![allow(missing_docs)] // Internal macro-expansion helpers are documented at the owning module boundary.
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program};

use super::fnlike::*;
use super::named_dispatch::*;
use super::objlike::*;
use super::output_token::*;
use super::*;

pub fn opt_named_macro_expansion(
    in_tok_types: &str,
    in_tok_starts: &str,
    in_tok_lens: &str,
    source_words: &str,
    macro_name_hashes: &str,
    macro_name_starts: &str,
    macro_name_lens: &str,
    macro_name_words: &str,
    macro_vals: &str,
    macro_sizes: &str,
    macro_kinds: &str,
    macro_param_counts: &str,
    macro_replacement_params: &str,
    out_tok_types: &str,
    out_tok_counts: &str,
    num_tokens: Expr,
    source_len: Expr,
    max_out_tokens: u32,
) -> Program {
    let tok_count = match &num_tokens {
        Expr::LitU32(n) => *n,
        _ => 1,
    };
    let tok_buffer_count = tok_count.max(1);
    let source_count = match &source_len {
        Expr::LitU32(n) => *n,
        _ => 1,
    }
    .max(1);
    let out_buffer_count = max_out_tokens.max(1);

    let process_current = emit_named_macro_dispatch(NamedMacroDispatchSpec {
        scan: NamedMacroScanSpec {
            in_tok_types,
            in_tok_starts,
            in_tok_lens,
            source_words,
            source_layout: MacroByteLayout::ExpandedU32,
            macro_name_hashes,
            macro_name_starts,
            macro_name_lens,
            macro_name_words,
            macro_name_layout: MacroByteLayout::ExpandedU32,
            macro_vals,
            macro_kinds,
            macro_param_counts,
            source_len: source_len.clone(),
            decode_variadic_param_count: false,
        },
        macro_sizes,
        num_tokens: num_tokens.clone(),
        unknown_passthrough: emit_one_output_token(
            out_tok_types,
            Expr::var("named_tok"),
            max_out_tokens,
        ),
        object_like: emit_object_like_replacement(
            macro_vals,
            macro_replacement_params,
            out_tok_types,
            max_out_tokens,
        ),
        function_name_passthrough: emit_one_output_token(
            out_tok_types,
            Expr::var("named_tok"),
            max_out_tokens,
        ),
        function_like: emit_function_like_replacement(
            in_tok_types,
            macro_vals,
            macro_replacement_params,
            out_tok_types,
            "macro_arg_starts",
            "macro_arg_ends",
            num_tokens.clone(),
            max_out_tokens,
        ),
    });

    Program::wrapped(
        vec![
            BufferDecl::storage(in_tok_types, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(tok_buffer_count),
            BufferDecl::storage(in_tok_starts, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(tok_buffer_count),
            BufferDecl::storage(in_tok_lens, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(tok_buffer_count),
            BufferDecl::storage(source_words, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(source_count),
            BufferDecl::storage(macro_name_hashes, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(macro_name_starts, 5, BufferAccess::ReadOnly, DataType::U32)
                .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(macro_name_lens, 6, BufferAccess::ReadOnly, DataType::U32)
                .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(macro_name_words, 7, BufferAccess::ReadOnly, DataType::U32)
                .with_count(0),
            BufferDecl::storage(macro_vals, 8, BufferAccess::ReadOnly, DataType::U32)
                .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(macro_sizes, 9, BufferAccess::ReadOnly, DataType::U32)
                .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(macro_kinds, 10, BufferAccess::ReadOnly, DataType::U32)
                .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(
                macro_param_counts,
                11,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(
                macro_replacement_params,
                12,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(MACRO_TABLE_SLOTS),
            BufferDecl::storage(out_tok_types, 13, BufferAccess::ReadWrite, DataType::U32)
                .with_count(out_buffer_count),
            BufferDecl::storage(out_tok_counts, 14, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
            BufferDecl::workgroup("macro_arg_starts", tok_buffer_count, DataType::U32),
            BufferDecl::workgroup("macro_arg_ends", tok_buffer_count, DataType::U32),
        ],
        [1, 1, 1],
        vec![emit_named_expansion_driver(NamedExpansionDriverSpec {
            op_id: "vyre-libs::parsing::opt_named_macro_expansion",
            body: process_current,
            num_tokens,
            out_tok_counts,
            out_source_counts: None,
        })],
    )
    .with_entry_op_id("vyre-libs::parsing::opt_named_macro_expansion")
    .with_non_composable_with_self(true)
}
