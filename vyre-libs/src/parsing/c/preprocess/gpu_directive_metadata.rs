//! GPU equivalent of `reference_c_preprocessor_directive_metadata`.
//!
//! Classifies every `TOK_PREPROC` token's directive kind on the GPU. For
//! tokens of any other type the output rows are zero-filled.
//!
//! ## Phase split (matches the v0.4 plan)
//!
//! - **17a (this file, today):** directive kind classification only  -
//!   walks the directive row source bytes per token, skips horizontal
//!   whitespace, expects `#`, reads the directive keyword, byte-compares
//!   against the 16 known names, emits the matched `TOK_PP_*` constant.
//!   Conditional value (`directive_values`) stays 0 for every token.
//!
//! - **17b (follow-up):** conditional-expression evaluator (`#if`,
//!   `#elif`, `#ifdef`, `#ifndef`) ported from the recursive-descent CPU
//!   parser to an iterative shunting-yard kernel that uses fixed-size
//!   per-thread operand and operator stacks. Lands as
//!   `gpu_conditional_value` in the same module.
//!
//! Both phases share this same kernel-input shape so callers do not have
//! to re-wire when 17b ships.
//!
//! ## Wire layout
//!
//! Inputs:
//!   - `tok_types` (U32)  -  token-kind id per token.
//!   - `tok_starts` (U32)  -  byte offset into `source` per token.
//!   - `tok_lens` (U32)  -  byte length per token (excludes any phase-2
//!     splices but includes the row's terminating newline).
//!   - `source`  -  original source bytes. [`gpu_directive_metadata`] keeps
//!     the packed `U32` ABI used by standalone preprocess kernels;
//!     [`gpu_directive_metadata_u8`](crate::parsing::c::preprocess::gpu_directive_metadata::gpu_directive_metadata_u8) consumes one raw `U8` element per byte for
//!     the resident preprocessing pipeline.
//!
//! Outputs:
//!   - `directive_kinds` (U32)  -  `TOK_PP_*` constant for `TOK_PREPROC`
//!     tokens; `0` for all other token types.
//!   - `directive_values` (U32)  -  conditional value (0/1). Always 0 in
//!     17a; populated by 17b's evaluator.
//!
//! Workgroup size is fixed at 256.
//!
//! ## Real-GPU lowering note
//!
//! vyre-lower's region-scope phi-merge drops nested-scope assigns to
//! outer-scope mutables (Q7 carrier-seed family bug  -  see
//! `vyre-q7-carrier-seed-bug.md`). The earlier loop-and-mutable
//! formulation of this kernel was correct under reference-eval but
//! returned `0` for every `TOK_PREPROC` token on real GPU because the
//! `hash_idx` / `kw_len` / `kind_out` outer-scope assigns inside the
//! hash-scan / kw-read loop bodies did not propagate back through the
//! WGSL phi-merge.
//!
//! This implementation uses **only** straight-line `let_bind` chains
//! and direct buffer stores  -  no loops, no outer-scope mutables. Every
//! intermediate value is bound once and read by name; every output is
//! a `Node::store` directly inside whatever conditional fires it.
//! The one mutability is the output buffer cell, which is pre-stored
//! to `0` and conditionally overwritten by exactly the matching
//! keyword arm (matches are mutually exclusive by length+content).

use super::gpu_directive_parse_shared::{
    keyword_match_expr, push_found_hash, push_hash_scan, push_keyword_bytes, push_keyword_start,
    source_buffer_element, DirectiveSourceLayout,
};
use crate::parsing::c::lex::tokens::{
    TOK_PP_DEFINE, TOK_PP_ELIF, TOK_PP_ELSE, TOK_PP_ENDIF, TOK_PP_ERROR, TOK_PP_IDENT, TOK_PP_IF,
    TOK_PP_IFDEF, TOK_PP_IFNDEF, TOK_PP_INCLUDE, TOK_PP_INCLUDE_NEXT, TOK_PP_LINE, TOK_PP_NULL,
    TOK_PP_PRAGMA, TOK_PP_SCCS, TOK_PP_UNDEF, TOK_PP_WARNING, TOK_PREPROC,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::parsing::c::preprocess::gpu_directive_metadata";

/// Canonical binding index for the input token-kind buffer.
pub const BINDING_TOK_TYPES: u32 = 0;
/// Canonical binding index for the input per-token byte-offset buffer.
pub const BINDING_TOK_STARTS: u32 = 1;
/// Canonical binding index for the input per-token byte-length buffer.
pub const BINDING_TOK_LENS: u32 = 2;
/// Canonical binding index for the input source-bytes buffer.
pub const BINDING_SOURCE: u32 = 3;
/// Canonical binding index for the output `directive_kinds` buffer.
pub const BINDING_DIRECTIVE_KINDS: u32 = 4;
/// Canonical binding index for the output `directive_values` buffer
/// (always zero-filled in 17a; populated by the 17b conditional
/// evaluator).
pub const BINDING_DIRECTIVE_VALUES: u32 = 5;

/// Maximum directive keyword length (`include_next` is the longest at 12
/// bytes). The kernel only inspects the first this many bytes after `#`
/// when classifying.
pub const MAX_KEYWORD_LEN: u32 = 12;



/// Build the 17a directive-classification `Program` over packed `DataType::U32`
/// source words.
///
/// Hybrid runtime/static-bound: kernel BODY uses `Expr::buf_len()` for
/// per-thread bounds and `safe_load`, `num_tokens` is kept ONLY for
/// output buffer sizing, `source_len` is unused.
#[must_use]
pub fn gpu_directive_metadata(num_tokens: u32, source_len: u32) -> Program {
    gpu_directive_metadata_with_source_layout(
        num_tokens,
        source_len,
        DirectiveSourceLayout::PackedU32,
    )
}

/// Build the 17a directive-classification `Program` over raw `DataType::U8`
/// source bytes.
#[must_use]
pub fn gpu_directive_metadata_u8(num_tokens: u32, source_len: u32) -> Program {
    gpu_directive_metadata_with_source_layout(num_tokens, source_len, DirectiveSourceLayout::RawU8)
}

fn gpu_directive_metadata_with_source_layout(
    num_tokens: u32,
    source_len: u32,
    source_layout: DirectiveSourceLayout,
) -> Program {
    let _ = source_len;
    let t = Expr::var("t");

    // ---- per-thread classify body (loop-free, mutation-free) ----
    //
    // Directive-line scan stages come from `gpu_directive_parse_shared`; this
    // builder only chooses the binding namespace and the stage order. Reading
    // the keyword bytes before the found-hash flag is this classifier's own
    // ordering and is deliberately not centralized.
    let mut classify: Vec<Node> = Vec::new();
    classify.push(Node::let_bind(
        "tok_start",
        Expr::load("tok_starts", t.clone()),
    ));
    push_hash_scan(&mut classify, source_layout, "s");
    push_keyword_start(&mut classify, source_layout, "p");
    push_keyword_bytes(&mut classify, source_layout, MAX_KEYWORD_LEN);
    push_found_hash(&mut classify);

    // Per-keyword stores. Each `if` is mutually exclusive with every
    // other (same first byte → different lengths or different later
    // bytes), so at most one fires per token.
    //
    // Null directive: `#` followed by no ident-continue byte. Fires
    // when `k_is_continue_0 == 0`. Other keywords all require
    // `k_is_continue_0 == 1`, so they don't conflict with null.
    let store_kind = |kind: u32| -> Vec<Node> {
        vec![Node::store("directive_kinds", t.clone(), Expr::u32(kind))]
    };
    let fire = |cond_u32: Expr, kind: u32| -> Node {
        // Both `found_hash` and `cond_u32` are u32 0/1; bitand stays
        // u32. Convert to bool for if_then via `eq u32(1)`.
        Node::if_then(
            Expr::eq(
                Expr::bitand(Expr::var("found_hash"), cond_u32),
                Expr::u32(1),
            ),
            store_kind(kind),
        )
    };

    // Null directive (kw_len == 0).
    classify.push(fire(
        Expr::select(
            Expr::eq(Expr::var("k_is_continue_0"), Expr::u32(0)),
            Expr::u32(1),
            Expr::u32(0),
        ),
        TOK_PP_NULL,
    ));

    // Match each known directive. include_next must be checked before
    // include because both share a 7-byte prefix; the trailing-byte
    // ident-continue check on include's k_7 ensures `include_next`
    // doesn't accidentally fire `include` (k_7 = '_' which IS
    // ident-continue, so `include` matches only when k_7 is NOT).
    classify.push(fire(
        keyword_match_expr(&[100, 101, 102, 105, 110, 101]),
        TOK_PP_DEFINE,
    ));
    classify.push(fire(
        keyword_match_expr(&[117, 110, 100, 101, 102]),
        TOK_PP_UNDEF,
    ));
    classify.push(fire(
        keyword_match_expr(&[105, 110, 99, 108, 117, 100, 101, 95, 110, 101, 120, 116]),
        TOK_PP_INCLUDE_NEXT,
    ));
    classify.push(fire(
        keyword_match_expr(&[105, 110, 99, 108, 117, 100, 101]),
        TOK_PP_INCLUDE,
    ));
    classify.push(fire(
        keyword_match_expr(&[105, 102, 110, 100, 101, 102]),
        TOK_PP_IFNDEF,
    ));
    classify.push(fire(
        keyword_match_expr(&[105, 102, 100, 101, 102]),
        TOK_PP_IFDEF,
    ));
    classify.push(fire(keyword_match_expr(&[105, 102]), TOK_PP_IF));
    classify.push(fire(keyword_match_expr(&[101, 108, 105, 102]), TOK_PP_ELIF));
    classify.push(fire(keyword_match_expr(&[101, 108, 115, 101]), TOK_PP_ELSE));
    classify.push(fire(
        keyword_match_expr(&[101, 110, 100, 105, 102]),
        TOK_PP_ENDIF,
    ));
    classify.push(fire(
        keyword_match_expr(&[112, 114, 97, 103, 109, 97]),
        TOK_PP_PRAGMA,
    ));
    classify.push(fire(keyword_match_expr(&[108, 105, 110, 101]), TOK_PP_LINE));
    classify.push(fire(
        keyword_match_expr(&[101, 114, 114, 111, 114]),
        TOK_PP_ERROR,
    ));
    classify.push(fire(
        keyword_match_expr(&[119, 97, 114, 110, 105, 110, 103]),
        TOK_PP_WARNING,
    ));
    classify.push(fire(
        keyword_match_expr(&[105, 100, 101, 110, 116]),
        TOK_PP_IDENT,
    ));
    classify.push(fire(keyword_match_expr(&[115, 99, 99, 115]), TOK_PP_SCCS));

    // ---- per-thread top-level body ----
    let body: Vec<Node> = vec![
        Node::let_bind("t", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(t.clone(), Expr::buf_len("tok_starts")),
            vec![
                Node::let_bind("tok_type", Expr::load("tok_types", t.clone())),
                // Pre-zero output cells. Classify path conditionally
                // overwrites `directive_kinds`; `directive_values` is
                // populated by the 17b evaluator.
                Node::store("directive_kinds", t.clone(), Expr::u32(0)),
                Node::store("directive_values", t.clone(), Expr::u32(0)),
                Node::if_then(
                    Expr::eq(Expr::var("tok_type"), Expr::u32(TOK_PREPROC)),
                    classify,
                ),
            ],
        ),
    ];

    let source_element = source_buffer_element(source_layout);

    Program::wrapped(
        vec![
            BufferDecl::storage(
                "tok_types",
                BINDING_TOK_TYPES,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(num_tokens.max(1)),
            BufferDecl::storage(
                "tok_starts",
                BINDING_TOK_STARTS,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(num_tokens.max(1)),
            BufferDecl::storage(
                "tok_lens",
                BINDING_TOK_LENS,
                BufferAccess::ReadOnly,
                DataType::U32,
            )
            .with_count(num_tokens.max(1)),
            BufferDecl::storage(
                "source",
                BINDING_SOURCE,
                BufferAccess::ReadOnly,
                source_element,
            )
            .with_count(0),
            BufferDecl::storage(
                "directive_kinds",
                BINDING_DIRECTIVE_KINDS,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(num_tokens.max(1)),
            BufferDecl::storage(
                "directive_values",
                BINDING_DIRECTIVE_VALUES,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(num_tokens.max(1)),
        ],
        [256, 1, 1],
        body,
    )
    .with_entry_op_id(OP_ID)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_id_is_canonical_and_stable() {
        assert_eq!(
            OP_ID,
            "vyre-libs::parsing::c::preprocess::gpu_directive_metadata"
        );
    }

    #[test]
    fn binding_indices_are_canonical_and_stable() {
        assert_eq!(BINDING_TOK_TYPES, 0);
        assert_eq!(BINDING_TOK_STARTS, 1);
        assert_eq!(BINDING_TOK_LENS, 2);
        assert_eq!(BINDING_SOURCE, 3);
        assert_eq!(BINDING_DIRECTIVE_KINDS, 4);
        assert_eq!(BINDING_DIRECTIVE_VALUES, 5);
    }

    #[test]
    fn build_program_returns_well_formed_program() {
        let p = gpu_directive_metadata(8, 64);
        assert_eq!(p.buffers().len(), 6);
        assert_eq!(p.workgroup_size(), [256, 1, 1]);
    }

    #[test]
    fn source_buffer_is_runtime_sized_not_source_length_specialized() {
        let p = gpu_directive_metadata(8, 64);
        let source = p
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "source")
            .expect("Fix: source buffer must exist after directive metadata allocation");
        assert_eq!(
            source.count, 0,
            "source must be runtime-sized so one directive classifier program serves all source lengths"
        );
    }

    #[test]
    fn source_buffer_layouts_preserve_packed_abi_and_raw_u8_variant() {
        let packed = gpu_directive_metadata(8, 64);
        let raw_u8 = gpu_directive_metadata_u8(8, 64);
        let packed_source = packed
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "source")
            .expect("Fix: packed directive metadata source buffer must exist");
        let raw_u8_source = raw_u8
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "source")
            .expect("Fix: raw-U8 directive metadata source buffer must exist");

        assert_eq!(packed_source.element(), DataType::U32);
        assert_eq!(packed_source.count, 0);
        assert_eq!(raw_u8_source.element(), DataType::U8);
        assert_eq!(raw_u8_source.count, 0);
    }

    #[test]
    fn max_keyword_len_covers_longest_directive() {
        // include_next is the longest at 12 ASCII bytes.
        assert!(MAX_KEYWORD_LEN >= 12);
    }
}
