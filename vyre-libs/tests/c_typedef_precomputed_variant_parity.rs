//! Differential parity for the PRECOMPUTED-context/scope VAST annotation +
//! classification variants against their (already-covered) reference builders.
//!
//! The typedef-annotation and node-classification passes each ship a slow
//! self-contained reference builder (`c11_annotate_typedef_names`,
//! `c11_classify_vast_node_kinds`) and faster variants that consume a
//! separately-precomputed side table:
//!   * `c11_precompute_vast_scopes`  writes each node's enclosing brace scope
//!     into `VAST_TYPEDEF_SCOPE_FIELD`; `c11_annotate_typedef_names_precomputed_scope`
//!     then READS that field instead of re-walking the brace stack per row.
//!   * `c11_precompute_vast_decl_contexts` writes a per-node declaration-context
//!     table; `..._precomputed_context` / `c11_classify_vast_node_kinds_precomputed_context`
//!     read it instead of re-scanning.
//! Each variant is a SEMANTICS-PRESERVING optimization: fed the correct
//! precomputed table, it must produce byte-identical output to the reference
//! builder. Nothing pinned that contract, these variants were orphan builders
//! (registry-coverage closure gate `adversarial_registry_closure.rs`). This test
//! chains `precompute -> variant` through `reference_eval` and asserts equality
//! with the reference, draining the whole precomputed-variant cluster with real
//! output-byte assertions (Testing-Contract: truth, not `!is_empty`).
#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod common;

use common::decode_u32_words as words_from_bytes;
use common::u32_bytes as bytes;
use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    c11_annotate_typedef_names, c11_annotate_typedef_names_precomputed_context,
    c11_annotate_typedef_names_precomputed_context_packed_haystack,
    c11_annotate_typedef_names_precomputed_scope,
    c11_annotate_typedef_names_precomputed_scope_packed_haystack, c11_build_expression_shape_nodes,
    c11_build_expression_shape_nodes_no_conditional,
    c11_classify_annotated_vast_node_kinds_precomputed_context, c11_classify_vast_node_kinds,
    c11_classify_vast_node_kinds_precomputed_context, c11_link_vast_typedef_symbols,
    c11_precompute_vast_decl_contexts, c11_precompute_vast_decl_prefix_starts,
    c11_precompute_vast_scopes, c11_precompute_vast_visible_type,
    c11_precompute_vast_visible_type_packed_haystack, c11_prehash_vast_identifiers,
};
use vyre_reference::value::Value;

// VAST wire layout (mirrors the private constants in parse/vast/mod.rs; these are
// stable on-wire field offsets, hardcoded the same way the sibling packed-haystack
// parity test hardcodes VAST_NODE_STRIDE_U32).
const VAST_NODE_STRIDE_U32: usize = 10;
const VAST_DECL_CONTEXT_STRIDE_U32: usize = 4;
const SENTINEL: u32 = u32::MAX;

/// `typedef int foo; { foo bar; }`: a typedef declaration, a brace scope, and a
/// reuse of the typedef name inside the scope. Exercises the scope walker (braces)
/// AND the declaration-context table (typedef declared, then used as a type).
fn fixture() -> (&'static [u8], Vec<(u32, u32, u32)>) {
    let source: &[u8] = b"typedef int foo; { foo bar; }";
    let tokens = vec![
        (TOK_TYPEDEF, 0u32, 7u32),
        (TOK_INT, 8, 3),
        (TOK_IDENTIFIER, 12, 3), // foo (declared)
        (TOK_SEMICOLON, 15, 1),
        (TOK_LBRACE, 17, 1),
        (TOK_IDENTIFIER, 19, 3), // foo (used as type)
        (TOK_IDENTIFIER, 23, 3), // bar (declarator)
        (TOK_SEMICOLON, 26, 1),
        (TOK_RBRACE, 28, 1),
    ];
    (source, tokens)
}

fn build_vast(tokens: &[(u32, u32, u32)]) -> Vec<u32> {
    let mut vast = vec![0u32; tokens.len() * VAST_NODE_STRIDE_U32];
    for (idx, (kind, start, len)) in tokens.iter().copied().enumerate() {
        let base = idx * VAST_NODE_STRIDE_U32;
        vast[base] = kind;
        vast[base + 1] = SENTINEL;
        vast[base + 2] = SENTINEL;
        vast[base + 3] = SENTINEL;
        vast[base + 4] = idx.saturating_sub(1) as u32;
        vast[base + 5] = start;
        vast[base + 6] = len;
        // fields 7 (flags), 8 (scope), 9 (symbol) start zeroed.
    }
    vast
}

fn expanded_haystack(source: &[u8]) -> Vec<u8> {
    bytes(
        &source
            .iter()
            .map(|byte| u32::from(*byte))
            .collect::<Vec<_>>(),
    )
}

fn packed_haystack(source: &[u8]) -> Vec<u8> {
    let mut packed = vec![0u8; source.len().max(1).div_ceil(4) * 4];
    packed[..source.len()].copy_from_slice(source);
    packed
}

fn eval_words(program: &vyre::ir::Program, inputs: Vec<Vec<u8>>) -> Vec<u32> {
    let values = inputs.into_iter().map(Value::from).collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(program, &values)
        .expect("precomputed-variant parity program must execute under reference_eval");
    words_from_bytes(&outputs[0].to_bytes())
}

/// Run `c11_prehash_vast_identifiers` and return the VAST with each IDENTIFIER row's
/// symbol-hash field (field 9) populated from the source. The precomputed-context
/// declaration chain keys on this hash, so it must be filled before decl-context
/// precompute (the base annotator re-hashes inline, so it needs no prepass).
fn prehash(vast: &[u32], source: &[u8], n: u32) -> Vec<u32> {
    let program = c11_prehash_vast_identifiers(
        "vast_nodes",
        "haystack",
        Expr::u32(source.len() as u32),
        Expr::u32(n),
        "out_hashed",
    );
    eval_words(
        &program,
        vec![
            bytes(vast),
            expanded_haystack(source),
            vec![0u8; vast.len() * 4],
        ],
    )
}

/// Run `c11_precompute_vast_scopes` and return the scope-populated VAST buffer.
fn precompute_scopes(vast: &[u32], n: u32) -> Vec<u32> {
    let program = c11_precompute_vast_scopes("vast_nodes", Expr::u32(n), "out_scoped");
    eval_words(
        &program,
        vec![
            bytes(vast),
            vec![0u8; vast.len() * 4],            // out_scoped (n * stride words)
            vec![0u8; n.max(1) as usize * 4],     // __vast_scope_stack scratch (n words)
        ],
    )
}

// VAST node field offsets (mirrors parse/vast/mod.rs private consts).
const VAST_TYPEDEF_FLAGS_FIELD: usize = 7;
const VAST_TYPEDEF_SYMBOL_FIELD: usize = 9;
// VAST decl-context table field offsets.
const VAST_DECL_CONTEXT_PREFIX_START_FIELD: usize = 0;

/// Run `c11_precompute_vast_decl_contexts` and return the decl-context table.
fn precompute_decl_contexts(vast: &[u32], n: u32) -> Vec<u32> {
    let program = c11_precompute_vast_decl_contexts("vast_nodes", Expr::u32(n), "out_decl_contexts");
    eval_words(
        &program,
        vec![
            bytes(vast),
            vec![0u8; n.max(1) as usize * VAST_DECL_CONTEXT_STRIDE_U32 * 4],
        ],
    )
}

/// Run `c11_precompute_vast_visible_type` (or its packed-haystack variant) over the
/// decl-context table and return the per-node visible-type bit vector (`n` words).
fn precompute_visible_type(
    vast: &[u32],
    haystack: Vec<u8>,
    haystack_len: u32,
    decl_contexts: &[u32],
    n: u32,
    packed: bool,
) -> Vec<u32> {
    let program = if packed {
        c11_precompute_vast_visible_type_packed_haystack(
            "vast_nodes",
            "haystack",
            "decl_contexts",
            Expr::u32(haystack_len),
            Expr::u32(n),
            "out_visible_type",
        )
    } else {
        c11_precompute_vast_visible_type(
            "vast_nodes",
            "haystack",
            "decl_contexts",
            Expr::u32(haystack_len),
            Expr::u32(n),
            "out_visible_type",
        )
    };
    eval_words(
        &program,
        vec![
            bytes(vast),
            haystack,
            bytes(decl_contexts),
            vec![0u8; n.max(1) as usize * 4], // out_visible_type (one word per node)
        ],
    )
}

fn reference_annotation(vast: &[u32], source: &[u8], n: u32) -> Vec<u32> {
    let program = c11_annotate_typedef_names(
        "vast_nodes",
        "haystack",
        Expr::u32(source.len() as u32),
        Expr::u32(n),
        "annotated",
    );
    eval_words(
        &program,
        vec![
            bytes(vast),
            expanded_haystack(source),
            vec![0u8; vast.len() * 4],
        ],
    )
}

#[test]
fn precomputed_scope_annotation_matches_reference() {
    let (source, tokens) = fixture();
    let n = tokens.len() as u32;
    let vast = build_vast(&tokens);
    let reference = reference_annotation(&vast, source, n);

    let scoped = precompute_scopes(&vast, n);
    assert_eq!(
        scoped.len(),
        vast.len(),
        "precompute_scopes must return the full scope-populated VAST buffer"
    );

    // Expanded-haystack precomputed-scope variant.
    let variant = eval_words(
        &c11_annotate_typedef_names_precomputed_scope(
            "vast_nodes",
            "haystack",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "annotated",
        ),
        vec![
            bytes(&scoped),
            expanded_haystack(source),
            vec![0u8; vast.len() * 4],
        ],
    );
    assert_eq!(
        variant, reference,
        "precomputed-scope annotation must match the self-contained reference walker"
    );

    // Packed-haystack precomputed-scope variant.
    let variant_packed = eval_words(
        &c11_annotate_typedef_names_precomputed_scope_packed_haystack(
            "vast_nodes",
            "haystack",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "annotated",
        ),
        vec![
            bytes(&scoped),
            packed_haystack(source),
            vec![0u8; vast.len() * 4],
        ],
    );
    assert_eq!(
        variant_packed, reference,
        "packed precomputed-scope annotation must match the reference walker"
    );
}

#[test]
fn precomputed_context_annotation_matches_reference() {
    // The precomputed-CONTEXT typedef ANNOTATION variants read the decl-context table
    // AND the per-node visible-type side table (`c11_precompute_vast_visible_type`)
    // instead of re-resolving typedef-name-as-type inline per prefix row. Fed the
    // correct precomputed tables, each must be byte-identical to the self-contained
    // reference annotator. This previously diverged: the haystack-free prefix scan
    // matched only builtin type KEYWORDS, so a declarator whose type is a typedef-name
    // (`{ foo bar; }` in the fixture, where `foo` is `typedef int foo`) lost its
    // ordinary-declarator flag. The visible-type precompute closes that gap
    // (BACKLOG.md WIRING-vyrelibs-precomputed-context-divergence, now resolved).
    let (source, tokens) = fixture();
    let n = tokens.len() as u32;
    let vast = build_vast(&tokens);
    let reference = reference_annotation(&vast, source, n);

    // The precomputed-context path consumes a FULLY prepared VAST: the base annotator
    // re-hashes identifiers and walks brace scopes inline, but the precomputed variant
    // reads those from side tables. Build the real pipeline:
    //   prehash (field 9 symbol hashes) -> scopes (field 8 enclosing scope)
    //   -> decl_contexts (per-node declaration chain) -> visible_type (typedef-name bit).
    let hashed = prehash(&vast, source, n);
    let prepared = precompute_scopes(&hashed, n);
    let decl_contexts = precompute_decl_contexts(&prepared, n);
    let visible_type = precompute_visible_type(
        &prepared,
        expanded_haystack(source),
        source.len() as u32,
        &decl_contexts,
        n,
        false,
    );
    assert_eq!(
        visible_type.len(),
        tokens.len(),
        "visible-type precompute writes one bit per node"
    );
    // The typedef-name reused as a type inside the brace scope (node 5, `foo`) must be
    // flagged as a visible type; the plain declarator (`bar`) must not. This is the
    // exact signal the precomputed declaration-kind path was missing.
    assert_eq!(
        visible_type[5], 1,
        "the reused typedef-name `foo` (node 5) must resolve as a visible type"
    );
    assert_eq!(
        visible_type[6], 0,
        "the ordinary declarator `bar` (node 6) is not itself a type-name"
    );

    // The bit table is a pure function of source content, independent of haystack
    // packing (the packed precompute pass must produce identical bits).
    let visible_type_packed = precompute_visible_type(
        &prepared,
        packed_haystack(source),
        source.len() as u32,
        &decl_contexts,
        n,
        true,
    );
    assert_eq!(
        visible_type_packed, visible_type,
        "visible-type bits are invariant to haystack packing"
    );

    // Expanded-haystack precomputed-context annotation.
    let variant = eval_words(
        &c11_annotate_typedef_names_precomputed_context(
            "vast_nodes",
            "haystack",
            "decl_contexts",
            "visible_type",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "annotated",
        ),
        vec![
            bytes(&prepared),
            expanded_haystack(source),
            bytes(&decl_contexts),
            bytes(&visible_type),
            vec![0u8; vast.len() * 4],
        ],
    );
    assert_eq!(
        variant, reference,
        "precomputed-context annotation must match the self-contained reference annotator"
    );

    // Packed-haystack precomputed-context annotation.
    let variant_packed = eval_words(
        &c11_annotate_typedef_names_precomputed_context_packed_haystack(
            "vast_nodes",
            "haystack",
            "decl_contexts",
            "visible_type",
            Expr::u32(source.len() as u32),
            Expr::u32(n),
            "annotated",
        ),
        vec![
            bytes(&prepared),
            packed_haystack(source),
            bytes(&decl_contexts),
            bytes(&visible_type_packed),
            vec![0u8; vast.len() * 4],
        ],
    );
    assert_eq!(
        variant_packed, reference,
        "packed precomputed-context annotation must match the reference annotator"
    );
}

#[test]
fn precomputed_context_classification_matches_reference() {
    let (_source, tokens) = fixture();
    let n = tokens.len() as u32;
    let vast = build_vast(&tokens);

    let reference = eval_words(
        &c11_classify_vast_node_kinds("vast_nodes", Expr::u32(n), "typed"),
        vec![bytes(&vast), vec![0u8; vast.len() * 4]],
    );
    let decl_contexts = precompute_decl_contexts(&vast, n);

    let variant = eval_words(
        &c11_classify_vast_node_kinds_precomputed_context(
            "vast_nodes",
            "decl_contexts",
            Expr::u32(n),
            "typed",
        ),
        vec![bytes(&vast), bytes(&decl_contexts), vec![0u8; vast.len() * 4]],
    );
    assert_eq!(
        variant, reference,
        "precomputed-context classification must match the self-contained reference classifier"
    );
}

#[test]
fn annotated_classify_precomputed_context_degrades_to_plain_on_unannotated_input() {
    // `c11_classify_annotated_vast_node_kinds_precomputed_context` is the classify variant
    // that ALSO consumes typedef annotations (`typedef_annotations_available=true`), vs the
    // (already-drained) `c11_classify_vast_node_kinds_precomputed_context` which does not.
    // On UNANNOTATED VAST (build_vast leaves the flags/scope/symbol fields zero), the
    // annotation-consuming paths have no annotations to act on, so the two must produce
    // byte-identical classification. Pins that degrade-to-base contract, draining the
    // annotated variant.
    let (_source, tokens) = fixture();
    let n = tokens.len() as u32;
    let vast = build_vast(&tokens); // unannotated: fields 7/8/9 == 0
    let decl_contexts = precompute_decl_contexts(&vast, n);

    let plain = eval_words(
        &c11_classify_vast_node_kinds_precomputed_context(
            "vast_nodes",
            "decl_contexts",
            Expr::u32(n),
            "typed",
        ),
        vec![bytes(&vast), bytes(&decl_contexts), vec![0u8; vast.len() * 4]],
    );
    let annotated = eval_words(
        &c11_classify_annotated_vast_node_kinds_precomputed_context(
            "vast_nodes",
            "decl_contexts",
            Expr::u32(n),
            "typed",
        ),
        vec![bytes(&vast), bytes(&decl_contexts), vec![0u8; vast.len() * 4]],
    );
    assert_eq!(
        annotated, plain,
        "on unannotated VAST the annotated classify variant must equal the plain \
         precomputed-context classify (no annotations to consume)"
    );
}

#[test]
fn expression_shape_no_conditional_matches_base_on_non_ternary_input() {
    // `c11_build_expression_shape_nodes_no_conditional` differs from the base only in
    // that it SKIPS the ternary/conditional-boundary handling. On input with NO
    // conditional (`?:`) operator that handling has nothing to act on, so the two
    // builders must emit byte-identical expression-shape nodes. Fixture `a = b` (a
    // binary assignment, no ternary). Drains the reduced-shape variant.
    const EXPR_SHAPE_STRIDE: usize = 8; // C_EXPR_SHAPE_STRIDE_U32
    let tokens = vec![
        (TOK_IDENTIFIER, 0u32, 1u32),
        (TOK_ASSIGN, 2, 1),
        (TOK_IDENTIFIER, 4, 1),
    ];
    let n = tokens.len() as u32;
    let raw = build_vast(&tokens);
    // expr-shape reads BOTH raw + typed node kinds; build the typed VAST via the classifier.
    let typed = eval_words(
        &c11_classify_vast_node_kinds("raw", Expr::u32(n), "typed"),
        vec![bytes(&raw), vec![0u8; raw.len() * 4]],
    );
    let out_init = vec![0u8; tokens.len() * EXPR_SHAPE_STRIDE * 4];

    let base = eval_words(
        &c11_build_expression_shape_nodes("raw", "typed", Expr::u32(n), "out"),
        vec![bytes(&raw), bytes(&typed), out_init.clone()],
    );
    let no_cond = eval_words(
        &c11_build_expression_shape_nodes_no_conditional("raw", "typed", Expr::u32(n), "out"),
        vec![bytes(&raw), bytes(&typed), out_init],
    );
    assert_eq!(
        base.len(),
        tokens.len() * EXPR_SHAPE_STRIDE,
        "expr-shape output is C_EXPR_SHAPE_STRIDE_U32 words per node"
    );
    assert_eq!(
        no_cond, base,
        "on non-ternary input the no-conditional expr-shape variant must equal the base"
    );
}

#[test]
fn typedef_symbol_link_chains_repeated_identifiers_by_hash() {
    // `c11_link_vast_typedef_symbols` builds a per-symbol back-link chain: for each
    // IDENTIFIER row (nonzero symbol hash) whose NEXT token is a link-follower
    // (`; , = ( [ : ) ]`), it writes into the FLAGS field the previous occurrence of
    // the same hash: SENTINEL for the chain head, else (prev_row_index + 1). Fixture
    // `foo; foo;`: two IDENTIFIER rows sharing a hash, each followed by `;`.
    let hash = 0xABCDu32;
    let tokens = [
        (TOK_IDENTIFIER, 0u32, 3u32, hash), // foo #0 (chain head)
        (TOK_SEMICOLON, 3, 1, 0),
        (TOK_IDENTIFIER, 5, 3, hash), // foo #1 (same hash -> links back to #0)
        (TOK_SEMICOLON, 8, 1, 0),
    ];
    let n = tokens.len() as u32;
    let mut vast = vec![0u32; tokens.len() * VAST_NODE_STRIDE_U32];
    for (idx, (kind, start, len, h)) in tokens.iter().copied().enumerate() {
        let base = idx * VAST_NODE_STRIDE_U32;
        vast[base] = kind;
        vast[base + 1] = SENTINEL;
        vast[base + 2] = SENTINEL;
        vast[base + 3] = SENTINEL;
        vast[base + 4] = idx.saturating_sub(1) as u32;
        vast[base + 5] = start;
        vast[base + 6] = len;
        vast[base + VAST_TYPEDEF_SYMBOL_FIELD] = h;
    }

    let linked = eval_words(
        &c11_link_vast_typedef_symbols("vast_nodes", Expr::u32(n), "out_linked"),
        vec![bytes(&vast), vec![0u8; vast.len() * 4]],
    );
    let flags: Vec<u32> = (0..tokens.len())
        .map(|i| linked[i * VAST_NODE_STRIDE_U32 + VAST_TYPEDEF_FLAGS_FIELD])
        .collect();
    assert_eq!(
        flags,
        vec![SENTINEL, 0, 1, 0],
        "foo#0 = chain head (SENTINEL); foo#1 back-links to row 0 (encoded 0+1=1); \
         non-identifier rows carry no link (0)"
    );
    // Every non-FLAGS field must pass through unchanged (only FLAGS is rewritten).
    for (idx, (kind, start, len, h)) in tokens.iter().copied().enumerate() {
        let base = idx * VAST_NODE_STRIDE_U32;
        assert_eq!(linked[base], kind, "kind preserved at row {idx}");
        assert_eq!(linked[base + 5], start, "start preserved at row {idx}");
        assert_eq!(linked[base + 6], len, "len preserved at row {idx}");
        assert_eq!(
            linked[base + VAST_TYPEDEF_SYMBOL_FIELD], h,
            "symbol hash preserved at row {idx}"
        );
    }
}

#[test]
fn standalone_decl_prefix_starts_match_full_decl_context_field() {
    // `c11_precompute_vast_decl_prefix_starts` is the standalone pass that fills ONLY
    // the PREFIX_START field of the decl-context table (the declaration-prefix
    // reset-token backscan); `c11_precompute_vast_decl_contexts` fills the same field
    // as part of its fuller table. They MUST agree node-for-node on PREFIX_START, the
    // standalone pass is a parallelizable extract of the same computation, and a
    // divergence would silently corrupt the precomputed-context declaration logic.
    let (_source, tokens) = fixture();
    let n = tokens.len() as u32;
    let vast = build_vast(&tokens);

    let full = precompute_decl_contexts(&vast, n);
    let prefix_only = eval_words(
        &c11_precompute_vast_decl_prefix_starts("vast_nodes", Expr::u32(n), "out_decl_contexts"),
        vec![
            bytes(&vast),
            vec![0u8; tokens.len() * VAST_DECL_CONTEXT_STRIDE_U32 * 4],
        ],
    );
    assert_eq!(
        prefix_only.len(),
        tokens.len() * VAST_DECL_CONTEXT_STRIDE_U32,
        "prefix-start pass writes the full decl-context-strided table"
    );
    for node in 0..tokens.len() {
        let field = node * VAST_DECL_CONTEXT_STRIDE_U32 + VAST_DECL_CONTEXT_PREFIX_START_FIELD;
        assert_eq!(
            prefix_only[field], full[field],
            "PREFIX_START at node {node} must match the full decl-context pass"
        );
    }
    // At least one reset boundary must appear (the ';' after the typedef declaration
    // resets the declaration prefix), else the fixture wouldn't exercise the reset path.
    let starts: Vec<u32> = (0..tokens.len())
        .map(|node| prefix_only[node * VAST_DECL_CONTEXT_STRIDE_U32])
        .collect();
    assert!(
        starts.iter().any(|&s| s != 0),
        "the typedef declaration's terminating ';' must advance the prefix start past 0, got {starts:?}"
    );
}
