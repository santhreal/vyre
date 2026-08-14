//! Shared C-token assembly and Program Graph row assertions for integration tests.

#![allow(deprecated)]

use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::keyword::reference_c_keyword_types;
use vyre_libs::parsing::c::lex::lexer::c11_lexer;
use vyre_libs::parsing::c::lex::tokens::{TOK_COMMENT, TOK_WHITESPACE};
use vyre_libs::parsing::c::lower::{c_lower_ast_to_pg_nodes, reference_ast_to_pg_nodes};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_expression_shape_nodes, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds, C_EXPR_ASSOC_NONE, C_EXPR_SHAPE_NONE,
    C_EXPR_SHAPE_STRIDE_U32,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all, pack_u32_slice};
use vyre_reference::value::Value;

const VAST_STRIDE_U32: usize = 10;
pub(crate) const PG_STRIDE_U32: usize = 6;

/// Token rows assembled from explicit lexeme fixtures.
pub(crate) struct Assembled {
    pub(crate) source: String,
    #[allow(dead_code)]
    pub(crate) raw_kinds: Vec<u32>,
    pub(crate) tok_types: Vec<u32>,
    pub(crate) tok_starts: Vec<u32>,
    pub(crate) tok_lens: Vec<u32>,
}

/// Assemble non-whitespace fixture rows and classify C keywords.
pub(crate) fn assemble(lexemes: &[(&str, u32)]) -> Assembled {
    let mut source = String::new();
    let mut tok_starts = Vec::new();
    let mut tok_lens = Vec::new();
    let mut raw_kinds = Vec::new();

    for (lexeme, kind) in lexemes {
        if *kind == TOK_WHITESPACE || *kind == TOK_COMMENT {
            source.push_str(lexeme);
            continue;
        }
        if !source.is_empty() && !source.ends_with('\n') {
            source.push(' ');
        }
        tok_starts.push(source.len() as u32);
        source.push_str(lexeme);
        tok_lens.push(lexeme.len() as u32);
        raw_kinds.push(*kind);
    }

    let tok_types =
        reference_c_keyword_types(&raw_kinds, &tok_starts, &tok_lens, source.as_bytes());
    Assembled {
        source,
        raw_kinds,
        tok_types,
        tok_starts,
        tok_lens,
    }
}

/// Find the first token row whose source span equals `needle`.
pub(crate) fn find_row_for_lexeme(assembled: &Assembled, needle: &str) -> usize {
    assembled
        .tok_starts
        .iter()
        .zip(&assembled.tok_lens)
        .position(|(start, len)| {
            let start = *start as usize;
            let end = start.saturating_add(*len as usize);
            assembled.source.as_bytes().get(start..end) == Some(needle.as_bytes())
        })
        .unwrap_or_else(|| panic!("lexeme {needle:?} not found in fixture"))
}

/// Read the kind field from one typed VAST row.
pub(crate) fn row_typed_kind(typed: &[u8], row: usize) -> u32 {
    word_at(typed, row * VAST_STRIDE_U32)
}

/// Assert a typed VAST row and its lowered Program Graph source span.
pub(crate) fn assert_pg_row(assembled: &Assembled, pg: &[u8], typed: &[u8], idx: usize, kind: u32) {
    assert_eq!(
        word_at(typed, idx * VAST_STRIDE_U32),
        kind,
        "typed kind at {idx}"
    );
    assert_eq!(word_at(pg, idx * PG_STRIDE_U32), kind, "PG kind at {idx}");
    assert_eq!(
        word_at(pg, idx * PG_STRIDE_U32 + 1),
        assembled.tok_starts[idx],
        "PG span_start at {idx}"
    );
    assert_eq!(
        word_at(pg, idx * PG_STRIDE_U32 + 2),
        assembled.tok_starts[idx] + assembled.tok_lens[idx],
        "PG span_end at {idx}"
    );
}

pub(crate) fn word_at(buf: &[u8], word: usize) -> u32 {
    let offset = word * 4;
    u32::from_le_bytes(
        buf[offset..offset + 4]
            .try_into()
            .expect("complete u32 word"),
    )
}

pub(crate) fn node_count_from_vast(vast: &[u8]) -> u32 {
    u32::try_from(vast.len() / (VAST_STRIDE_U32 * 4)).unwrap_or_default()
}

/// Lower typed VAST to Program Graph rows through the executable lowerer under
/// the reference oracle, the arm a GPU parity test compares against.
pub(crate) fn run_reference_pg_lower(typed_vast: &[u8]) -> Vec<u8> {
    let num_nodes = node_count_from_vast(typed_vast);
    let program = c_lower_ast_to_pg_nodes("vast_nodes", Expr::u32(num_nodes), "pg_nodes");
    let output_len = num_nodes.saturating_mul(PG_STRIDE_U32 as u32).max(1) as usize * 4;
    let values = [
        Value::from(typed_vast.to_vec()),
        Value::from(vec![0; output_len]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &values)
        .unwrap_or_else(|e| panic!("C AST PG lowerer must execute on CPU: {e}"));
    assert_eq!(outputs.len(), 1);
    outputs[0].to_bytes()
}

/// Every buffer the four-stage CPU pipeline produces for one token stream.
pub(crate) struct PipelineOut {
    pub(crate) raw_vast: Vec<u8>,
    pub(crate) typed_vast: Vec<u8>,
    pub(crate) expr_shape: Vec<u8>,
    pub(crate) pg: Vec<u8>,
}

/// Build VAST, classify it, derive expression shapes, and lower to Program Graph
/// rows, asserting the executable lowerer agrees with the byte oracle.
pub(crate) fn run_cpu_pipeline(assembled: &Assembled) -> PipelineOut {
    let raw_vast = reference_c11_build_vast_nodes(
        &assembled.tok_types,
        &assembled.tok_starts,
        &assembled.tok_lens,
    );
    let typed_vast = reference_c11_classify_vast_node_kinds(&raw_vast);
    let expr_shape = reference_c11_build_expression_shape_nodes(&raw_vast, &typed_vast);
    let pg = run_reference_pg_lower(&typed_vast);
    assert_eq!(
        pg,
        reference_ast_to_pg_nodes(&typed_vast),
        "executable PG lowerer must match byte oracle"
    );
    PipelineOut {
        raw_vast,
        typed_vast,
        expr_shape,
        pg,
    }
}

/// Assert the fixture's own lexemes re-lex to the kinds it declares, so a
/// hand-built stream cannot assert a tokenization the lexer does not produce.
pub(crate) fn assert_lex_matches_non_ws(assembled: &Assembled) {
    let kinds = c_grammar_gen::lex_c11_max_munch_kinds(assembled.source.as_bytes())
        .expect("lex fixture source");
    let filtered: Vec<u32> = kinds
        .into_iter()
        .filter(|k| *k != TOK_WHITESPACE && *k != TOK_COMMENT)
        .collect();
    assert_eq!(
        filtered, assembled.raw_kinds,
        "hand-built fixture must match max-munch lexer (no fake tokenization)"
    );
}

/// A structural or preprocessing row carries no expression shape.
pub(crate) fn assert_shape_none(expr_shape: &[u8], idx: usize) {
    let base = idx * C_EXPR_SHAPE_STRIDE_U32 as usize;
    assert_eq!(
        word_at(expr_shape, base),
        C_EXPR_SHAPE_NONE,
        "preproc/structural rows stay shape-none"
    );
    assert_eq!(word_at(expr_shape, base + 4), C_EXPR_ASSOC_NONE);
}

/// Widen source bytes to the one-byte-per-word haystack the lexer op reads.
pub(crate) fn haystack_words(source: &[u8]) -> Vec<u32> {
    source.iter().copied().map(u32::from).collect()
}

/// Run the GPU lexer `c11_lexer` through the reference oracle and return the
/// compact token stream trimmed to the emitted count.
pub(crate) fn run_c11_lexer(
    source: &[u8],
    haystack_len: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let program = c11_lexer(
        "haystack",
        "out_tok_types",
        "out_tok_starts",
        "out_tok_lens",
        "out_counts",
        haystack_len,
    );
    let zero_buf = vec![0u8; haystack_len as usize * 4];
    let inputs = [
        Value::from(pack_u32_slice(&haystack_words(source))),
        Value::from(zero_buf.clone()),
        Value::from(zero_buf.clone()),
        Value::from(zero_buf),
        Value::from(vec![0u8; 4]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("c11_lexer must execute under the reference oracle");
    assert_eq!(
        outputs.len(),
        4,
        "expected [tok_types, tok_starts, tok_lens, counts]"
    );
    let tok_types = decode_u32_le_bytes_all(&outputs[0].to_bytes());
    let tok_starts = decode_u32_le_bytes_all(&outputs[1].to_bytes());
    let tok_lens = decode_u32_le_bytes_all(&outputs[2].to_bytes());
    let tok_count = decode_u32_le_bytes_all(&outputs[3].to_bytes())
        .first()
        .copied()
        .unwrap_or(0);
    (
        tok_types[..tok_count as usize].to_vec(),
        tok_starts[..tok_count as usize].to_vec(),
        tok_lens[..tok_count as usize].to_vec(),
        tok_count,
    )
}
