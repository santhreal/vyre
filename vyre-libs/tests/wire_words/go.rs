//! Shared harness for driving the Go frontend from tests.
//!
//! The tokenizer is a five-stage chain and every stage's output feeds the next,
//! so a test that wants tokens has to reproduce the whole chain. Two suites
//! need it (`go_frontend_corpus` and `go_tokenizer_semantics`), and a second
//! copy would drift the moment the chain gains a stage, so it lives here once.

#![allow(dead_code)]

use vyre_libs::parsing::go::lex::{
    go_compact_tokens, go_lexer, go_quote_flags, go_scan_emit_flags, go_scan_quote_flags,
};
use vyre_reference::value::Value;

/// Widen a source string to one u32 lane per byte, the layout the Go IR reads.
pub(crate) fn pack_source(source: &str) -> Vec<u8> {
    source
        .as_bytes()
        .iter()
        .flat_map(|byte| u32::from(*byte).to_le_bytes())
        .collect()
}

/// A zeroed output buffer of `words` u32 lanes.
pub(crate) fn zeroed_u32_words(words: usize) -> Vec<u8> {
    vec![0u8; words * 4]
}

/// Execute a program under the reference interpreter, binding inputs by position.
pub(crate) fn run(program: &vyre::Program, inputs: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    vyre_reference::reference_eval(program, &vyre_reference::reference_inputs(program, inputs))
        .expect("reference execution must succeed")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect()
}

/// Run a program, binding inputs by BUFFER NAME rather than by position.
///
/// The prefix-scan chain declares intermediate buffers whose names and count
/// depend on the input size (it switches between a single-block scan and a
/// three-pass Blelloch chain above `BLOCK_LANES`). Supplying its inputs
/// positionally would hardcode that internal shape into the tests and break the
/// moment the scan is retuned. Named buffers get the caller's bytes; everything
/// else gets zeros sized from its own declaration.
pub(crate) fn run_by_name(program: &vyre::Program, named: &[(&str, Vec<u8>)]) -> Vec<Vec<u8>> {
    let inputs: Vec<Vec<u8>> = program
        .buffers()
        .iter()
        .filter(|decl| decl.access() != vyre::ir::BufferAccess::Workgroup)
        .map(|decl| {
            named
                .iter()
                .find(|(name, _)| *name == decl.name())
                .map(|(_, bytes)| bytes.clone())
                .unwrap_or_else(|| {
                    let bytes = decl
                        .static_byte_len()
                        .ok()
                        .flatten()
                        .unwrap_or_else(|| panic!("{} must be statically sized", decl.name()));
                    vec![0u8; bytes]
                })
        })
        .collect();
    run(program, inputs)
}

/// The dense token stream: types, starts, lengths, and how many are valid.
pub(crate) struct DenseTokens {
    pub(crate) types: Vec<u8>,
    pub(crate) starts: Vec<u8>,
    pub(crate) lens: Vec<u8>,
    pub(crate) count: usize,
}

/// Run the five-stage Go tokenizer and return the dense, source-ordered stream.
///
/// `go_lexer` emits sparse, one slot per source byte, so the tokens have to be
/// scanned and compacted before any extractor can read `tok_types[t + 1]` as
/// "the next token". The quote-parity pre-pass runs first because no single
/// byte lane can tell an opening quote from a closing one. See the module docs
/// on `go_lexer` for why each stage exists.
pub(crate) fn tokenize(source: &str) -> DenseTokens {
    let haystack_words = source.len().max(1);

    // Quote parity first: a `"` opens a literal only when an even number of
    // quotes precede it, which no single byte can decide on its own.
    let quote_flags = go_quote_flags("haystack", "quote_flags", haystack_words as u32);
    let flags = run(
        &quote_flags,
        vec![pack_source(source), zeroed_u32_words(haystack_words)],
    );
    let quote_scan = go_scan_quote_flags("quote_flags", "quote_ranks", haystack_words as u32);
    let quote_scan_outputs = run_by_name(&quote_scan, &[("quote_flags", flags[0].clone())]);
    let quote_ranks = quote_scan_outputs[vyre_reference::output_index(&quote_scan, "quote_ranks")
        .expect("the quote scan must return its quote_ranks buffer")]
    .clone();

    let lexer = go_lexer(
        "haystack",
        "quote_ranks",
        "sparse_types",
        "sparse_starts",
        "sparse_lens",
        "emit_flags",
        haystack_words as u32,
    );
    let sparse = run(
        &lexer,
        vec![
            pack_source(source),
            quote_ranks,
            zeroed_u32_words(haystack_words),
            zeroed_u32_words(haystack_words),
            zeroed_u32_words(haystack_words),
            zeroed_u32_words(haystack_words),
        ],
    );

    let scan = go_scan_emit_flags("emit_flags", "emit_offsets", haystack_words as u32);
    let scan_outputs = run_by_name(&scan, &[("emit_flags", sparse[3].clone())]);
    // The scan chain returns several buffers; locate the one we asked for by
    // name through the interpreter's own output-selection predicate rather
    // than guessing a position.
    let emit_offsets = scan_outputs[vyre_reference::output_index(&scan, "emit_offsets")
        .expect("the scan program must return its emit_offsets buffer")]
    .clone();

    let compact = go_compact_tokens(
        "sparse_types",
        "sparse_starts",
        "sparse_lens",
        "emit_flags",
        "emit_offsets",
        "out_tok_types",
        "out_tok_starts",
        "out_tok_lens",
        "out_counts",
        haystack_words as u32,
    );
    let dense = run(
        &compact,
        vec![
            sparse[0].clone(),
            sparse[1].clone(),
            sparse[2].clone(),
            sparse[3].clone(),
            emit_offsets,
            zeroed_u32_words(haystack_words),
            zeroed_u32_words(haystack_words),
            zeroed_u32_words(haystack_words),
            zeroed_u32_words(1),
        ],
    );

    DenseTokens {
        count: super::decode_u32_words(&dense[3])[0] as usize,
        types: dense[0].clone(),
        starts: dense[1].clone(),
        lens: dense[2].clone(),
    }
}
