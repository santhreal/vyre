//! The C lexer program driven through the reference oracle.
//!
//! A raw-source contract does not hand-build a token stream: it runs
//! `c11_lexer` over the source bytes, trims the four output buffers to the token
//! count the kernel reported, and drops the whitespace and comment rows before
//! it promotes keywords and builds a VAST. Five suites carried their own copy of
//! that run, so `dup-scan` read every pair of them as thirty duplicated lines,
//! and a change to the oracle's buffer order had five call sites to fix.
//!
//! [`run_c11_lexer`] is the trimmed stream, [`run_c11_lexer_promoted`] the same
//! stream with keywords promoted, [`lex_significant_tokens`] the stream a VAST
//! build takes, and [`classify_raw_source`] the whole CPU pipeline from source
//! bytes to classified rows.

use super::rows::haystack_words;
use vyre_libs::parsing::c::lex::keyword::reference_c_keyword_types;
use vyre_libs::parsing::c::lex::lexer::c11_lexer;
use vyre_libs::parsing::c::lex::tokens::{TOK_COMMENT, TOK_WHITESPACE};
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_annotate_typedef_names, reference_c11_build_vast_nodes,
    reference_c11_classify_vast_node_kinds,
};
use vyre_primitives::wire::decode_u32_le_bytes_all as decode_u32_words;
use vyre_reference::value::Value;

/// The raw `[tok_types, tok_starts, tok_lens, counts]` buffers `c11_lexer`
/// wrote, untrimmed.
///
/// Public because one contract asserts the kernel never leaves every output
/// buffer zero for non-empty input, which is a claim about the buffers rather
/// than about the token stream in them.
pub(crate) fn c11_lexer_outputs(source: &[u8], haystack_len: u32) -> Vec<Value> {
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
        Value::from(haystack_words(source)),
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
    outputs
}

/// The token stream `c11_lexer` emitted, trimmed to the count it reported.
pub(crate) fn run_c11_lexer(
    source: &[u8],
    haystack_len: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let outputs = c11_lexer_outputs(source, haystack_len);
    let tok_types = decode_u32_words(&outputs[0].to_bytes());
    let tok_starts = decode_u32_words(&outputs[1].to_bytes());
    let tok_lens = decode_u32_words(&outputs[2].to_bytes());
    let tok_count = decode_u32_words(&outputs[3].to_bytes())
        .first()
        .copied()
        .unwrap_or(0);
    let live = tok_count as usize;
    (
        tok_types[..live].to_vec(),
        tok_starts[..live].to_vec(),
        tok_lens[..live].to_vec(),
        tok_count,
    )
}

/// [`run_c11_lexer`] with the identifier rows promoted to their keyword kinds.
pub(crate) fn run_c11_lexer_promoted(
    source: &[u8],
    haystack_len: u32,
) -> (Vec<u32>, Vec<u32>, Vec<u32>, u32) {
    let (tok_types, tok_starts, tok_lens, tok_count) = run_c11_lexer(source, haystack_len);
    let promoted = reference_c_keyword_types(&tok_types, &tok_starts, &tok_lens, source);
    (promoted, tok_starts, tok_lens, tok_count)
}

/// [`run_c11_lexer`] with the whitespace and comment rows dropped.
///
/// The VAST builder reads a stream of significant tokens only, so every
/// raw-source contract filters here before it builds.
pub(crate) fn lex_significant_tokens(source: &[u8]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (raw_types, raw_starts, raw_lens, _) = run_c11_lexer(source, source.len() as u32);
    let mut types = Vec::with_capacity(raw_types.len());
    let mut starts = Vec::with_capacity(raw_types.len());
    let mut lens = Vec::with_capacity(raw_types.len());
    for (idx, kind) in raw_types.iter().copied().enumerate() {
        if kind != TOK_WHITESPACE && kind != TOK_COMMENT {
            types.push(kind);
            starts.push(raw_starts[idx]);
            lens.push(raw_lens[idx]);
        }
    }
    (types, starts, lens)
}

/// The stream and the classified rows a raw-source contract asserts against.
pub(crate) struct ClassifiedSource {
    /// Keyword-promoted token kinds, whitespace and comments already dropped.
    pub(crate) tok_types: Vec<u32>,
    /// Source byte offset per token.
    pub(crate) tok_starts: Vec<u32>,
    /// Source byte width per token.
    pub(crate) tok_lens: Vec<u32>,
    /// VAST rows after typedef annotation and node-kind classification.
    pub(crate) typed_vast: Vec<u8>,
}

/// Raw source bytes through lex, keyword promotion, VAST build, typedef
/// annotation and classification, the CPU arm of the C frontend.
pub(crate) fn classify_raw_source(source: &[u8]) -> ClassifiedSource {
    let (raw_types, tok_starts, tok_lens) = lex_significant_tokens(source);
    let tok_types = reference_c_keyword_types(&raw_types, &tok_starts, &tok_lens, source);
    let raw_vast = reference_c11_build_vast_nodes(&tok_types, &tok_starts, &tok_lens);
    let annotated = reference_c11_annotate_typedef_names(&raw_vast, source);
    let typed_vast = reference_c11_classify_vast_node_kinds(&annotated);
    ClassifiedSource {
        tok_types,
        tok_starts,
        tok_lens,
        typed_vast,
    }
}
