//! Differential parity for the c11 lexer PERF-VARIANTS against the tested dense base.
//!
//! `c11_lexer` (dense, sequential, covered by c11_lexer_naga_validation +
//! gpu_pipeline_lex_classify_roundtrip + the preprocess pipeline tests) is the
//! reference. Its parallel/sparse/ranked/single-pass regular variants were orphan
//! builders (registry-coverage closure gate `adversarial_registry_closure.rs`).
//! Every one lexes the SAME source into the SAME token set; they differ only in the
//! OUTPUT LAYOUT:
//!   * dense / regular / ranked  -> COMPACTED (index = token number, count in out_counts[0])
//!   * sparse                    -> POSITION-INDEXED (index = byte position, empty slots zero)
//! So this normalizes every variant to a canonical token list, the (type,start,len)
//! triples of every slot with len>0, sorted by start, and asserts each equals the
//! dense base's list (real token bytes, not `!is_empty`). A directive-free source is
//! used so the `no_directives` variants agree with the full base.
#![cfg(feature = "c-parser")]
mod common;

use common::u32_bytes as bytes;
use vyre::ir::Program;
use vyre_libs::parsing::c::lex::lexer::{
    c11_lex_regular_single_pass, c11_lexer, c11_lexer_regular, c11_lexer_regular_ranked,
    c11_lexer_regular_sparse, c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives,
    c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan,
};
use vyre_reference::value::Value;

/// A directive-free C fragment: `int x=42;` → tokens int, x, =, 42, ;.
const SOURCE: &[u8] = b"int x=42;";

fn expanded_haystack(source: &[u8]) -> Vec<u8> {
    bytes(
        &source
            .iter()
            .map(|b| u32::from(*b))
            .collect::<Vec<_>>(),
    )
}

fn packed_haystack(source: &[u8]) -> Vec<u8> {
    let mut packed = vec![0u8; source.len().max(1).div_ceil(4) * 4];
    packed[..source.len()].copy_from_slice(source);
    packed
}

fn decode(bytes_out: &[u8]) -> Vec<u32> {
    bytes_out
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Run a lexer program and normalize its token columns to a canonical
/// `(type, start, len)` list: every output slot with a non-zero length, sorted by
/// start position. This collapses the compacted vs position-indexed layout
/// difference so dense/regular/ranked/sparse become directly comparable.
fn normalized_tokens(program: &Program, haystack: Vec<u8>) -> Vec<(u32, u32, u32)> {
    let n = SOURCE.len();
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(haystack),
            Value::from(vec![0u8; n * 4]), // out_tok_types (binding 1)
            Value::from(vec![0u8; n * 4]), // out_tok_starts (binding 2)
            Value::from(vec![0u8; n * 4]), // out_tok_lens (binding 3)
            // out_counts (binding 4): reference_eval sizes RW buffers to the dispatch
            // extent, so provide n words even though only [0] is written.
            Value::from(vec![0u8; n * 4]),
        ],
    )
    .expect("lexer program must execute under reference_eval");
    let types = decode(&outputs[0].to_bytes());
    let starts = decode(&outputs[1].to_bytes());
    let lens = decode(&outputs[2].to_bytes());
    let mut toks: Vec<(u32, u32, u32)> = (0..types.len())
        .filter(|&i| lens.get(i).copied().unwrap_or(0) > 0)
        .map(|i| (types[i], starts[i], lens[i]))
        .collect();
    toks.sort_by_key(|&(_, start, _)| start);
    toks
}

#[test]
fn regular_variants_match_dense_base_token_stream() {
    let n = SOURCE.len() as u32;
    let reference = normalized_tokens(
        &c11_lexer("haystack", "types", "starts", "lens", "counts", n),
        expanded_haystack(SOURCE),
    );
    // Sanity: the dense base must actually tokenize `int x=42;` into 5 tokens
    // (int, x, =, 42, ;) (otherwise the whole differential is vacuous).
    assert_eq!(
        reference.len(),
        5,
        "dense base must lex `int x=42;` into 5 tokens, got {reference:?}"
    );

    let expanded_variants: [(&str, Program); 3] = [
        (
            "regular",
            c11_lexer_regular("haystack", "types", "starts", "lens", "counts", n),
        ),
        (
            "ranked",
            c11_lexer_regular_ranked("haystack", "types", "starts", "lens", "counts", n),
        ),
        (
            "sparse",
            c11_lexer_regular_sparse("haystack", "types", "starts", "lens", "counts", n),
        ),
    ];
    for (label, program) in expanded_variants {
        let got = normalized_tokens(&program, expanded_haystack(SOURCE));
        assert_eq!(
            got, reference,
            "lexer variant `{label}` token stream must match the dense base"
        );
    }

    // single-pass = regular lex + digraph rewrite; `int x=42;` has no digraphs, so
    // the token stream is identical to the base. digraph_capacity=8 is ample.
    let single_pass = normalized_tokens(
        &c11_lex_regular_single_pass("haystack", "types", "starts", "lens", "counts", n, 8),
        expanded_haystack(SOURCE),
    );
    assert_eq!(
        single_pass, reference,
        "single-pass regular lexer token stream must match the dense base"
    );

    // Packed-haystack sparse variants (source stored 4 bytes / u32 word).
    let packed_variants: [(&str, Program); 2] = [
        (
            "packed_no_directives",
            c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives(
                "haystack", "types", "starts", "lens", "counts", n,
            ),
        ),
        (
            "packed_no_directives_no_backscan",
            c11_lexer_regular_sparse_packed_haystack_with_flags_no_directives_no_backscan(
                "haystack", "types", "starts", "lens", "counts", n,
            ),
        ),
    ];
    for (label, program) in packed_variants {
        let got = normalized_tokens(&program, packed_haystack(SOURCE));
        assert_eq!(
            got, reference,
            "packed sparse lexer variant `{label}` token stream must match the dense base"
        );
    }
}
