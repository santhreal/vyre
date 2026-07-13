//! Behavioral parity for `go_extract_channel_creations`.
//!
//! The builder scans the Go token stream for the `make ( chan` pattern (identifier
//! "make", `(`, identifier "chan") and records the `make` token's (start, len) span
//! into `out_ops`, bumping `out_counts[0]` by `GO_SPAN_RECORD_WORDS` per hit via an
//! atomic add. It was an orphan builder (registry-coverage closure gate
//! `adversarial_registry_closure.rs`). This pins it with real output bytes on a
//! positive (`make(chan int)`) and a negative (`make(int)`) case, proving both the
//! match logic (identifier-spelling + `(` + `chan`) and the span record it emits.
#![cfg(feature = "go-parser")]

use vyre::ir::Expr;
use vyre_libs::parsing::go::lex::{TOK_IDENTIFIER, TOK_LPAREN};
use vyre_libs::parsing::go::parse::ast_ops::go_extract_channel_creations;
use vyre_libs::parsing::go::parse::structure::GO_SPAN_RECORD_WORDS;
use vyre_reference::value::Value;

fn pack(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn unpack(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// One u32 word per source byte (the builder loads `haystack[start+off] & 0xFF`).
fn expanded(source: &[u8]) -> Vec<u8> {
    pack(&source.iter().map(|b| u32::from(*b)).collect::<Vec<_>>())
}

/// Returns (out_ops, out_counts) after running the extractor.
fn run(
    source: &[u8],
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let n = tok_types.len();
    let program = go_extract_channel_creations(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "haystack",
        Expr::u32(n as u32),
        "out_ops",
        "out_counts",
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(tok_types)),
            Value::from(pack(tok_starts)),
            Value::from(pack(tok_lens)),
            Value::from(expanded(source)),
            Value::from(vec![0u8; n * GO_SPAN_RECORD_WORDS as usize * 4]), // out_ops
            Value::from(vec![0u8; n * 4]), // out_counts (sized to dispatch extent)
        ],
    )
    .expect("go_extract_channel_creations must execute under reference_eval");
    (unpack(&outputs[0].to_bytes()), unpack(&outputs[1].to_bytes()))
}

#[test]
fn make_chan_is_recorded_as_a_channel_creation_span() {
    // `make(chan int)`: make(0,4) ((4,1) chan(5,4) int(10,3) )(13,1).
    let source = b"make(chan int)";
    let tok_types = [TOK_IDENTIFIER, TOK_LPAREN, TOK_IDENTIFIER, TOK_IDENTIFIER, TOK_LPAREN];
    let tok_starts = [0u32, 4, 5, 10, 13];
    let tok_lens = [4u32, 1, 4, 3, 1];

    let (out_ops, out_counts) = run(source, &tok_types, &tok_starts, &tok_lens);
    assert_eq!(
        out_counts[0], GO_SPAN_RECORD_WORDS,
        "exactly one channel creation must be recorded (counts bumped by one record)"
    );
    // The record holds the `make` token span: start=0, len=4.
    assert_eq!(
        &out_ops[0..2],
        &[0, 4],
        "the recorded span must be the `make` token's (start, len)"
    );
}

#[test]
fn make_without_chan_records_nothing() {
    // `make(int)`: make( followed by `int`, not `chan`: no channel creation.
    let source = b"make(int)";
    let tok_types = [TOK_IDENTIFIER, TOK_LPAREN, TOK_IDENTIFIER, TOK_LPAREN];
    let tok_starts = [0u32, 4, 5, 8];
    let tok_lens = [4u32, 1, 3, 1];

    let (_out_ops, out_counts) = run(source, &tok_types, &tok_starts, &tok_lens);
    assert_eq!(
        out_counts[0], 0,
        "`make(int)` is not a channel creation, no span recorded"
    );
}
