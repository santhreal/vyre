//! Integration contracts for C declarator shape extraction in structural parser stages.

#![cfg(feature = "c-parser")]

use vyre_primitives::wire::pack_u32_slice as words_to_bytes;

#[path = "c_ast_gpu_parity_support/mod.rs"]
mod c_ast_gpu_parity_support;

use c_ast_gpu_parity_support::dispatch_gpu_program;
use vyre::ir::Expr;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::structure::{c11_extract_calls, c11_extract_functions};

const SENTINEL: u32 = u32::MAX;

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn paired(len: usize, pairs: &[(usize, usize)]) -> Vec<u32> {
    let mut out = vec![SENTINEL; len];
    for &(left, right) in pairs {
        out[left] = right as u32;
        out[right] = left as u32;
    }
    out
}

/// Decode records from a SPARSE record array.
///
/// `c11_extract_functions` and `c11_extract_calls` do not compact. Each matching
/// token `t` writes its record at slot `t * record_words` and every unoccupied
/// slot is zeroed, so `out_counts[0]` is the array CAPACITY
/// (`num_tokens * record_words`), not a match count. Compaction is a separate
/// downstream stage, which
/// is why these tests decode by walking slots and skipping the empty ones, the
/// same way `vyre-libs/tests/c11_function_extractor_contracts.rs` does.
///
/// An all-zero record is unambiguously an empty slot for both record kinds: a
/// function record needs a return-type prefix so its name is never token 0 and
/// its body end is never 0, and a call record carries a non-zero span.
fn decode_records(records: &[u32], capacity: u32, record_words: usize) -> Vec<Vec<u32>> {
    let slots = capacity as usize / record_words;
    (0..slots)
        .map(|slot| records[slot * record_words..(slot + 1) * record_words].to_vec())
        .filter(|record| record.iter().any(|&word| word != 0))
        .collect()
}

/// Assert the sparse capacity the extractor reports, then decode.
///
/// Reading the capacity as though it were a match count is exactly the mistake
/// this file used to make, so the contract is asserted here rather than assumed.
fn decoded_records(out: &(Vec<u32>, u32), tok_count: usize, record_words: usize) -> Vec<Vec<u32>> {
    let (records, capacity) = out;
    assert_eq!(
        *capacity as usize,
        tok_count * record_words,
        "sparse extractors report the array capacity in out_counts[0], not a match count"
    );
    decode_records(records, *capacity, record_words)
}

fn run_extract_functions(
    tok_types: &[u32],
    paren_pairs: &[u32],
    brace_pairs: &[u32],
) -> (Vec<u32>, u32) {
    let program = c11_extract_functions(
        "tok_types",
        "paren_pairs",
        "brace_pairs",
        Expr::u32(tok_types.len() as u32),
        "out_functions",
        "out_counts",
    );
    let tok_bytes = words_to_bytes(tok_types);
    let paren_bytes = words_to_bytes(paren_pairs);
    let brace_bytes = words_to_bytes(brace_pairs);
    let count_bytes = words_to_bytes(&[0]);
    let outputs = dispatch_gpu_program(
        "GPU C function extraction",
        program,
        vec![tok_bytes, paren_bytes, brace_bytes, count_bytes],
    );
    assert_eq!(outputs.len(), 2);
    (bytes_to_words(&outputs[0]), bytes_to_words(&outputs[1])[0])
}

fn run_extract_calls(
    tok_types: &[u32],
    paren_pairs: &[u32],
    function_records: &[u32],
) -> (Vec<u32>, u32) {
    let program = c11_extract_calls(
        "tok_types",
        "paren_pairs",
        "functions",
        Expr::u32(tok_types.len() as u32),
        Expr::u32((function_records.len() / 3) as u32),
        "out_calls",
        "out_counts",
    );
    let tok_bytes = words_to_bytes(tok_types);
    let paren_bytes = words_to_bytes(paren_pairs);
    let function_bytes = words_to_bytes(function_records);
    let count_bytes = words_to_bytes(&[0]);
    let outputs = dispatch_gpu_program(
        "GPU C call extraction",
        program,
        vec![tok_bytes, paren_bytes, function_bytes, count_bytes],
    );
    assert_eq!(outputs.len(), 2);
    (bytes_to_words(&outputs[0]), bytes_to_words(&outputs[1])[0])
}

#[test]
fn typedef_return_function_definition_is_a_function_record() {
    let tok_types = [
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_RBRACE,
    ];
    let paren_pairs = paired(tok_types.len(), &[(2, 4)]);
    let brace_pairs = paired(tok_types.len(), &[(5, 6)]);

    let out = run_extract_functions(&tok_types, &paren_pairs, &brace_pairs);

    let records = decoded_records(&out, tok_types.len(), 3);

    assert_eq!(
        records,
        vec![vec![1, 5, 6]],
        "`typedef_name f(void) {{}}` must record f and its body span"
    );
}

#[test]
fn tagged_return_function_definition_is_a_function_record() {
    let tok_types = [
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_RBRACE,
    ];
    let paren_pairs = paired(tok_types.len(), &[(3, 5)]);
    let brace_pairs = paired(tok_types.len(), &[(6, 7)]);

    let out = run_extract_functions(&tok_types, &paren_pairs, &brace_pairs);

    let records = decoded_records(&out, tok_types.len(), 3);

    assert_eq!(
        records,
        vec![vec![2, 6, 7]],
        "`struct tag f(void) {{}}` must record f and its body span"
    );
}

#[test]
fn parenthesized_function_declarator_definition_is_a_function_record() {
    let tok_types = [
        TOK_INT,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_RBRACE,
    ];
    let paren_pairs = paired(tok_types.len(), &[(1, 3), (4, 6)]);
    let brace_pairs = paired(tok_types.len(), &[(7, 8)]);

    let out = run_extract_functions(&tok_types, &paren_pairs, &brace_pairs);

    let records = decoded_records(&out, tok_types.len(), 3);

    assert_eq!(
        records,
        vec![vec![2, 7, 8]],
        "`int (f)(void) {{}}` must record f and its body span"
    );
}

#[test]
fn function_pointer_declarator_is_not_a_pointer_call() {
    let tok_types = [
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let paren_pairs = paired(tok_types.len(), &[(1, 4), (5, 7)]);
    let out = run_extract_calls(&tok_types, &paren_pairs, &[0, 0, 0]);

    assert_eq!(
        decoded_records(&out, tok_types.len(), 4),
        Vec::<Vec<u32>>::new(),
        "`int (*fp)(int);` must not emit a pointer-call record"
    );
}

#[test]
fn abstract_function_pointer_parameter_is_not_a_pointer_call() {
    let tok_types = [
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let paren_pairs = paired(tok_types.len(), &[(2, 10), (4, 6), (7, 9)]);
    let out = run_extract_calls(&tok_types, &paren_pairs, &[0, 0, 0]);

    assert_eq!(
        decoded_records(&out, tok_types.len(), 4),
        Vec::<Vec<u32>>::new(),
        "`void f(int (*)(int));` must not emit a pointer-call record"
    );
}

#[test]
fn parenthesized_pointer_call_still_emits_call_record() {
    let tok_types = [
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let paren_pairs = paired(tok_types.len(), &[(0, 3), (4, 6)]);
    let out = run_extract_calls(&tok_types, &paren_pairs, &[0, 0, 0]);

    assert_eq!(
        decoded_records(&out, tok_types.len(), 4),
        vec![vec![SENTINEL, 2, 4, 6]],
        "`(*fp)(arg);` must keep emitting a pointer-call record"
    );
}

/// Two calls in a row must both survive the pre-loop sparse zeroing.
///
/// The extractor zeroes every slot before the match loop and then writes each
/// match at `t * 4`. If the zeroing raced the writes, or if two lanes computed
/// the same slot, one of these records would come back all-zero and decode away.
#[test]
fn consecutive_direct_calls_each_keep_their_own_sparse_slot() {
    let tok_types = [
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let paren_pairs = paired(tok_types.len(), &[(1, 2), (5, 6)]);
    let out = run_extract_calls(&tok_types, &paren_pairs, &[SENTINEL, 0, 0]);

    let mut records = decoded_records(&out, tok_types.len(), 4);
    records.sort_by_key(|record| record[1]);

    assert_eq!(
        records
            .iter()
            .map(|record| record[1..].to_vec())
            .collect::<Vec<_>>(),
        vec![vec![0, 1, 2], vec![4, 5, 6]],
        "both call records must survive the pre-loop sparse row clearing"
    );
}
