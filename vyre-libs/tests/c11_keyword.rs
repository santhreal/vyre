//! Nonzero tests for the C11 keyword promotion pass.

#![cfg(feature = "c-parser")]
#![allow(deprecated)]
mod wire_words;

use wire_words::decode_u32_words as words_from_bytes;
use wire_words::u32_bytes as bytes;
use vyre_libs::parsing::c::lex::keyword::{
    c_keyword, c_keyword_map_words, c_keyword_packed_haystack, fnv1a32, reference_c_keyword_types,
    C_KEYWORDS,
};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_reference::value::Value;

#[test]
fn keyword_hash_table_has_no_shadowing_collisions() {
    let mut seen = std::collections::HashMap::new();
    for (keyword, token) in C_KEYWORDS {
        let hash = fnv1a32(keyword.as_bytes());
        if let Some((prior_keyword, prior_token)) = seen.insert(hash, (*keyword, *token)) {
            assert_eq!(
                (prior_keyword, prior_token),
                (*keyword, *token),
                "FNV-1a keyword collision would silently shadow {prior_keyword:?} with {keyword:?}"
            );
        }
    }
}

fn haystack_words(source: &[u8]) -> Vec<u32> {
    source.iter().map(|byte| u32::from(*byte)).collect()
}

fn packed_haystack_bytes(source: &[u8]) -> Vec<u8> {
    let word_bytes = source.len().max(1).div_ceil(4) * 4;
    let mut out = vec![0u8; word_bytes];
    out[..source.len()].copy_from_slice(source);
    out
}

#[test]
fn packed_keyword_pass_promotes_keywords_without_expanded_haystack() {
    let source = b"int return __asm__ volatile";
    let tok_types = [TOK_IDENTIFIER; 4];
    let tok_starts = [0u32, 4, 11, 19];
    let tok_lens = [3u32, 6, 7, 8];
    let expected = reference_c_keyword_types(&tok_types, &tok_starts, &tok_lens, source);
    assert_eq!(
        expected,
        vec![TOK_INT, TOK_RETURN, TOK_GNU_ASM, TOK_VOLATILE]
    );

    let keyword_map = c_keyword_map_words();
    let program = c_keyword_packed_haystack(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "counts",
        "haystack",
        "keyword_map",
        tok_types.len() as u32,
        C_KEYWORDS.len() as u32,
        source.len() as u32,
    );
    let input_bytes = [
        bytes(&tok_types),
        bytes(&tok_starts),
        bytes(&tok_lens),
        bytes(&[tok_types.len() as u32]),
        packed_haystack_bytes(source),
        bytes(&keyword_map),
    ];
    let values = input_bytes
        .iter()
        .cloned()
        .map(Value::from)
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("packed keyword Program must execute under the reference oracle");
    assert_eq!(
        words_from_bytes(&outputs[0].to_bytes()),
        expected,
        "packed GPU keyword pass IR must promote the same tokens as the expanded oracle path"
    );
}

/// Every spelling in `C_KEYWORDS` must promote to its declared token, through the CPU oracle and
/// through the emitted pass alike. The case list is read off the table at run time, so a spelling
/// added there without a working promotion turns this red instead of escaping coverage.
#[test]
fn keyword_pass_promotes_every_table_entry() {
    let mut source = Vec::new();
    let mut tok_starts = Vec::new();
    let mut tok_lens = Vec::new();
    let mut declared = Vec::new();
    for (keyword, token) in C_KEYWORDS {
        if !source.is_empty() {
            source.push(b' ');
        }
        tok_starts.push(source.len() as u32);
        tok_lens.push(keyword.len() as u32);
        source.extend_from_slice(keyword.as_bytes());
        declared.push(*token);
    }
    let tok_types = vec![TOK_IDENTIFIER; C_KEYWORDS.len()];

    let expected = reference_c_keyword_types(&tok_types, &tok_starts, &tok_lens, &source);
    assert_eq!(
        expected, declared,
        "the CPU keyword oracle must promote every table spelling to the token the table declares"
    );

    let keyword_map = c_keyword_map_words();
    let program = c_keyword(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "counts",
        "haystack",
        "keyword_map",
        tok_types.len() as u32,
        C_KEYWORDS.len() as u32,
        source.len() as u32,
    );
    let input_bytes = [
        bytes(&tok_types),
        bytes(&tok_starts),
        bytes(&tok_lens),
        bytes(&[tok_types.len() as u32]),
        bytes(&haystack_words(&source)),
        bytes(&keyword_map),
    ];
    let values = input_bytes
        .iter()
        .cloned()
        .map(Value::from)
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("keyword Program must execute under the reference oracle");
    assert_eq!(
        words_from_bytes(&outputs[0].to_bytes()),
        declared,
        "the emitted keyword pass must promote every table spelling like the CPU oracle"
    );
}

#[test]
fn keyword_pass_promotes_keywords_in_c_source_context() {
    let source = b"int main(void) { return _Bool; volatile x; __attribute__((cold)); asm volatile(\"nop\"); __asm__ __volatile__(\"mfence\"); }";
    let tok_types = [
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_STRING,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_STRING,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_starts = [
        0u32, 4, 8, 9, 13, 15, 17, 24, 29, 31, 40, 41, 43, 56, 57, 58, 62, 63, 64, 66, 70, 78, 79,
        84, 85, 87, 95, 107, 108, 116, 117, 119,
    ];
    let tok_lens = [
        3u32, 4, 1, 4, 1, 1, 6, 5, 1, 8, 1, 1, 13, 1, 1, 4, 1, 1, 1, 3, 8, 1, 5, 1, 1, 7, 12, 1, 8,
        1, 1, 1,
    ];
    let expected = reference_c_keyword_types(&tok_types, &tok_starts, &tok_lens, source);
    assert_eq!(expected[0], TOK_INT);
    assert_eq!(expected[1], TOK_IDENTIFIER);
    assert_eq!(expected[3], TOK_VOID);
    assert_eq!(expected[6], TOK_RETURN);
    assert_eq!(expected[7], TOK_BOOL);
    assert_eq!(expected[9], TOK_VOLATILE);
    assert_eq!(expected[10], TOK_IDENTIFIER);
    assert_eq!(expected[12], TOK_GNU_ATTRIBUTE);
    assert_eq!(expected[15], TOK_IDENTIFIER);
    assert_eq!(expected[19], TOK_GNU_ASM);
    assert_eq!(expected[20], TOK_VOLATILE);
    assert_eq!(expected[25], TOK_GNU_ASM);
    assert_eq!(expected[26], TOK_VOLATILE);

    let keyword_map = c_keyword_map_words();
    let program = c_keyword(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "counts",
        "haystack",
        "keyword_map",
        tok_types.len() as u32,
        C_KEYWORDS.len() as u32,
        source.len() as u32,
    );
    let input_bytes = [
        bytes(&tok_types),
        bytes(&tok_starts),
        bytes(&tok_lens),
        bytes(&[tok_types.len() as u32]),
        bytes(&haystack_words(source)),
        bytes(&keyword_map),
    ];
    let values = input_bytes
        .iter()
        .cloned()
        .map(Value::from)
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("keyword Program must execute under the reference oracle");
    assert_eq!(
        words_from_bytes(&outputs[0].to_bytes()),
        expected,
        "GPU keyword pass IR must promote the same tokens as the CPU oracle"
    );

    let builtin_source = b"__builtin_constant_p __builtin_choose_expr __builtin_types_compatible_p";
    let builtin_tok_types = [TOK_IDENTIFIER, TOK_IDENTIFIER, TOK_IDENTIFIER];
    let builtin_tok_starts = [0u32, 21, 43];
    let builtin_tok_lens = [20u32, 21, 28];
    let builtin_expected = reference_c_keyword_types(
        &builtin_tok_types,
        &builtin_tok_starts,
        &builtin_tok_lens,
        builtin_source,
    );
    assert_eq!(
        builtin_expected,
        vec![
            TOK_BUILTIN_CONSTANT_P,
            TOK_BUILTIN_CHOOSE_EXPR,
            TOK_BUILTIN_TYPES_COMPATIBLE_P
        ],
        "GNU builtin spellings must promote to dedicated parser tokens"
    );

    let linux_type_source = b"__auto_type __int128 typeof_unqual __typeof_unqual __typeof_unqual__";
    let linux_type_tok_types = [TOK_IDENTIFIER; 5];
    let linux_type_tok_starts = [0u32, 12, 21, 35, 51];
    let linux_type_tok_lens = [11u32, 8, 13, 15, 17];
    let linux_type_expected = reference_c_keyword_types(
        &linux_type_tok_types,
        &linux_type_tok_starts,
        &linux_type_tok_lens,
        linux_type_source,
    );
    assert_eq!(
        linux_type_expected,
        vec![
            TOK_GNU_AUTO_TYPE,
            TOK_GNU_INT128,
            TOK_GNU_TYPEOF_UNQUAL,
            TOK_GNU_TYPEOF_UNQUAL,
            TOK_GNU_TYPEOF_UNQUAL
        ],
        "GNU/Linux declaration type spellings must promote before type propagation"
    );

    let program = c_keyword(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "counts",
        "haystack",
        "keyword_map",
        builtin_tok_types.len() as u32,
        C_KEYWORDS.len() as u32,
        builtin_source.len() as u32,
    );
    let input_bytes = [
        bytes(&builtin_tok_types),
        bytes(&builtin_tok_starts),
        bytes(&builtin_tok_lens),
        bytes(&[builtin_tok_types.len() as u32]),
        bytes(&haystack_words(builtin_source)),
        bytes(&keyword_map),
    ];
    let values = input_bytes
        .iter()
        .cloned()
        .map(Value::from)
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("builtin keyword Program must execute under the reference oracle");
    assert_eq!(
        words_from_bytes(&outputs[0].to_bytes()),
        builtin_expected,
        "GPU keyword pass IR must promote GNU builtins like the CPU oracle"
    );

    let gnu_qualifier_source = b"__const __const__ __signed __signed__ __volatile __thread";
    let gnu_qualifier_raw = [TOK_IDENTIFIER; 6];
    let gnu_qualifier_starts = [0u32, 8, 18, 27, 38, 49];
    let gnu_qualifier_lens = [7u32, 9, 8, 10, 10, 8];
    let gnu_qualifier_expected = reference_c_keyword_types(
        &gnu_qualifier_raw,
        &gnu_qualifier_starts,
        &gnu_qualifier_lens,
        gnu_qualifier_source,
    );
    assert_eq!(
        gnu_qualifier_expected,
        vec![
            TOK_CONST,
            TOK_CONST,
            TOK_SIGNED,
            TOK_SIGNED,
            TOK_VOLATILE,
            TOK_THREAD_LOCAL,
        ],
        "GNU/Linux qualifier aliases must promote to their canonical C token kinds"
    );

    let program = c_keyword(
        "tok_types",
        "tok_starts",
        "tok_lens",
        "counts",
        "haystack",
        "keyword_map",
        gnu_qualifier_raw.len() as u32,
        C_KEYWORDS.len() as u32,
        gnu_qualifier_source.len() as u32,
    );
    let input_bytes = [
        bytes(&gnu_qualifier_raw),
        bytes(&gnu_qualifier_starts),
        bytes(&gnu_qualifier_lens),
        bytes(&[gnu_qualifier_raw.len() as u32]),
        bytes(&haystack_words(gnu_qualifier_source)),
        bytes(&keyword_map),
    ];
    let values = input_bytes
        .iter()
        .cloned()
        .map(Value::from)
        .collect::<Vec<_>>();
    let outputs = vyre_reference::reference_eval(&program, &values)
        .expect("GNU qualifier keyword Program must execute under the reference oracle");
    assert_eq!(
        words_from_bytes(&outputs[0].to_bytes()),
        gnu_qualifier_expected,
        "GPU keyword pass IR must promote GNU qualifier aliases like the CPU oracle"
    );
}
