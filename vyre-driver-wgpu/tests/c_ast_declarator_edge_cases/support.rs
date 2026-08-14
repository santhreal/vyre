// GPU/CPU parity tests for difficult C declarator edge cases.
//
// Coverage:
//   * arrays of function pointers
//   * function returning pointer to function
//   * nested type qualifiers (const, volatile, restrict, _Atomic)
//   * parameter typedef shadowing
//   * abstract declarators in casts and sizeof
//   * K&R-style function declarations
//   * deeply parenthesised declarators

pub(crate) use crate::gemini_named_fixtures::{ident, named_fixture, tok, NamedFixture};
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::{
    reference_c11_build_vast_nodes, reference_c11_classify_vast_node_kinds,
};

pub(crate) use crate::c_ast_gpu_parity_support::node_count_from_vast;
use crate::c_ast_gpu_parity_support::{
    run_gpu_classifier_with_count as run_gpu_classifier, run_gpu_full_typedef_annotation,
    starts_for_lens,
};

pub(crate) const TYPEDEF_FLAGS_FIELD: usize = 7;
pub(crate) const ORDINARY_FLAG_DECL: u32 = 1 << 2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn run_gpu_annotate(raw_vast: &[u8], haystack: &[u8], _node_count: u32) -> Vec<u8> {
    run_gpu_full_typedef_annotation(haystack, raw_vast)
}

pub(crate) fn cpu_gpu_classified(
    tok_types: &[u32],
    tok_starts: &[u32],
    tok_lens: &[u32],
) -> Vec<u8> {
    let raw = reference_c11_build_vast_nodes(tok_types, tok_starts, tok_lens);
    let expected = reference_c11_classify_vast_node_kinds(&raw);
    let gpu = run_gpu_classifier(&raw, node_count_from_vast(&raw));
    assert_eq!(
        gpu, expected,
        "GPU classifier must match CPU for declarator fixture"
    );
    expected
}

// ---------------------------------------------------------------------------
// Atom helpers for fixtures that need a haystack (typedef annotation)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// ```c
/// int (*handlers[4])(void *ctx, int opcode);
/// ```
pub(crate) fn fixture_array_of_function_pointers() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_VOID,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// int (*f(int))(float);
/// ```
pub(crate) fn fixture_function_returning_fnptr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_FLOAT_KW,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// const int * const * volatile p;
/// ```
pub(crate) fn fixture_nested_qualifiers() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_CONST,
        TOK_INT,
        TOK_STAR,
        TOK_CONST,
        TOK_STAR,
        TOK_VOLATILE,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// typedef int T;
/// void f(int T) {
///   T * y;
/// }
/// ```
pub(crate) fn fixture_parameter_typedef_shadowing() -> NamedFixture {
    named_fixture(&[
        tok(TOK_TYPEDEF),
        tok(TOK_INT),
        ident("T"),
        tok(TOK_SEMICOLON),
        tok(TOK_VOID),
        ident("f"),
        tok(TOK_LPAREN),
        tok(TOK_INT),
        ident("T"),
        tok(TOK_RPAREN),
        tok(TOK_LBRACE),
        ident("T"),
        tok(TOK_STAR),
        ident("y"),
        tok(TOK_SEMICOLON),
        tok(TOK_RBRACE),
    ])
}

/// ```c
/// (void (*)(int))p;
/// ```
pub(crate) fn fixture_abstract_declarator_cast() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_LPAREN,
        TOK_VOID,
        TOK_LPAREN,
        TOK_STAR,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// sizeof(void (*)(int));
/// ```
pub(crate) fn fixture_abstract_declarator_sizeof() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_SIZEOF,
        TOK_LPAREN,
        TOK_VOID,
        TOK_LPAREN,
        TOK_STAR,
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// int f(a, b)
/// int a;
/// int b;
/// {
///   return a;
/// }
/// ```
pub(crate) fn fixture_kr_function() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_COMMA,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_LBRACE,
        TOK_RETURN,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// int (((*p)));
/// ```
pub(crate) fn fixture_deeply_parenthesised_pointer() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// ```c
/// volatile int * const arr[8];
/// ```
pub(crate) fn fixture_qualified_pointer_array() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_VOLATILE,
        TOK_INT,
        TOK_STAR,
        TOK_CONST,
        TOK_IDENTIFIER,
        TOK_LBRACKET,
        TOK_INTEGER,
        TOK_RBRACKET,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
