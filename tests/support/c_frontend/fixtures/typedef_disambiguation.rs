//! Token fixtures for C typedef and name disambiguation: cast versus multiply, nested shadowing,
//! tag versus typedef names, and declarator contexts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::rows::starts_for_lens;
use vyre_libs::parsing::c::lex::tokens::*;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// typedef int T;
/// void f(void) {
///   (T)*p;   -- cast expression: T is a typedef name
///   (x)*p;   -- multiplication: x is a variable, not a type
/// }
pub(crate) fn fixture_typedef_cast_vs_expr_multiply() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_TYPEDEF,
        TOK_INT,
        TOK_IDENTIFIER, // T
        TOK_SEMICOLON,
        TOK_VOID,
        TOK_IDENTIFIER, // f
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_LPAREN,     // (T)
        TOK_IDENTIFIER, // T
        TOK_RPAREN,
        TOK_STAR,
        TOK_IDENTIFIER, // p
        TOK_SEMICOLON,
        TOK_LPAREN,     // (x)
        TOK_IDENTIFIER, // x
        TOK_RPAREN,
        TOK_STAR,
        TOK_IDENTIFIER, // p
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// typedef int T;
/// void f(void) {
///   {
///     int T;   -- shadows the typedef
///     T * b;   -- multiplication, not pointer declaration
///   }
/// }
pub(crate) fn fixture_typedef_shadowing_nested() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_TYPEDEF,
        TOK_INT,
        TOK_IDENTIFIER, // T
        TOK_SEMICOLON,
        TOK_VOID,
        TOK_IDENTIFIER, // f
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER, // T (variable)
        TOK_SEMICOLON,
        TOK_IDENTIFIER, // T
        TOK_STAR,
        TOK_IDENTIFIER, // b
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// struct S { int x; };
/// typedef struct S S;
/// void f(void) {
///   struct S *a;   -- tag name in declaration
///   S *b;          -- typedef name in declaration
/// }
pub(crate) fn fixture_struct_tag_vs_typedef() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER, // S
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER, // x
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_SEMICOLON,
        TOK_TYPEDEF,
        TOK_STRUCT,
        TOK_IDENTIFIER, // S
        TOK_IDENTIFIER, // S
        TOK_SEMICOLON,
        TOK_VOID,
        TOK_IDENTIFIER, // f
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_STRUCT,
        TOK_IDENTIFIER, // S (tag)
        TOK_STAR,
        TOK_IDENTIFIER, // a
        TOK_SEMICOLON,
        TOK_IDENTIFIER, // S (typedef)
        TOK_STAR,
        TOK_IDENTIFIER, // b
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// void f(void) {
///   int *a[10];      -- array of pointers
///   int (*a)[10];    -- pointer to array
///   int *f(int);     -- function returning pointer
///   int (*f)(int);   -- pointer to function
/// }
pub(crate) fn fixture_declarator_contexts() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_VOID,
        TOK_IDENTIFIER, // f
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_INT,
        TOK_STAR,
        TOK_IDENTIFIER, // a
        TOK_LBRACKET,
        TOK_INTEGER, // 10
        TOK_RBRACKET,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER, // a
        TOK_RPAREN,
        TOK_LBRACKET,
        TOK_INTEGER, // 10
        TOK_RBRACKET,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_STAR,
        TOK_IDENTIFIER, // f
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER, // f
        TOK_RPAREN,
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_SEMICOLON,
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
