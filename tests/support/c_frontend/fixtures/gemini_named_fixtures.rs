//! Named token fixtures for the Gemini C AST contracts: typedef shadowing, cast versus multiply,
//! nested function pointers, tag separation, GNU attributes, and compound literals.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::rows::starts_for_lens;
use vyre_libs::parsing::c::lex::tokens::*;
use vyre_libs::parsing::c::parse::vast::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
pub(crate) enum Atom {
    Tok(u32),
    Ident(&'static str),
}

pub(crate) struct NamedFixture {
    pub(crate) tok_types: Vec<u32>,
    pub(crate) tok_starts: Vec<u32>,
    pub(crate) tok_lens: Vec<u32>,
    pub(crate) haystack: Vec<u8>,
}

pub(crate) fn tok(token: u32) -> Atom {
    Atom::Tok(token)
}

pub(crate) fn ident(name: &'static str) -> Atom {
    Atom::Ident(name)
}

pub(crate) fn named_fixture(atoms: &[Atom]) -> NamedFixture {
    let mut tok_types = Vec::with_capacity(atoms.len());
    let mut tok_starts = Vec::with_capacity(atoms.len());
    let mut tok_lens = Vec::with_capacity(atoms.len());
    let mut haystack = Vec::new();
    let mut cursor = 0u32;

    for atom in atoms {
        match atom {
            Atom::Tok(token) => {
                tok_types.push(*token);
                tok_starts.push(0);
                tok_lens.push(0);
            }
            Atom::Ident(name) => {
                tok_types.push(TOK_IDENTIFIER);
                tok_starts.push(cursor);
                tok_lens.push(name.len() as u32);
                haystack.extend_from_slice(name.as_bytes());
                cursor = cursor.saturating_add(name.len() as u32);
            }
        }
    }

    NamedFixture {
        tok_types,
        tok_starts,
        tok_lens,
        haystack,
    }
}

pub(crate) fn annotated_named_vast(fix: &NamedFixture) -> Vec<u8> {
    let raw = reference_c11_build_vast_nodes(&fix.tok_types, &fix.tok_starts, &fix.tok_lens);
    reference_c11_annotate_typedef_names(&raw, &fix.haystack)
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// typedef int T;
/// void f() {
///   T x;
///   {
///     float T;
///     T = 1.0f;
///   }
///   T y;
/// }
pub(crate) fn fixture_typedef_shadowing() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_TYPEDEF,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 0-3: typedef int T;
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN, // 4-8: void f(void)
        TOK_LBRACE, // 9: {
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 10-12: T x;
        TOK_LBRACE,    // 13: {
        TOK_FLOAT_KW,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 14-16: float T;
        TOK_IDENTIFIER,
        TOK_ASSIGN,
        TOK_FLOAT,
        TOK_SEMICOLON, // 17-20: T = 1.0f;
        TOK_RBRACE,    // 21: }
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 22-24: T y;
        TOK_RBRACE,    // 25: }
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// typedef int T;
/// void f() {
///   (T)*x;  // cast
///   int T;
///   (T)*x;  // multiply
/// }
pub(crate) fn fixture_cast_vs_multiply() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_TYPEDEF,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 0-3: typedef int T;
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN, // 4-8
        TOK_LBRACE, // 9
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 10-15: (T)*x;
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 16-18: int T;
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // 19-24: (T)*x;
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// int (*(*f)(int))(float);
pub(crate) fn fixture_nested_fnptr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_INT,
        TOK_LPAREN,
        TOK_STAR,
        TOK_LPAREN,
        TOK_STAR,
        TOK_IDENTIFIER,
        TOK_RPAREN,
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

/// struct T { int x; };
/// typedef int T;
/// void f() {
///   struct T a;
///   T b;
/// }
pub(crate) fn fixture_tag_separation() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_LBRACE,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_RBRACE,
        TOK_SEMICOLON,
        TOK_TYPEDEF,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_SEMICOLON,
        TOK_VOID,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_VOID,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_STRUCT,
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // struct T a;
        TOK_IDENTIFIER,
        TOK_IDENTIFIER,
        TOK_SEMICOLON, // T b;
        TOK_RBRACE,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// __attribute__((pure)) int g(int x);
pub(crate) fn fixture_gnu_attributes() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_GNU_ATTRIBUTE,
        TOK_LPAREN,
        TOK_LPAREN,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_RPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_LPAREN,
        TOK_INT,
        TOK_IDENTIFIER,
        TOK_RPAREN,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}

/// (int){1}
pub(crate) fn fixture_compound_literal() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tok_types = vec![
        TOK_LPAREN,
        TOK_INT,
        TOK_RPAREN,
        TOK_LBRACE,
        TOK_INTEGER,
        TOK_RBRACE,
        TOK_SEMICOLON,
    ];
    let tok_lens = vec![1; tok_types.len()];
    let tok_starts = starts_for_lens(&tok_lens);
    (tok_types, tok_starts, tok_lens)
}
