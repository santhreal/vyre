//! Named token fixtures for the Gemini C AST contracts: typedef shadowing, cast versus multiply,
//! nested function pointers, tag separation, GNU attributes, and compound literals.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_rows;
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
    c_rows(
        "TYPEDEF INT IDENTIFIER SEMICOLON VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         IDENTIFIER IDENTIFIER SEMICOLON LBRACE FLOAT_KW IDENTIFIER SEMICOLON IDENTIFIER \
         ASSIGN FLOAT SEMICOLON RBRACE IDENTIFIER IDENTIFIER SEMICOLON RBRACE",
    )
}

/// typedef int T;
/// void f() {
///   (T)*x;  // cast
///   int T;
///   (T)*x;  // multiply
/// }
pub(crate) fn fixture_cast_vs_multiply() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "TYPEDEF INT IDENTIFIER SEMICOLON VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         LPAREN IDENTIFIER RPAREN STAR IDENTIFIER SEMICOLON INT IDENTIFIER SEMICOLON \
         LPAREN IDENTIFIER RPAREN STAR IDENTIFIER SEMICOLON RBRACE",
    )
}

/// int (*(*f)(int))(float);
pub(crate) fn fixture_nested_fnptr() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "INT LPAREN STAR LPAREN STAR IDENTIFIER RPAREN LPAREN INT RPAREN RPAREN LPAREN \
         FLOAT_KW RPAREN SEMICOLON",
    )
}

/// struct T { int x; };
/// typedef int T;
/// void f() {
///   struct T a;
///   T b;
/// }
pub(crate) fn fixture_tag_separation() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT IDENTIFIER LBRACE INT IDENTIFIER SEMICOLON RBRACE SEMICOLON TYPEDEF INT \
         IDENTIFIER SEMICOLON VOID IDENTIFIER LPAREN VOID RPAREN LBRACE STRUCT IDENTIFIER \
         IDENTIFIER SEMICOLON IDENTIFIER IDENTIFIER SEMICOLON RBRACE",
    )
}

/// __attribute__((pure)) int g(int x);
pub(crate) fn fixture_gnu_attributes() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "GNU_ATTRIBUTE LPAREN LPAREN IDENTIFIER RPAREN RPAREN INT IDENTIFIER LPAREN INT \
         IDENTIFIER RPAREN SEMICOLON",
    )
}

/// (int){1}
pub(crate) fn fixture_compound_literal() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("LPAREN INT RPAREN LBRACE INTEGER RBRACE SEMICOLON")
}
