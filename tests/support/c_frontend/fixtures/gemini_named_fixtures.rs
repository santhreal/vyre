//! Named token fixtures for the Gemini C AST contracts: typedef shadowing, cast versus multiply,
//! nested function pointers, tag separation, GNU attributes, and compound literals.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

use crate::c_frontend::spelling::c_rows;

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
