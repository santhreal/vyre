//! Token fixtures for C typedef and name disambiguation: cast versus multiply, nested shadowing,
//! tag versus typedef names, and declarator contexts.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// typedef int T;
/// void f(void) {
///   (T)*p;   -- cast expression: T is a typedef name
///   (x)*p;   -- multiplication: x is a variable, not a type
/// }
use crate::c_frontend::spelling::c_rows;
pub(crate) fn fixture_typedef_cast_vs_expr_multiply() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "TYPEDEF INT IDENTIFIER SEMICOLON VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         LPAREN IDENTIFIER RPAREN STAR IDENTIFIER SEMICOLON LPAREN IDENTIFIER RPAREN STAR \
         IDENTIFIER SEMICOLON RBRACE",
    )
}

/// typedef int T;
/// void f(void) {
///   {
///     int T;   -- shadows the typedef
///     T * b;   -- multiplication, not pointer declaration
///   }
/// }
pub(crate) fn fixture_typedef_shadowing_nested() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "TYPEDEF INT IDENTIFIER SEMICOLON VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         LBRACE INT IDENTIFIER SEMICOLON IDENTIFIER STAR IDENTIFIER SEMICOLON RBRACE \
         RBRACE",
    )
}

/// struct S { int x; };
/// typedef struct S S;
/// void f(void) {
///   struct S *a;   -- tag name in declaration
///   S *b;          -- typedef name in declaration
/// }
pub(crate) fn fixture_struct_tag_vs_typedef() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT IDENTIFIER LBRACE INT IDENTIFIER SEMICOLON RBRACE SEMICOLON TYPEDEF \
         STRUCT IDENTIFIER IDENTIFIER SEMICOLON VOID IDENTIFIER LPAREN VOID RPAREN LBRACE \
         STRUCT IDENTIFIER STAR IDENTIFIER SEMICOLON IDENTIFIER STAR IDENTIFIER SEMICOLON \
         RBRACE",
    )
}

/// void f(void) {
///   int *a[10];      -- array of pointers
///   int (*a)[10];    -- pointer to array
///   int *f(int);     -- function returning pointer
///   int (*f)(int);   -- pointer to function
/// }
pub(crate) fn fixture_declarator_contexts() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "VOID IDENTIFIER LPAREN VOID RPAREN LBRACE INT STAR IDENTIFIER LBRACKET INTEGER \
         RBRACKET SEMICOLON INT LPAREN STAR IDENTIFIER RPAREN LBRACKET INTEGER RBRACKET \
         SEMICOLON INT STAR IDENTIFIER LPAREN INT RPAREN SEMICOLON INT LPAREN STAR \
         IDENTIFIER RPAREN LPAREN INT RPAREN SEMICOLON RBRACE",
    )
}
