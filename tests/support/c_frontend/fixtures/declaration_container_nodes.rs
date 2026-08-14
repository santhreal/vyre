//! Token fixtures for C declaration container nodes: struct, union, enum, typedef, function,
//! bitfield, and static assert declarations.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.


/// ```c
/// struct S { int x; };
/// ```

use crate::c_frontend::spelling::c_rows;
pub(crate) fn fixture_struct_definition() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("STRUCT IDENTIFIER LBRACE INT IDENTIFIER SEMICOLON RBRACE SEMICOLON")
}

/// ```c
/// struct S;
/// ```
pub(crate) fn fixture_struct_forward_declaration() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("STRUCT IDENTIFIER SEMICOLON")
}

/// ```c
/// union U { int i; float f; };
/// ```
pub(crate) fn fixture_union_definition() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "UNION IDENTIFIER LBRACE INT IDENTIFIER SEMICOLON FLOAT_KW IDENTIFIER SEMICOLON \
         RBRACE SEMICOLON",
    )
}

/// ```c
/// union U;
/// ```
pub(crate) fn fixture_union_forward_declaration() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("UNION IDENTIFIER SEMICOLON")
}

/// ```c
/// enum E { A, B };
/// ```
pub(crate) fn fixture_enum_definition() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("ENUM IDENTIFIER LBRACE IDENTIFIER COMMA IDENTIFIER RBRACE SEMICOLON")
}

/// ```c
/// enum E;
/// ```
pub(crate) fn fixture_enum_forward_declaration() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("ENUM IDENTIFIER SEMICOLON")
}

/// ```c
/// typedef int T;
/// ```
pub(crate) fn fixture_typedef_declaration() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("TYPEDEF INT IDENTIFIER SEMICOLON")
}

/// ```c
/// int f(void) { return 0; }
/// ```
pub(crate) fn fixture_function_definition() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("INT IDENTIFIER LPAREN VOID RPAREN LBRACE RETURN INTEGER SEMICOLON RBRACE")
}

/// ```c
/// int f(void);
/// ```
pub(crate) fn fixture_function_prototype() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("INT IDENTIFIER LPAREN VOID RPAREN SEMICOLON")
}

/// ```c
/// struct { int a : 4; unsigned int : 0; };
/// ```
pub(crate) fn fixture_bitfield() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows(
        "STRUCT LBRACE INT IDENTIFIER COLON INTEGER SEMICOLON UNSIGNED INT COLON INTEGER \
         SEMICOLON RBRACE SEMICOLON",
    )
}

/// ```c
/// _Static_assert(1, "ok");
/// ```
pub(crate) fn fixture_static_assert() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    c_rows("STATIC_ASSERT LPAREN INTEGER COMMA STRING RPAREN SEMICOLON")
}
