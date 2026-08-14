//! Token fixtures for expression-shape gaps: prefix and postfix unary operators, casts, member
//! access, subscripts, designators, and GNU case ranges.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.


/// ```c
/// int a = ++x;
/// int b = y--;
/// int c = &z;
/// int d = *w;
/// int e = +v;
/// int f = -u;
/// int g = ~t;
/// int h = !s;
/// ```

use crate::c_frontend::spelling::c_kinds;
pub(crate) fn fixture_unary_prefix_and_postfix() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "INT IDENTIFIER ASSIGN INC IDENTIFIER SEMICOLON INT IDENTIFIER ASSIGN IDENTIFIER \
         DEC SEMICOLON INT IDENTIFIER ASSIGN AMP IDENTIFIER SEMICOLON INT IDENTIFIER \
         ASSIGN STAR IDENTIFIER SEMICOLON INT IDENTIFIER ASSIGN PLUS IDENTIFIER SEMICOLON \
         INT IDENTIFIER ASSIGN MINUS IDENTIFIER SEMICOLON INT IDENTIFIER ASSIGN TILDE \
         IDENTIFIER SEMICOLON INT IDENTIFIER ASSIGN BANG IDENTIFIER SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

/// ```c
/// int a = (int)x;
/// ```
pub(crate) fn fixture_cast_expr() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds("INT IDENTIFIER ASSIGN LPAREN INT RPAREN IDENTIFIER SEMICOLON");
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

/// ```c
/// int a = s.m;
/// int b = p->m;
/// ```
pub(crate) fn fixture_member_access() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "INT IDENTIFIER ASSIGN IDENTIFIER DOT IDENTIFIER SEMICOLON INT IDENTIFIER ASSIGN \
         IDENTIFIER ARROW IDENTIFIER SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

/// ```c
/// int a = arr[0];
/// ```
pub(crate) fn fixture_array_subscript() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
         INT IDENTIFIER ASSIGN IDENTIFIER LBRACKET INTEGER RBRACKET SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

/// ```c
/// struct S s = { .x = 1, [0] = 2 };
/// ```
pub(crate) fn fixture_designated_initializer() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "STRUCT IDENTIFIER IDENTIFIER ASSIGN LBRACE DOT IDENTIFIER ASSIGN INTEGER COMMA \
         LBRACKET INTEGER RBRACKET ASSIGN INTEGER RBRACE SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

/// ```c
/// int a[] = { [0 ... 1] = 2 };
/// ```
pub(crate) fn fixture_array_range_designator() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "INT IDENTIFIER LBRACKET RBRACKET ASSIGN LBRACE LBRACKET INTEGER ELLIPSIS INTEGER \
         RBRACKET ASSIGN INTEGER RBRACE SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

/// ```c
/// switch(x) { case 1 ... 5: break; }
/// ```
pub(crate) fn fixture_gnu_case_range() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "SWITCH LPAREN IDENTIFIER RPAREN LBRACE CASE INTEGER ELLIPSIS INTEGER COLON BREAK \
         SEMICOLON RBRACE",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
