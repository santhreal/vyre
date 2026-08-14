//! Token fixtures for GNU builtin expressions and C11 `_Generic` selections.
//!
//! The CPU contracts in `vyre-libs/tests` and the backend parity arm in the
//! driver crate build the same token streams, so the fixtures have one owner
//! here rather than a copy per crate.

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

use crate::c_frontend::spelling::c_kinds;
pub(crate) fn builtin_constant_p_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds("BUILTIN_CONSTANT_P LPAREN IDENTIFIER RPAREN SEMICOLON");
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn builtin_choose_expr_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types =
        c_kinds("BUILTIN_CHOOSE_EXPR LPAREN INTEGER COMMA INTEGER COMMA INTEGER RPAREN SEMICOLON");
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn builtin_types_compatible_p_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds("BUILTIN_TYPES_COMPATIBLE_P LPAREN INT COMMA LONG RPAREN SEMICOLON");
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn generic_selection_fixture() -> (Vec<u32>, Vec<u32>) {
    let tok_types = c_kinds(
        "GENERIC LPAREN IDENTIFIER COMMA INT COLON INTEGER COMMA DEFAULT COLON INTEGER \
         RPAREN SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}

pub(crate) fn nested_builtin_fixture() -> (Vec<u32>, Vec<u32>) {
    // __builtin_choose_expr(1, __builtin_constant_p(2), 0);
    let tok_types = c_kinds(
        "BUILTIN_CHOOSE_EXPR LPAREN INTEGER COMMA BUILTIN_CONSTANT_P LPAREN INTEGER \
         RPAREN COMMA INTEGER RPAREN SEMICOLON",
    );
    let tok_lens = vec![1; tok_types.len()];
    (tok_types, tok_lens)
}
