//! Raw token streams shared by the two initializer-designator families.
//!
//! `c_ast_initializer_designator_e2e` covers the constructs a C program writes;
//! `c_ast_initializer_designator_deep_contracts` covers the corners the lowerer
//! gets wrong. Two families of one construct group means a stream that belongs to
//! both, and a union field designator is that stream: each family spelled it
//! itself, with the same tokens, so a change to what a union designator lowers to
//! had two places to be recorded and one of them would stay behind.
//!
//! A stream lives here only when both families index it. A stream one family owns
//! stays in that family, because moving it here would put a construct's operands
//! away from the contract that reads them by row.

/// ```c
/// union U u = { .f = 42 };
/// ```
///
/// The e2e family asserts the initializer list and the member access it lowers
/// to; the deep family asserts the same rows survive the property-graph
/// lowering. Row positions are therefore load bearing in both, which is the
/// reason the stream cannot differ between them.
pub(crate) fn union_field_designator() -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    crate::c_frontend::spelling::c_rows(
        "UNION:5 IDENTIFIER IDENTIFIER ASSIGN \
         LBRACE DOT IDENTIFIER ASSIGN INTEGER:2 RBRACE SEMICOLON",
    )
}
