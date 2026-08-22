//! Substrate-neutral parsing primitives (AST, delimiters, bracket matching).

pub mod ast;

pub(crate) fn ascii_whitespace_expr(value: vyre_foundation::ir::Expr) -> vyre_foundation::ir::Expr {
    use vyre_foundation::ir::Expr;
    let byte_eq = |candidate: Expr, byte: u8| Expr::eq(candidate, Expr::u32(u32::from(byte)));
    Expr::or(
        byte_eq(value.clone(), b' '),
        Expr::or(
            byte_eq(value.clone(), b'\n'),
            Expr::or(byte_eq(value.clone(), b'\r'), byte_eq(value, b'\t')),
        ),
    )
}
