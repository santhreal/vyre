//! Structural C11 parser passes.

/// Declaration specifier + declarator extraction.
pub mod declarations;
mod gnu_builtin_catalog;
/// GNU builtin recognition pass.
pub mod gnu_builtins;
/// `asm` / `__asm__` inline-assembly extraction.
pub mod inline_asm;
/// Function / struct / enum structural pass.
pub mod structure;
/// GPU statement-bound extraction used by AST windowing.
pub mod structure_statement;
/// Token stream to packed VAST rows.
pub mod vast;
mod vast_kinds;

pub(crate) fn token_range_expr(token: &vyre_foundation::ir::Expr, lo: u32, hi: u32) -> vyre_foundation::ir::Expr {
    use vyre_foundation::ir::Expr;
    if lo == hi {
        Expr::eq(token.clone(), Expr::u32(lo))
    } else {
        Expr::and(
            Expr::ge(token.clone(), Expr::u32(lo)),
            Expr::le(token.clone(), Expr::u32(hi)),
        )
    }
}

pub(crate) fn merged_token_ranges(values: &[u32]) -> Vec<(u32, u32)> {
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    sorted.dedup();

    let mut ranges: Vec<(u32, u32)> = Vec::new();
    for value in sorted {
        match ranges.last_mut() {
            Some((_, hi)) if hi.checked_add(1) == Some(value) => *hi = value,
            _ => ranges.push((value, value)),
        }
    }
    ranges
}
