use vyre_foundation::ir::{Expr, Program};

/// Build a Tier-3 elementwise u32 logical binary op.
#[must_use]
pub(crate) fn build_logical_binary<F>(
    op_id: &'static str,
    a: &str,
    b: &str,
    out: &str,
    size: u32,
    op: F,
) -> Program
where
    F: Fn(Expr, Expr) -> Expr,
{
    crate::math::elementwise::u32_elementwise_binary(op_id, a, b, out, size, op)
}
