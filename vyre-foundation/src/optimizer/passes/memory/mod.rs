//! Memory / data-layout catalog (Phase 4C).
//!
//! Buffer + load/store rewrites: const-buffer folding, dead-buffer
//! elimination, vector/coalescing layout hint promotion, and the
//! decode→scan storage-to-workgroup handoff fusion.

/// Compile-time constant-buffer load folding.
pub mod const_buffer_fold;
/// Remove declared buffers that cannot affect any output.
pub mod dead_buffer_elim;
/// Drop sibling `Node::Store` whose value is overwritten before any
/// reader can observe it (ROADMAP A20).
pub mod dead_store_elim;
/// Decode→scan storage-to-workgroup handoff fusion.
pub mod decode_scan_fuse;
/// Hoist `Let(name, Load(ro_buf, idx))` out of common branch
/// prefixes when `ro_buf` is declared `BufferAccess::ReadOnly`
/// (ROADMAP A15  -  buffer-aliasing-fact-aware load elision via
/// the trivial alias proof).
pub mod read_only_load_hoist;
/// Replace `Let(name, Load(b, i))` with the value of an immediately
/// preceding `Store(b, i, v)` when no intervening node could observe
/// or mutate `b` between the two (ROADMAP A22).
pub mod store_to_load_forward;
/// Proven-safe vector/coalescing layout hint promotion.
pub mod vectorization;

fn expr_touches_buffer(
    expr: &crate::ir::Expr,
    buffer: &crate::ir::Ident,
    foreign_compare_exchange_touches: bool,
) -> bool {
    use crate::ir::{AtomicOp, Expr};
    let recurse = |child| expr_touches_buffer(child, buffer, foreign_compare_exchange_touches);
    match expr {
        Expr::Load {
            buffer: other,
            index,
        } => other == buffer || recurse(index),
        Expr::BufLen { buffer: other } | Expr::BufferRef { buffer: other } => other == buffer,
        Expr::Atomic {
            buffer: other,
            index,
            expected,
            value,
            op,
            ..
        } => {
            other == buffer
                || recurse(index)
                || (foreign_compare_exchange_touches
                    && matches!(
                        op,
                        AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak
                    ))
                || expected.as_deref().is_some_and(recurse)
                || recurse(value)
        }
        Expr::BinOp { left, right, .. } => recurse(left) || recurse(right),
        Expr::UnOp { operand, .. } | Expr::Cast { value: operand, .. } => recurse(operand),
        Expr::Fma { a, b, c } => recurse(a) || recurse(b) || recurse(c),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => recurse(cond) || recurse(true_val) || recurse(false_val),
        Expr::Call { args, .. } => args.iter().any(recurse),
        Expr::SubgroupReduce { value, .. } => recurse(value),
        Expr::SubgroupShuffle { value, lane } => recurse(value) || recurse(lane),
        Expr::SubgroupBallot { cond } => recurse(cond),
        Expr::Opaque(_) => true,
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => false,
    }
}
