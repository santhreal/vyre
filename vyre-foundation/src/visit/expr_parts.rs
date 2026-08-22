//! What an `Expr` variant carries.
//!
//! The value-namespace counterpart of [`super::node`]: which operands an
//! expression variant holds, which buffer it names and in which direction, and
//! the two sub-expression walks built directly on those answers. Exhaustive
//! with no catch-all arm for the same reason.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::expr::Ident;
use smallvec::SmallVec;

/// What an expression does to the buffer it names.
#[derive(Debug, Clone, Copy)]
pub enum ExprBufferRef<'a> {
    /// Names no buffer.
    None,
    /// Reads the named buffer, or reads its metadata.
    Read(&'a Ident),
    /// Reads and writes the named buffer: an atomic read-modify-write.
    ReadWrite(&'a Ident),
    /// An out-of-tree extension, whose buffer references core cannot enumerate.
    /// A caller whose answer has to be sound must treat it as touching every
    /// buffer.
    Unknown,
}

/// The buffer `expr` names, and what it does to it.
///
/// The expression half of [`super::node_buffer_refs`]. `Expr::Atomic` is the case every
/// buffer-set walk in this crate had recorded as a pure read, which is the
/// direction that loses: a dependency walk that believes an atomic only reads
/// sees no conflict with a store to the same buffer.
#[must_use]
pub fn expr_buffer_ref(expr: &Expr) -> ExprBufferRef<'_> {
    match expr {
        Expr::Atomic { buffer, .. } => ExprBufferRef::ReadWrite(buffer),
        Expr::Load { buffer, .. } | Expr::BufLen { buffer } | Expr::BufferRef { buffer } => {
            ExprBufferRef::Read(buffer)
        }
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::BinOp { .. }
        | Expr::UnOp { .. }
        | Expr::Call { .. }
        | Expr::Select { .. }
        | Expr::Cast { .. }
        | Expr::Fma { .. }
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupShuffle { .. }
        | Expr::SubgroupReduce { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => ExprBufferRef::None,
        Expr::Opaque(_) => ExprBufferRef::Unknown,
    }
}

/// Every operand expression of `expr`, in source order.
///
/// This is the ONE owner of the question "which expression variants contain
/// other expressions", the [`super::child_bodies`] of the value namespace. Adding an
/// `Expr` variant fails to compile in [`expr_children`], and that failure is
/// the mechanism that keeps every expression walk in the crate correct.
///
/// At most three operands are held inline and the argument list of an
/// [`Expr::Call`] is borrowed as a slice, so enumerating children allocates
/// nothing. The whole record is `Copy`.
#[derive(Debug, Clone, Copy)]
pub struct ExprChildren<'a> {
    /// Fixed operand positions, in source order. `None` is an absent optional
    /// operand (`Expr::Atomic::expected`) and is skipped by [`Self::iter`].
    direct: [Option<&'a Expr>; 3],
    /// Call arguments, in source order. Empty for every other variant.
    args: &'a [Expr],
}

impl<'a> ExprChildren<'a> {
    const NONE: Self = Self {
        direct: [None, None, None],
        args: &[],
    };

    const fn one(first: &'a Expr) -> Self {
        Self {
            direct: [Some(first), None, None],
            args: &[],
        }
    }

    const fn two(first: &'a Expr, second: &'a Expr) -> Self {
        Self {
            direct: [Some(first), Some(second), None],
            args: &[],
        }
    }

    const fn three(first: &'a Expr, second: &'a Expr, third: &'a Expr) -> Self {
        Self {
            direct: [Some(first), Some(second), Some(third)],
            args: &[],
        }
    }

    /// The operands in source order.
    ///
    /// The iterator is double-ended, so a stack-based walk that wants children
    /// popped in source order pushes `iter().rev()`.
    pub fn iter(self) -> impl DoubleEndedIterator<Item = &'a Expr> + Clone {
        self.direct.into_iter().flatten().chain(self.args.iter())
    }
}

/// The operands of `expr`, in source order.
///
/// Exhaustive with no catch-all arm, deliberately. Adding an `Expr` variant
/// fails to compile here, and that failure is the point: it forces the author
/// to say which of the new variant's positions a walk owes a visit. A walk that
/// re-derives this with its own `match expr` ending in `_ => {}` classifies a
/// new variant as a leaf, which is how an operand stops being renamed,
/// substituted, counted as a live use, or folded.
#[inline]
#[must_use]
pub fn expr_children(expr: &Expr) -> ExprChildren<'_> {
    match expr {
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufferRef { .. }
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize
        | Expr::Opaque(_) => ExprChildren::NONE,
        Expr::Load { index, .. }
        | Expr::UnOp { operand: index, .. }
        | Expr::Cast { value: index, .. }
        | Expr::SubgroupBallot { cond: index }
        | Expr::SubgroupReduce { value: index, .. } => ExprChildren::one(index),
        Expr::BinOp { left, right, .. } => ExprChildren::two(left, right),
        Expr::SubgroupShuffle { value, lane } => ExprChildren::two(value, lane),
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => ExprChildren::three(cond, true_val, false_val),
        Expr::Fma { a, b, c } => ExprChildren::three(a, b, c),
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => ExprChildren {
            direct: [Some(index), expected.as_deref(), Some(value)],
            args: &[],
        },
        Expr::Call { args, .. } => ExprChildren {
            direct: [None, None, None],
            args,
        },
    }
}

/// True when `expr` or any sub-expression satisfies `pred`.
///
/// Children come from [`expr_children`], so a new operand-carrying variant is
/// covered without touching this function. The walk is an explicit worklist,
/// short-circuiting on the first match, so an adversarially deep expression
/// cannot overflow the native stack.
#[must_use]
pub fn any_subexpr(expr: &Expr, pred: &mut impl FnMut(&Expr) -> bool) -> bool {
    let mut stack: SmallVec<[&Expr; 32]> = SmallVec::new();
    stack.push(expr);
    while let Some(current) = stack.pop() {
        if pred(current) {
            return true;
        }
        stack.extend(expr_children(current).iter().rev());
    }
    false
}

/// Visit `expr` and every sub-expression below it, in source pre-order.
///
/// This is the collector counterpart of [`any_subexpr`]: it visits every node
/// rather than stopping at the first match, so a collector cannot accidentally
/// be written on an early-exit search and lose the operands after the first
/// hit. Children come from [`expr_children`], so a new operand-carrying variant
/// is covered without touching this function, and the walk is an explicit
/// worklist so an adversarially deep expression cannot overflow the native
/// stack.
pub fn for_each_subexpr<'a>(expr: &'a Expr, visit: &mut impl FnMut(&'a Expr)) {
    let mut stack: SmallVec<[&'a Expr; 32]> = SmallVec::new();
    stack.push(expr);
    while let Some(current) = stack.pop() {
        visit(current);
        stack.extend(expr_children(current).iter().rev());
    }
}
