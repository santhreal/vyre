//! What an `Expr` variant carries.
//!
//! The value-namespace counterpart of [`super::node`]: which operands an
//! expression variant holds, which buffer it names and in which direction, and
//! the two sub-expression walks built directly on those answers. Exhaustive
//! with no catch-all arm for the same reason.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::op_signature::{AtomicOp, BinOp, SubgroupReduceOp, UnOp};
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
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
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

/// The buffer-name position an expression holds, as a rewrite can replace it.
#[derive(Debug)]
pub enum ExprBufferName<'a> {
    /// Names no buffer.
    None,
    /// Holds this buffer name.
    Named(&'a mut Ident),
    /// An out-of-tree extension, whose buffer references core cannot
    /// enumerate. A rewrite that has to be complete refuses rather than
    /// leaving a reference it could not inspect.
    Unknown,
}

/// The buffer name `expr` holds, borrowed for replacement.
///
/// The write direction of [`expr_buffer_ref`], and exhaustive for the same
/// reason: a variant that names a buffer and answers [`ExprBufferName::None`]
/// here keeps a reference to a name a rename has already retired, which lowers
/// as a load from a buffer the program does not declare.
pub fn expr_buffer_name_mut(expr: &mut Expr) -> ExprBufferName<'_> {
    match expr {
        Expr::Atomic { buffer, .. }
        | Expr::Load { buffer, .. }
        | Expr::BufLen { buffer }
        | Expr::BufferRef { buffer } => ExprBufferName::Named(buffer),
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
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
        | Expr::SubgroupSize => ExprBufferName::None,
        Expr::Opaque(_) => ExprBufferName::Unknown,
    }
}

/// The worklist an in-place expression rewrite drives.
///
/// Sized like the borrowing [`push_expr_children`] stack, so an expression of
/// ordinary depth is rewritten without touching the heap.
pub type ExprStackMut<'a> = SmallVec<[&'a mut Expr; 16]>;

/// Push every operand of `expr` onto a rewriting worklist, in source order.
///
/// The write direction of [`expr_children`], exhaustive for the same reason: a
/// variant whose operands are not offered here is a subtree no in-place
/// rewrite reaches.
pub fn push_expr_children_mut<'a>(expr: &'a mut Expr, sink: &mut ExprStackMut<'a>) {
    match expr {
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufferRef { .. }
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize
        | Expr::Opaque(_) => {}
        Expr::Load { index, .. }
        | Expr::UnOp { operand: index, .. }
        | Expr::Cast { value: index, .. }
        | Expr::SubgroupBallot { cond: index }
        | Expr::SubgroupReduce { value: index, .. } => sink.push(index),
        Expr::BinOp { left, right, .. } => {
            sink.push(left);
            sink.push(right);
        }
        Expr::SubgroupShuffle { value, lane } => {
            sink.push(value);
            sink.push(lane);
        }
        Expr::Select {
            cond,
            true_val,
            false_val,
        } => {
            sink.push(cond);
            sink.push(true_val);
            sink.push(false_val);
        }
        Expr::Fma { a, b, c } => {
            sink.push(a);
            sink.push(b);
            sink.push(c);
        }
        Expr::Atomic {
            index,
            expected,
            value,
            ..
        } => {
            sink.push(index);
            if let Some(expected) = expected.as_deref_mut() {
                sink.push(expected);
            }
            sink.push(value);
        }
        Expr::Call { args, .. } => sink.extend(args.iter_mut()),
    }
}

/// The cross-invocation combine an expression applies.
#[derive(Debug, Clone, Copy)]
pub enum ExprCombine<'a> {
    /// A read-modify-write applied by every invocation that reaches it, to the
    /// element of `buffer` its index selects.
    Atomic {
        /// Operator the read-modify-write applies.
        op: &'a AtomicOp,
        /// Buffer whose element the operator combines into.
        buffer: &'a Ident,
    },
    /// A reduction across the lanes of one subgroup, over a computed value whose
    /// element type the expression does not state.
    Subgroup {
        /// Operator the reduction applies.
        op: SubgroupReduceOp,
    },
    /// An out-of-tree extension, whose combine core cannot enumerate. A caller
    /// whose answer has to be sound must treat it as combining in an order it
    /// cannot prove.
    Unknown,
}

/// The combine `expr` applies across invocations, if it applies one.
///
/// The question is which expressions produce a result that depends on the order
/// invocations reach them, because a schedule that reorders invocations is legal
/// over such an expression only when the operator's laws say the order does not
/// matter. Exhaustive with no catch-all arm: an expression variant that combines
/// and is classified as combining nothing would make every reordering schedule
/// look legal over it.
#[must_use]
pub fn expr_combine(expr: &Expr) -> Option<ExprCombine<'_>> {
    match expr {
        Expr::Atomic { op, buffer, .. } => Some(ExprCombine::Atomic { op, buffer }),
        Expr::SubgroupReduce { op, .. } => Some(ExprCombine::Subgroup { op: *op }),
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::Var(_)
        | Expr::BufferRef { .. }
        | Expr::Load { .. }
        | Expr::BufLen { .. }
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
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
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => None,
        Expr::Opaque(_) => Some(ExprCombine::Unknown),
    }
}

/// Where the magnitude of an expression's value comes from.
///
/// The question a loop bound asks. `Node::Loop` runs for as many iterations as
/// its `to` expression states, so a bound that reaches a buffer element runs
/// for as long as that element says, and one out-of-contract `u32` read from a
/// producer buffer asks for four billion iterations: hours in the reference
/// interpreter, a watchdog reset on a device.
///
/// The classification is provenance, not size. A value derived only from
/// literals, launch geometry and declared buffer extents is fixed when the
/// program is built, however large it is. A value derived from buffer contents
/// is fixed by whatever ran before it.
#[derive(Debug, Clone, Copy)]
pub enum ExprMagnitude<'a> {
    /// Fixed when the program is built: a literal, a launch-geometry index, or
    /// a subgroup fact the target declares.
    HostFact,
    /// The declared extent of the named buffer.
    BufferExtent(&'a Ident),
    /// Whatever the named buffer holds.
    BufferElement(&'a Ident),
    /// The value the named binding carries.
    Binding(&'a Ident),
    /// At most the number of bits in one element, whatever the operand holds.
    BitCount,
    /// Zero or one, whatever the operands hold.
    Predicate,
    /// At most the smallest operand, so the result is fixed at build time when
    /// any one operand is. This is the arm a clamp against a buffer extent
    /// relies on.
    LeastOperand,
    /// A function of every operand, so the result is fixed at build time only
    /// when all of them are.
    AllOperands,
    /// Core cannot attribute the value: an out-of-tree extension, a call whose
    /// callee this crate does not resolve, or an operator with no recorded
    /// decision in [`bin_op_magnitude`] or [`un_op_magnitude`].
    Unknown,
}

/// Where the magnitude of `expr` comes from.
///
/// Exhaustive with no catch-all arm, for the reason [`expr_children`] is: a new
/// `Expr` variant defaulting to [`ExprMagnitude::HostFact`] would let a loop
/// bound built from it read as fixed when the program is built.
#[inline]
#[must_use]
pub fn expr_magnitude(expr: &Expr) -> ExprMagnitude<'_> {
    match expr {
        Expr::LitU32(_)
        | Expr::LitI32(_)
        | Expr::LitF32(_)
        | Expr::LitBool(_)
        | Expr::InvocationId { .. }
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
        | Expr::WorkgroupId { .. }
        | Expr::LocalId { .. }
        | Expr::SubgroupLocalId
        | Expr::SubgroupSize => ExprMagnitude::HostFact,
        Expr::BufLen { buffer } => ExprMagnitude::BufferExtent(buffer),
        Expr::Var(name) => ExprMagnitude::Binding(name),
        Expr::Load { buffer, .. } | Expr::Atomic { buffer, .. } => {
            ExprMagnitude::BufferElement(buffer)
        }
        Expr::UnOp { op, .. } => un_op_magnitude(op),
        Expr::BinOp { op, .. } => bin_op_magnitude(op),
        Expr::Cast { .. }
        | Expr::Select { .. }
        | Expr::Fma { .. }
        | Expr::SubgroupShuffle { .. } => ExprMagnitude::AllOperands,
        Expr::BufferRef { .. }
        | Expr::Call { .. }
        | Expr::SubgroupBallot { .. }
        | Expr::SubgroupReduce { .. }
        | Expr::Opaque(_) => ExprMagnitude::Unknown,
    }
}

/// Where the magnitude of a binary operator's result comes from.
///
/// `BinOp` is `#[non_exhaustive]` and owned by another crate, so this match
/// carries a catch-all. The catch-all answers [`ExprMagnitude::Unknown`], so an
/// operator added without a decision here makes every loop bound built from it
/// unattributable, which is a finding rather than silence.
#[must_use]
pub fn bin_op_magnitude(op: &BinOp) -> ExprMagnitude<'static> {
    match op {
        BinOp::BitAnd | BinOp::Mod | BinOp::Min => ExprMagnitude::LeastOperand,
        BinOp::Eq
        | BinOp::Ne
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Le
        | BinOp::Ge
        | BinOp::And
        | BinOp::Or => ExprMagnitude::Predicate,
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::WrappingAdd
        | BinOp::WrappingSub
        | BinOp::BitOr
        | BinOp::BitXor
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::AbsDiff
        | BinOp::Max
        | BinOp::SaturatingAdd
        | BinOp::SaturatingSub
        | BinOp::SaturatingMul
        | BinOp::RotateLeft
        | BinOp::RotateRight
        | BinOp::MulHigh => ExprMagnitude::AllOperands,
        _ => ExprMagnitude::Unknown,
    }
}

/// Where the magnitude of a unary operator's result comes from.
///
/// Catch-all for the same reason [`bin_op_magnitude`] carries one.
#[must_use]
pub fn un_op_magnitude(op: &UnOp) -> ExprMagnitude<'static> {
    match op {
        UnOp::Popcount | UnOp::Clz | UnOp::Ctz => ExprMagnitude::BitCount,
        UnOp::LogicalNot | UnOp::IsNan | UnOp::IsInf | UnOp::IsFinite | UnOp::Sign => {
            ExprMagnitude::Predicate
        }
        UnOp::Negate | UnOp::Abs | UnOp::Floor | UnOp::Ceil | UnOp::Round | UnOp::Trunc => {
            ExprMagnitude::AllOperands
        }
        _ => ExprMagnitude::Unknown,
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

    /// True when the expression has no child operands.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.direct[0].is_none() && self.args.is_empty()
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
        | Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. }
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

/// True when `expr` has no operand expressions.
#[inline]
#[must_use]
pub fn expr_is_leaf(expr: &Expr) -> bool {
    expr_children(expr).is_empty()
}

/// Push every operand of `expr` onto an order-insensitive worklist.
pub fn push_expr_children<'a>(expr: &'a Expr, stack: &mut SmallVec<[&'a Expr; 16]>) {
    stack.extend(expr_children(expr).iter());
}
