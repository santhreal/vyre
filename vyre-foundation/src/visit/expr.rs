use crate::ir_inner::model::expr::{Expr, ExprNode, Ident};
use crate::ir_inner::model::types::{AtomicOp, BinOp, DataType, UnOp};
use crate::visit::VisitOrder;
use std::ops::ControlFlow;

/// Visitor over [`Expr`] trees.
///
/// Implementors must handle every core variant explicitly. This is
/// intentional: `Expr` is `#[non_exhaustive]`, so a new variant must
/// become a compile error in every visitor instead of silently
/// disappearing behind a default body.
///
/// Traversal order is explicit:
/// - [`visit_preorder`] visits the current expression before its children.
/// - [`visit_postorder`] visits children before the current expression.
///
/// Visitors that want pass-through recursion can call
/// [`ExprVisitor::walk_children_default`] from a variant method.
pub trait ExprVisitor {
    /// Break payload returned when traversal short-circuits.
    type Break;

    /// Integer literal (`u32`).
    fn visit_lit_u32(&mut self, _expr: &Expr, _value: u32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Integer literal (`i32`).
    fn visit_lit_i32(&mut self, _expr: &Expr, _value: i32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Float literal (`f32`).
    fn visit_lit_f32(&mut self, _expr: &Expr, _value: f32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Bool literal.
    fn visit_lit_bool(&mut self, _expr: &Expr, _value: bool) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Variable reference.
    fn visit_var(&mut self, _expr: &Expr, _name: &Ident) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Buffer load (`buffer[index]`).
    fn visit_load(
        &mut self,
        _expr: &Expr,
        _buffer: &Ident,
        _index: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Buffer length.
    fn visit_buf_len(&mut self, _expr: &Expr, _buffer: &Ident) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// A whole buffer named as a call argument.
    fn visit_buffer_ref(&mut self, _expr: &Expr, _buffer: &Ident) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Invocation id axis (`gid.{x,y,z}`).
    fn visit_invocation_id(&mut self, _expr: &Expr, _axis: u32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Workgroup id axis.
    fn visit_workgroup_id(&mut self, _expr: &Expr, _axis: u32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Local id axis within the workgroup.
    fn visit_local_id(&mut self, _expr: &Expr, _axis: u32) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Subgroup invocation id (lane index within subgroup).
    fn visit_subgroup_local_id(&mut self, _expr: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Subgroup size.
    fn visit_subgroup_size(&mut self, _expr: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Binary operation.
    fn visit_bin_op(
        &mut self,
        _expr: &Expr,
        _op: &BinOp,
        _left: &Expr,
        _right: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Unary operation.
    fn visit_un_op(
        &mut self,
        _expr: &Expr,
        _op: &UnOp,
        _operand: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Function call.
    fn visit_call(
        &mut self,
        _expr: &Expr,
        _op_id: &str,
        _args: &[Expr],
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Sequence-valued extension hook.
    ///
    /// Core IR does not currently emit a dedicated `Expr::Sequence`
    /// variant, but downstream visitor implementations must still opt in
    /// explicitly so a sequence node cannot compile behind a silent
    /// default body.
    fn visit_sequence(&mut self, _parts: &[Expr]) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Fused multiply-add (`a * b + c`).
    fn visit_fma(
        &mut self,
        _expr: &Expr,
        _a: &Expr,
        _b: &Expr,
        _c: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Ternary `select(cond, true_val, false_val)`.
    fn visit_select(
        &mut self,
        _expr: &Expr,
        _cond: &Expr,
        _true_val: &Expr,
        _false_val: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Numeric cast.
    fn visit_cast(
        &mut self,
        _expr: &Expr,
        _target: &DataType,
        _value: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Atomic operation on a shared buffer.
    fn visit_atomic(
        &mut self,
        _expr: &Expr,
        _op: &AtomicOp,
        _buffer: &Ident,
        _index: &Expr,
        _expected: Option<&Expr>,
        _value: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Subgroup ballot.
    fn visit_subgroup_ballot(&mut self, _expr: &Expr, _cond: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Subgroup shuffle.
    fn visit_subgroup_shuffle(
        &mut self,
        _expr: &Expr,
        _value: &Expr,
        _lane: &Expr,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Subgroup add.
    fn visit_subgroup_add(&mut self, _expr: &Expr, _value: &Expr) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }
    /// Downstream opaque expression extension.
    fn visit_opaque_expr(
        &mut self,
        _expr: &Expr,
        _extension: &dyn ExprNode,
    ) -> ControlFlow<Self::Break> {
        ControlFlow::Continue(())
    }

    /// Recursively walk this expression's children using the requested order.
    fn walk_children_default(&mut self, expr: &Expr, order: VisitOrder) -> ControlFlow<Self::Break>
    where
        Self: Sized,
    {
        walk_expr_children_default(self, expr, order)
    }
}

/// Kind of direct buffer access observed while traversing an expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExprBufferAccess {
    /// Read dependency through `Expr::Load`, `Expr::BufLen`, or
    /// `Expr::BufferRef`.
    Load,
    /// Read-modify-write through `Expr::Atomic`.
    Atomic,
}

struct ExprBufferAccessVisitor<F> {
    visitor: F,
}

impl<F> ExprVisitor for ExprBufferAccessVisitor<F>
where
    F: FnMut(ExprBufferAccess, &Ident),
{
    type Break = std::convert::Infallible;

    fn visit_load(
        &mut self,
        _expr: &Expr,
        buffer: &Ident,
        _index: &Expr,
    ) -> ControlFlow<Self::Break> {
        (self.visitor)(ExprBufferAccess::Load, buffer);
        ControlFlow::Continue(())
    }

    fn visit_buf_len(&mut self, _expr: &Expr, buffer: &Ident) -> ControlFlow<Self::Break> {
        (self.visitor)(ExprBufferAccess::Load, buffer);
        ControlFlow::Continue(())
    }

    fn visit_buffer_ref(&mut self, _expr: &Expr, buffer: &Ident) -> ControlFlow<Self::Break> {
        (self.visitor)(ExprBufferAccess::Load, buffer);
        ControlFlow::Continue(())
    }

    fn visit_atomic(
        &mut self,
        _expr: &Expr,
        _op: &AtomicOp,
        buffer: &Ident,
        _index: &Expr,
        _expected: Option<&Expr>,
        _value: &Expr,
    ) -> ControlFlow<Self::Break> {
        (self.visitor)(ExprBufferAccess::Atomic, buffer);
        ControlFlow::Continue(())
    }
}

/// Visit every direct data, metadata, whole-buffer, and atomic buffer target.
pub fn visit_expr_buffer_accesses(expr: &Expr, visitor: impl FnMut(ExprBufferAccess, &Ident)) {
    let mut visitor = ExprBufferAccessVisitor { visitor };
    let _ = visit_preorder(&mut visitor, expr);
}

mod traversal;

pub use traversal::{visit_expr, visit_postorder, visit_preorder, walk_expr_children_default};
