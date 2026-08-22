//! The one static type walker, and the environment it reads free names from.

use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::op_signature::DataType;
use crate::ir_inner::model::program::BufferDecl;
use crate::validate::Binding;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use vyre_spec::BinOpResult;

/// Environment the one expression type walker reads its free names from.
///
/// [`expr_type`] is the only answer in this crate to "what type does this
/// expression have". Its consumers - IR validation, the optimizer fact cache,
/// and autodiff forward-type recording - differ only in where a scalar's type
/// and a buffer's element type come from, and in whether they want the type of
/// every subexpression the walk resolves along the way.
pub(crate) trait TypeEnv {
    /// Static type of the scalar named `name`, or `None` when it is unbound or
    /// its binding type could not be inferred.
    fn var_type(&self, name: &str) -> Option<DataType>;

    /// Declared element type of the buffer named `name`.
    fn buffer_element(&self, name: &str) -> Option<DataType>;

    /// Observe the resolved type of every expression the walk visits, in
    /// post-order, subexpressions included.
    ///
    /// The walk enters every child even when the parent's type does not depend
    /// on it, so an implementor that records these gets the type of a whole
    /// expression tree out of one traversal.
    fn on_typed(&mut self, _expr: &Expr, _ty: Option<&DataType>) {}
}

/// [`TypeEnv`] backed by the validator's buffer table and lexical scope.
pub(crate) struct ScopeTypes<'a> {
    buffers: &'a FxHashMap<&'a str, &'a BufferDecl>,
    scope: &'a FxHashMap<crate::ir::Ident, Binding>,
}

impl<'a> ScopeTypes<'a> {
    #[inline]
    pub(crate) fn new(
        buffers: &'a FxHashMap<&'a str, &'a BufferDecl>,
        scope: &'a FxHashMap<crate::ir::Ident, Binding>,
    ) -> Self {
        Self { buffers, scope }
    }
}

impl TypeEnv for ScopeTypes<'_> {
    #[inline]
    fn var_type(&self, name: &str) -> Option<DataType> {
        self.scope.get(name).map(|binding| binding.ty.clone())
    }

    #[inline]
    fn buffer_element(&self, name: &str) -> Option<DataType> {
        self.buffers.get(name).map(|buffer| buffer.element.clone())
    }
}

/// Infer the static type of an expression, if it can be determined from the IR.
///
/// The walk is iterative so a deep expression cannot overflow the stack, and it
/// visits every child so [`TypeEnv::on_typed`] sees the whole tree.
#[inline]
#[expect(
    clippy::too_many_lines,
    reason = "iterative expression type inference keeps every Expr variant in one non-recursive dispatch table to preserve stack-safety and exhaustiveness"
)]
pub(crate) fn expr_type<E: TypeEnv + ?Sized>(expr: &Expr, env: &mut E) -> Option<DataType> {
    /// How the result of an expression is produced once its children are typed.
    enum Combine {
        /// Drop the given number of child results; the answer is the value left
        /// underneath, which is either a type pushed before the children were
        /// entered or the first child's own type.
        Drop(usize),
        /// Unify two arithmetic operand types.
        Arith,
        /// `cond`, `true_val`, `false_val`: the two arms must agree.
        Select,
        /// `a`, `b`, `c`: fused multiply-add is F32 only.
        Fma,
    }

    enum Frame<'a> {
        Enter(&'a Expr),
        Combine(&'a Expr, Combine),
    }

    type Frames<'a> = SmallVec<[Frame<'a>; 32]>;

    fn plan<'a, I>(expr: &'a Expr, combine: Combine, children: I, frames: &mut Frames<'a>)
    where
        I: IntoIterator<Item = &'a Expr>,
        I::IntoIter: DoubleEndedIterator,
    {
        frames.push(Frame::Combine(expr, combine));
        for child in children.into_iter().rev() {
            frames.push(Frame::Enter(child));
        }
    }

    use crate::ir_inner::model::op_signature::UnOp;

    let mut frames: Frames<'_> = SmallVec::new();
    frames.push(Frame::Enter(expr));
    let mut values: SmallVec<[Option<DataType>; 32]> = SmallVec::new();
    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Enter(expr) => match expr {
                Expr::LitU32(_)
                | Expr::BufLen { .. }
                | Expr::InvocationId { .. }
                | Expr::WorkgroupId { .. }
                | Expr::LocalId { .. }
                | Expr::SubgroupLocalId
                | Expr::SubgroupSize => {
                    values.push(Some(DataType::U32));
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                // A buffer reference names a buffer rather than producing a
                // value, so it has no type. Reporting one would let it pass
                // an operand typecheck it must never pass.
                Expr::BufferRef { .. } => {
                    values.push(None);
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                Expr::LitI32(_) => {
                    values.push(Some(DataType::I32));
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                Expr::LitF32(_) => {
                    values.push(Some(DataType::F32));
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                Expr::LitBool(_) => {
                    values.push(Some(DataType::Bool));
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                Expr::Var(name) => {
                    values.push(env.var_type(name.as_str()));
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                Expr::Opaque(extension) => {
                    values.push(extension.result_type());
                    plan(expr, Combine::Drop(0), [], &mut frames);
                }
                Expr::Load { buffer, index } => {
                    values.push(env.buffer_element(buffer.as_str()));
                    plan(expr, Combine::Drop(1), [index.as_ref()], &mut frames);
                }
                Expr::Cast { target, value } => {
                    values.push(Some(target.clone()));
                    plan(expr, Combine::Drop(1), [value.as_ref()], &mut frames);
                }
                Expr::Call { args, .. } => {
                    values.push(None);
                    plan(expr, Combine::Drop(args.len()), args.iter(), &mut frames);
                }
                Expr::BinOp { op, left, right } => {
                    let operands = [left.as_ref(), right.as_ref()];
                    match op.result_class() {
                        BinOpResult::Numeric => plan(expr, Combine::Arith, operands, &mut frames),
                        // Logical And/Or and all comparisons evaluate to Bool.
                        // The reference interpreter produces Value::Bool here, so
                        // the static type must match or programs like `(a && b) + 1`
                        // pass validation and then fail at interpreter time.
                        BinOpResult::Predicate => {
                            values.push(Some(DataType::Bool));
                            plan(expr, Combine::Drop(2), operands, &mut frames);
                        }
                        // Bitwise, modulo, shift, rotate, unsigned absolute
                        // difference, multiply-high, and the wave operators are
                        // integer-typed, and an extension operator has no
                        // declared result type. U32 is the safe default for both:
                        // the operand-checker already rejects non-integer
                        // operands, and an answer here keeps the enclosing
                        // operator's mixed-type check armed.
                        BinOpResult::Integer | BinOpResult::Extension => {
                            values.push(Some(DataType::U32));
                            plan(expr, Combine::Drop(2), operands, &mut frames);
                        }
                    }
                }
                Expr::UnOp { op, operand } => {
                    let operand = [operand.as_ref()];
                    match op {
                        UnOp::Negate
                        | UnOp::BitNot
                        | UnOp::Popcount
                        | UnOp::Clz
                        | UnOp::Ctz
                        | UnOp::ReverseBits => plan(expr, Combine::Drop(0), operand, &mut frames),
                        // LogicalNot produces Bool. Integer lowering emits
                        // `x == 0u`, which also yields Bool.
                        UnOp::LogicalNot | UnOp::IsNan | UnOp::IsInf | UnOp::IsFinite => {
                            values.push(Some(DataType::Bool));
                            plan(expr, Combine::Drop(1), operand, &mut frames);
                        }
                        UnOp::Sin
                        | UnOp::Cos
                        | UnOp::Exp
                        | UnOp::Log
                        | UnOp::Log2
                        | UnOp::Exp2
                        | UnOp::Tan
                        | UnOp::Acos
                        | UnOp::Asin
                        | UnOp::Atan
                        | UnOp::Tanh
                        | UnOp::Sinh
                        | UnOp::Cosh
                        | UnOp::Abs
                        | UnOp::Sqrt
                        | UnOp::InverseSqrt
                        | UnOp::Reciprocal
                        | UnOp::Floor
                        | UnOp::Ceil
                        | UnOp::Round
                        | UnOp::Trunc
                        | UnOp::Sign => {
                            values.push(Some(DataType::F32));
                            plan(expr, Combine::Drop(1), operand, &mut frames);
                        }
                        // Lane unpacking and extension operators have no
                        // statically known result type.
                        _ => {
                            values.push(None);
                            plan(expr, Combine::Drop(1), operand, &mut frames);
                        }
                    }
                }
                Expr::Select {
                    cond,
                    true_val,
                    false_val,
                } => plan(
                    expr,
                    Combine::Select,
                    [cond.as_ref(), true_val.as_ref(), false_val.as_ref()],
                    &mut frames,
                ),
                Expr::Fma { a, b, c } => plan(
                    expr,
                    Combine::Fma,
                    [a.as_ref(), b.as_ref(), c.as_ref()],
                    &mut frames,
                ),
                Expr::Atomic {
                    index,
                    expected,
                    value,
                    ..
                } => {
                    values.push(Some(DataType::U32));
                    let operands = [index.as_ref()]
                        .into_iter()
                        .chain(expected.as_deref())
                        .chain([value.as_ref()]);
                    plan(
                        expr,
                        Combine::Drop(2 + usize::from(expected.is_some())),
                        operands,
                        &mut frames,
                    );
                }
                Expr::SubgroupBallot { cond } => {
                    values.push(Some(DataType::U32));
                    plan(expr, Combine::Drop(1), [cond.as_ref()], &mut frames);
                }
                // Both operations produce the same type as their value operand.
                Expr::SubgroupShuffle { value, lane } => plan(
                    expr,
                    Combine::Drop(1),
                    [value.as_ref(), lane.as_ref()],
                    &mut frames,
                ),
                Expr::SubgroupReduce { value, .. } => {
                    plan(expr, Combine::Drop(0), [value.as_ref()], &mut frames);
                }
            },
            Frame::Combine(expr, combine) => {
                match combine {
                    Combine::Drop(count) => {
                        values.truncate(values.len().saturating_sub(count));
                    }
                    Combine::Arith => {
                        let right = values.pop().unwrap_or(None);
                        let left = values.pop().unwrap_or(None);
                        // Operands unify, or the result falls back to U32. The
                        // fallback is deliberately confident: answering `None`
                        // for a mismatched or unknown operand pair would
                        // disarm the mixed-type and saturating-operand checks
                        // of every enclosing operator.
                        values.push(Some(match (left, right) {
                            (Some(left), Some(right)) if left == right => left,
                            _ => DataType::U32,
                        }));
                    }
                    Combine::Select => {
                        let false_ty = values.pop().unwrap_or(None);
                        let true_ty = values.pop().unwrap_or(None);
                        values.pop();
                        values.push(if true_ty == false_ty { true_ty } else { None });
                    }
                    Combine::Fma => {
                        let c = values.pop().unwrap_or(None);
                        let b = values.pop().unwrap_or(None);
                        let a = values.pop().unwrap_or(None);
                        values.push(
                            (a == Some(DataType::F32)
                                && b == Some(DataType::F32)
                                && c == Some(DataType::F32))
                            .then_some(DataType::F32),
                        );
                    }
                }
                env.on_typed(expr, values.last().and_then(Option::as_ref));
            }
        }
    }
    values.pop().flatten()
}
