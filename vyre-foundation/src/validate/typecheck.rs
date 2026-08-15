use crate::ir_inner::model::expr::Expr;
use crate::ir_inner::model::program::BufferDecl;
use crate::ir_inner::model::spec_types::{BinOp, DataType};
use crate::validate::{err, Binding, ValidationError};
use crate::validate::{ValidationLocation, ValidationPhase};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[inline]
#[expect(
    clippy::too_many_lines,
    reason = "binary operator validation is kept as one exhaustive BinOp policy table so type-safety edits review the complete operator surface"
)]
pub(crate) fn validate_binop_operands(
    op: BinOp,
    left: &Expr,
    right: &Expr,
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    let left_ty = expr_type(left, &mut ScopeTypes::new(buffers, scope));
    let right_ty = expr_type(right, &mut ScopeTypes::new(buffers, scope));

    match op {
        // Arithmetic: U32, I32, and F32 are all valid in target-text.
        // Bool is NOT  -  `(a && b) + 1` must be rejected at validation time.
        // Operand types must also match: `u32 + f32` is silently ambiguous
        // today and must be rejected (VAL-003).
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::SaturatingAdd
        | BinOp::SaturatingSub
        | BinOp::SaturatingMul
        | BinOp::Min
        | BinOp::Max
        | BinOp::AbsDiff => {
            if matches!(op, BinOp::Div) && expr_is_static_zero(right) {
                errors.push(err("V044", ValidationPhase::Type, ValidationLocation::Program, "binary operation `Div` has a statically-zero divisor"
                        .to_string(), "guard the divisor, use Select to substitute a non-zero value, or reject the input before building IR."
                        .to_string()));
            }
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                if matches!(l, DataType::U64 | DataType::I64)
                    || matches!(r, DataType::U64 | DataType::I64)
                {
                    errors.push(err("V084", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` received left=`{l}`, right=`{r}`. 64-bit integer arithmetic is outside vyre-foundation's cross-backend arithmetic contract"
                    ), format!(
                        "express the operation as a U32 pair with explicit carry/borrow, or use a backend-specific op whose schema declares native 64-bit arithmetic."
                    )));
                }

                if matches!(
                    op,
                    BinOp::SaturatingAdd | BinOp::SaturatingSub | BinOp::SaturatingMul
                ) && (l != &DataType::U32 || r != &DataType::U32)
                {
                    errors.push(err("V085", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "Saturating arithmetic `{op:?}` received left=`{l}`, right=`{r}`; legal set is only U32 in the current lowering"
                        )
                            .to_string(), format!(
                            "cast both operands to U32, or clamp explicitly for I32/F32."
                        )
                            .to_string()));
                }

                if matches!(op, BinOp::AbsDiff) && (l == &DataType::I32 || r == &DataType::I32) {
                    errors.push(err("V086", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "AbsDiff has left=`{l}`, right=`{r}` and can overflow (i32::MIN - i32::MAX invokes target-text signed-integer UB)"
                        )
                            .to_string(), format!(
                            "cast operands to U32 before AbsDiff, or rewrite as an explicit branch."
                        )
                            .to_string()));
                }
            }
            for (side, ty) in [("left", &left_ty), ("right", &right_ty)] {
                if let Some(ty) = ty {
                    if matches!(ty, DataType::Bool) {
                        errors.push(err("V087", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `{op:?}` {side} operand has type `{ty}`, but numeric arithmetic expects one of `u32`, `i32`, or `f32`"
                        ), format!(
                            "cast the operand to U32 or I32 before arithmetic, or rewrite to avoid mixing logical and arithmetic operators."
                        )));
                    }
                }
            }
            // VAL-003: reject mixed numeric types. target-text has no implicit
            // promotion; `a: u32 + b: f32` must be a cast at the call site,
            // not a silent validator pass.
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                let both_numeric = matches!(l, DataType::U32 | DataType::I32 | DataType::F32)
                    && matches!(r, DataType::U32 | DataType::I32 | DataType::F32);
                if both_numeric && l != r {
                    errors.push(err("V088", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` operands have mismatched numeric types: left=`{l}`, right=`{r}` (legal set: U32, I32, F32)"
                    ), format!(
                        "cast one operand so both sides share a type (target-text has no implicit promotion)."
                    )));
                }
            }
        }
        // Modulo: target emitters support total unsigned modulo and signed
        // modulo with explicit zero/overflow guards, so both operands must be
        // integer operands of the same width.
        BinOp::Mod => {
            if expr_is_static_zero(right) {
                errors.push(err("V044", ValidationPhase::Type, ValidationLocation::Program, "binary operation `Mod` has a statically-zero divisor"
                        .to_string(), "guard the divisor, use Select to substitute a non-zero value, or reject the input before building IR."
                        .to_string()));
            }
            for (side, ty) in [("left", left_ty.as_ref()), ("right", right_ty.as_ref())] {
                if let Some(ty) = ty {
                    if !matches!(ty, DataType::U32 | DataType::I32) {
                        errors.push(err("V089", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `Mod` {side} operand must be `u32` or `i32`, got `{ty}`. Legal set for Mod is integer-only"
                        ), format!(
                            "cast both operands to the same integer type before modulo."
                        )));
                    }
                }
            }
            if let (Some(left), Some(right)) = (&left_ty, &right_ty) {
                if matches!(left, DataType::U32 | DataType::I32)
                    && matches!(right, DataType::U32 | DataType::I32)
                    && left != right
                {
                    errors.push(err("V090", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `Mod` operands have mismatched integer types: left=`{left}`, right=`{right}`"
                    ), format!(
                        "cast one operand so both sides share the same integer type."
                    )));
                }
            }
        }
        // Bitwise: target-text `&` / `|` / `^` require integer operands of the same type.
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor => {
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                if !matches!(l, DataType::U32 | DataType::I32) {
                    errors.push(err("V091", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` left operand has type `{l}`; legal integer set is `u32` or `i32`"
                    ), format!(
                        "cast the left operand to U32 or I32."
                    )));
                }
                if !matches!(r, DataType::U32 | DataType::I32) {
                    errors.push(err("V092", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` right operand has type `{r}`; legal integer set is `u32` or `i32`"
                    ), format!(
                        "cast the right operand to U32 or I32."
                    )));
                }
                if l != r {
                    errors.push(err("V093", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary operation `{op:?}` operands have mismatched integer types: left=`{l}`, right=`{r}`. Integer operands must match and belong to `u32` or `i32`"
                    ), format!(
                        "cast both operands to the same integer type."
                    )));
                }
            }
        }
        // Shifts and rotates: target-text masks the right operand with `& 31u`,
        // so both sides must be u32. Rotates share the same typing  -
        // left is the bit-pattern, right is the rotation count in bits.
        BinOp::Shl | BinOp::Shr | BinOp::RotateLeft | BinOp::RotateRight => {
            for (side, ty) in [("left", left_ty), ("right", right_ty)] {
                if let Some(ty) = ty {
                    if !matches!(ty, DataType::U32) {
                        errors.push(err("V094", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `{op:?}` {side} operand has type `{ty}`; shift/rotate operands must be `u32`"
                        ), format!(
                            "cast the operand to U32 before shifting/rotating."
                        )));
                    }
                }
            }
        }
        // Logical And/Or: target-text lowers via `!= 0u`, so only u32 and bool are valid.
        BinOp::And | BinOp::Or => {
            for (side, ty) in [("left", left_ty), ("right", right_ty)] {
                if let Some(ty) = ty {
                    if !matches!(ty, DataType::U32 | DataType::Bool) {
                        errors.push(err("V095", ValidationPhase::Type, ValidationLocation::Program, format!(
                            "binary operation `{op:?}` {side} operand has type `{ty}`; logical And/Or operands must be `u32` or `bool`"
                        ), format!(
                            "cast the operand to U32 or Bool."
                        )));
                    }
                }
            }
        }
        // Comparisons: target-text requires both operands to have the same type.
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
            if let (Some(l), Some(r)) = (&left_ty, &right_ty) {
                if l != r {
                    errors.push(err("V096", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "binary comparison `{op:?}` operands have mismatched types: left=`{l}`, right=`{r}`. Comparisons require matching types"
                    ), format!(
                        "cast both operands to the same type before comparing."
                    )));
                }
            }
        }
        BinOp::Shuffle | BinOp::Ballot | BinOp::WaveReduce | BinOp::WaveBroadcast => {
            errors.push(err("V097", ValidationPhase::Type, ValidationLocation::Program, format!(
                "binary operation `{op:?}` requires backend subgroup semantics (`supports_subgroup_ops() == true`) before foundation validation can guarantee safety"
            ), format!(
                "validate with ValidationOptions::with_backend(backend) where `backend.supports_subgroup_ops() == true`, or remove `{op:?}` before lowering."
            )));
        }
        _ => {}
    }
}

#[inline]
fn expr_is_static_zero(expr: &Expr) -> bool {
    match expr {
        Expr::LitU32(0) | Expr::LitI32(0) => true,
        Expr::LitF32(value) => *value == 0.0,
        Expr::Cast { value, .. } => expr_is_static_zero(value),
        _ => false,
    }
}

#[inline]
pub(crate) fn validate_unop_operand(
    op: &crate::ir_inner::model::spec_types::UnOp,
    expr: &Expr,
    buffers: &FxHashMap<&str, &BufferDecl>,
    scope: &FxHashMap<crate::ir::Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
    if let Some(ty) = expr_type(expr, &mut ScopeTypes::new(buffers, scope)) {
        match op {
            crate::ir_inner::model::spec_types::UnOp::Negate => {
                if matches!(ty, DataType::I32) {
                    errors.push(err("V098", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `Negate` operand has type `{ty}`, but legal total Negate types are `u32` and `f32`; raw i32 negation has the i32::MIN overflow case"
                    ), format!(
                        "use `0 - x` for wrapping i32 negation, cast to U32 before Negate, or guard with Select(i32::MIN, 0, -x)."
                    )));
                } else if !matches!(ty, DataType::U32 | DataType::F32) {
                    errors.push(err("V099", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`, but legal set is U32, I32, or F32"
                    ), format!(
                        "cast or rewrite the operand to U32/I32/F32."
                    )));
                }
            }
            crate::ir_inner::model::spec_types::UnOp::LogicalNot => {
                if !matches!(ty, DataType::U32 | DataType::Bool) {
                    errors.push(err("V100", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `LogicalNot` operand has type `{ty}`; legal set is `u32` or `bool`"
                    ), format!(
                        "cast or rewrite the operand to produce U32 or Bool."
                    )));
                }
            }
            crate::ir_inner::model::spec_types::UnOp::BitNot
            | crate::ir_inner::model::spec_types::UnOp::Popcount
            | crate::ir_inner::model::spec_types::UnOp::Clz
            | crate::ir_inner::model::spec_types::UnOp::Ctz
            | crate::ir_inner::model::spec_types::UnOp::ReverseBits => {
                // VAL-004: U64 operands are valid for every bitwise-unary
                // op. The reference interpreter handles Value::U64 for
                // BitNot/Popcount/Clz/Ctz/ReverseBits and target-text ≥ the 64-bit
                // extension emits the right intrinsics. Previously the
                // validator rejected U64 and forced an avoidable down-cast.
                if !matches!(ty, DataType::U32 | DataType::I32 | DataType::U64) {
                    errors.push(err("V101", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`; legal integer set is `u32`, `i32`, or `u64`"
                    ), format!(
                        "cast or rewrite the operand to produce U32, I32, or U64."
                    )));
                }
            }
            crate::ir_inner::model::spec_types::UnOp::Sin
            | crate::ir_inner::model::spec_types::UnOp::Cos
            | crate::ir_inner::model::spec_types::UnOp::Exp
            | crate::ir_inner::model::spec_types::UnOp::Log
            | crate::ir_inner::model::spec_types::UnOp::Log2
            | crate::ir_inner::model::spec_types::UnOp::Exp2
            | crate::ir_inner::model::spec_types::UnOp::Tan
            | crate::ir_inner::model::spec_types::UnOp::Acos
            | crate::ir_inner::model::spec_types::UnOp::Asin
            | crate::ir_inner::model::spec_types::UnOp::Atan
            | crate::ir_inner::model::spec_types::UnOp::Tanh
            | crate::ir_inner::model::spec_types::UnOp::Sinh
            | crate::ir_inner::model::spec_types::UnOp::Cosh
            | crate::ir_inner::model::spec_types::UnOp::Abs
            | crate::ir_inner::model::spec_types::UnOp::Sqrt
            | crate::ir_inner::model::spec_types::UnOp::InverseSqrt
            | crate::ir_inner::model::spec_types::UnOp::Reciprocal
            | crate::ir_inner::model::spec_types::UnOp::Floor
            | crate::ir_inner::model::spec_types::UnOp::Ceil
            | crate::ir_inner::model::spec_types::UnOp::Round
            | crate::ir_inner::model::spec_types::UnOp::Trunc
            | crate::ir_inner::model::spec_types::UnOp::Sign
            | crate::ir_inner::model::spec_types::UnOp::IsNan
            | crate::ir_inner::model::spec_types::UnOp::IsInf
            | crate::ir_inner::model::spec_types::UnOp::IsFinite => {
                if ty != DataType::F32 {
                    errors.push(err("V102", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`; legal set for math ops is `f32`"
                    ), format!(
                        "cast or rewrite the operand to produce F32."
                    )));
                }
            }
            crate::ir_inner::model::spec_types::UnOp::Unpack4Low
            | crate::ir_inner::model::spec_types::UnOp::Unpack4High
            | crate::ir_inner::model::spec_types::UnOp::Unpack8Low
            | crate::ir_inner::model::spec_types::UnOp::Unpack8High => {
                // VAL-004: nibble/byte unpack ops extract a masked, shifted lane
                // from a 32-bit integer word, emit lowers them to
                // `(v >> shift) & mask` and the reference interpreter mirrors it,
                // so operand and result are 32-bit integers. These previously
                // fell through to the `_` catch-all and were rejected as "not
                // recognized" even though that message LISTS them as valid and
                // every backend lowers them: a validator rejecting ops it emits.
                if !matches!(ty, DataType::U32 | DataType::I32) {
                    errors.push(err("V103", ValidationPhase::Type, ValidationLocation::Program, format!(
                        "unary operation `{op:?}` operand has type `{ty}`; unpack ops require a 32-bit integer (`u32` or `i32`) word"
                    ), format!(
                        "cast or rewrite the operand to produce U32 or I32."
                    )));
                }
            }
            _ => {
                errors.push(err("V104", ValidationPhase::Type, ValidationLocation::Program, format!(
                    "unary operation `{op:?}` is not recognized"
                ), format!(
                    "use a known UnOp variant from this enum (`Negate`, `LogicalNot`, `BitNot`, `Popcount`, `Clz`, `Ctz`, `ReverseBits`, `Sin`, `Cos`, `Exp`, `Log`, `Log2`, `Exp2`, `Tan`, `Acos`, `Asin`, `Atan`, `Tanh`, `Sinh`, `Cosh`, `Abs`, `Sqrt`, `InverseSqrt`, `Reciprocal`, `Floor`, `Ceil`, `Round`, `Trunc`, `Sign`, `IsNan`, `IsInf`, `IsFinite`, `Unpack4Low`, `Unpack4High`, `Unpack8Low`, `Unpack8High`)."
                )));
            }
        }
    }
}

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

    use crate::ir_inner::model::spec_types::UnOp;

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
                    match op {
                        BinOp::Add
                        | BinOp::Sub
                        | BinOp::Mul
                        | BinOp::Div
                        | BinOp::SaturatingAdd
                        | BinOp::SaturatingSub
                        | BinOp::SaturatingMul
                        | BinOp::Min
                        | BinOp::Max => plan(expr, Combine::Arith, operands, &mut frames),
                        // Logical And/Or and all comparisons evaluate to Bool.
                        // The reference interpreter produces Value::Bool here, so
                        // the static type must match or programs like `(a && b) + 1`
                        // pass validation and then fail at interpreter time.
                        BinOp::And
                        | BinOp::Or
                        | BinOp::Eq
                        | BinOp::Ne
                        | BinOp::Lt
                        | BinOp::Gt
                        | BinOp::Le
                        | BinOp::Ge => {
                            values.push(Some(DataType::Bool));
                            plan(expr, Combine::Drop(2), operands, &mut frames);
                        }
                        // Bitwise, modulo, shift, rotate, unsigned absolute
                        // difference, multiply-high, and the wave operators are
                        // integer-typed, and an extension operator has no
                        // declared result type. U32 is the safe default for all
                        // of them: the operand-checker already rejects
                        // non-integer operands, and an answer here keeps the
                        // enclosing operator's mixed-type check armed.
                        _ => {
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

#[cfg(test)]
#[path = "typecheck_critical_test.rs"]
mod typecheck_critical_test;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;

    fn empty_buffers() -> FxHashMap<&'static str, &'static BufferDecl> {
        FxHashMap::default()
    }
    fn empty_scope() -> FxHashMap<crate::ir::Ident, Binding> {
        FxHashMap::default()
    }

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::BinOp {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    fn ty(expr: &Expr) -> Option<DataType> {
        expr_type(expr, &mut ScopeTypes::new(&empty_buffers(), &empty_scope()))
    }

    #[test]
    fn and_or_type_is_bool() {
        for op in [BinOp::And, BinOp::Or] {
            let e = bin(op, Expr::LitBool(true), Expr::LitBool(false));
            assert_eq!(
                ty(&e),
                Some(DataType::Bool),
                "And/Or must type as Bool (reference interpreter produces Value::Bool)"
            );
        }
    }

    #[test]
    fn comparisons_type_is_bool() {
        for op in [
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Lt,
            BinOp::Gt,
            BinOp::Le,
            BinOp::Ge,
        ] {
            let e = bin(op, Expr::LitU32(1), Expr::LitU32(2));
            assert_eq!(ty(&e), Some(DataType::Bool), "comparison must type as Bool");
        }
    }

    #[test]
    fn bitwise_type_is_integer() {
        let e = bin(BinOp::BitAnd, Expr::LitU32(1), Expr::LitU32(2));
        assert_eq!(ty(&e), Some(DataType::U32));
    }

    #[test]
    fn bool_plus_int_is_rejected() -> Result<(), String> {
        // Regression for REF-002: `(a && b) + 1`  -  previously accepted because
        // And was typed U32. Now And types as Bool, so arithmetic must reject.
        let and_expr = bin(BinOp::And, Expr::LitBool(true), Expr::LitBool(false));
        let add_expr = bin(BinOp::Add, and_expr, Expr::LitU32(1));
        let mut errors = Vec::new();
        if let Expr::BinOp { op, left, right } = &add_expr {
            validate_binop_operands(
                *op,
                left,
                right,
                &empty_buffers(),
                &empty_scope(),
                &mut errors,
            );
        } else {
            return Err("expected BinOp".to_string());
        }
        assert_eq!(
            errors.len(),
            1,
            "bool + int must produce exactly one type error"
        );
        assert!(
            errors[0].message().contains("Bool") || errors[0].message().contains("type"),
            "type error must mention Bool mismatch: {}",
            errors[0].message()
        );
        Ok(())
    }

    #[test]
    fn div_by_static_zero_is_rejected() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Div,
            &Expr::LitU32(9),
            &Expr::LitU32(0),
            &empty_buffers(),
            &empty_scope(),
            &mut errors,
        );
        assert!(errors.iter().any(|error| error.code().as_str() == "V044"));
    }

    #[test]
    fn div_by_casted_static_zero_is_rejected() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Div,
            &Expr::LitU32(9),
            &Expr::Cast {
                target: DataType::U32,
                value: Box::new(Expr::LitI32(0)),
            },
            &empty_buffers(),
            &empty_scope(),
            &mut errors,
        );
        assert!(errors.iter().any(|error| error.code().as_str() == "V044"));
    }

    #[test]
    fn mod_by_static_zero_is_rejected() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Mod,
            &Expr::LitU32(9),
            &Expr::LitU32(0),
            &empty_buffers(),
            &empty_scope(),
            &mut errors,
        );
        assert!(errors.iter().any(|error| error.code().as_str() == "V044"));
    }
}
