//! Binary operation emission, including the synthesized forms naga has no
//! single operator for.

use naga::{BinaryOperator, Expression, Literal};
use vyre_foundation::ir::BinOp;
use vyre_lower::KernelOp;

use super::super::op_lookup::{binary_math_function, binary_operator};
use super::super::BodyBuilder;
use crate::EmitError;

impl BodyBuilder<'_> {
    /// `BinOpKind` emit  -  bool-vs-numeric widening, literal-pool fold,
    /// and Math-builtin routing live here to keep `emit_op` flat.
    pub(in crate::emitter) fn emit_binop(
        &mut self,
        op: &KernelOp,
        binop: BinOp,
    ) -> Result<(), EmitError> {
        let left = self.value_operand(op, 0)?;
        let right = self.value_operand(op, 1)?;
        // 64-bit gate: U64/I64 are backed by vec2<u32> (the vec2_u32_ty handle).
        // Componentwise bitwise AND/OR/XOR on the pair are mathematically
        // correct, but add/sub/mul/compare/shift need carry/borrow propagation
        // between the low and high words, a componentwise vec2 op would be
        // SILENTLY WRONG arithmetic. Fail closed (Law 10) rather than emit it.
        let lhs_is_u64 = self
            .value_type_operand(op, 0)
            .map(|h| h == self.types.vec2_u32_ty)
            .unwrap_or(false);
        let rhs_is_u64 = self
            .value_type_operand(op, 1)
            .map(|h| h == self.types.vec2_u32_ty)
            .unwrap_or(false);
        if (lhs_is_u64 || rhs_is_u64)
            && !matches!(binop, BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor)
        {
            return Err(EmitError::NagaConstructionFailed(format!(
                "64-bit (U64/I64) `{binop:?}` is not lowered: the vec2<u32> backing \
                 carries no carry/borrow between the low and high words, so a \
                 componentwise op would be silently wrong. Only bitwise AND/OR/XOR \
                 are supported on 64-bit values. Fix: add a carry-propagating U64 \
                 emulation pass before this op reaches Naga emission."
            )));
        }
        if let Some(folded) = self.fold_literal_binop(left, right, binop) {
            let ty = self.binary_result_type(op, binop)?;
            return self.bind_result_typed(op, folded, ty);
        }
        let mut effective_binop = binop;
        let mut left_eff = left;
        let mut right_eff = right;
        if matches!(
            binop,
            BinOp::And | BinOp::Or | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
        ) {
            let left_ty = self.value_type_operand(op, 0).ok();
            let right_ty = self.value_type_operand(op, 1).ok();
            let left_naga_kind = self.scalar_kind_of_expression(left, 0);
            let right_naga_kind = self.scalar_kind_of_expression(right, 0);
            let left_is_bool = match left_naga_kind {
                Some(naga::ScalarKind::Bool) => true,
                Some(_) => false,
                None => match left_ty {
                    Some(ty) => ty == self.types.bool_ty,
                    None => self.is_bool_expression(left),
                },
            };
            let right_is_bool = match right_naga_kind {
                Some(naga::ScalarKind::Bool) => true,
                Some(_) => false,
                None => match right_ty {
                    Some(ty) => ty == self.types.bool_ty,
                    None => self.is_bool_expression(right),
                },
            };
            if left_is_bool && right_is_bool {
                // both bool → keep bool; binary_operator emits bitwise And/Or
            } else if !left_is_bool && !right_is_bool {
                // both numeric → bitwise as-is
            } else {
                let left_widen_ty = if left_is_bool {
                    Some(self.types.bool_ty)
                } else {
                    left_ty.or(Some(self.types.u32_ty))
                };
                let right_widen_ty = if right_is_bool {
                    Some(self.types.bool_ty)
                } else {
                    right_ty.or(Some(self.types.u32_ty))
                };
                left_eff = self.coerce_to_u32(left, left_widen_ty);
                right_eff = self.coerce_to_u32(right, right_widen_ty);
                effective_binop = match binop {
                    BinOp::And => BinOp::BitAnd,
                    BinOp::Or => BinOp::BitOr,
                    other => other,
                };
            }
        }
        let left_kind = self.scalar_kind_of_expression(left_eff, 0);
        let right_kind = self.scalar_kind_of_expression(right_eff, 0);
        // Comparison and arithmetic BinOps require numeric (non-Bool)
        // operands in WGSL. When the carrier-publish round-trip exposes
        // Bool-typed Loads on either arm, naga rejects with
        // `InvalidBinaryOperandTypes`. Coerce both arms to u32 for the
        // affected ops; Eq/Ne/And/Or are bool-friendly and are routed
        // through the bool-widening branch above.
        let comparison_or_arith = matches!(
            binop,
            BinOp::Lt
                | BinOp::Gt
                | BinOp::Le
                | BinOp::Ge
                | BinOp::Add
                | BinOp::Sub
                | BinOp::Mul
                | BinOp::Div
                | BinOp::Mod
                | BinOp::Shl
                | BinOp::Shr
                | BinOp::Min
                | BinOp::Max
                | BinOp::WrappingAdd
                | BinOp::WrappingSub
                | BinOp::AbsDiff
                | BinOp::RotateLeft
                | BinOp::RotateRight
                | BinOp::MulHigh
                | BinOp::SaturatingAdd
                | BinOp::SaturatingSub
                | BinOp::SaturatingMul
        );
        if comparison_or_arith {
            if matches!(left_kind, Some(naga::ScalarKind::Bool)) {
                left_eff = self.coerce_to_u32(left_eff, Some(self.types.bool_ty));
            }
            if matches!(right_kind, Some(naga::ScalarKind::Bool)) {
                right_eff = self.coerce_to_u32(right_eff, Some(self.types.bool_ty));
            }
        }
        let left_kind = self.scalar_kind_of_expression(left_eff, 0);
        let right_kind = self.scalar_kind_of_expression(right_eff, 0);
        // Shifts are the exception to operand-kind unification: the amount must
        // stay u32 regardless of the value's signedness (the value's signedness
        // selects arithmetic vs logical shift). Coercing the amount to the
        // value's kind here would force `i32 >> 1`'s amount to Sint; the shift
        // block below owns the amount's type (coerce-to-u32 + bit-width mask).
        let is_shift = matches!(effective_binop, BinOp::Shl | BinOp::Shr);
        if !is_shift {
            if let (Some(lk), Some(rk)) = (left_kind, right_kind) {
                if lk != rk {
                    let target = match lk {
                        naga::ScalarKind::Bool => self.types.bool_ty,
                        naga::ScalarKind::Sint => self.types.i32_ty,
                        naga::ScalarKind::Float => self.types.f32_ty,
                        _ => self.types.u32_ty,
                    };
                    right_eff = self.coerce_value_to_type(right_eff, target);
                }
            }
        }
        // Backend contract: the shift amount is taken modulo the bit width (32).
        // The reference oracle masks (`right & 31`) and PTX masks
        // (`and.b32 …,31`), but a bare naga ShiftLeft/ShiftRight leaves an
        // amount >= 32 undefined per the SPIR-V/WGSL shift rules, a silent
        // CPU/GPU divergence (Law 10). Mask here so the wgpu/spirv/metal path
        // matches PTX and the oracle. A known in-range constant amount (the
        // `x >> 16` hot path) is left untouched, it would fold to itself, so
        // the mask only costs an `& 31` on genuinely variable shift counts.
        // (u64 shifts never reach here: the 64-bit gate fails them closed.)
        if is_shift {
            let amount_in_range = matches!(
                self.function.expressions.try_get(right_eff),
                Ok(Expression::Literal(Literal::U32(v))) if *v < 32
            );
            if !amount_in_range {
                right_eff = self.coerce_value_to_type(right_eff, self.types.u32_ty);
                let mask31 = self.append_expr(Expression::Literal(Literal::U32(31)));
                right_eff = self.append_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: right_eff,
                    right: mask31,
                });
            }
        }
        // naga's `BinaryOperator::Modulo` lowers to an UNSIGNED remainder on the
        // SPIR-V backend even for SIGNED operands, a vendored-naga bug verified
        // on the 5090: `rem(i32, i32)` of (-7, 3) returned 0 (== unsigned
        // 0xFFFF_FFF9 % 3), not the signed -1, while `div(i32, i32)` of (-7, 3)
        // correctly returned -2 (naga's `Divide` DOES pick SDiv). naga's signed
        // Divide is trustworthy, so synthesize the signed remainder from the
        // truncating-division identity `a - (a / b) * b` (-7 - (-2)*3 = -1),
        // bypassing the buggy Modulo. Unsigned Mod keeps naga's `Modulo` (correct
        // for Uint, plus the divisor-zero guard below). The result type of a Mod
        // is its operand type (`binary_result_type` -> operand 0), so an i32_ty
        // result means signed operands. wgpu/spirv/metal all route through this
        // emitter, so the one fix covers every naga-derived backend.
        let signed_mod = matches!(effective_binop, BinOp::Mod)
            && self.binary_result_type(op, effective_binop)? == self.types.i32_ty;
        let value = if signed_mod {
            let quotient = self.append_expr(Expression::Binary {
                op: BinaryOperator::Divide,
                left: left_eff,
                right: right_eff,
            });
            let product = self.append_expr(Expression::Binary {
                op: BinaryOperator::Multiply,
                left: quotient,
                right: right_eff,
            });
            self.append_expr(Expression::Binary {
                op: BinaryOperator::Subtract,
                left: left_eff,
                right: product,
            })
        } else if let Some(value) = self.emit_synthetic_binop(effective_binop, left_eff, right_eff)
        {
            value
        } else if let Some(fun) = binary_math_function(effective_binop) {
            self.append_expr(Expression::Math {
                fun,
                arg: left_eff,
                arg1: Some(right_eff),
                arg2: None,
                arg3: None,
            })
        } else {
            let naga_op = binary_operator(effective_binop)?;
            self.append_expr(Expression::Binary {
                op: naga_op,
                left: left_eff,
                right: right_eff,
            })
        };
        // Div/Mod by zero is backend-divergent: naga 25 overrides a zero
        // divisor to 1 (so `x / 0 == x`, `x % 0 == 0`), while PTX leaves it to
        // unspecified hardware. The vyre-reference oracle documents a single
        // total contract (`u32 x / 0 == u32::MAX`, `x % 0 == 0`) with explicit
        // tests, so a bare Naga `Divide` makes the wgpu backend silently
        // disagree with its own oracle. Force the oracle contract here so every
        // backend is uniform and the CPU oracle stays sound (Law 10). Only the
        // unsigned divisor is guarded, signed div-by-zero / INT_MIN÷-1 are
        // rejected upstream as undefined backend semantics.
        let value = if matches!(binop, BinOp::Div | BinOp::Mod)
            && matches!(right_kind, Some(naga::ScalarKind::Uint))
        {
            let zero = self.append_expr(Expression::Literal(Literal::U32(0)));
            let divisor_is_zero = self.append_expr(Expression::Binary {
                op: BinaryOperator::Equal,
                left: right_eff,
                right: zero,
            });
            let sentinel = if matches!(binop, BinOp::Div) {
                self.append_expr(Expression::Literal(Literal::U32(u32::MAX)))
            } else {
                zero
            };
            self.append_expr(Expression::Select {
                condition: divisor_is_zero,
                accept: sentinel,
                reject: value,
            })
        } else {
            value
        };
        let ty = self.binary_result_type(op, binop)?;
        self.bind_result_typed(op, value, ty)
    }

    fn emit_synthetic_binop(
        &mut self,
        binop: BinOp,
        left: naga::Handle<Expression>,
        right: naga::Handle<Expression>,
    ) -> Option<naga::Handle<Expression>> {
        match binop {
            BinOp::AbsDiff => {
                let left_lt_right = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Less,
                    left,
                    right,
                });
                let hi = self.append_expr(Expression::Select {
                    condition: left_lt_right,
                    accept: right,
                    reject: left,
                });
                let lo = self.append_expr(Expression::Select {
                    condition: left_lt_right,
                    accept: left,
                    reject: right,
                });
                Some(self.append_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: hi,
                    right: lo,
                }))
            }
            BinOp::RotateLeft | BinOp::RotateRight => {
                let mask = self.append_expr(Expression::Literal(Literal::U32(31)));
                let shift = self.append_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: right,
                    right: mask,
                });
                let thirty_two = self.append_expr(Expression::Literal(Literal::U32(32)));
                let inv_raw = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: thirty_two,
                    right: shift,
                });
                let inv = self.append_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: inv_raw,
                    right: mask,
                });
                let (left_shift, right_shift) = if matches!(binop, BinOp::RotateLeft) {
                    (shift, inv)
                } else {
                    (inv, shift)
                };
                let lhs = self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left,
                    right: left_shift,
                });
                let rhs = self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left,
                    right: right_shift,
                });
                Some(self.append_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: lhs,
                    right: rhs,
                }))
            }
            BinOp::MulHigh => Some(self.emit_u32_mul_high(left, right)),
            BinOp::SaturatingAdd => {
                let sum = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left,
                    right,
                });
                let overflow = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Less,
                    left: sum,
                    right: left,
                });
                let max = self.append_expr(Expression::Literal(Literal::U32(u32::MAX)));
                Some(self.append_expr(Expression::Select {
                    condition: overflow,
                    accept: max,
                    reject: sum,
                }))
            }
            BinOp::SaturatingSub => {
                let underflow = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Less,
                    left,
                    right,
                });
                let diff = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left,
                    right,
                });
                let zero = self.append_expr(Expression::Literal(Literal::U32(0)));
                Some(self.append_expr(Expression::Select {
                    condition: underflow,
                    accept: zero,
                    reject: diff,
                }))
            }
            BinOp::SaturatingMul => {
                let zero = self.append_expr(Expression::Literal(Literal::U32(0)));
                let max = self.append_expr(Expression::Literal(Literal::U32(u32::MAX)));
                let right_ne_zero = self.append_expr(Expression::Binary {
                    op: BinaryOperator::NotEqual,
                    left: right,
                    right: zero,
                });
                let one = self.append_expr(Expression::Literal(Literal::U32(1)));
                let divisor = self.append_expr(Expression::Select {
                    condition: right_ne_zero,
                    accept: right,
                    reject: one,
                });
                let limit = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: max,
                    right: divisor,
                });
                let left_gt_limit = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Greater,
                    left,
                    right: limit,
                });
                let overflow = self.append_expr(Expression::Binary {
                    op: BinaryOperator::LogicalAnd,
                    left: right_ne_zero,
                    right: left_gt_limit,
                });
                let product = self.append_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left,
                    right,
                });
                Some(self.append_expr(Expression::Select {
                    condition: overflow,
                    accept: max,
                    reject: product,
                }))
            }
            _ => None,
        }
    }

    fn emit_u32_mul_high(
        &mut self,
        left: naga::Handle<Expression>,
        right: naga::Handle<Expression>,
    ) -> naga::Handle<Expression> {
        let mask16 = self.append_expr(Expression::Literal(Literal::U32(0xffff)));
        let shift16 = self.append_expr(Expression::Literal(Literal::U32(16)));
        let al = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left,
            right: mask16,
        });
        let ah = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left,
            right: shift16,
        });
        let bl = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: right,
            right: mask16,
        });
        let bh = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: right,
            right: shift16,
        });
        let p0 = self.append_expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: al,
            right: bl,
        });
        let p1 = self.append_expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: ah,
            right: bl,
        });
        let p2 = self.append_expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: al,
            right: bh,
        });
        let p3 = self.append_expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: ah,
            right: bh,
        });
        let p0_hi = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: p0,
            right: shift16,
        });
        let p1_lo = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: p1,
            right: mask16,
        });
        let p2_lo = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: p2,
            right: mask16,
        });
        let mid_a = self.append_expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: p0_hi,
            right: p1_lo,
        });
        let mid_b = self.append_expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: mid_a,
            right: p2_lo,
        });
        let carry = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: mid_b,
            right: shift16,
        });
        let p1_hi = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: p1,
            right: shift16,
        });
        let p2_hi = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: p2,
            right: shift16,
        });
        let high_a = self.append_expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: p3,
            right: p1_hi,
        });
        let high_b = self.append_expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: high_a,
            right: p2_hi,
        });
        self.append_expr(Expression::Binary {
            op: BinaryOperator::Add,
            left: high_b,
            right: carry,
        })
    }
}
