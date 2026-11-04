//! Cast operation emission and narrow-scalar truncation.

use naga::{BinaryOperator, Expression, Literal, ScalarKind};
use vyre_foundation::ir::DataType;
use vyre_lower::KernelOp;

use super::super::op_lookup::scalar_cast_target;
use super::super::BodyBuilder;
use crate::EmitError;

impl BodyBuilder<'_> {
    pub(super) fn emit_cast(&mut self, op: &KernelOp, target: &DataType) -> Result<(), EmitError> {
        let expr = self.value_operand(op, 0)?;
        // A multi-word-backed SOURCE (U64/I64/Vec2U32 -> vec2<u32>,
        // Vec4U32 -> vec4<u32>) cannot go through the scalar `As` path or
        // the scalar-source widening path below: both assume a scalar
        // source, so a plain `As` over the whole vector (or coercing the
        // vector to u32) yields invalid WGSL (InvalidStoreTypes / Compose
        // arity). Lower EVERY cast from a wide source by extracting its
        // lanes explicitly, matching the reference + PTX:
        //   * 2-word target (U64/I64/Vec2U32) -> low two words;
        //   * Vec4U32 -> the four words (identity, vec4 source only);
        //   * scalar integer -> low word (lane 0), truncated, then the
        //     scalar cast reinterprets it (u64->u32 keeps the low 32 bits);
        //   * F32 -> reconstruct (low | high<<32) then ConvertUToF, so the
        //     high word is NOT dropped (matches the reference u64 as f32);
        //   * Bool -> truthy over ALL source words: OR the lanes, != 0.
        let source_lanes = self.value_type_operand(op, 0).ok().and_then(|h| {
            if h == self.types.vec4_u32_ty {
                Some(4u32)
            } else if h == self.types.vec2_u32_ty {
                Some(2u32)
            } else {
                None
            }
        });
        if let Some(lanes) = source_lanes {
            let lane_handles: Vec<_> = (0..lanes)
                .map(|i| {
                    self.append_expr(Expression::AccessIndex {
                        base: expr,
                        index: i,
                    })
                })
                .collect();
            let low = lane_handles[0];
            match target {
                DataType::U64 | DataType::I64 | DataType::Vec2U32 => {
                    let composed = self.append_expr(Expression::Compose {
                        ty: self.types.vec2_u32_ty,
                        components: vec![low, lane_handles[1]],
                    });
                    return self.bind_result_typed(op, composed, self.types.vec2_u32_ty);
                }
                DataType::Vec4U32 => {
                    if lanes < 4 {
                        return Err(EmitError::NagaConstructionFailed(format!(
                            "cast to Vec4U32 from a {lanes}-word source is not \
                             representable: only a 4-word (Vec4U32) source can \
                             widen to Vec4U32. Fix: route through a Vec2U32 \
                             intermediate or zero-fill the upper lanes explicitly."
                        )));
                    }
                    let composed = self.append_expr(Expression::Compose {
                        ty: self.types.vec4_u32_ty,
                        components: lane_handles,
                    });
                    return self.bind_result_typed(op, composed, self.types.vec4_u32_ty);
                }
                DataType::F32 => {
                    let low_u64 = self.append_expr(Expression::As {
                        expr: low,
                        kind: ScalarKind::Uint,
                        convert: Some(8),
                    });
                    let high_u64 = self.append_expr(Expression::As {
                        expr: lane_handles[1],
                        kind: ScalarKind::Uint,
                        convert: Some(8),
                    });
                    let shift = self.append_expr(Expression::Literal(naga::Literal::U32(32)));
                    let shift_u64 = self.append_expr(Expression::As {
                        expr: shift,
                        kind: ScalarKind::Uint,
                        convert: Some(8),
                    });
                    let high_shifted = self.append_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: high_u64,
                        right: shift_u64,
                    });
                    let full = self.append_expr(Expression::Binary {
                        op: BinaryOperator::InclusiveOr,
                        left: low_u64,
                        right: high_shifted,
                    });
                    let value = self.append_expr(Expression::As {
                        expr: full,
                        kind: ScalarKind::Float,
                        convert: Some(4),
                    });
                    let ty = self.type_for_data_type(target)?;
                    return self.bind_result_typed(op, value, ty);
                }
                DataType::Bool => {
                    let mut merged = low;
                    for &lane in &lane_handles[1..] {
                        merged = self.append_expr(Expression::Binary {
                            op: BinaryOperator::InclusiveOr,
                            left: merged,
                            right: lane,
                        });
                    }
                    let zero = self.append_expr(Expression::Literal(naga::Literal::U32(0)));
                    let value = self.append_expr(Expression::Binary {
                        op: BinaryOperator::NotEqual,
                        left: merged,
                        right: zero,
                    });
                    let ty = self.type_for_data_type(target)?;
                    return self.bind_result_typed(op, value, ty);
                }
                _ => {
                    let (kind, width) = scalar_cast_target(target)?;
                    let value = self.append_expr(Expression::As {
                        expr: low,
                        kind,
                        convert: Some(width),
                    });
                    // A 64-bit source narrowing to u8/u16/i8/i16 keeps
                    // only the low word's low bits, truncate the `As`
                    // result to the narrow target width (matches the
                    // oracle's `u64 as u8` low-byte semantics).
                    let value = self.apply_narrow_mask(value, target);
                    let ty = self.type_for_data_type(target)?;
                    return self.bind_result_typed(op, value, ty);
                }
            }
        }
        // Detect a float source via BOTH the bound type handle AND the
        // scalar-kind resolver: `value_type_operand == f32_ty` catches
        // buffer loads (scalar_kind_of_expression returns None through an
        // Access->Load chain), while `scalar_kind_of_expression == Float`
        // catches computed floats (arithmetic results) whose type handle
        // may not be a literal `f32_ty`. Either alone leaves a silent-skip
        // hole. Neither over-matches a non-float.
        let source_is_f32 = self
            .value_type_operand(op, 0)
            .map(|h| h == self.types.f32_ty)
            .unwrap_or(false)
            || self.scalar_kind_of_expression(expr, 0) == Some(ScalarKind::Float);
        // A float source converts numerically ONLY to u32/i32 (saturating,
        // below), bool (truthy), or f32 (identity). The foundation cast
        // table (`validate::cast::cast_is_valid`) rejects f32 -> {U8, U16,
        // I8, I16, U64, I64, Vec2U32, Vec4U32} for exactly this reason. But
        // the Program-compat `emit_module` path does NOT run full
        // validation, so such a cast can reach here, and the paths below
        // would SILENTLY miscompile it: the U64/I64/Vec2U32 wide path
        // reinterprets the float through a u32 coerce (dropping the high
        // word), and a narrow int target (U8/U16/I8/I16, all backed by a
        // 32-bit scalar) takes a bare `As` that skips the saturating guard
        // (NaN -> undefined, overflow -> FClamp divergence) AND does not
        // narrow. Fail closed (Law 10), mirroring the `Bytes` arm in
        // `scalar_cast_target`, and name the fix.
        if source_is_f32
            && matches!(
                target,
                DataType::U8
                    | DataType::U16
                    | DataType::I8
                    | DataType::I16
                    | DataType::U64
                    | DataType::I64
                    | DataType::Vec2U32
                    | DataType::Vec4U32
            )
        {
            return Err(EmitError::NagaConstructionFailed(format!(
                "cast from f32 to `{target:?}` has no defined conversion: a \
                 float source converts only to u32/i32 (saturating), bool \
                 (truthy), or f32. Fix: cast the f32 to u32 or i32 first, then \
                 narrow or widen the integer."
            )));
        }
        if matches!(target, DataType::U64 | DataType::I64 | DataType::Vec2U32) {
            // WGSL has no native 64-bit integer; U64/I64 are backed by
            // vec2<u32> (low word `.x`, high word `.y`). The low word is
            // always the source's 32-bit pattern. The HIGH word depends
            // on what the cast means:
            //   * `Vec2U32` is a STRUCTURAL 2-word vector, lane 1 is
            //     zero-filled (matches the reference `widen_to_words` /
            //     `cast_to_vec2` zero-pad), never sign-extended.
            //   * `U64`/`I64` are 64-bit INTEGERS, the high word must
            //     extend per the SOURCE's signedness. A signed (i32)
            //     source SIGN-extends so a negative value carries its
            //     full two's-complement high word (matching the PTX
            //     `cvt.s64.s32` path and Rust `i32 as i64`); an unsigned
            //     source zero-extends. Zeroing the high word
            //     unconditionally, as this did before, silently turned
            //     every negative `i32 -> i64/u64` into a large positive
            //     value (Law 10 miscompile). This stays componentwise
            //     (the high word is derived from the low word's sign bit,
            //     no cross-lane carry), unlike 64-bit arithmetic which
            //     emit_binop still rejects until a carry pass lands.
            let src_is_signed = matches!(
                self.scalar_kind_of_expression(expr, 0),
                Some(ScalarKind::Sint)
            );
            let low = self.coerce_value_to_type(expr, self.types.u32_ty);
            let high = if src_is_signed && matches!(target, DataType::U64 | DataType::I64) {
                // sign_bit = low >> 31 (logical shift on a u32 → 0 or 1);
                // high = sign_bit * 0xFFFF_FFFF → 0x0000_0000 when the
                // sign bit is clear, 0xFFFF_FFFF when set. No branch, no
                // carry (a pure componentwise sign replicate).
                let thirty_one = self.append_expr(Expression::Literal(naga::Literal::U32(31)));
                let sign_bit = self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: low,
                    right: thirty_one,
                });
                let all_ones =
                    self.append_expr(Expression::Literal(naga::Literal::U32(0xFFFF_FFFF)));
                self.append_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: sign_bit,
                    right: all_ones,
                })
            } else {
                self.append_expr(Expression::Literal(naga::Literal::U32(0)))
            };
            let value = self.append_expr(Expression::Compose {
                ty: self.types.vec2_u32_ty,
                components: vec![low, high],
            });
            return self.bind_result_typed(op, value, self.types.vec2_u32_ty);
        }
        let (kind, width) = scalar_cast_target(target)?;
        // Float->{U32,I32}: naga's SPIR-V backend lowers `As` as
        // `FClamp(x, min_repr, max_repr)` then ConvertFTo{U,S}. That
        // DIVERGES from the reference oracle (Rust saturating `as`) two
        // ways, both confirmed on a live 5090:
        //   1. OVERFLOW: FClamp pins to the largest *f32-representable*
        //      value <= target max (i32 -> 2147483520, u32 -> 4294967040),
        //      but the oracle + PTX `cvt.rzi.sat` saturate to the exact
        //      INTEGER max (i32::MAX, u32::MAX). Diverges for EVERY
        //      positive overflow including +inf.
        //   2. NaN: `FClamp(NaN, ..)` is SPIR-V-undefined (observed
        //      i32::MIN on the 5090); the oracle defines NaN -> 0.
        // Restore the reference's saturating semantics explicitly:
        // NaN -> 0, x >= 2^bits -> INT_MAX. The in-range and
        // negative-overflow paths already match (min_repr is exactly
        // f32-representable, so naga's FClamp low bound == the oracle),
        // so they keep naga's `As`. Only a Float source to a 32-bit int
        // target is rewritten; integer->int and narrow targets are
        // unchanged.
        //
        // `source_is_f32` was computed once above (it also gates the
        // fail-closed guard that rejects float -> non-{u32,i32,bool,f32}
        // targets), so by here the only float targets left are U32/I32.
        if source_is_f32 && matches!(target, DataType::U32 | DataType::I32) {
            let converted = self.append_expr(Expression::As {
                expr,
                kind,
                convert: Some(width),
            });
            // is_finite_ordered = (x == x): TRUE for every non-NaN value,
            // FALSE for NaN. We must NOT use `x != x` here: naga lowers a
            // float `!=` to the ORDERED `FOrdNotEqual`, and
            // `FOrdNotEqual(NaN, NaN)` is FALSE (ordered comparisons are
            // false whenever an operand is NaN), the canonical `x != x`
            // NaN idiom is silently dead in naga. `Equal` lowers to
            // `FOrdEqual`, and `FOrdEqual(NaN, NaN)` is likewise FALSE, so
            // `x == x` is the portable not-NaN predicate.
            let is_not_nan = self.append_expr(Expression::Binary {
                op: BinaryOperator::Equal,
                left: expr,
                right: expr,
            });
            let (overflow_threshold, int_max, int_zero) = match target {
                DataType::I32 => (
                    naga::Literal::F32(2_147_483_648.0), // 2^31
                    naga::Literal::I32(i32::MAX),
                    naga::Literal::I32(0),
                ),
                // U32 (the only other arm the outer `matches!` allows).
                _ => (
                    naga::Literal::F32(4_294_967_296.0), // 2^32
                    naga::Literal::U32(u32::MAX),
                    naga::Literal::U32(0),
                ),
            };
            let threshold = self.append_expr(Expression::Literal(overflow_threshold));
            // too_high = (x >= 2^bits): true for positive overflow and
            // +inf; false for NaN (ordered compare is false for NaN), so
            // the outer not-NaN select still wins for NaN regardless of
            // `converted`/`saturated_high`.
            let too_high = self.append_expr(Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: expr,
                right: threshold,
            });
            let max_lit = self.append_expr(Expression::Literal(int_max));
            let saturated_high = self.append_expr(Expression::Select {
                condition: too_high,
                accept: max_lit,
                reject: converted,
            });
            let zero_lit = self.append_expr(Expression::Literal(int_zero));
            // non-NaN -> the saturated conversion; NaN -> 0 (the oracle's
            // `NaN as {i32,u32}`).
            let value = self.append_expr(Expression::Select {
                condition: is_not_nan,
                accept: saturated_high,
                reject: zero_lit,
            });
            let ty = self.type_for_data_type(target)?;
            return self.bind_result_typed(op, value, ty);
        }
        let value = self.append_expr(Expression::As {
            expr,
            kind,
            convert: Some(width),
        });
        // Integer narrowing (u32 -> u8/u16/i8/i16): the `As` above keeps
        // the full 32-bit word; mask/sign-extend to the target width so
        // the emitter matches Rust `as` and the reference oracle.
        let value = self.apply_narrow_mask(value, target);
        let ty = self.type_for_data_type(target)?;
        self.bind_result_typed(op, value, ty)
    }

    /// Truncate a 32-bit scalar cast result to a narrow integer target's width.
    ///
    /// WGSL has no native 8/16-bit integer scalar, so `scalar_cast_target` backs
    /// U8/U16 with a `Uint` (u32) and I8/I16 with a `Sint` (i32) register. The
    /// bare `As` that produces that register does NOT discard the high bits, a
    /// u32 source to a U8 target stays the full word, so a narrowing cast would
    /// silently keep the high bits, diverging from the V035 contract ("narrowing
    /// cast may truncate high bits"), Rust `as u8/u16/i8/i16`, and the reference
    /// oracle (`cast_value`). This applies the missing truncation:
    ///   * U8/U16: bitwise-AND with the low-`width` mask (`& 0xFF` / `& 0xFFFF`).
    ///   * I8/I16: truncate then SIGN-extend from the new top bit via
    ///     `(x << shift) >> shift` with an arithmetic (Sint) right shift, the
    ///     same idiom `emit_byte_element_load` uses for an I8 buffer byte, so e.g.
    ///     `200 as i8 == -56`, `-1 as i8 == -1`. The signed `>>` lowers to a naga
    ///     arithmetic shift (the shift-emit fix keeps the Sint value's signedness
    ///     while forcing the amount to u32).
    /// A non-narrow target returns the value unchanged.
    fn apply_narrow_mask(
        &mut self,
        value: naga::Handle<Expression>,
        target: &DataType,
    ) -> naga::Handle<Expression> {
        match target {
            DataType::U8 | DataType::U16 => {
                let mask = if matches!(target, DataType::U8) {
                    0xFFu32
                } else {
                    0xFFFFu32
                };
                let mask_lit = self.append_expr(Expression::Literal(Literal::U32(mask)));
                self.append_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: value,
                    right: mask_lit,
                })
            }
            DataType::I8 | DataType::I16 => {
                let shift = if matches!(target, DataType::I8) {
                    24u32
                } else {
                    16u32
                };
                let shift_lit = self.append_expr(Expression::Literal(Literal::U32(shift)));
                let shifted_left = self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: value,
                    right: shift_lit,
                });
                self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: shifted_left,
                    right: shift_lit,
                })
            }
            _ => value,
        }
    }
}
