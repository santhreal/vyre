//! Byte-element load and store emission over word-addressed bindings.

use naga::{BinaryOperator, Expression, Span, Statement};
use vyre_foundation::ir::DataType;
use vyre_lower::KernelOp;

use super::super::BodyBuilder;
use super::diagnostics::non_byte_load_route_message;
use crate::EmitError;

impl BodyBuilder<'_> {
    /// Emit a Load on a byte-element binding (DataType::U8 / DataType::I8).
    ///
    /// Reference-eval treats U8/I8 buffers as byte-addressed; the WGSL
    /// backend has no native byte storage, so the underlying buffer is
    /// `array<u32>` (per `setup::scalar_type`). To honor the IR-level
    /// byte semantics, the emitter computes
    ///
    /// ```text
    /// word_index = index >> 2
    /// shift      = (index & 3) << 3
    /// byte       = (buffer[word_index] >> shift) & 0xff
    /// ```
    ///
    /// For `I8`, the extracted byte is sign-extended via the
    /// `(byte << 24) >> 24` bitcast pattern (arithmetic shift on i32
    /// preserves the sign bit).
    pub(in crate::emitter) fn emit_byte_element_load(
        &mut self,
        op: &KernelOp,
        slot: u32,
        data_type: DataType,
    ) -> Result<(), EmitError> {
        // The IR-level index is a byte address. Translate it to a word
        // index for naga's `array<u32>` Access expression.
        let raw_index = self.value_operand(op, 1)?;
        let byte_index = self.coerce_value_to_type(raw_index, self.types.u32_ty);
        let two = self.literal_u32(2);
        let three = self.literal_u32(3);
        let eight = self.literal_u32(8);
        let mask_ff = self.literal_u32(0xff);
        let word_index = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: byte_index,
            right: two,
        });
        let lane_in_word = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: byte_index,
            right: three,
        });
        let shift_bits = self.append_expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: lane_in_word,
            right: eight,
        });
        let pointer = self.binding_element_pointer_by_slot(slot, word_index)?;
        let word_bits = self.append_expr(Expression::Load { pointer });
        // The byte-extract shift+mask is UNSIGNED bit manipulation. An `I8`
        // buffer is backed by `array<i32>` (scalar_type maps I8 -> i32_ty), so
        // the loaded word is Sint; masking it with the u32 `0xff`/shift literals
        // would emit `And(i32, u32)` / `ShiftRight(i32, u32)` which naga rejects
        // (InvalidBinaryOperandTypes) (the I8 byte-extract emitted invalid WGSL).
        // Reinterpret the word's bits to u32 so the whole extraction is u32; the
        // I8 case re-signs only at the final `(byte << 24) as i32 >> 24` step.
        // `U8` is already `array<u32>`, so its word needs no reinterpret.
        let word = if matches!(data_type, DataType::I8) {
            self.append_expr(Expression::As {
                expr: word_bits,
                kind: naga::ScalarKind::Uint,
                convert: None,
            })
        } else {
            word_bits
        };
        let shifted = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: word,
            right: shift_bits,
        });
        let byte_u32 = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: shifted,
            right: mask_ff,
        });
        match data_type {
            DataType::U8 => {
                // Result type tracked in binding_types is u32_ty (per
                // setup::scalar_type's U8 → u32_ty mapping); the
                // extracted byte is already a u32 in the [0, 255]
                // range so it is wire-correct as-is.
                let ty = self.types.u32_ty;
                self.bind_result_typed(op, byte_u32, ty)
            }
            DataType::I8 => {
                // Sign-extend the [0, 255] u32 byte to a 32-bit signed
                // value via `((byte << 24) as i32) >> 24` (arithmetic
                // shift on i32 propagates the sign bit).
                let twenty_four = self.literal_u32(24);
                let shifted_left = self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: byte_u32,
                    right: twenty_four,
                });
                let as_i32 = self.append_expr(Expression::As {
                    expr: shifted_left,
                    kind: naga::ScalarKind::Sint,
                    convert: None,
                });
                let signed = self.append_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: as_i32,
                    right: twenty_four,
                });
                let ty = self.types.i32_ty;
                self.bind_result_typed(op, signed, ty)
            }
            other => Err(EmitError::InvalidBinding {
                slot,
                reason: non_byte_load_route_message(other),
            }),
        }
    }

    /// Emit a Store on a byte-element binding (DataType::U8 / DataType::I8).
    ///
    /// WGSL has no native byte storage; the underlying buffer is
    /// `array<u32>`. To store one byte at byte address `index` without
    /// clobbering the three adjacent bytes packed in the same u32, the
    /// emitter computes:
    ///
    /// ```text
    /// word_index = index >> 2
    /// shift      = (index & 3) << 3
    /// word       = buffer[word_index]
    /// cleared    = word & ~(0xff << shift)
    /// buffer[word_index] = cleared | ((value & 0xff) << shift)
    /// ```
    ///
    /// **Concurrency:** the read-modify-write is non-atomic. Two
    /// invocations writing different bytes of the same u32 word can race
    /// and lose one byte. This matches the existing convention for
    /// non-atomic word stores; callers needing safe concurrent byte
    /// stores should keep one invocation per word (the common pattern
    /// for output buffers indexed by `GlobalInvocationId`) or migrate to
    /// `Expr::Atomic` ops on a U32 buffer with explicit byte packing.
    pub(in crate::emitter) fn emit_byte_element_store(
        &mut self,
        op: &KernelOp,
        slot: u32,
    ) -> Result<(), EmitError> {
        let raw_index = self.value_operand(op, 1)?;
        let raw_value = self.value_operand(op, 2)?;
        let byte_index = self.coerce_value_to_type(raw_index, self.types.u32_ty);
        let value_u32 = self.coerce_value_to_type(raw_value, self.types.u32_ty);
        let two = self.literal_u32(2);
        let three = self.literal_u32(3);
        let eight = self.literal_u32(8);
        let mask_ff = self.literal_u32(0xff);
        let word_index = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: byte_index,
            right: two,
        });
        let lane_in_word = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: byte_index,
            right: three,
        });
        let shift_bits = self.append_expr(Expression::Binary {
            op: BinaryOperator::Multiply,
            left: lane_in_word,
            right: eight,
        });
        // (0xff << shift)  -  the byte mask in u32-word position.
        let lane_mask = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftLeft,
            left: mask_ff,
            right: shift_bits,
        });
        // ~(0xff << shift)  -  invert to clear the target byte.
        let cleared_mask = self.append_expr(Expression::Unary {
            op: naga::UnaryOperator::BitwiseNot,
            expr: lane_mask,
        });
        // (value & 0xff) << shift  -  value byte in u32-word position.
        let value_byte = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: value_u32,
            right: mask_ff,
        });
        let value_in_word = self.append_expr(Expression::Binary {
            op: BinaryOperator::ShiftLeft,
            left: value_byte,
            right: shift_bits,
        });
        // An `I8` buffer is backed by `array<i32>` (scalar_type maps I8 ->
        // i32_ty). The read-modify-write below is UNSIGNED bit manipulation
        // (mask/clear/merge with u32 literals), so on a Sint word it would emit
        // `And(i32, u32)` which naga rejects. Reinterpret the loaded word to u32
        // for the RMW, then reinterpret the merged result back to i32 before the
        // Store (whose value must match the array<i32> element type). `U8` is
        // already `array<u32>`, so it needs neither reinterpret.
        let is_signed_byte = matches!(self.binding_data_types.get(&slot), Some(DataType::I8));
        // Read-modify-write the u32 word.
        let pointer = self.binding_element_pointer_by_slot(slot, word_index)?;
        let word_bits = self.append_expr(Expression::Load { pointer });
        let word = if is_signed_byte {
            self.append_expr(Expression::As {
                expr: word_bits,
                kind: naga::ScalarKind::Uint,
                convert: None,
            })
        } else {
            word_bits
        };
        let cleared = self.append_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: word,
            right: cleared_mask,
        });
        let merged = self.append_expr(Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: cleared,
            right: value_in_word,
        });
        // Re-sign the merged u32 word to the buffer's i32 element type for I8.
        let store_value = if is_signed_byte {
            self.append_expr(Expression::As {
                expr: merged,
                kind: naga::ScalarKind::Sint,
                convert: None,
            })
        } else {
            merged
        };
        // Re-emit the pointer Access expression: naga's Statement::Store
        // requires a pointer that is in scope at the store site, and
        // the earlier `pointer` handle was consumed by the `Load`
        // we emitted above.
        let store_pointer = self.binding_element_pointer_by_slot(slot, word_index)?;
        self.function.body.push(
            Statement::Store {
                pointer: store_pointer,
                value: store_value,
            },
            Span::UNDEFINED,
        );
        Ok(())
    }
}
