//! The `KernelOpKind` dispatch: every variant routes from here to the emit
//! helper that owns it, plus the two helpers only this dispatch calls.

use std::fmt::Write as _;

use naga::{BinaryOperator, Expression, Literal, LocalVariable, ScalarKind, Span, Statement};
use smallvec::SmallVec;
use vyre_foundation::ir::{DataType, UnOp};
use vyre_lower::{KernelBody, KernelOp, KernelOpKind, LiteralValue};

use super::super::op_lookup::{
    barrier_flags, naga_literal, unary_math_function, unary_operator, unpack_shift_mask,
};
use super::super::BodyBuilder;
use super::diagnostics::{
    call_reached_message, missing_binding_slot_message, missing_literal_pool_index_message,
    opaque_expression_message, opaque_node_message, wide_literal_kind_gate_message,
    wide_literal_payload_message,
};
use super::OpDispatchRoute;
use crate::EmitError;

macro_rules! with_route_kind {
    ($op:expr, $route:expr, $pattern:pat => $body:expr) => {
        match &$op.kind {
            $pattern => $body,
            _ => Err(route_mismatch($route)),
        }
    };
}

fn route_mismatch(route: OpDispatchRoute) -> EmitError {
    let mut message: String = Default::default();
    message.push_str("internal Naga op-dispatch route mismatch for ");
    let _ = write!(&mut message, "{route:?}");
    EmitError::InvalidDescriptor(message)
}

impl BodyBuilder<'_> {
    pub(in crate::emitter) fn emit_op(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
    ) -> Result<(), EmitError> {
        let route = self.op_dispatch_routes.route(&op.kind);
        match route {
            OpDispatchRoute::Literal => {
                let literal_index = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("literal op missing literal-pool index".into())
                })?;
                let literal = body.literals.get(literal_index as usize).ok_or_else(|| {
                    EmitError::InvalidDescriptor(missing_literal_pool_index_message(literal_index))
                })?;
                let handle = if let LiteralValue::F32(value) = literal {
                    if value.is_finite() {
                        self.append_expr(Expression::Literal(naga_literal(literal)?))
                    } else {
                        // Naga's `Literal::F32` rejects NaN/Inf even though
                        // WGSL can represent the exact bit pattern via
                        // `bitcast<f32>(u32_bits)`. Preserve the IR literal
                        // byte-for-byte instead of weakening ops that use
                        // `-inf` as a sentinel, e.g. top-k initializers.
                        let bits = self.append_expr(Expression::Literal(Literal::U32(
                            value.to_bits(),
                        )));
                        self.append_expr(Expression::As {
                            expr: bits,
                            kind: ScalarKind::Float,
                            convert: None,
                        })
                    }
                } else {
                    self.append_expr(Expression::Literal(naga_literal(literal)?))
                };
                let ty = self.literal_type(literal);
                self.bind_result_typed(op, handle, ty)
            }
            OpDispatchRoute::LocalInvocationId => self.emit_builtin_axis(op, self.builtins.local),
            OpDispatchRoute::GlobalInvocationId => self.emit_builtin_axis(op, self.builtins.global),
            OpDispatchRoute::WorkgroupId => self.emit_builtin_axis(op, self.builtins.workgroup),
            OpDispatchRoute::SubgroupLocalId => {
                self.emit_scalar_builtin(op, self.builtins.subgroup_local, "SubgroupLocalId")
            }
            OpDispatchRoute::SubgroupSize => {
                self.emit_scalar_builtin(op, self.builtins.subgroup_size, "SubgroupSize")
            }
            OpDispatchRoute::LoopIndex => with_route_kind!(
                op,
                route,
                KernelOpKind::LoopIndex { loop_var } => self.emit_loop_index(op, loop_var)
            ),
            OpDispatchRoute::BufferLength => {
                let slot = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("BufferLength op missing binding slot".into())
                })?;
                let value = self.buffer_len_expr(slot)?;
                self.bind_result_typed(op, value, self.types.u32_ty)
            }
            OpDispatchRoute::Load => {
                let slot = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor(missing_binding_slot_message(&op.kind))
                })?;
                // Byte-element bindings (U8/I8) are packed into array<u32>
                // by the WGSL emitter (no native byte storage). The IR-level
                // index is a byte address (matching reference-eval); extract
                // the right lane from the loaded word so the wire-correct
                // byte reaches the consumer.
                let data_type = self.binding_data_types.get(&slot).cloned();
                if let Some(dt @ (DataType::U8 | DataType::I8)) = data_type {
                    return self.emit_byte_element_load(op, slot, dt);
                }
                let pointer = self.binding_element_pointer(op, 0, 1)?;
                let value = self.append_expr(Expression::Load { pointer });
                let ty =
                    *self
                        .binding_types
                        .get(&slot)
                        .ok_or_else(|| EmitError::InvalidBinding {
                            slot,
                            reason: "no scalar type was recorded for this slot".into(),
                        })?;
                self.bind_result_typed(op, value, ty)
            }
            OpDispatchRoute::Store => {
                let slot = self.slot_operand(op, 0)?;
                // Byte-element bindings (U8/I8) need a read-modify-write
                // through the array<u32> word so the byte at `index`
                // changes without clobbering the three adjacent bytes
                // packed into the same u32. Naive Store would write the
                // value as a u32 to the byte address, corrupting the
                // surrounding bytes  -  the same byte/word-addressing
                // mismatch the LoadGlobal byte-extract path closed.
                let data_type = self.binding_data_types.get(&slot).cloned();
                if matches!(data_type, Some(DataType::U8) | Some(DataType::I8)) {
                    return self.emit_byte_element_store(op, slot);
                }
                let pointer = self.binding_element_pointer(op, 0, 1)?;
                let raw_value = self.value_operand(op, 2)?;
                let value = match self.binding_types.get(&slot).copied() {
                    Some(ty) => self.coerce_value_to_type(raw_value, ty),
                    None => raw_value,
                };
                self.function
                    .body
                    .push(Statement::Store { pointer, value }, Span::UNDEFINED);
                Ok(())
            }
            OpDispatchRoute::VectorLoad => with_route_kind!(
                op,
                route,
                KernelOpKind::VectorLoadGlobal { width } => {
                    let slot = *op.operands.first().ok_or_else(|| {
                        EmitError::InvalidDescriptor(missing_binding_slot_message(&op.kind))
                    })?;
                    let base_index = self.value_operand(op, 1)?;
                    let ty = *self
                        .binding_types
                        .get(&slot)
                        .ok_or_else(|| EmitError::InvalidBinding {
                            slot,
                            reason: "no scalar type was recorded for this slot".into(),
                        })?;
                    let mut lane_exprs = SmallVec::<[naga::Handle<Expression>; 4]>::new();
                    for i in 0..*width {
                        let offset_expr = if i == 0 {
                            base_index
                        } else {
                            let i_lit = self.literal_u32(i as u32);
                            self.append_expr(Expression::Binary {
                                op: BinaryOperator::Add,
                                left: base_index,
                                right: i_lit,
                            })
                        };
                        let pointer = self.binding_element_pointer_by_slot(slot, offset_expr)?;
                        let val = self.append_expr(Expression::Load { pointer });
                        lane_exprs.push(val);
                    }
                    if let Some(res) = op.result {
                        self.vector_lanes.insert(res, (lane_exprs, ty));
                    }
                    Ok(())
                }
            ),
            OpDispatchRoute::VectorStore => with_route_kind!(
                op,
                route,
                KernelOpKind::VectorStoreGlobal { width: _ } => {
                    let slot = self.slot_operand(op, 0)?;
                    let base_index = self.value_operand(op, 1)?;
                    let binding_ty = self.binding_types.get(&slot).copied();
                    for i in 0..(op.operands.len().saturating_sub(2)) {
                        let offset_expr = if i == 0 {
                            base_index
                        } else {
                            let i_lit = self.literal_u32(i as u32);
                            self.append_expr(Expression::Binary {
                                op: BinaryOperator::Add,
                                left: base_index,
                                right: i_lit,
                            })
                        };
                        let pointer = self.binding_element_pointer_by_slot(slot, offset_expr)?;
                        let raw_value = self.value_operand(op, 2 + i)?;
                        let value = match binding_ty {
                            Some(ty) => self.coerce_value_to_type(raw_value, ty),
                            None => raw_value,
                        };
                        self.function
                            .body
                            .push(Statement::Store { pointer, value }, Span::UNDEFINED);
                    }
                    Ok(())
                }
            ),
            OpDispatchRoute::ExtractLane => with_route_kind!(
                op,
                route,
                KernelOpKind::ExtractLane { lane } => {
                    let vec_id = *op.operands.first().ok_or_else(|| {
                        EmitError::InvalidDescriptor("ExtractLane missing vector operand".into())
                    })?;
                    let (lanes, ty) = self.vector_lanes.get(&vec_id).ok_or_else(|| {
                        EmitError::InvalidDescriptor(format!(
                            "ExtractLane: vector value {vec_id} has no recorded vector lanes"
                        ))
                    })?;
                    let val = *lanes.get(*lane as usize).ok_or_else(|| {
                        EmitError::InvalidDescriptor(format!(
                            "ExtractLane: lane index {lane} out of bounds for vector {vec_id}"
                        ))
                    })?;
                    let ty = *ty;
                    self.bind_result_typed(op, val, ty)
                }
            ),
            OpDispatchRoute::Copy => {
                let value = self.value_operand(op, 0)?;
                let ty = self.value_type_operand(op, 0)?;
                let local = self.function.local_variables.append(
                    LocalVariable {
                        name: None,
                        ty,
                        init: None,
                    },
                    Span::UNDEFINED,
                );
                let value = self.coerce_value_to_type(value, ty);
                let pointer = self.append_expr(Expression::LocalVariable(local));
                self.function
                    .body
                    .push(Statement::Store { pointer, value }, Span::UNDEFINED);
                let pointer = self.append_expr(Expression::LocalVariable(local));
                let snapshot = self.append_expr(Expression::Load { pointer });
                self.bind_result_typed(op, snapshot, ty)
            }
            OpDispatchRoute::BinOpKind => with_route_kind!(
                op,
                route,
                KernelOpKind::BinOpKind(binop) => self.emit_binop(op, *binop)
            ),
            OpDispatchRoute::UnOpKind => with_route_kind!(op, route, KernelOpKind::UnOpKind(unop) => {
                let expr = self.value_operand(op, 0)?;
                let ty = match unop {
                    UnOp::LogicalNot | UnOp::IsNan | UnOp::IsInf | UnOp::IsFinite => {
                        self.types.bool_ty
                    }
                    _ => self.value_type_operand(op, 0)?,
                };
                // 64-bit gate (mirrors the binop gate): U64/I64 are backed by
                // vec2<u32>. A Naga unary applied to the pair runs PER-WORD, so
                // popcount/clz/ctz/reverse/negate on a 64-bit value would be
                // SILENTLY WRONG (popcount/clz/ctz count a single word;
                // reverse_bits reverses each word without swapping them; negate
                // carries no borrow). Only bitwise NOT is correct componentwise.
                // Fail closed (Law 10) rather than emit a per-word result.
                let operand_is_u64 = self
                    .value_type_operand(op, 0)
                    .map(|h| h == self.types.vec2_u32_ty)
                    .unwrap_or(false);
                if operand_is_u64 && !matches!(unop, UnOp::BitNot) {
                    return Err(EmitError::NagaConstructionFailed(format!(
                        "64-bit (U64/I64) unary `{unop:?}` is not lowered: the \
                         vec2<u32> backing would apply it per-word, so the 64-bit \
                         result would be silently wrong (popcount/clz/ctz count a \
                         single word; reverse_bits does not swap words; negate \
                         carries no borrow). Only bitwise NOT is correct \
                         componentwise on a 64-bit value. Fix: add a cross-word \
                         U64 emulation pass before this op reaches Naga emission."
                    )));
                }
                // Naga's `LogicalNot` requires a Bool operand. When the
                // operand was published via a u32 carrier local (e.g. a
                // bool result that was bind_result_typed as u32 because
                // an upstream op flagged it as numeric), the cached Load
                // returns u32 and naga rejects with
                // `InvalidUnaryOperandType(LogicalNot, ...)`. Coerce
                // explicitly via the same path used for `if` conditions.
                let expr = if matches!(unop, UnOp::LogicalNot) {
                    self.ensure_bool_condition(expr)
                } else {
                    expr
                };
                let value = if matches!(unop, UnOp::Reciprocal) {
                    let one = self.append_expr(Expression::Literal(Literal::F32(1.0)));
                    self.append_expr(Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: one,
                        right: expr,
                    })
                } else if matches!(unop, UnOp::IsNan) {
                    self.append_expr(Expression::Binary {
                        op: BinaryOperator::NotEqual,
                        left: expr,
                        right: expr,
                    })
                } else if matches!(unop, UnOp::IsInf | UnOp::IsFinite) {
                    let abs = self.append_expr(Expression::Math {
                        fun: naga::MathFunction::Abs,
                        arg: expr,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    });
                    let max = self.append_expr(Expression::Literal(Literal::F32(f32::MAX)));
                    let op = if matches!(unop, UnOp::IsFinite) {
                        BinaryOperator::LessEqual
                    } else {
                        BinaryOperator::Greater
                    };
                    self.append_expr(Expression::Binary {
                        op,
                        left: abs,
                        right: max,
                    })
                } else if let Some((shift, mask)) = unpack_shift_mask(unop) {
                    // Nibble/byte unpack has no Naga intrinsic; lower to an
                    // explicit `(value >> shift) & mask` on u32 (semantics match
                    // ir_eval). Without this the emitter rejected with "unary op
                    // `Unpack4Low` has no direct Naga unary operator".
                    //
                    // Unpack is UNSIGNED bit extraction. Reinterpret the source
                    // to u32 first so a signed source (e.g. a load from an i32
                    // buffer, whose kind does not resolve through the
                    // `Load(Access)` chain) does not emit `ShiftRight(i32, u32)`
                    // / `And(i32, u32)`, which naga rejects. A source already
                    // known to be u32 is left untouched (no redundant bitcast).
                    let expr = if matches!(
                        self.scalar_kind_of_expression(expr, 0),
                        Some(ScalarKind::Uint)
                    ) {
                        expr
                    } else {
                        self.append_expr(Expression::As {
                            expr,
                            kind: ScalarKind::Uint,
                            convert: None,
                        })
                    };
                    let shifted = if shift == 0 {
                        expr
                    } else {
                        let shift_lit =
                            self.append_expr(Expression::Literal(Literal::U32(shift)));
                        self.append_expr(Expression::Binary {
                            op: BinaryOperator::ShiftRight,
                            left: expr,
                            right: shift_lit,
                        })
                    };
                    let mask_lit = self.append_expr(Expression::Literal(Literal::U32(mask)));
                    self.append_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: shifted,
                        right: mask_lit,
                    })
                } else if let Some(fun) = unary_math_function(unop) {
                    self.append_expr(Expression::Math {
                        fun,
                        arg: expr,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    })
                } else if matches!(unop, UnOp::Negate)
                    && self.scalar_kind_of_expression(expr, 0) == Some(ScalarKind::Uint)
                {
                    // WGSL/naga reject unary minus on an UNSIGNED operand
                    // (`InvalidUnaryOperandType(Negate)`), but vyre's typecheck
                    // legalises integer `Negate` exactly for `u32` (the total,
                    // wrapping case, raw i32 negate is rejected upstream for its
                    // i32::MIN overflow), and the reference oracle defines it as
                    // `0u32.wrapping_sub(v)`. Emitting `Unary(Negate, u32)` made
                    // every u32 negate validate + run correctly on the CPU oracle
                    // yet HARD-FAIL at GPU dispatch, a front-end-accepts /
                    // backend-can't-emit coherence gap (Law 10). naga's `Subtract`
                    // on a Uint wraps in two's complement, so synthesize the
                    // unsigned negate as `0u - v`, matching the oracle exactly.
                    let zero = self.append_expr(Expression::Literal(Literal::U32(0)));
                    self.append_expr(Expression::Binary {
                        op: BinaryOperator::Subtract,
                        left: zero,
                        right: expr,
                    })
                } else {
                    let naga_op = unary_operator(unop)?;
                    self.append_expr(Expression::Unary { op: naga_op, expr })
                };
                self.bind_result_typed(op, value, ty)
            }),
            OpDispatchRoute::Cast => with_route_kind!(op, route, KernelOpKind::Cast { target } => {
                self.emit_cast(op, target)
            }),
            OpDispatchRoute::Select => {
                let condition = self.value_operand(op, 0)?;
                let accept = self.value_operand(op, 1)?;
                let reject = self.value_operand(op, 2)?;
                let condition = self.ensure_bool_condition(condition);
                let ty = self.value_type_operand(op, 1)?;
                // Coerce reject to accept's scalar type. Without this,
                // when accept and reject were each `bind_result_typed`-d
                // with different scalar kinds (e.g. accept=u32 from a
                // numeric op, reject=bool from a comparison), naga
                // rejects the Select with `SelectValuesTypeMismatch`.
                // The pre-publish path masked this by inlining one arm
                // as a literal; explicit `LocalVariable + Load` round-
                // tripping (Q7 carrier mechanism) exposes the mismatch.
                let reject = self.coerce_value_to_type(reject, ty);
                let accept = self.coerce_value_to_type(accept, ty);
                let value = self.append_expr(Expression::Select {
                    condition,
                    accept,
                    reject,
                });
                self.bind_result_typed(op, value, ty)
            }
            OpDispatchRoute::Fma => {
                let arg = self.value_operand(op, 0)?;
                let arg1 = Some(self.value_operand(op, 1)?);
                let arg2 = Some(self.value_operand(op, 2)?);
                let value = self.append_expr(Expression::Math {
                    fun: naga::MathFunction::Fma,
                    arg,
                    arg1,
                    arg2,
                    arg3: None,
                });
                let ty = self.value_type_operand(op, 0)?;
                self.bind_result_typed(op, value, ty)
            }
            OpDispatchRoute::StructuredIfThen => {
                self.emit_structured_if(body, op, &[1])
            }
            OpDispatchRoute::StructuredIfThenElse => {
                self.emit_structured_if(body, op, &[1, 2])
            }
            OpDispatchRoute::StructuredBlock => {
                self.emit_structured_block(body, op)
            }
            OpDispatchRoute::StructuredForLoop => with_route_kind!(
                op,
                route,
                KernelOpKind::StructuredForLoop { loop_var } => {
                self.emit_structured_for_loop(body, op, loop_var)
                }
            ),
            OpDispatchRoute::AsyncLoad => self.emit_async_load(op),
            OpDispatchRoute::AsyncStore => self.emit_async_store(op),
            OpDispatchRoute::AsyncWait => {
                self.function.body.push(
                    Statement::Barrier(naga::Barrier::STORAGE | naga::Barrier::WORK_GROUP),
                    Span::UNDEFINED,
                );
                Ok(())
            }
            OpDispatchRoute::Trap => with_route_kind!(
                op,
                route,
                KernelOpKind::Trap { tag } => self.emit_trap(op, tag)
            ),
            // Resume is a runtime sequencing marker that the Naga backend
            // treats as a no-op. The Trap protocol in this backend emits an
            // unconditional Return after the sidecar write (see emit_trap),
            // so any statements after a Trap are not executed. Resume carries
            // sequencing intent for higher-level passes (scheduling, analysis)
            // but does not map to a Naga IR statement. On backends with real
            // continuations (e.g. PTX setmaxnreg + bar.sync) this must emit
            // the continuation resume instruction.
            OpDispatchRoute::Resume => Ok(()),
            OpDispatchRoute::Barrier => with_route_kind!(op, route, KernelOpKind::Barrier { ordering } => {
                let barrier = barrier_flags(*ordering, body)?;
                self.function
                    .body
                    .push(Statement::Barrier(barrier), Span::UNDEFINED);
                Ok(())
            }),
            OpDispatchRoute::Return => {
                self.function
                    .body
                    .push(Statement::Return { value: None }, Span::UNDEFINED);
                Ok(())
            }
            OpDispatchRoute::SubgroupBallot => {
                self.emit_subgroup_ballot(op)
            }
            OpDispatchRoute::SubgroupReduce => {
                self.emit_subgroup_reduce(op)
            }
            OpDispatchRoute::SubgroupShuffle => {
                self.emit_subgroup_shuffle(op)
            }
            OpDispatchRoute::SubgroupBroadcast => {
                self.emit_subgroup_broadcast(op)
            }
            OpDispatchRoute::Atomic => with_route_kind!(op, route, KernelOpKind::Atomic {
                op: atomic_op,
                ordering: _,
            } => {
                self.emit_atomic(op, *atomic_op)
            }),
            // IndirectDispatch has no Naga lowering. Naga compute shaders fix
            // the workgroup size in the @workgroup_size attribute at
            // compile time. Writing a dispatch-count buffer at runtime
            // (the IndirectDispatch semantic) is not a shader-internal
            // operation in the WGSL/Naga model, it must be done by the
            // host before launching the next dispatch. Fix: perform the
            // indirect count buffer write on the host side (or via a
            // separate count-kernel dispatch) rather than embedding it in
            // the main compute shader.
            OpDispatchRoute::IndirectDispatch => Err(EmitError::InvalidDescriptor(
                "IndirectDispatch reached the Naga emitter. Naga compute shaders cannot write \
                 dispatch count buffers from within a shader; the workgroup size is fixed at \
                 compile time. Fix: compute and write the indirect count buffer on the host, or \
                 via a dedicated count-kernel dispatch, before launching the indirect dispatch."
                    .into(),
            )),
            OpDispatchRoute::MatrixMma => Err(EmitError::InvalidDescriptor(
                "MatrixMma reached descriptor Naga emission. Fix: route MatrixMma through a concrete tensor-core backend or lower it before Naga emission.".into(),
            )),
            OpDispatchRoute::Call => with_route_kind!(
                op,
                route,
                KernelOpKind::Call { op_id } => {
                    Err(EmitError::InvalidDescriptor(call_reached_message(op_id.as_ref())))
                }
            ),
            OpDispatchRoute::OpaqueExpr => with_route_kind!(op, route, KernelOpKind::OpaqueExpr(data) => {
                self.emit_opaque_expr(op, data.extension_id, &data.extension_kind, &data.payload)
            }),
            OpDispatchRoute::OpaqueNode => with_route_kind!(
                op,
                route,
                KernelOpKind::OpaqueNode(data) => Err(EmitError::InvalidDescriptor(
                    opaque_node_message(&data.extension_kind, data.payload.len())
                ))
            ),
            OpDispatchRoute::LoopCarrierInit => with_route_kind!(
                op,
                route,
                KernelOpKind::LoopCarrierInit { name } => self.emit_loop_carrier_init(op, name)
            ),
            OpDispatchRoute::LoopCarrier => with_route_kind!(
                op,
                route,
                KernelOpKind::LoopCarrier { name } => self.emit_loop_carrier_read(op, name)
            ),
            OpDispatchRoute::LoopCarrierEnd => with_route_kind!(
                op,
                route,
                KernelOpKind::LoopCarrierEnd { name } => self.emit_loop_carrier_end(op, name)
            ),
        }
    }

    pub(in crate::emitter) fn global_invocation_axis(
        &mut self,
        axis: u32,
    ) -> naga::Handle<Expression> {
        let base = self.append_expr(Expression::FunctionArgument(self.builtins.global));
        self.append_expr(Expression::AccessIndex { base, index: axis })
    }

    pub(in crate::emitter) fn emit_opaque_expr(
        &mut self,
        op: &KernelOp,
        extension_id: u32,
        extension_kind: &str,
        payload: &[u8],
    ) -> Result<(), EmitError> {
        if matches!(
            extension_kind,
            "vyre.literal.u64" | "vyre.literal.i64" | "vyre.literal.f64"
        ) {
            let bytes: [u8; 8] = payload.try_into().map_err(|_| {
                EmitError::InvalidDescriptor(wide_literal_payload_message(
                    extension_kind,
                    payload.len(),
                ))
            })?;
            let (literal, ty) = match extension_kind {
                // Emit the full 64-bit literal directly. Naga's IR supports
                // Literal::U64 and the type handle u64_ty is already
                // registered in TypeHandles. Previously this narrowed to u32,
                // which silently produced the wrong type (and hard-errored for
                // values above u32::MAX), diverging from f64 which already
                // used Literal::F64. Callers that ask for vyre.literal.u64
                // always want a u64 result.
                "vyre.literal.u64" => {
                    let value = u64::from_le_bytes(bytes);
                    (Literal::U64(value), self.types.u64_ty)
                }
                // Emit the full 64-bit signed literal directly, matching the
                // u64 fix above. Previously narrowed to i32 and hard-errored
                // for values outside i32 range.
                "vyre.literal.i64" => {
                    let value = i64::from_le_bytes(bytes);
                    (Literal::I64(value), self.types.i64_ty)
                }
                "vyre.literal.f64" => (Literal::F64(f64::from_le_bytes(bytes)), self.types.f64_ty),
                other => {
                    return Err(EmitError::InvalidDescriptor(
                        wide_literal_kind_gate_message(other),
                    ));
                }
            };
            let value = self.append_expr(Expression::Literal(literal));
            return self.bind_result_typed(op, value, ty);
        }
        Err(EmitError::InvalidDescriptor(opaque_expression_message(
            extension_kind,
            extension_id,
        )))
    }
}
