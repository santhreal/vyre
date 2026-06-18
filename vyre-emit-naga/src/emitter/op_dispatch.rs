//! `KernelOp` → naga emit dispatcher. The big match arm  -  every
//! `KernelOpKind` variant routes to its emit helper from here. Plus the
//! two helpers that only `emit_op` calls (`global_invocation_axis`,
//! `emit_opaque_expr`).

use std::fmt::Write as _;
use std::mem::{self, Discriminant};

use naga::{BinaryOperator, Expression, Literal, LocalVariable, ScalarKind, Span, Statement};
use rustc_hash::FxHashMap;
use vyre_foundation::ir::{BinOp, DataType, UnOp};
use vyre_lower::{KernelBody, KernelOp, KernelOpKind, LiteralValue};

use super::op_lookup::{
    barrier_flags, binary_math_function, binary_operator, naga_literal, scalar_cast_target,
    unary_math_function, unary_operator,
};
use super::BodyBuilder;
use crate::EmitError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OpDispatchRoute {
    Literal,
    LocalInvocationId,
    GlobalInvocationId,
    WorkgroupId,
    SubgroupLocalId,
    SubgroupSize,
    LoopIndex,
    BufferLength,
    Load,
    Store,
    Copy,
    BinOpKind,
    UnOpKind,
    Cast,
    Select,
    Fma,
    StructuredIfThen,
    StructuredIfThenElse,
    StructuredBlock,
    StructuredForLoop,
    AsyncLoad,
    AsyncStore,
    AsyncWait,
    Trap,
    Resume,
    Barrier,
    Return,
    SubgroupBallot,
    SubgroupAdd,
    SubgroupShuffle,
    Atomic,
    IndirectDispatch,
    MatrixMma,
    Call,
    OpaqueExpr,
    OpaqueNode,
    LoopCarrierInit,
    LoopCarrier,
    LoopCarrierEnd,
}

pub(super) struct OpDispatchRouteCache {
    routes: FxHashMap<Discriminant<KernelOpKind>, OpDispatchRoute>,
    #[cfg(test)]
    hits: usize,
}

impl Default for OpDispatchRouteCache {
    fn default() -> Self {
        Self {
            routes: FxHashMap::default(),
            #[cfg(test)]
            hits: 0,
        }
    }
}

impl OpDispatchRouteCache {
    fn route(&mut self, kind: &KernelOpKind) -> OpDispatchRoute {
        let key = mem::discriminant(kind);
        if let Some(route) = self.routes.get(&key).copied() {
            #[cfg(test)]
            {
                self.hits += 1;
            }
            return route;
        }
        let route = classify_op_dispatch_route(kind);
        self.routes.insert(key, route);
        route
    }
}

#[cfg(test)]
pub(crate) fn op_dispatch_route_cache_probe(kinds: &[KernelOpKind]) -> (bool, usize) {
    let mut cache = OpDispatchRouteCache::default();
    let mut parity = true;
    for kind in kinds {
        let uncached = classify_op_dispatch_route(kind);
        let cached = cache.route(kind);
        parity &= uncached == cached;
    }
    (parity, cache.hits)
}

fn classify_op_dispatch_route(kind: &KernelOpKind) -> OpDispatchRoute {
    match kind {
        KernelOpKind::Literal => OpDispatchRoute::Literal,
        KernelOpKind::LocalInvocationId => OpDispatchRoute::LocalInvocationId,
        KernelOpKind::GlobalInvocationId => OpDispatchRoute::GlobalInvocationId,
        KernelOpKind::WorkgroupId => OpDispatchRoute::WorkgroupId,
        KernelOpKind::SubgroupLocalId => OpDispatchRoute::SubgroupLocalId,
        KernelOpKind::SubgroupSize => OpDispatchRoute::SubgroupSize,
        KernelOpKind::LoopIndex { .. } => OpDispatchRoute::LoopIndex,
        KernelOpKind::BufferLength => OpDispatchRoute::BufferLength,
        KernelOpKind::LoadGlobal | KernelOpKind::LoadShared | KernelOpKind::LoadConstant => {
            OpDispatchRoute::Load
        }
        KernelOpKind::StoreGlobal | KernelOpKind::StoreShared => OpDispatchRoute::Store,
        KernelOpKind::Copy => OpDispatchRoute::Copy,
        KernelOpKind::BinOpKind(_) => OpDispatchRoute::BinOpKind,
        KernelOpKind::UnOpKind(_) => OpDispatchRoute::UnOpKind,
        KernelOpKind::Cast { .. } => OpDispatchRoute::Cast,
        KernelOpKind::Select => OpDispatchRoute::Select,
        KernelOpKind::Fma => OpDispatchRoute::Fma,
        KernelOpKind::StructuredIfThen => OpDispatchRoute::StructuredIfThen,
        KernelOpKind::StructuredIfThenElse => OpDispatchRoute::StructuredIfThenElse,
        KernelOpKind::StructuredBlock | KernelOpKind::Region { .. } => {
            OpDispatchRoute::StructuredBlock
        }
        KernelOpKind::StructuredForLoop { .. } => OpDispatchRoute::StructuredForLoop,
        KernelOpKind::AsyncLoad { .. } => OpDispatchRoute::AsyncLoad,
        KernelOpKind::AsyncStore { .. } => OpDispatchRoute::AsyncStore,
        KernelOpKind::AsyncWait { .. } => OpDispatchRoute::AsyncWait,
        KernelOpKind::Trap { .. } => OpDispatchRoute::Trap,
        KernelOpKind::Resume { .. } => OpDispatchRoute::Resume,
        KernelOpKind::Barrier { .. } => OpDispatchRoute::Barrier,
        KernelOpKind::Return => OpDispatchRoute::Return,
        KernelOpKind::SubgroupBallot => OpDispatchRoute::SubgroupBallot,
        KernelOpKind::SubgroupAdd => OpDispatchRoute::SubgroupAdd,
        KernelOpKind::SubgroupShuffle => OpDispatchRoute::SubgroupShuffle,
        KernelOpKind::Atomic { .. } => OpDispatchRoute::Atomic,
        KernelOpKind::IndirectDispatch { .. } => OpDispatchRoute::IndirectDispatch,
        KernelOpKind::MatrixMma { .. } => OpDispatchRoute::MatrixMma,
        KernelOpKind::Call { .. } => OpDispatchRoute::Call,
        KernelOpKind::OpaqueExpr(_) => OpDispatchRoute::OpaqueExpr,
        KernelOpKind::OpaqueNode(_) => OpDispatchRoute::OpaqueNode,
        KernelOpKind::LoopCarrierInit { .. } => OpDispatchRoute::LoopCarrierInit,
        KernelOpKind::LoopCarrier { .. } => OpDispatchRoute::LoopCarrier,
        KernelOpKind::LoopCarrierEnd { .. } => OpDispatchRoute::LoopCarrierEnd,
    }
}

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

fn missing_literal_pool_index_message(literal_index: u32) -> String {
    let mut message: String = Default::default();
    message.push_str("literal op references missing literal-pool index ");
    let _ = write!(&mut message, "{literal_index}");
    message
}

fn missing_binding_slot_message(kind: &KernelOpKind) -> String {
    let mut message: String = Default::default();
    let _ = write!(&mut message, "{kind:?} missing binding slot");
    message
}

fn non_byte_load_route_message(data_type: DataType) -> String {
    let mut message: String = Default::default();
    message.push_str("emit_byte_element_load called with non-byte DataType ");
    let _ = write!(&mut message, "{data_type:?}");
    message.push_str("; this is an emitter routing bug");
    message
}

fn call_reached_message(op_id: &str) -> String {
    let mut message: String = Default::default();
    message.push_str("Call op `");
    message.push_str(op_id);
    message.push_str(
        "` reached descriptor Naga emission. Fix: expand calls into KernelDescriptor ops before emission.",
    );
    message
}

fn opaque_node_message(extension_kind: &str, payload_len: usize) -> String {
    let mut message: String = Default::default();
    message.push_str("opaque node `");
    message.push_str(extension_kind);
    message.push_str("` with ");
    let _ = write!(&mut message, "{payload_len}");
    message.push_str(
        " payload bytes has no descriptor Naga lowering. Fix: lower this extension into concrete KernelDescriptor ops before descriptor emission.",
    );
    message
}

fn wide_literal_payload_message(extension_kind: &str, payload_len: usize) -> String {
    let mut message: String = Default::default();
    message.push_str("wide-literal opaque `");
    message.push_str(extension_kind);
    message.push_str("` carries ");
    let _ = write!(&mut message, "{payload_len}");
    message.push_str(
        " payload bytes, expected 8. Fix: encode literals through Expr::u64/i64/f64 builders.",
    );
    message
}

fn wide_literal_kind_gate_message(kind: &str) -> String {
    let mut message: String = Default::default();
    message.push_str("wide-literal kind `");
    message.push_str(kind);
    message.push_str(
        "` reached descriptor opaque emission after the kind gate. Fix: update the kind gate and decoder together.",
    );
    message
}

fn opaque_expression_message(extension_kind: &str, extension_id: u32) -> String {
    let mut message: String = Default::default();
    message.push_str("opaque expression `");
    message.push_str(extension_kind);
    message.push_str("` (id=");
    let _ = write!(&mut message, "{extension_id:#010x}");
    message.push_str(
        ") has no descriptor Naga lowering. Fix: lower this extension into concrete KernelDescriptor ops or add a descriptor extension emitter before Naga emission.",
    );
    message
}

impl BodyBuilder<'_> {
    pub(super) fn emit_body(&mut self, body: &KernelBody) -> Result<(), EmitError> {
        for op in &body.ops {
            self.emit_op(body, op)?;
        }
        Ok(())
    }

    pub(super) fn emit_op(&mut self, body: &KernelBody, op: &KernelOp) -> Result<(), EmitError> {
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
                } else if let Some(fun) = unary_math_function(unop) {
                    self.append_expr(Expression::Math {
                        fun,
                        arg: expr,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    })
                } else {
                    let naga_op = unary_operator(unop)?;
                    self.append_expr(Expression::Unary { op: naga_op, expr })
                };
                self.bind_result_typed(op, value, ty)
            }),
            OpDispatchRoute::Cast => with_route_kind!(op, route, KernelOpKind::Cast { target } => {
                let expr = self.value_operand(op, 0)?;
                let (kind, width) = scalar_cast_target(target)?;
                let value = self.append_expr(Expression::As {
                    expr,
                    kind,
                    convert: Some(width),
                });
                let ty = self.type_for_data_type(target)?;
                self.bind_result_typed(op, value, ty)
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
            // AsyncWait is a documented no-op in the Naga backend. The Naga
            // backend lowers AsyncLoad and AsyncStore as fully synchronous
            // counted copy loops — the copy completes before the next op
            // executes. There is no deferred or out-of-order DMA in this path,
            // so no fence or barrier is needed: the copy is already done.
            // Backends that use real hardware async DMA (e.g. PTX cp.async)
            // must emit a hardware-level wait instruction here.
            OpDispatchRoute::AsyncWait => Ok(()),
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
                let barrier = barrier_flags(*ordering)?;
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
            OpDispatchRoute::SubgroupAdd => {
                self.emit_subgroup_add(op)
            }
            OpDispatchRoute::SubgroupShuffle => {
                self.emit_subgroup_shuffle(op)
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
            // operation in the WGSL/Naga model — it must be done by the
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

    /// Emit `Statement::Block` for `StructuredBlock` / `Region` with the
    /// same Q7 carrier-publish machinery as `emit_structured_if` and
    /// `emit_structured_for_loop`. Any SSA id produced inside the
    /// region's child body that the parent body references after the
    /// region must round-trip through a function-local: the in-region
    /// `Statement::Emit` lives inside the closed inner block, and the
    /// post-region reader needs a fresh `Load` whose Emit lives in the
    /// parent block. Without this, naga's WGSL writer emits `let _eN =
    /// ...;` inside the inner block and the post-region read of `_eN`
    /// trips `no definition in scope` validation. The lowering's
    /// Region phi-merge handles source-level NAMED carriers; this
    /// handles UNNAMED in-region SSA results that escape  -  exactly the
    /// `vyre_loop_carry_<id>` carrier path Loop/If already use.
    pub(super) fn emit_structured_block(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
    ) -> Result<(), EmitError> {
        let prior_carriers = self.snapshot_loop_carriers();
        let op_pos = body
            .ops
            .iter()
            .position(|candidate| std::ptr::eq(candidate, op))
            .unwrap_or(body.ops.len());
        let child_body_idxs: Vec<u32> = op.operands.iter().take(1).copied().collect();
        let new_targets = self.collect_child_carried_ids(body, op_pos, &child_body_idxs);

        let mut pre_init: Vec<(u32, naga::Handle<Expression>)> = Vec::default();
        for id in &new_targets {
            self.loop_carrier_targets.insert(*id);
            if let Some(handle) = self.value_handle_for_id(*id) {
                pre_init.push((*id, handle));
            }
        }
        for (id, init_handle) in &pre_init {
            let local = self.allocate_carrier_local(*id, init_handle);
            let local_ty = self.function.local_variables[local].ty;
            let init = self.coerce_value_to_type(*init_handle, local_ty);
            let pointer = self.append_expr(Expression::LocalVariable(local));
            self.function.body.push(
                Statement::Store {
                    pointer,
                    value: init,
                },
                Span::UNDEFINED,
            );
        }

        let block = self.child_block(body, op, 0)?;
        self.function
            .body
            .push(Statement::Block(block), Span::UNDEFINED);

        for id in &new_targets {
            if let Some(local) = self.loop_carrier_locals.get(id).copied() {
                let pointer = self.append_expr(Expression::LocalVariable(local));
                let load = self.append_expr(Expression::Load { pointer });
                self.values.insert(*id, load);
            }
        }
        self.restore_loop_carriers(prior_carriers);
        Ok(())
    }

    /// Emit `Statement::If { accept, reject }` for `StructuredIfThen`
    /// (`child_indices=&[1]`) and `StructuredIfThenElse`
    /// (`child_indices=&[1, 2]`) with the same Q7 carrier-publish
    /// machinery that `emit_structured_for_loop` uses. Without the
    /// publish, any value bound inside the if-body and read after the
    /// if surfaces as `no definition in scope for identifier _eN` from
    /// naga's WGSL writer (the `let _eN = ...;` binding lives inside
    /// the if-body's scope; the post-if reader is outside it).
    pub(super) fn emit_structured_if(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
        child_indices: &[usize],
    ) -> Result<(), EmitError> {
        let prior_carriers = self.snapshot_loop_carriers();
        let op_pos = body
            .ops
            .iter()
            .position(|candidate| std::ptr::eq(candidate, op))
            .unwrap_or(body.ops.len());
        let child_body_idxs: Vec<u32> = child_indices
            .iter()
            .filter_map(|i| op.operands.get(*i).copied())
            .collect();
        let new_targets = self.collect_child_carried_ids(body, op_pos, &child_body_idxs);

        // Pre-if init: for any new carrier whose id had a prior SSA
        // value bound in the parent scope, seed the carrier local so a
        // reader inside the if (or after it on the not-taken path) sees
        // the pre-if value. value_handle_for_id materializes the prior
        // value via fresh Load when the cached handle's emit-block has
        // closed; otherwise it returns the cached handle directly.
        let mut pre_init: Vec<(u32, naga::Handle<Expression>)> = Vec::default();
        for id in &new_targets {
            self.loop_carrier_targets.insert(*id);
            if let Some(handle) = self.value_handle_for_id(*id) {
                pre_init.push((*id, handle));
            }
        }
        for (id, init_handle) in &pre_init {
            let local = self.allocate_carrier_local(*id, init_handle);
            let local_ty = self.function.local_variables[local].ty;
            let init = self.coerce_value_to_type(*init_handle, local_ty);
            let pointer = self.append_expr(Expression::LocalVariable(local));
            self.function.body.push(
                Statement::Store {
                    pointer,
                    value: init,
                },
                Span::UNDEFINED,
            );
        }

        let condition = self.value_operand(op, 0)?;
        let condition = self.ensure_bool_condition(condition);
        let accept = self.child_block(body, op, child_indices[0])?;
        let reject = if child_indices.len() > 1 {
            self.child_block(body, op, child_indices[1])?
        } else {
            naga::Block::new()
        };
        self.function.body.push(
            Statement::If {
                condition,
                accept,
                reject,
            },
            Span::UNDEFINED,
        );

        // Post-if rebind: re-Load every carrier from its function-scope
        // local in the parent block so any subsequent reader resolves
        // to a Load whose Statement::Emit is in the current (parent)
        // body  -  not the now-closed if-body's expression range.
        for id in &new_targets {
            if let Some(local) = self.loop_carrier_locals.get(id).copied() {
                let pointer = self.append_expr(Expression::LocalVariable(local));
                let load = self.append_expr(Expression::Load { pointer });
                self.values.insert(*id, load);
            }
        }
        self.restore_loop_carriers(prior_carriers);
        Ok(())
    }

    /// `BinOpKind` emit  -  bool-vs-numeric widening, literal-pool fold,
    /// and Math-builtin routing live here to keep `emit_op` flat.
    fn emit_binop(&mut self, op: &KernelOp, binop: BinOp) -> Result<(), EmitError> {
        let left = self.value_operand(op, 0)?;
        let right = self.value_operand(op, 1)?;
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
        let value =
            if let Some(value) = self.emit_synthetic_binop(effective_binop, left_eff, right_eff) {
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

    pub(super) fn global_invocation_axis(&mut self, axis: u32) -> naga::Handle<Expression> {
        let base = self.append_expr(Expression::FunctionArgument(self.builtins.global));
        self.append_expr(Expression::AccessIndex { base, index: axis })
    }

    pub(super) fn emit_opaque_expr(
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
    fn emit_byte_element_load(
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
        let word = self.append_expr(Expression::Load { pointer });
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
    fn emit_byte_element_store(&mut self, op: &KernelOp, slot: u32) -> Result<(), EmitError> {
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
        // Read-modify-write the u32 word.
        let pointer = self.binding_element_pointer_by_slot(slot, word_index)?;
        let word = self.append_expr(Expression::Load { pointer });
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
        // Re-emit the pointer Access expression: naga's Statement::Store
        // requires a pointer that is in scope at the store site, and
        // the earlier `pointer` handle was consumed by the `Load`
        // we emitted above.
        let store_pointer = self.binding_element_pointer_by_slot(slot, word_index)?;
        self.function.body.push(
            Statement::Store {
                pointer: store_pointer,
                value: merged,
            },
            Span::UNDEFINED,
        );
        Ok(())
    }
}
