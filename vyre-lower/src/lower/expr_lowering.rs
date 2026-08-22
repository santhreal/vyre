//! Expression lowering: every `Expr` form that reaches a descriptor operand.

use crate::descriptor::{KernelBody, KernelOp, KernelOpKind, LiteralValue, OpaqueExprData};
use crate::error::LowerError;
use vyre_foundation::ir::{AtomicOp, BinOp, Expr, SubgroupReduceOp};

use super::body_assembly::opaque_extension_id;
use super::LowerCtx;

impl LowerCtx {
    pub(super) fn lower_expr(
        &mut self,
        expr: &Expr,
        body: &mut KernelBody,
    ) -> Result<u32, LowerError> {
        match expr {
            Expr::LitU32(value) => self.literal(body, LiteralValue::U32(*value)),
            Expr::LitI32(value) => self.literal(body, LiteralValue::I32(*value)),
            Expr::LitF32(value) => self.literal(body, LiteralValue::F32(*value)),
            Expr::LitBool(value) => self.literal(body, LiteralValue::Bool(*value)),
            Expr::Var(name) => self.scope.get(name).ok_or_else(|| {
                LowerError::InvalidProgram(format!(
                    "variable `{name}` is referenced before binding. Fix: emit a Let/Assign before use."
                ))
            }),
            Expr::Load { buffer, index } => {
                let slot = self.buffer_slot(buffer)?;
                let index_id = self.lower_expr(index, body)?;
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: self.load_kind(slot),
                    operands: vec![slot, index_id],
                    result: Some(result),
                });
                Ok(result)
            }
            Expr::BufLen { buffer } => {
                let slot = self.buffer_slot(buffer)?;
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::BufferLength,
                    operands: vec![slot],
                    result: Some(result),
                });
                Ok(result)
            }
            Expr::InvocationId { axis } => {
                self.builtin_axis(body, KernelOpKind::GlobalInvocationId, *axis)
            }
            Expr::WorkgroupId { axis } => {
                self.builtin_axis(body, KernelOpKind::WorkgroupId, *axis)
            }
            Expr::LocalId { axis } => {
                self.builtin_axis(body, KernelOpKind::LocalInvocationId, *axis)
            }
            Expr::BinOp { op, left, right } => {
                // Subgroup/wave ops are spelled as binary ops at the Program
                // level but have dedicated subgroup KernelOps (and emit to
                // `subgroup*` dialect statements, not a BinaryOperator). Route
                // them before the generic `BinOpKind` path, which has no emitter
                // and would fail closed at emit. Operand contract mirrors the
                // canonical `Expr::Subgroup*` lowering below.
                match op {
                    BinOp::Shuffle => {
                        let value_id = self.lower_expr(left, body)?;
                        let lane_id = self.lower_expr(right, body)?;
                        self.binary(body, KernelOpKind::SubgroupShuffle, value_id, lane_id)
                    }
                    BinOp::WaveBroadcast => {
                        let value_id = self.lower_expr(left, body)?;
                        let lane_id = self.lower_expr(right, body)?;
                        self.binary(body, KernelOpKind::SubgroupBroadcast, value_id, lane_id)
                    }
                    BinOp::Ballot => {
                        // Ballot is unary (predicate). The binary spelling's
                        // right operand is unused, so it is not lowered.
                        let cond_id = self.lower_expr(left, body)?;
                        self.unary(body, KernelOpKind::SubgroupBallot, cond_id)
                    }
                    BinOp::WaveReduce => {
                        // The only subgroup reduce KernelOp is Add (sum-reduce),
                        // and the binary spelling carries no reduce-op selector,
                        // so WaveReduce lowers to a subgroup sum across the wave.
                        let value_id = self.lower_expr(left, body)?;
                        self.unary(
                            body,
                            KernelOpKind::SubgroupReduce {
                                op: SubgroupReduceOp::Add,
                            },
                            value_id,
                        )
                    }
                    _ => {
                        let left_id = self.lower_expr(left, body)?;
                        let right_id = self.lower_expr(right, body)?;
                        self.binary(body, KernelOpKind::BinOpKind(*op), left_id, right_id)
                    }
                }
            }
            Expr::UnOp { op, operand } => {
                let operand_id = self.lower_expr(operand, body)?;
                self.unary(body, KernelOpKind::UnOpKind(op.clone()), operand_id)
            }
            Expr::Call { op_id, args } => {
                let mut operands = Vec::with_capacity(args.len());
                for arg in args {
                    operands.push(self.lower_expr(arg, body)?);
                }
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Call {
                        op_id: op_id.shared_text(),
                    },
                    operands,
                    result: Some(result),
                });
                Ok(result)
            }
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => {
                let cond_id = self.lower_expr(cond, body)?;
                let true_id = self.lower_expr(true_val, body)?;
                let false_id = self.lower_expr(false_val, body)?;
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Select,
                    operands: vec![cond_id, true_id, false_id],
                    result: Some(result),
                });
                Ok(result)
            }
            Expr::Cast { target, value } => {
                let value_id = self.lower_expr(value, body)?;
                self.unary(
                    body,
                    KernelOpKind::Cast {
                        target: target.clone(),
                    },
                    value_id,
                )
            }
            Expr::Fma { a, b, c } => {
                let a_id = self.lower_expr(a, body)?;
                let b_id = self.lower_expr(b, body)?;
                let c_id = self.lower_expr(c, body)?;
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Fma,
                    operands: vec![a_id, b_id, c_id],
                    result: Some(result),
                });
                Ok(result)
            }
            Expr::Atomic {
                op,
                buffer,
                index,
                expected,
                value,
                ordering,
            } => {
                let slot = self.buffer_slot(buffer)?;
                let index_id = self.lower_expr(index, body)?;
                let value_id = self.lower_expr(value, body)?;
                let operands = if matches!(
                    op,
                    AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak
                ) {
                    let Some(expected) = expected else {
                        return Err(LowerError::InvalidProgram(
                            "atomic compare-exchange is missing expected value. Fix: set Expr::Atomic.expected.".into(),
                        ));
                    };
                    let expected_id = self.lower_expr(expected, body)?;
                    vec![slot, index_id, expected_id, value_id]
                } else {
                    vec![slot, index_id, value_id]
                };
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::Atomic {
                        op: *op,
                        ordering: *ordering,
                    },
                    operands,
                    result: Some(result),
                });
                Ok(result)
            }
            Expr::SubgroupBallot { cond } => {
                let cond_id = self.lower_expr(cond, body)?;
                self.unary(body, KernelOpKind::SubgroupBallot, cond_id)
            }
            Expr::SubgroupShuffle { value, lane } => {
                let value_id = self.lower_expr(value, body)?;
                let lane_id = self.lower_expr(lane, body)?;
                self.binary(body, KernelOpKind::SubgroupShuffle, value_id, lane_id)
            }
            Expr::SubgroupReduce { op, value } => {
                let value_id = self.lower_expr(value, body)?;
                self.unary(body, KernelOpKind::SubgroupReduce { op: *op }, value_id)
            }
            Expr::SubgroupLocalId => self.simple_result(body, KernelOpKind::SubgroupLocalId),
            Expr::SubgroupSize => self.simple_result(body, KernelOpKind::SubgroupSize),
            Expr::Opaque(extension) => {
                let result = self.alloc_value()?;
                body.ops.push(KernelOp {
                    kind: KernelOpKind::OpaqueExpr(Box::new(OpaqueExprData {
                        extension_id: opaque_extension_id(&**extension),
                        extension_kind: extension.extension_kind().to_owned(),
                        payload: extension.wire_payload(),
                    })),
                    operands: Vec::new(),
                    result: Some(result),
                });
                Ok(result)
            }
            // A buffer reference is consumed by inlining, which rebinds the
            // callee's parameter onto this buffer. Reaching lowering means
            // the call around it was never inlined, so say that rather than
            // ask for a descriptor mapping that must never exist: naming a
            // buffer is not an operation a kernel can perform.
            Expr::BufferRef { buffer } => Err(LowerError::UnsupportedConstruct(format!(
                "reference to buffer `{buffer}` reached lowering. It is only legal as an argument to a composite op, where composition expansion consumes it. Fix: route this Program through `vyre_lower::lower_verified` and register the callee's composition body."
            ))),
            other => Err(LowerError::UnsupportedConstruct(format!(
                "expression `{other:?}` has no KernelDescriptor lowering. Fix: add a descriptor op mapping."
            ))),
        }
    }
}

// Inline: covers items in the crate-private `descriptor` module, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::super::lower;
    use crate::descriptor::{KernelBody, KernelOpKind};
    use vyre_foundation::ir::{DataType, Program};

    #[test]
    fn lower_opaque_expr_preserves_kind_and_payload() {
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u64(42))],
        );

        let desc = lower(&program).expect("Fix: opaque literals must descriptor-lower");
        fn find_opaque_expr(body: &KernelBody) -> Option<(&String, &Vec<u8>)> {
            body.ops
                .iter()
                .find_map(|op| match &op.kind {
                    KernelOpKind::OpaqueExpr(data) => Some((&data.extension_kind, &data.payload)),
                    _ => None,
                })
                .or_else(|| body.child_bodies.iter().find_map(find_opaque_expr))
        }

        let opaque =
            find_opaque_expr(&desc.body).expect("Fix: opaque expression op must be present");
        assert_eq!(opaque.0, "vyre.literal.u64");
        assert_eq!(opaque.1, &42u64.to_le_bytes().to_vec());
    }
}
