use super::super::CalleeExpander;
use crate::error::IrResult as Result;
use crate::ir::{AtomicOp, BinOp, Expr, Ident, Node, UnOp};

impl CalleeExpander<'_> {
    #[inline]
    pub(crate) fn expr(&mut self, expr: &Expr) -> Result<(Vec<Node>, Expr)> {
        match expr {
            Expr::Var(name) => Ok((Vec::new(), Expr::var(self.rename_use(name)))),
            Expr::Load { buffer, index } => self.load(buffer, index),
            Expr::BufferRef { buffer } => Ok((
                Vec::new(),
                Expr::BufferRef {
                    buffer: self.rebound_buffer(buffer).unwrap_or_else(|| buffer.clone()),
                },
            )),
            Expr::BufLen { buffer } if self.output_name == *buffer => {
                Ok((Vec::new(), Expr::u32(1)))
            }
            // A buffer-reference argument keeps a real buffer behind the
            // callee's parameter, so its length is the caller buffer's
            // length. Only a scalar argument collapses the parameter to a
            // single value, and only then is the length 1.
            Expr::BufLen { buffer } if self.input_args.contains_key(buffer) => {
                Ok((
                    Vec::new(),
                    match self.rebound_buffer(buffer) {
                        Some(caller) => Expr::BufLen { buffer: caller },
                        None => Expr::u32(1),
                    },
                ))
            }
            // A nested call's arguments are callee expressions: they can name a
            // callee local, a callee parameter bound to a scalar, or a
            // parameter bound to one of the caller's buffers. Expanding them
            // under the caller's policy renamed the locals and did nothing
            // else, so a parameter name survived into a program whose buffer
            // table never declared it. Only the call itself belongs to the
            // caller, which resolves and expands it.
            Expr::Call { op_id, args } => {
                let mut prefix = Vec::new();
                let mut expanded = Vec::with_capacity(args.len());
                for arg in args {
                    let (statements, value) = self.expr(arg)?;
                    prefix.extend(statements);
                    expanded.push(value);
                }
                let (statements, value) = self.ctx.expand_call(op_id, &expanded)?;
                prefix.extend(statements);
                Ok((prefix, value))
            }
            Expr::InvocationId { .. }
            | Expr::WorkgroupId { .. }
            | Expr::LocalId { .. }
            | Expr::SubgroupLocalId
            | Expr::SubgroupSize => {
                Err(crate::error::IrError::lowering(
                    "inliner cannot inline a callee that references \
                     InvocationId / WorkgroupId / LocalId / SubgroupLocalId / SubgroupSize: \
                     these built-ins are per-invocation and cannot be passed as callee arguments. \
                     Fix: hoist the built-in read to the call site and pass it as an explicit \
                     argument before inlining.".to_string(),
                ))
            }
            Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF32(_)
            | Expr::LitBool(_)
            | Expr::BufLen { .. }
            | Expr::LogicalIndex { .. }
            | Expr::LogicalTileId { .. }
            | Expr::LogicalWithinTileId { .. } => Ok((Vec::new(), expr.clone())),
            Expr::BinOp { op, left, right } => self.binop(*op, left, right),
            Expr::UnOp { op, operand } => self.unop(op.clone(), operand),
            Expr::Fma { a, b, c } => self.fma(a, b, c),
            Expr::Select {
                cond,
                true_val,
                false_val,
            } => self.select(cond, true_val, false_val),
            Expr::Cast { target, value } => {
                let (prefix, value) = self.expr(value)?;
                Ok((
                    prefix,
                    Expr::Cast {
                        target: target.clone(),
                        value: Box::new(value),
                    },
                ))
            }
            Expr::Atomic {
                op,
                buffer,
                index,
                expected,
                value,
                ordering,
            } => self.atomic(*op, buffer, index, expected.as_deref(), value, *ordering),
            &Expr::SubgroupBallot { .. } | &Expr::SubgroupShuffle { .. } | &Expr::SubgroupReduce { .. } => {
                Err(crate::error::IrError::lowering(
                    "inliner cannot expand subgroup intrinsics; RFC 0004 gates this on target builder 25+. Fix: avoid inlining across subgroup-op boundaries.".to_string(),
                ))
            }
        Expr::Opaque(extension) => Err(crate::error::IrError::lowering(format!(
                "inliner cannot expand opaque expression extension `{}`/`{}`. Fix: lower the extension to core Expr variants before inlining.",
                extension.extension_kind(),
                extension.debug_identity()
            ))),
        }
    }

    /// Return the caller buffer a callee parameter was bound to, if the call
    /// site passed a buffer reference rather than a scalar.
    ///
    /// A scalar argument replaces every read of the parameter with the value
    /// itself. A [`Expr::BufferRef`] argument instead retargets the access at
    /// the caller's buffer and keeps the index, which is what lets a callee
    /// index a table it does not own.
    #[inline]
    pub(crate) fn rebound_buffer(&self, param: &Ident) -> Option<Ident> {
        match self.input_args.get(param) {
            Some(Expr::BufferRef { buffer }) => Some(buffer.clone()),
            _ => None,
        }
    }

    #[inline]
    pub(crate) fn load(&mut self, buffer: &Ident, index: &Expr) -> Result<(Vec<Node>, Expr)> {
        if let Some(caller) = self.rebound_buffer(buffer) {
            let (prefix, index) = self.expr(index)?;
            return Ok((
                prefix,
                Expr::Load {
                    buffer: caller,
                    index: Box::new(index),
                },
            ));
        }
        if let Some(arg) = self.input_args.get(buffer) {
            return Ok((Vec::new(), arg.clone()));
        }
        let (prefix, index) = self.expr(index)?;
        Ok((
            prefix,
            Expr::Load {
                buffer: buffer.into(),
                index: Box::new(index),
            },
        ))
    }

    #[inline]
    pub(crate) fn binop(
        &mut self,
        op: BinOp,
        left: &Expr,
        right: &Expr,
    ) -> Result<(Vec<Node>, Expr)> {
        let (mut prefix, left) = self.expr(left)?;
        let (right_prefix, right) = self.expr(right)?;
        prefix.extend(right_prefix);
        Ok((
            prefix,
            Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
        ))
    }

    #[inline]
    pub(crate) fn unop(&mut self, op: UnOp, operand: &Expr) -> Result<(Vec<Node>, Expr)> {
        let (prefix, operand) = self.expr(operand)?;
        Ok((
            prefix,
            Expr::UnOp {
                op,
                operand: Box::new(operand),
            },
        ))
    }

    #[inline]
    fn ternary_operands(
        &mut self,
        first: &Expr,
        second: &Expr,
        third: &Expr,
    ) -> Result<(Vec<Node>, [Expr; 3])> {
        let (mut prefix, first) = self.expr(first)?;
        let (second_prefix, second) = self.expr(second)?;
        let (third_prefix, third) = self.expr(third)?;
        prefix.extend(second_prefix);
        prefix.extend(third_prefix);
        Ok((prefix, [first, second, third]))
    }

    #[inline]
    pub(crate) fn fma(&mut self, a: &Expr, b: &Expr, c: &Expr) -> Result<(Vec<Node>, Expr)> {
        let (prefix, [a, b, c]) = self.ternary_operands(a, b, c)?;
        Ok((
            prefix,
            Expr::Fma {
                a: Box::new(a),
                b: Box::new(b),
                c: Box::new(c),
            },
        ))
    }

    #[inline]
    pub(crate) fn select(
        &mut self,
        cond: &Expr,
        true_val: &Expr,
        false_val: &Expr,
    ) -> Result<(Vec<Node>, Expr)> {
        let (prefix, [cond, true_val, false_val]) =
            self.ternary_operands(cond, true_val, false_val)?;
        Ok((
            prefix,
            Expr::Select {
                cond: Box::new(cond),
                true_val: Box::new(true_val),
                false_val: Box::new(false_val),
            },
        ))
    }

    #[inline]
    pub(crate) fn atomic(
        &mut self,
        op: AtomicOp,
        buffer: &Ident,
        index: &Expr,
        expected: Option<&Expr>,
        value: &Expr,
        ordering: crate::memory_model::MemoryOrdering,
    ) -> Result<(Vec<Node>, Expr)> {
        let (mut prefix, index) = self.expr(index)?;
        let (expected_prefix, expected) = match expected {
            Some(expected) => {
                let (prefix, expected) = self.expr(expected)?;
                (prefix, Some(Box::new(expected)))
            }
            None => (Vec::new(), None),
        };
        let (value_prefix, value) = self.expr(value)?;
        prefix.extend(expected_prefix);
        prefix.extend(value_prefix);
        Ok((
            prefix,
            Expr::Atomic {
                op,
                buffer: self.rebound_buffer(buffer).unwrap_or_else(|| buffer.into()),
                index: Box::new(index),
                expected,
                value: Box::new(value),
                ordering,
            },
        ))
    }
}
