//! Body and structured-block emission: the op loop, blocks, and if/else.

use naga::{Span, Statement};
use vyre_lower::{KernelBody, KernelOp, KernelOpKind};

use super::super::BodyBuilder;
use crate::EmitError;

impl BodyBuilder<'_> {
    pub(in crate::emitter) fn emit_body(&mut self, body: &KernelBody) -> Result<(), EmitError> {
        for op in &body.ops {
            self.emit_op(body, op)?;
            // A `Trap` lowers to an unconditional `Return` for every lane (see
            // emit_trap), and a `Return` terminates the block. naga rejects any
            // statement after a `Return` in the same block
            // (`InstructionsAfterReturn`), so stop emitting this block's
            // remaining ops (they are unreachable by the trap/return semantics).
            if matches!(op.kind, KernelOpKind::Trap { .. } | KernelOpKind::Return) {
                break;
            }
        }
        Ok(())
    }

    /// Emit `Statement::Block` for `StructuredBlock` / `Region`.
    ///
    /// The child body is operand 0 and runs inside the carrier scope, so any
    /// SSA id it produces that the parent references afterwards round-trips
    /// through a function-local. The lowering's Region phi-merge handles
    /// source-level NAMED carriers; the carrier scope handles the UNNAMED
    /// in-region results that escape.
    pub(in crate::emitter) fn emit_structured_block(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
    ) -> Result<(), EmitError> {
        self.with_carrier_scope(body, op, &[0], |builder| {
            let block = builder.child_block(body, op, 0)?;
            builder
                .function
                .body
                .push(Statement::Block(block), Span::UNDEFINED);
            Ok(())
        })
    }

    /// Emit `Statement::If { accept, reject }` for `StructuredIfThen`
    /// (`child_indices=&[1]`) and `StructuredIfThenElse`
    /// (`child_indices=&[1, 2]`), inside the carrier scope those child bodies
    /// need: a value bound in an if-arm and read after the if otherwise
    /// surfaces as `no definition in scope for identifier _eN` from naga's
    /// WGSL writer, because the `let _eN = ...;` binding lives in the arm's
    /// scope and the reader does not.
    pub(in crate::emitter) fn emit_structured_if(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
        child_indices: &[usize],
    ) -> Result<(), EmitError> {
        self.with_carrier_scope(body, op, child_indices, |builder| {
            let condition = builder.value_operand(op, 0)?;
            let condition = builder.ensure_bool_condition(condition);
            let accept = builder.child_block(body, op, child_indices[0])?;
            let reject = if child_indices.len() > 1 {
                builder.child_block(body, op, child_indices[1])?
            } else {
                naga::Block::new()
            };
            builder.function.body.push(
                Statement::If {
                    condition,
                    accept,
                    reject,
                },
                Span::UNDEFINED,
            );
            Ok(())
        })
    }
}
