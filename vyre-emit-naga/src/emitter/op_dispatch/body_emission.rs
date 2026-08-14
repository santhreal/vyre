//! Body and structured-block emission: the op loop, blocks, and if/else.

use naga::{Expression, Span, Statement};
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
    pub(in crate::emitter) fn emit_structured_block(
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
    pub(in crate::emitter) fn emit_structured_if(
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
}
