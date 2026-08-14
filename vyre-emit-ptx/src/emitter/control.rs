use std::fmt::Write as _;

use vyre_lower::{KernelBody, KernelOp, KernelOpKind};

use super::schedule::is_schedulable_pure_op;
use super::BodyCtx;
use crate::reg::PtxType;
use crate::EmitError;

const MAX_PREDICATED_BODY_OPS: usize = 4;

impl BodyCtx<'_> {
    pub(super) fn emit_region(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
        generator: &str,
    ) -> Result<(), EmitError> {
        let _ = writeln!(self.text, "    // region: {generator}");
        if let Some(child_id) = op.operands.first() {
            if let Some(child) = body.child_bodies.get(*child_id as usize) {
                self.emit_body(child)?;
            }
        }
        Ok(())
    }

    pub(super) fn emit_structured_block(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
    ) -> Result<(), EmitError> {
        if let Some(child_id) = op.operands.first() {
            if let Some(child) = body.child_bodies.get(*child_id as usize) {
                self.emit_body(child)?;
            }
        }
        Ok(())
    }

    /// True when `id` was proven grid-uniform.
    ///
    /// An unknown id answers `false`. The set only ever records positive
    /// proofs, so "absent" means "not proven" and every caller stays
    /// conservative.
    pub(super) fn is_uniform(&self, id: u32) -> bool {
        self.uniform_results.contains(&id)
    }

    fn operands_uniform(&self, operands: &[u32]) -> bool {
        operands.iter().all(|id| self.is_uniform(*id))
    }

    /// Record whether this op's result is identical in every invocation of the
    /// grid.
    ///
    /// Called for every op as it is emitted. The descriptor is SSA-ordered, so
    /// each operand is already classified when its consumer arrives.
    ///
    /// Only [`Self::emit_return`] consumes this, and it needs a proof rather
    /// than a guess, so every unlisted op kind is treated as varying. Being
    /// wrong in the permissive direction here produces a divergent branch and a
    /// hung kernel; being wrong in the conservative direction produces a
    /// compile-time refusal, so the default is deliberately `false`.
    pub(super) fn record_uniformity(&mut self, op: &KernelOp) {
        let Some(result) = op.result else {
            return;
        };
        let uniform = match &op.kind {
            // Compile-time constants and per-launch invariants. `BufferLength`
            // and `SubgroupSize` are fixed for the whole dispatch.
            KernelOpKind::Literal | KernelOpKind::BufferLength | KernelOpKind::SubgroupSize => true,
            // Pure value computation: uniform exactly when every input is.
            // `operands` for these kinds are all result ids.
            KernelOpKind::Copy
            | KernelOpKind::Cast { .. }
            | KernelOpKind::UnOpKind(_)
            | KernelOpKind::BinOpKind(_)
            | KernelOpKind::Fma
            | KernelOpKind::Select => self.operands_uniform(&op.operands),
            // Global and constant memory is one address space for the entire
            // grid, so every invocation reading a grid-uniform address observes
            // the same value. Operand 0 is an inline binding slot, not a result
            // id, so only the index (operand 1) is classified.
            //
            // This inherits the IR's requirement that a value steering control
            // flow is not concurrently written without synchronization, which is
            // the same requirement that makes the source-level conditional
            // meaningful at all; a grid barrier before the read is what
            // establishes it in practice.
            //
            // `LoadShared` is deliberately absent: shared memory is per-CTA, so
            // equal addresses in different CTAs are different storage and the
            // value is not grid-uniform.
            KernelOpKind::LoadGlobal | KernelOpKind::LoadConstant => op
                .operands
                .get(1)
                .is_some_and(|index| self.is_uniform(*index)),
            // Everything else is unproven, which includes every invocation id
            // (`LocalInvocationId`, `GlobalInvocationId`, `SubgroupLocalId`),
            // `WorkgroupId` (uniform within a CTA but NOT across the grid, and a
            // whole CTA leaving early strands the others at a grid barrier),
            // every subgroup op, `LoadShared`, and `Atomic` (which returns the
            // pre-op value and so differs between the invocations that race).
            _ => false,
        };
        if uniform {
            self.uniform_results.insert(result);
        }
    }

    /// Lower `Node::Return` to a branch to the kernel's single exit label.
    ///
    /// `finish_with_return` emits `$L_exit:` followed by `ret;` once at the end
    /// of the kernel, and `Trap` already branches there, so the target and the
    /// precedent both exist.
    ///
    /// REFUSES when control flow reached this point through a condition that was
    /// not proven grid-uniform. Emitting the branch then would let some
    /// invocations leave while others continue, and any later `bar.sync` or
    /// cooperative grid barrier would wait forever on invocations that already
    /// returned. A compile-time refusal is loud and findable; a hang is neither.
    ///
    /// It must also never silently skip the op. This arm previously emitted
    /// nothing at all, which made every nested `Return` in the tree a no-op:
    /// programs kept running past their exit, which cost work without changing
    /// answers and so stayed invisible to every correctness test.
    pub(super) fn emit_return(&mut self) -> Result<(), EmitError> {
        if self.nonuniform_cond_depth > 0 {
            return Err(EmitError::InvalidDescriptor(
                "Node::Return under a condition that is not provably uniform across the grid \
                 cannot be lowered to PTX. A Return becomes `bra $L_exit`, so if only some \
                 invocations take it, the ones that left can never reach a later `bar.sync` or \
                 cooperative grid barrier and the ones that stayed block on them forever. This \
                 emitter proves uniformity only for values built from literals, buffer lengths, \
                 the subgroup size, and loads from global or constant memory at a uniform index; \
                 anything derived from an invocation id, a workgroup id, a subgroup op, shared \
                 memory, or an atomic's returned value is treated as varying. Fix: gate the exit \
                 on a grid-uniform value (the established shape is a flag word written with \
                 `atomic_or` and read back after a barrier, as \
                 vyre_primitives::fixpoint::persistent_fixpoint::persistent_fixpoint_grid does), \
                 or express the per-invocation case as a guarded body instead of an early return."
                    .to_string(),
            ));
        }
        let _ = writeln!(self.text, "    bra $L_exit;");
        Ok(())
    }

    pub(super) fn emit_structured_if_then(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
    ) -> Result<(), EmitError> {
        let cond_id = *op
            .operands
            .first()
            .ok_or_else(|| EmitError::InvalidDescriptor("StructuredIfThen missing cond".into()))?;
        let body_id = *op.operands.get(1).ok_or_else(|| {
            EmitError::InvalidDescriptor("StructuredIfThen missing body index".into())
        })?;
        let cond_reg = self.lookup_operand(cond_id)?;
        if let Some(child) = body.child_bodies.get(body_id as usize) {
            if child.ops.len() <= MAX_PREDICATED_BODY_OPS {
                let pred = self.pred_from_boolish(cond_reg);
                if self.emit_predicated_store_body(child, pred, false)? {
                    return Ok(());
                }
            }
        }
        let branch_pred = self.pred_from_boolish(cond_reg);
        let end_label = self.alloc_label("if_end");
        let _ = writeln!(self.text, "    @!{branch_pred} bra {end_label};");
        if let Some(child) = body.child_bodies.get(body_id as usize) {
            let divergent = !self.is_uniform(cond_id);
            if divergent {
                self.nonuniform_cond_depth = self.nonuniform_cond_depth.saturating_add(1);
            }
            let emitted = self.emit_body(child);
            if divergent {
                self.nonuniform_cond_depth = self.nonuniform_cond_depth.saturating_sub(1);
            }
            emitted?;
        }
        let _ = writeln!(self.text, "{end_label}:");
        Ok(())
    }

    pub(super) fn emit_structured_if_then_else(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
    ) -> Result<(), EmitError> {
        let cond_id = *op.operands.first().ok_or_else(|| {
            EmitError::InvalidDescriptor("StructuredIfThenElse missing cond".into())
        })?;
        let then_id = *op.operands.get(1).ok_or_else(|| {
            EmitError::InvalidDescriptor("StructuredIfThenElse missing then index".into())
        })?;
        let else_id = *op.operands.get(2).ok_or_else(|| {
            EmitError::InvalidDescriptor("StructuredIfThenElse missing else index".into())
        })?;
        let cond_reg = self.lookup_operand(cond_id)?;
        if let (Some(then_body), Some(else_body)) = (
            body.child_bodies.get(then_id as usize),
            body.child_bodies.get(else_id as usize),
        ) {
            if then_body.ops.len() <= MAX_PREDICATED_BODY_OPS
                && else_body.ops.len() <= MAX_PREDICATED_BODY_OPS
                && predicated_store_body_supported(then_body)
                && predicated_store_body_supported(else_body)
            {
                let pred = self.pred_from_boolish(cond_reg);
                let then_emitted = self.emit_predicated_store_body(then_body, pred, false)?;
                let else_emitted = self.emit_predicated_store_body(else_body, pred, true)?;
                if then_emitted && else_emitted {
                    return Ok(());
                }
            }
        }
        let branch_pred = self.pred_from_boolish(cond_reg);
        let else_label = self.alloc_label("if_else");
        let end_label = self.alloc_label("if_end");
        let _ = writeln!(self.text, "    @!{branch_pred} bra {else_label};");
        let divergent = !self.is_uniform(cond_id);
        if divergent {
            self.nonuniform_cond_depth = self.nonuniform_cond_depth.saturating_add(1);
        }
        let arms = (|| -> Result<(), EmitError> {
            if let Some(child) = body.child_bodies.get(then_id as usize) {
                self.emit_body(child)?;
            }
            let _ = writeln!(self.text, "    bra {end_label};");
            let _ = writeln!(self.text, "{else_label}:");
            if let Some(child) = body.child_bodies.get(else_id as usize) {
                self.emit_body(child)?;
            }
            Ok(())
        })();
        if divergent {
            self.nonuniform_cond_depth = self.nonuniform_cond_depth.saturating_sub(1);
        }
        arms?;
        let _ = writeln!(self.text, "{end_label}:");
        Ok(())
    }

    fn emit_predicated_store_body(
        &mut self,
        child: &KernelBody,
        pred: crate::reg::Reg,
        negate: bool,
    ) -> Result<bool, EmitError> {
        if !predicated_store_body_supported(child) {
            return Ok(false);
        }
        let mut emitted_store = false;
        for op in &child.ops {
            if matches!(
                op.kind,
                KernelOpKind::StoreGlobal | KernelOpKind::StoreShared
            ) {
                emitted_store |= self.emit_predicated_store(op, pred, negate)?;
            } else {
                self.emit_op(child, op)?;
            }
        }
        Ok(emitted_store)
    }

    pub(super) fn emit_structured_for_loop(
        &mut self,
        body: &KernelBody,
        op: &KernelOp,
        loop_var: &str,
    ) -> Result<(), EmitError> {
        let lo_id = *op
            .operands
            .first()
            .ok_or_else(|| EmitError::InvalidDescriptor("StructuredForLoop missing lo".into()))?;
        let hi_id = *op
            .operands
            .get(1)
            .ok_or_else(|| EmitError::InvalidDescriptor("StructuredForLoop missing hi".into()))?;
        let body_id = *op.operands.get(2).ok_or_else(|| {
            EmitError::InvalidDescriptor("StructuredForLoop missing body index".into())
        })?;
        let lo_reg = self.lookup_operand(lo_id)?;
        let hi_reg = self.lookup_operand(hi_id)?;
        let var_reg = self.alloc(PtxType::U32);
        let cond_reg = self.alloc(PtxType::Bool);
        let one_reg = self.alloc(PtxType::U32);
        let head = self.alloc_label("for_head");
        let exit = self.alloc_label("for_exit");
        let _ = writeln!(self.text, "    // for {loop_var} in [{lo_reg}, {hi_reg})");
        let _ = writeln!(self.text, "    mov.u32    {var_reg}, {lo_reg};");
        let _ = writeln!(self.text, "    mov.u32    {one_reg}, 1;");
        let _ = writeln!(self.text, "{head}:");
        let _ = writeln!(
            self.text,
            "    setp.ge.u32 {cond_reg}, {var_reg}, {hi_reg};"
        );
        let _ = writeln!(self.text, "    @{cond_reg} bra {exit};");
        self.loop_indices.insert(loop_var.into(), var_reg);
        if let Some(child) = body.child_bodies.get(body_id as usize) {
            // A trip count built from non-uniform bounds diverges: invocations
            // leave the loop on different iterations, so a `Return` in the body
            // is reached by only some of them even with no conditional in sight.
            let divergent = !(self.is_uniform(lo_id) && self.is_uniform(hi_id));
            if divergent {
                self.nonuniform_cond_depth = self.nonuniform_cond_depth.saturating_add(1);
            }
            self.grid_sync_loop_depth = self.grid_sync_loop_depth.saturating_add(1);
            let emitted = self.emit_body(child);
            self.grid_sync_loop_depth = self.grid_sync_loop_depth.saturating_sub(1);
            if divergent {
                self.nonuniform_cond_depth = self.nonuniform_cond_depth.saturating_sub(1);
            }
            emitted?;
        }
        self.loop_indices.remove(loop_var);
        let _ = writeln!(self.text, "    add.u32    {var_reg}, {var_reg}, {one_reg};");
        let _ = writeln!(self.text, "    bra {head};");
        let _ = writeln!(self.text, "{exit}:");
        Ok(())
    }

    pub(super) fn emit_loop_index(
        &mut self,
        op: &KernelOp,
        loop_var: &str,
    ) -> Result<(), EmitError> {
        let reg = *self.loop_indices.get(loop_var).ok_or_else(|| {
            EmitError::InvalidDescriptor(format!(
                "LoopIndex `{loop_var}` appeared outside its StructuredForLoop"
            ))
        })?;
        self.bind_result(op, reg)
    }

    /// Close the kernel: drain any cp.async group the descriptor never
    /// waited on, then land the single exit label and `ret`.
    pub(super) fn finish_with_return(&mut self) {
        if !self.pending_cp_async_tags.is_empty() {
            let _ = writeln!(
                self.text,
                "    // implicit cp.async drain for descriptors missing AsyncWait"
            );
            let _ = writeln!(self.text, "    cp.async.wait_group 0;");
            let _ = writeln!(self.text, "    membar.cta;");
            self.pending_cp_async_tags.clear();
        }
        self.text.push_str("$L_exit:\n");
        self.text.push_str("    ret;\n");
    }
}

fn predicated_store_body_supported(body: &KernelBody) -> bool {
    let mut has_store = false;
    for op in &body.ops {
        if matches!(
            op.kind,
            KernelOpKind::StoreGlobal | KernelOpKind::StoreShared
        ) {
            has_store = true;
            continue;
        }
        if !is_schedulable_pure_op(op) {
            return false;
        }
    }
    has_store
}
