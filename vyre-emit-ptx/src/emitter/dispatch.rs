//! Op-kind dispatch.
//!
//! Owns the single `match` from `KernelOpKind` to the lowering that emits it,
//! plus the short lowerings that are one instruction and need no module of
//! their own. Anything longer than a handful of instructions is delegated to
//! the sibling module named for that concept.

use std::fmt::Write as _;

use vyre_foundation::ir::{DataType, MemoryOrdering};
use vyre_lower::{KernelBody, KernelOp, KernelOpKind, LiteralValue};

use super::memory::AsyncCopyDirection;
use super::operand_decode::{read_store_operands, read_two_operands};
use super::BodyCtx;
use crate::reg::{PtxType, Reg};
use crate::EmitError;

impl BodyCtx<'_> {
    pub(super) fn emit_op(&mut self, body: &KernelBody, op: &KernelOp) -> Result<(), EmitError> {
        use KernelOpKind::*;
        // Classify before emitting. Operands are already classified because the
        // descriptor is SSA-ordered, and `emit_return` needs the enclosing
        // conditions' verdicts to already be recorded.
        self.record_uniformity(op);
        match &op.kind {
            Literal => {
                let pool_idx = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("Literal op missing pool index".into())
                })? as usize;
                let lit = body.literals.get(pool_idx).ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!("Literal pool index {pool_idx} OOR"))
                })?;
                let (reg, lit_text) = self.alloc_literal(lit);
                let _ = writeln!(
                    self.text,
                    "    mov.{}    {reg}, {lit_text};",
                    reg.0.ptx_type_str()
                );
                self.bind_result(op, reg)?;
                if let (Some(result), LiteralValue::U32(value)) = (op.result, lit) {
                    self.u32_literals.insert(result, *value);
                }
            }
            Copy => {
                let operand_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("Copy missing operand".into()))?;
                let source = self.lookup_operand(operand_id)?;
                let snapshot = self.alloc(source.0);
                let _ = writeln!(
                    self.text,
                    "    mov.{}    {snapshot}, {source};",
                    source.0.ptx_type_str()
                );
                self.bind_result(op, snapshot)?;
            }
            LocalInvocationId | GlobalInvocationId => {
                let axis_idx = *op.operands.first().unwrap_or(&0);
                let reg = if matches!(op.kind, LocalInvocationId) {
                    match axis_idx {
                        0 => Reg(PtxType::U32, 2),
                        1 => Reg(PtxType::U32, 6),
                        _ => Reg(PtxType::U32, 24),
                    }
                } else {
                    match axis_idx {
                        0 => Reg(PtxType::U32, 3),
                        1 => Reg(PtxType::U32, 7),
                        _ => Reg(PtxType::U32, 25),
                    }
                };
                self.bind_result(op, reg)?;
            }
            WorkgroupId => {
                let axis_idx = *op.operands.first().unwrap_or(&0);
                let reg = match axis_idx {
                    0 => Reg(PtxType::U32, 0),
                    1 => Reg(PtxType::U32, 4),
                    _ => Reg(PtxType::U32, 8),
                };
                self.bind_result(op, reg)?;
            }
            LoadGlobal | LoadShared | LoadConstant => {
                let (binding_slot, index_op_id) = read_two_operands(op, "Load")?;
                let binding = self.binding_for_slot(binding_slot)?;
                let element_type = binding.element_type.clone();
                let memory_class = binding.memory_class;
                let elem_ty = PtxType::from_dtype(&element_type)?;
                let address = self.emit_memory_address(
                    binding_slot,
                    index_op_id,
                    &element_type,
                    memory_class,
                )?;
                let load_space = self.load_space_for(binding_slot, memory_class);
                let val_reg = self.emit_load_value(address, load_space, &element_type, elem_ty)?;
                self.bind_result(op, val_reg)?;
            }
            BufferLength => {
                let binding_slot = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("BufferLength missing slot".into())
                })?;
                // Checked arithmetic: slot byte offset in the params buffer is
                // `4 + slot * 4`. An overflowing slot would wrap silently and
                // produce a wrong PTX address; reject it with a structured error.
                let byte_offset = binding_slot
                    .checked_mul(4)
                    .and_then(|v| v.checked_add(4))
                    .ok_or_else(|| EmitError::InvalidBinding {
                        slot: binding_slot,
                        reason: "BufferLength slot byte offset (4 + slot * 4) overflows u32".into(),
                    })?;
                let reg = self.alloc(PtxType::U32);
                let _ = writeln!(
                    self.text,
                    "    ld.global.u32    {reg}, [%rd0 + {byte_offset}];"
                );
                self.bind_result(op, reg)?;
            }
            StoreGlobal | StoreShared => {
                let (binding_slot, index_op_id, value_op_id) = read_store_operands(op)?;
                let binding = self.binding_for_slot(binding_slot)?;
                let element_type = binding.element_type.clone();
                let memory_class = binding.memory_class;
                let elem_ty = PtxType::from_dtype(&element_type)?;
                let value_reg = self.coerce_for_store(self.lookup_operand(value_op_id)?, elem_ty);
                let address = self.emit_memory_address(
                    binding_slot,
                    index_op_id,
                    &element_type,
                    memory_class,
                )?;
                let guard =
                    self.store_guard_for_index(binding_slot, index_op_id, memory_class, None)?;
                self.emit_store_value(guard, address, &element_type, value_reg)?;
            }
            VectorLoadGlobal { width } => {
                let (binding_slot, index_op_id) = read_two_operands(op, "VectorLoadGlobal")?;
                let binding = self.binding_for_slot(binding_slot)?;
                let element_type = binding.element_type.clone();
                let memory_class = binding.memory_class;
                let elem_ty = PtxType::from_dtype(&element_type)?;
                let vector_ty = match (element_type.min_bytes(), elem_ty) {
                    (1, _) => PtxType::U32,
                    (2, _) => PtxType::B16,
                    (_, ty) => ty,
                };
                let final_addr =
                    self.emit_global_address_operand(binding_slot, index_op_id, &element_type)?;
                let load_space = self.load_space_for(binding_slot, memory_class);
                let mut regs = smallvec::SmallVec::<[crate::reg::Reg; 4]>::new();
                for _ in 0..*width {
                    regs.push(self.alloc(vector_ty));
                }
                let (mnemonic_prefix, cache_suffix) = if load_space == ".global"
                    && self.read_only_cache_slots.contains(&binding_slot)
                {
                    ("ld.global", ".nc")
                } else {
                    ("ld.global", "")
                };
                let _ = write!(
                    self.text,
                    "    {mnemonic_prefix}{cache_suffix}.v{}.{}    ",
                    width,
                    vector_ty.ptx_type_str()
                );
                crate::reg::write_reg_tuple(&mut self.text, &regs);
                self.text.push_str(", ");
                self.write_mem_operand(final_addr)?;
                self.text.push_str(";\n");
                let mut canonical_regs = smallvec::SmallVec::<[crate::reg::Reg; 4]>::new();
                for &reg in &regs {
                    let canonical = if matches!(element_type, DataType::Bool) {
                        let pred = self.alloc(PtxType::Bool);
                        let _ = writeln!(self.text, "    setp.ne.u32    {pred}, {reg}, 0;");
                        pred
                    } else {
                        self.canonicalize_f32(reg)
                    };
                    canonical_regs.push(canonical);
                }
                if let Some(res) = op.result {
                    self.vector_regs.insert(res, canonical_regs);
                }
            }
            VectorStoreGlobal { width } => {
                let binding_slot = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("VectorStoreGlobal missing slot".into())
                })?;
                let index_op_id = *op.operands.get(1).ok_or_else(|| {
                    EmitError::InvalidDescriptor("VectorStoreGlobal missing index".into())
                })?;
                let binding = self.binding_for_slot(binding_slot)?;
                let element_type = binding.element_type.clone();
                let elem_ty = PtxType::from_dtype(&element_type)?;
                let vector_ty = match (element_type.min_bytes(), elem_ty) {
                    (1, _) => PtxType::U32,
                    (2, _) => PtxType::B16,
                    (_, ty) => ty,
                };
                let final_addr =
                    self.emit_global_address_operand(binding_slot, index_op_id, &element_type)?;
                let mut regs = smallvec::SmallVec::<[crate::reg::Reg; 4]>::new();
                for &val_id in &op.operands[2..] {
                    let value = self.lookup_operand(val_id)?;
                    regs.push(if matches!(element_type, DataType::Bool) {
                        let pred = self.pred_from_boolish(value);
                        let word = self.alloc(PtxType::U32);
                        let _ = writeln!(self.text, "    selp.u32    {word}, 1, 0, {pred};");
                        word
                    } else if elem_ty == PtxType::F32 {
                        self.canonicalize_f32(value)
                    } else {
                        value
                    });
                }
                let _ = write!(
                    self.text,
                    "    st.global.v{}.{}    ",
                    width,
                    vector_ty.ptx_type_str()
                );
                self.write_mem_operand(final_addr)?;
                self.text.push_str(", ");
                crate::reg::write_reg_tuple(&mut self.text, &regs);
                self.text.push_str(";\n");
            }
            ExtractLane { lane } => {
                let vec_id = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("ExtractLane missing vector operand".into())
                })?;
                let regs = self.vector_regs.get(&vec_id).ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!(
                        "ExtractLane: vector value {vec_id} has no recorded vector registers"
                    ))
                })?;
                let reg = *regs.get(*lane as usize).ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!(
                        "ExtractLane: lane index {lane} out of bounds for vector {vec_id}"
                    ))
                })?;
                self.bind_result(op, reg)?;
            }
            BinOpKind(bin_op) => {
                let left_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("BinOp missing left".into()))?;
                let right_id = *op
                    .operands
                    .get(1)
                    .ok_or_else(|| EmitError::InvalidDescriptor("BinOp missing right".into()))?;
                let left = self.lookup_operand(left_id)?;
                let right = self.lookup_operand(right_id)?;
                if matches!(bin_op, vyre_foundation::ir::BinOp::Mul) {
                    if let Some(constant) = self.u32_literals.get(&right_id).copied() {
                        if let Some(result) = self.emit_small_u32_const_mul(left, constant) {
                            self.bind_result(op, result)?;
                            return Ok(());
                        }
                    }
                    if let Some(constant) = self.u32_literals.get(&left_id).copied() {
                        if let Some(result) = self.emit_small_u32_const_mul(right, constant) {
                            self.bind_result(op, result)?;
                            return Ok(());
                        }
                    }
                }
                if matches!(bin_op, vyre_foundation::ir::BinOp::Div) {
                    if let Some(divisor) = self.u32_literals.get(&right_id).copied() {
                        if let Some(result) = self.emit_fast_u32_const_div(left, divisor) {
                            self.bind_result(op, result)?;
                            return Ok(());
                        }
                    }
                }
                if matches!(bin_op, vyre_foundation::ir::BinOp::Mod) {
                    if let Some(divisor) = self.u32_literals.get(&right_id).copied() {
                        if let Some(result) = self.emit_fast_u32_const_mod(left, divisor) {
                            self.bind_result(op, result)?;
                            return Ok(());
                        }
                    }
                }
                let (result, _result_ty) = self.emit_binop(*bin_op, left, right)?;
                self.bind_result(op, result)?;
            }
            UnOpKind(un_op) => {
                let operand_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("UnOp missing operand".into()))?;
                let operand = self.lookup_operand(operand_id)?;
                let result = self.emit_unop(un_op, operand)?;
                self.bind_result(op, result)?;
            }
            Return => {
                self.emit_return()?;
            }
            Barrier { ordering } => {
                if ordering.requires_grid_sync() {
                    if self.options.cooperative_grid_sync {
                        self.emit_grid_sync_barrier()?;
                    } else {
                        return Err(EmitError::InvalidDescriptor(
                            "MemoryOrdering::GridSync cannot be lowered to a CTA-scope `bar.sync 0` \
                             barrier: that would silently downgrade whole-grid synchronization to \
                             one workgroup and lose cross-block visibility. Enable \
                             PtxEmitOptions::cooperative_grid_sync (only when the consumer guarantees \
                             a cooperative launch) to emit a native cooperative grid barrier, or split \
                             the program at the GridSync barrier."
                                .to_string(),
                        ));
                    }
                } else if matches!(
                    ordering,
                    MemoryOrdering::Acquire | MemoryOrdering::Release | MemoryOrdering::AcqRel
                ) {
                    // A one-sided or two-sided acquire/release names global memory
                    // visibility, not thread convergence. `membar.gl` publishes the
                    // preceding stores at device scope without stalling the CTA.
                    let _ = writeln!(self.text, "    membar.gl;");
                } else {
                    // `SeqCst` is a full barrier within the issuing workgroup, and any
                    // ordering added to the non-exhaustive enum later gets the strongest
                    // CTA-scope construct until it is mapped deliberately.
                    let _ = writeln!(self.text, "    bar.sync 0;");
                }
            }
            Region { generator } => {
                self.emit_region(body, op, generator)?;
            }
            StructuredBlock => {
                self.emit_structured_block(body, op)?;
            }
            StructuredIfThen => {
                self.emit_structured_if_then(body, op)?;
            }
            StructuredIfThenElse => {
                self.emit_structured_if_then_else(body, op)?;
            }
            StructuredForLoop { loop_var } => {
                self.emit_structured_for_loop(body, op, loop_var)?;
            }
            LoopIndex { loop_var } => {
                self.emit_loop_index(op, loop_var)?;
            }
            LoopCarrierInit { name } => {
                let seed_id = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!(
                        "LoopCarrierInit `{name}` missing seed operand"
                    ))
                })?;
                let seed_reg = self.lookup_operand(seed_id)?;
                let carrier = self
                    .named_carriers
                    .get(name)
                    .copied()
                    .unwrap_or_else(|| self.alloc(seed_reg.0));
                self.named_carriers.insert(name.clone(), carrier);
                let _ = writeln!(
                    self.text,
                    "    mov.{}    {carrier}, {seed_reg};",
                    carrier.0.ptx_type_str()
                );
            }
            LoopCarrier { name } => {
                let carrier = *self.named_carriers.get(name).ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!(
                        "LoopCarrier `{name}` read before LoopCarrierInit allocated its register"
                    ))
                })?;
                if let Some(result_id) = op.result {
                    self.named_carrier_result_ids
                        .insert(result_id, name.clone());
                }
                let snapshot = self.alloc(carrier.0);
                let _ = writeln!(
                    self.text,
                    "    mov.{}    {snapshot}, {carrier};",
                    carrier.0.ptx_type_str()
                );
                self.bind_result(op, snapshot)?;
            }
            LoopCarrierEnd { name } => {
                let value_id = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!(
                        "LoopCarrierEnd `{name}` missing value operand"
                    ))
                })?;
                let value = self.lookup_operand(value_id)?;
                let carrier = *self.named_carriers.get(name).ok_or_else(|| {
                    EmitError::InvalidDescriptor(format!(
                        "LoopCarrierEnd `{name}` written before LoopCarrierInit allocated its register"
                    ))
                })?;
                let _ = writeln!(
                    self.text,
                    "    mov.{}    {carrier}, {value};",
                    carrier.0.ptx_type_str()
                );
            }
            Cast { target } => {
                let operand_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("Cast missing operand".into()))?;
                let src = self.lookup_operand(operand_id)?;
                let dst = self.emit_cast(src, target)?;
                self.bind_result(op, dst)?;
            }
            Select => {
                let cond_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("Select missing cond".into()))?;
                let true_id = *op.operands.get(1).ok_or_else(|| {
                    EmitError::InvalidDescriptor("Select missing true_val".into())
                })?;
                let false_id = *op.operands.get(2).ok_or_else(|| {
                    EmitError::InvalidDescriptor("Select missing false_val".into())
                })?;
                let cond = self.pred_from_boolish(self.lookup_operand(cond_id)?);
                let t = self.lookup_operand(true_id)?;
                let f = self.lookup_operand(false_id)?;
                let dst = self.alloc(t.0);
                if matches!(t.0, crate::reg::PtxType::Bool) {
                    // PTX `selp` does not accept `.pred`. Lower predicate
                    // select as: dst = (cond AND t) OR ((NOT cond) AND f).
                    let not_cond = self.alloc(crate::reg::PtxType::Bool);
                    let pick_t = self.alloc(crate::reg::PtxType::Bool);
                    let pick_f = self.alloc(crate::reg::PtxType::Bool);
                    let _ = writeln!(self.text, "    not.pred    {not_cond}, {cond};");
                    let _ = writeln!(self.text, "    and.pred    {pick_t}, {cond}, {t};");
                    let _ = writeln!(self.text, "    and.pred    {pick_f}, {not_cond}, {f};");
                    let _ = writeln!(self.text, "    or.pred    {dst}, {pick_t}, {pick_f};");
                } else {
                    let _ = writeln!(
                        self.text,
                        "    selp.{}    {dst}, {t}, {f}, {cond};",
                        t.0.ptx_type_str()
                    );
                }
                self.bind_result(op, dst)?;
            }
            AsyncLoad { tag } => {
                // Operands: [src_binding, dst_binding, offset_op_id, size_op_id]
                let src_slot = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncLoad missing src slot".into())
                })?;
                let dst_slot = *op.operands.get(1).ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncLoad missing dst slot".into())
                })?;
                let offset_id = *op.operands.get(2).ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncLoad missing offset".into())
                })?;
                let size_id = *op
                    .operands
                    .get(3)
                    .ok_or_else(|| EmitError::InvalidDescriptor("AsyncLoad missing size".into()))?;
                let _ = writeln!(
                    self.text,
                    "    // async_load tag={tag} src=slot{src_slot} dst=slot{dst_slot}"
                );
                if !self.emit_cp_async_load_loop(tag, src_slot, dst_slot, offset_id, size_id)? {
                    self.emit_async_copy_loop(
                        tag,
                        src_slot,
                        dst_slot,
                        offset_id,
                        size_id,
                        AsyncCopyDirection::Load,
                    )?;
                }
            }
            AsyncStore { tag } => {
                let src_slot = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncStore missing src slot".into())
                })?;
                let dst_slot = *op.operands.get(1).ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncStore missing dst slot".into())
                })?;
                let offset_id = *op.operands.get(2).ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncStore missing offset".into())
                })?;
                let size_id = *op.operands.get(3).ok_or_else(|| {
                    EmitError::InvalidDescriptor("AsyncStore missing size".into())
                })?;
                let _ = writeln!(
                    self.text,
                    "    // async_store tag={tag} src=slot{src_slot} dst=slot{dst_slot}"
                );
                self.emit_async_copy_loop(
                    tag,
                    src_slot,
                    dst_slot,
                    offset_id,
                    size_id,
                    AsyncCopyDirection::Store,
                )?;
            }
            AsyncWait { tag } => {
                let _ = writeln!(self.text, "    // async_wait tag={tag}");
                if !self.emit_cp_async_wait_for_tag(tag) {
                    let _ = writeln!(self.text, "    membar.cta;");
                }
            }
            SubgroupBallot => {
                let cond_id = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("SubgroupBallot missing cond".into())
                })?;
                let cond = self.lookup_operand(cond_id)?;
                let result = self.alloc(PtxType::U32);
                let pred = self.pred_from_boolish(cond);
                let mask = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    activemask.b32    {mask};");
                let _ = writeln!(
                    self.text,
                    "    vote.sync.ballot.b32    {result}, {pred}, {mask};"
                );
                self.bind_result(op, result)?;
            }
            // Shuffle and Broadcast both lower to `shfl.sync.idx.b32` reading
            // `value` from the lane named by `lane`. They differ only in operand
            // uniformity (broadcast's lane is uniform across the wave), which is
            // the caller's contract, not the instruction's, so the PTX is
            // identical.
            SubgroupShuffle | SubgroupBroadcast => {
                let value_id = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("subgroup shuffle/broadcast missing value".into())
                })?;
                let lane_id = *op.operands.get(1).ok_or_else(|| {
                    EmitError::InvalidDescriptor("subgroup shuffle/broadcast missing lane".into())
                })?;
                let value = self.lookup_operand(value_id)?;
                let lane = self.lookup_operand(lane_id)?;
                let result = self.alloc(value.0);
                let mask = self.alloc(PtxType::U32);
                let lane_mask = self.subgroup_lane_mask();
                let _ = writeln!(self.text, "    activemask.b32    {mask};");
                if value.0 == PtxType::F32 {
                    let bits = self.alloc(PtxType::U32);
                    let shuffled_bits = self.alloc(PtxType::U32);
                    let _ = writeln!(self.text, "    mov.b32    {bits}, {value};");
                    let _ = writeln!(
                        self.text,
                        "    shfl.sync.idx.b32    {shuffled_bits}, {bits}, {lane}, 0x{lane_mask:x}, {mask};"
                    );
                    let _ = writeln!(self.text, "    mov.b32    {result}, {shuffled_bits};");
                } else {
                    let _ = writeln!(
                        self.text,
                        "    shfl.sync.idx.b32    {result}, {value}, {lane}, 0x{lane_mask:x}, {mask};"
                    );
                }
                self.bind_result(op, result)?;
            }
            SubgroupReduce { op: reduce_op } => {
                let value_id = *op.operands.first().ok_or_else(|| {
                    EmitError::InvalidDescriptor("SubgroupReduce missing value".into())
                })?;
                let value = self.lookup_operand(value_id)?;
                let result = self.emit_subgroup_reduce(*reduce_op, value)?;
                self.bind_result(op, result)?;
            }
            SubgroupLocalId => {
                let result = self.alloc(PtxType::U32);
                let _ = writeln!(self.text, "    mov.u32    {result}, %laneid;");
                self.bind_result(op, result)?;
            }
            SubgroupSize => {
                let result = self.alloc(PtxType::U32);
                let subgroup_size = self.options.subgroup_size;
                let _ = writeln!(self.text, "    mov.u32    {result}, {subgroup_size};");
                self.bind_result(op, result)?;
            }
            Atomic {
                op: atomic_op,
                ordering: _,
            } => {
                self.emit_atomic(op, *atomic_op)?;
            }
            Fma => {
                let a_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("Fma missing a".into()))?;
                let b_id = *op
                    .operands
                    .get(1)
                    .ok_or_else(|| EmitError::InvalidDescriptor("Fma missing b".into()))?;
                let c_id = *op
                    .operands
                    .get(2)
                    .ok_or_else(|| EmitError::InvalidDescriptor("Fma missing c".into()))?;
                let a = self.lookup_operand(a_id)?;
                let b = self.lookup_operand(b_id)?;
                let c = self.lookup_operand(c_id)?;
                let dst = self.alloc(a.0);
                let _ = writeln!(
                    self.text,
                    "    fma.rn.{}    {dst}, {a}, {b}, {c};",
                    a.0.ptx_type_str()
                );
                self.bind_result(op, dst)?;
            }
            MatrixMma {
                shape,
                a_layout,
                b_layout,
                a_type,
                b_type,
                accum_type,
            } => {
                let outputs = self.emit_matrix_mma(
                    op,
                    *shape,
                    *a_layout,
                    *b_layout,
                    *a_type,
                    *b_type,
                    *accum_type,
                )?;
                self.bind_consecutive_results(op, &outputs)?;
            }
            Trap { tag } => {
                let address_id = *op
                    .operands
                    .first()
                    .ok_or_else(|| EmitError::InvalidDescriptor("Trap missing address".into()))?;
                self.emit_trap(tag, address_id)?;
            }
            Resume { tag } => {
                let _ = writeln!(self.text, "    // resume tag: {tag}");
            }
            IndirectDispatch { .. } => {
                return Err(EmitError::UnsupportedOp(KernelOp {
                    kind: op.kind.clone(),
                    operands: op.operands.clone(),
                    result: op.result,
                }));
            }
            other => {
                return Err(EmitError::UnsupportedOp(KernelOp {
                    kind: other.clone(),
                    operands: op.operands.clone(),
                    result: op.result,
                }));
            }
        }
        Ok(())
    }
}
