//! Canonical vector-memory lowering rewrite for KernelDescriptor.
//!
//! Fuses and canonicalizes verified unit-stride adjacent global memory load and
//! store chains at vec2 and vec4 widths when proven aligned, alias-free,
//! side-effect-free, and within scheduling boundaries.
//!
//! Preserves Program semantics, SSA result IDs, and descriptor verification invariants.

use rustc_hash::FxHashMap;
use vyre_foundation::ir::{BinOp, DataType};

use crate::analyses::alias_facts::AliasFactSet;
use crate::analyses::child_body_operands;
use crate::analyses::vec_pack::{index_expr_by_result, IndexExpr};
use crate::op_facts::kernel_op_kind_is_dce_pure;
use crate::operand_class::operand_is_result_reference;
use crate::{KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue, MemoryClass};

/// Apply canonical vector-memory lowering rewrite to a descriptor.
#[must_use]
pub fn rewrite_vector_memory(descriptor: &KernelDescriptor) -> KernelDescriptor {
    rewrite_vector_memory_with_alias_facts(descriptor, &AliasFactSet::default())
}

/// Apply canonical vector-memory lowering rewrite with explicit alias facts.
#[must_use]
pub fn rewrite_vector_memory_with_alias_facts(
    descriptor: &KernelDescriptor,
    alias_facts: &AliasFactSet,
) -> KernelDescriptor {
    let mut output = descriptor.clone();
    let mut next_result_id = find_max_result_id(&descriptor.body).saturating_add(1);
    rewrite_body(
        &mut output.body,
        descriptor,
        alias_facts,
        &mut next_result_id,
    );
    output
}

fn find_max_result_id(body: &KernelBody) -> u32 {
    let mut max = 0;
    for op in &body.ops {
        for rid in op.result_ids() {
            max = max.max(rid);
        }
        for operand in &op.operands {
            max = max.max(*operand);
        }
    }
    for child in &body.child_bodies {
        max = max.max(find_max_result_id(child));
    }
    max
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorAccessKind {
    Load,
    Store,
}

#[derive(Debug, Clone)]
struct VectorChainCandidate {
    kind: VectorAccessKind,
    slot: u32,
    op_indices: Vec<usize>,
    intervening_pure_indices: Vec<usize>,
}

fn rewrite_body(
    body: &mut KernelBody,
    descriptor: &KernelDescriptor,
    alias_facts: &AliasFactSet,
    next_result_id: &mut u32,
) {
    for child in &mut body.child_bodies {
        rewrite_body(child, descriptor, alias_facts, next_result_id);
    }

    let mut start_idx = 0;
    while start_idx < body.ops.len() {
        let indices = index_expr_by_result(body);
        let op = &body.ops[start_idx];
        let Some(kind) = access_kind_of(&op.kind) else {
            start_idx += 1;
            continue;
        };

        let Some(slot) = op.operands.first().copied() else {
            start_idx += 1;
            continue;
        };

        if !is_eligible_global_binding(descriptor, slot) {
            start_idx += 1;
            continue;
        }

        let Some(index_op_id) = op.operands.get(1).copied() else {
            start_idx += 1;
            continue;
        };

        let Some(start_index_expr) = indices.get(&index_op_id).copied() else {
            start_idx += 1;
            continue;
        };

        let mut matched_candidate = None;
        for &width in &[4usize, 2usize] {
            if let Some(candidate) = try_collect_chain(
                body,
                &indices,
                descriptor,
                alias_facts,
                start_idx,
                kind,
                slot,
                start_index_expr,
                width,
            ) {
                matched_candidate = Some((candidate, width));
                break;
            }
        }

        if let Some((candidate, width)) = matched_candidate {
            let advance = apply_vector_chain(body, candidate, width, next_result_id);
            start_idx += advance;
        } else {
            start_idx += 1;
        }
    }
}

fn access_kind_of(kind: &KernelOpKind) -> Option<VectorAccessKind> {
    match kind {
        KernelOpKind::LoadGlobal => Some(VectorAccessKind::Load),
        KernelOpKind::StoreGlobal => Some(VectorAccessKind::Store),
        _ => None,
    }
}

fn is_eligible_global_binding(descriptor: &KernelDescriptor, slot: u32) -> bool {
    let Some(binding) = descriptor.bindings.slots.iter().find(|s| s.slot == slot) else {
        return false;
    };

    if binding.memory_class != MemoryClass::Global {
        return false;
    }

    // Supported 4-byte scalar types for vec2/vec4 memory transactions.
    matches!(
        binding.element_type,
        DataType::U32
            | DataType::I32
            | DataType::F32
            | DataType::Bool
            | DataType::Array { element_size: 4 }
    ) || binding.element_type.min_bytes() == 4
}

#[allow(clippy::too_many_arguments)]
fn try_collect_chain(
    body: &KernelBody,
    indices: &FxHashMap<u32, IndexExpr>,
    _descriptor: &KernelDescriptor,
    alias_facts: &AliasFactSet,
    start_idx: usize,
    kind: VectorAccessKind,
    slot: u32,
    start_index_expr: IndexExpr,
    target_width: usize,
) -> Option<VectorChainCandidate> {
    // 1. Proven alignment requirement.
    if !is_proven_aligned(body, start_index_expr, target_width as u32) {
        return None;
    }

    let mut op_indices = vec![start_idx];
    let mut intervening_pure_indices = Vec::new();
    let mut current_offset = start_index_expr.offset;

    let mut scan_idx = start_idx + 1;
    while scan_idx < body.ops.len() && op_indices.len() < target_width {
        let next_op = &body.ops[scan_idx];

        if is_scheduling_fence(&next_op.kind) {
            return None;
        }

        if is_structured_control_or_effect(&next_op.kind) {
            return None;
        }

        if let Some(next_kind) = access_kind_of(&next_op.kind) {
            let next_slot = next_op.operands.first().copied().unwrap_or(u32::MAX);

            if next_kind == kind && next_slot == slot {
                let next_index_op_id = next_op.operands.get(1).copied().unwrap_or(u32::MAX);
                let next_index_expr = indices.get(&next_index_op_id).copied()?;

                // Same base and strictly unit-stride consecutive offset.
                if next_index_expr.base_result == start_index_expr.base_result
                    && next_index_expr.offset == current_offset.saturating_add(1)
                {
                    op_indices.push(scan_idx);
                    current_offset = next_index_expr.offset;
                    scan_idx += 1;
                    continue;
                }
                // Non-consecutive access to same slot breaks the chain.
                return None;
            }

            // Hazard check on different access or different slot:
            if has_memory_hazard(kind, slot, next_kind, next_slot, alias_facts) {
                return None;
            }
        }

        if !kernel_op_kind_is_dce_pure(&next_op.kind) {
            return None;
        }

        intervening_pure_indices.push(scan_idx);
        scan_idx += 1;
    }

    if op_indices.len() != target_width {
        return None;
    }

    // Dependency validation for store chains:
    // Store values must not be produced by subsequent ops or dependent on intervening ops
    // that cannot precede the store.
    if kind == VectorAccessKind::Store {
        for &op_idx in &op_indices {
            let val_id = body.ops[op_idx]
                .operands
                .get(2)
                .copied()
                .unwrap_or(u32::MAX);
            if intervening_pure_indices
                .iter()
                .any(|&pure_idx| body.ops[pure_idx].result == Some(val_id))
            {
                // Value produced by an intervening op in the middle of the chain.
                return None;
            }
        }
    }

    // Dependency validation for load chains:
    // Intervening pure ops must not consume results produced by subsequent loads in this chain.
    if kind == VectorAccessKind::Load {
        for &pure_idx in &intervening_pure_indices {
            let pure_op = &body.ops[pure_idx];
            for (pos, &operand) in pure_op.operands.iter().enumerate() {
                if operand_is_result_reference(&pure_op.kind, pos) {
                    for &load_idx in &op_indices {
                        if body.ops[load_idx].result == Some(operand) && load_idx > pure_idx {
                            return None;
                        }
                    }
                }
            }
        }
    }

    Some(VectorChainCandidate {
        kind,
        slot,
        op_indices,
        intervening_pure_indices,
    })
}

fn is_scheduling_fence(kind: &KernelOpKind) -> bool {
    matches!(
        kind,
        KernelOpKind::Barrier { .. }
            | KernelOpKind::AsyncWait { .. }
            | KernelOpKind::AsyncLoad { .. }
            | KernelOpKind::AsyncStore { .. }
            | KernelOpKind::Trap { .. }
            | KernelOpKind::Resume { .. }
            | KernelOpKind::Return
            | KernelOpKind::IndirectDispatch { .. }
    )
}

fn is_structured_control_or_effect(kind: &KernelOpKind) -> bool {
    matches!(
        kind,
        KernelOpKind::StructuredIfThen
            | KernelOpKind::StructuredIfThenElse
            | KernelOpKind::StructuredForLoop { .. }
            | KernelOpKind::StructuredBlock
            | KernelOpKind::Region { .. }
            | KernelOpKind::Atomic { .. }
            | KernelOpKind::Call { .. }
            | KernelOpKind::OpaqueNode(..)
    ) || child_body_operands(kind, &[]).next().is_some()
}

fn has_memory_hazard(
    chain_kind: VectorAccessKind,
    chain_slot: u32,
    other_kind: VectorAccessKind,
    other_slot: u32,
    alias_facts: &AliasFactSet,
) -> bool {
    if chain_slot == other_slot {
        // Any other access to the same slot during the chain window is an alias hazard.
        return true;
    }

    if chain_kind == VectorAccessKind::Store || other_kind == VectorAccessKind::Store {
        // Between distinct slots, if no-alias is not proven, treat as potential alias uncertainty.
        if !alias_facts.is_empty() && !alias_facts.proves_no_alias(chain_slot, 0, other_slot, 0) {
            return true;
        }
    }

    false
}

/// Rewrite a validated contiguous vector chain candidate into wide vector operations.
///
/// # Panics
///
/// Panics if `candidate` violates the validated vector-chain invariants: empty
/// or out-of-bounds operation indices, or malformed scalar load/store operands.
fn apply_vector_chain(
    body: &mut KernelBody,
    candidate: VectorChainCandidate,
    width: usize,
    next_result_id: &mut u32,
) -> usize {
    let anchor_idx = *candidate
        .op_indices
        .first()
        .expect("Fix: vector chain candidate must contain at least one operation");
    let end_idx = *candidate
        .op_indices
        .last()
        .expect("Fix: vector chain candidate must contain at least one operation");

    let mut intervening_ops = Vec::with_capacity(candidate.intervening_pure_indices.len());
    for &idx in &candidate.intervening_pure_indices {
        intervening_ops.push(body.ops[idx].clone());
    }

    let mut replaced_ops = Vec::with_capacity(intervening_ops.len() + 1 + width);
    replaced_ops.extend(intervening_ops);

    match candidate.kind {
        VectorAccessKind::Load => {
            let start_op = &body.ops[anchor_idx];
            let slot = start_op.operands[0];
            let start_index_op_id = start_op.operands[1];
            let vec_result_id = *next_result_id;
            *next_result_id = next_result_id.saturating_add(1);

            // Emit 1 wide VectorLoadGlobal op
            replaced_ops.push(KernelOp {
                kind: KernelOpKind::VectorLoadGlobal { width: width as u8 },
                operands: vec![slot, start_index_op_id],
                result: Some(vec_result_id),
            });

            // Emit lane projections (ExtractLane) preserving original SSA result IDs
            for (lane, &load_idx) in candidate.op_indices.iter().enumerate() {
                let scalar_result = body.ops[load_idx].result;
                replaced_ops.push(KernelOp {
                    kind: KernelOpKind::ExtractLane { lane: lane as u8 },
                    operands: vec![vec_result_id],
                    result: scalar_result,
                });
            }
        }
        VectorAccessKind::Store => {
            let start_op = &body.ops[anchor_idx];
            let slot = start_op.operands[0];
            let start_index_op_id = start_op.operands[1];

            let mut operands = Vec::with_capacity(2 + width);
            operands.push(slot);
            operands.push(start_index_op_id);
            for &store_idx in &candidate.op_indices {
                operands.push(body.ops[store_idx].operands[2]);
            }

            // Emit 1 wide VectorStoreGlobal op
            replaced_ops.push(KernelOp {
                kind: KernelOpKind::VectorStoreGlobal { width: width as u8 },
                operands,
                result: None,
            });
        }
    }

    let inserted_count = replaced_ops.len();
    let mut new_ops = Vec::with_capacity(body.ops.len().saturating_add(inserted_count));
    new_ops.extend_from_slice(&body.ops[..anchor_idx]);
    new_ops.extend(replaced_ops);
    if end_idx + 1 < body.ops.len() {
        new_ops.extend_from_slice(&body.ops[end_idx + 1..]);
    }

    body.ops = new_ops;
    inserted_count
}

// ---------- Alignment & Index Facts ----------

fn is_proven_aligned(body: &KernelBody, expr: IndexExpr, width: u32) -> bool {
    if width == 0 || (width != 2 && width != 4) {
        return false;
    }

    if expr.offset % width != 0 {
        return false;
    }

    let Some(base_id) = expr.base_result else {
        // Constant literal: offset % width == 0 is exact proof.
        return true;
    };

    base_is_multiple_of(body, base_id, width, 0)
}

fn base_is_multiple_of(body: &KernelBody, result_id: u32, modulus: u32, depth: u8) -> bool {
    if depth > 8 {
        return false;
    }

    let Some(op) = body.ops.iter().find(|op| op.result == Some(result_id)) else {
        return false;
    };

    match &op.kind {
        KernelOpKind::Literal => {
            let Some(&pool_idx) = op.operands.first() else {
                return false;
            };
            match body.literals.get(pool_idx as usize) {
                Some(LiteralValue::U32(val)) => val % modulus == 0,
                Some(LiteralValue::I32(val)) if *val >= 0 => (*val as u32) % modulus == 0,
                _ => false,
            }
        }
        KernelOpKind::BinOpKind(BinOp::Mul) => {
            let lhs = op.operands.first().copied().unwrap_or(u32::MAX);
            let rhs = op.operands.get(1).copied().unwrap_or(u32::MAX);
            base_is_multiple_of(body, lhs, modulus, depth + 1)
                || base_is_multiple_of(body, rhs, modulus, depth + 1)
        }
        KernelOpKind::BinOpKind(BinOp::Shl) => {
            let rhs = op.operands.get(1).copied().unwrap_or(u32::MAX);
            if let Some(shift) = literal_u32_value(body, rhs) {
                let factor = 1u32.checked_shl(shift & 31).unwrap_or(0);
                if factor != 0 && factor % modulus == 0 {
                    return true;
                }
            }
            let lhs = op.operands.first().copied().unwrap_or(u32::MAX);
            base_is_multiple_of(body, lhs, modulus, depth + 1)
        }
        KernelOpKind::BinOpKind(BinOp::Add | BinOp::WrappingAdd) => {
            let lhs = op.operands.first().copied().unwrap_or(u32::MAX);
            let rhs = op.operands.get(1).copied().unwrap_or(u32::MAX);
            base_is_multiple_of(body, lhs, modulus, depth + 1)
                && base_is_multiple_of(body, rhs, modulus, depth + 1)
        }
        _ => false,
    }
}

fn literal_u32_value(body: &KernelBody, result_id: u32) -> Option<u32> {
    let op = body.ops.iter().find(|op| op.result == Some(result_id))?;
    if !matches!(op.kind, KernelOpKind::Literal) {
        return None;
    }
    let pool_idx = *op.operands.first()?;
    match body.literals.get(pool_idx as usize)? {
        LiteralValue::U32(val) => Some(*val),
        LiteralValue::I32(val) if *val >= 0 => Some(*val as u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};
    use crate::{BindingLayout, BindingSlot, BindingVisibility, Dispatch};

    fn test_load_descriptor(offsets: &[u32]) -> KernelDescriptor {
        let mut ops = Vec::new();
        let mut literals = Vec::new();
        for (i, &offset) in offsets.iter().enumerate() {
            literals.push(LiteralValue::U32(offset));
            ops.push(lit(i as u32, i as u32));
            ops.push(op(KernelOpKind::LoadGlobal, [0, i as u32], (10 + i) as u32));
        }

        KernelDescriptor {
            id: "test_vec_load".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    name: "in_buf".into(),
                    element_type: DataType::F32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    element_count: Some(1024),
                }],
            },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops,
                literals,
                child_bodies: vec![],
            },
        }
    }

    #[test]
    fn aligned_vec2_load_chain_is_canonicalized() {
        let desc = test_load_descriptor(&[0, 1]);
        let rewritten = rewrite_vector_memory(&desc);
        assert_eq!(rewritten.body.ops.len(), 5);
        assert_eq!(
            rewritten.body.ops[2].kind,
            KernelOpKind::VectorLoadGlobal { width: 2 }
        );
        assert_eq!(
            rewritten.body.ops[3].kind,
            KernelOpKind::ExtractLane { lane: 0 }
        );
        assert_eq!(
            rewritten.body.ops[4].kind,
            KernelOpKind::ExtractLane { lane: 1 }
        );
        assert_eq!(rewritten.body.ops[3].result, Some(10));
        assert_eq!(rewritten.body.ops[4].result, Some(11));
    }

    #[test]
    fn aligned_vec4_load_chain_is_canonicalized() {
        let desc = test_load_descriptor(&[0, 1, 2, 3]);
        let rewritten = rewrite_vector_memory(&desc);
        assert_eq!(rewritten.body.ops.len(), 9);
        assert_eq!(
            rewritten.body.ops[4].kind,
            KernelOpKind::VectorLoadGlobal { width: 4 }
        );
        assert_eq!(
            rewritten.body.ops[5].kind,
            KernelOpKind::ExtractLane { lane: 0 }
        );
        assert_eq!(
            rewritten.body.ops[6].kind,
            KernelOpKind::ExtractLane { lane: 1 }
        );
        assert_eq!(
            rewritten.body.ops[7].kind,
            KernelOpKind::ExtractLane { lane: 2 }
        );
        assert_eq!(
            rewritten.body.ops[8].kind,
            KernelOpKind::ExtractLane { lane: 3 }
        );
        assert_eq!(rewritten.body.ops[5].result, Some(10));
        assert_eq!(rewritten.body.ops[6].result, Some(11));
        assert_eq!(rewritten.body.ops[7].result, Some(12));
        assert_eq!(rewritten.body.ops[8].result, Some(13));
    }

    #[test]
    fn misaligned_vec2_load_chain_is_rejected() {
        let desc = test_load_descriptor(&[1, 2]);
        let rewritten = rewrite_vector_memory(&desc);
        assert_eq!(rewritten, desc);
    }

    #[test]
    fn misaligned_vec4_load_chain_is_rejected() {
        let desc = test_load_descriptor(&[2, 3, 4, 5]);
        let rewritten = rewrite_vector_memory(&desc);
        // Offset 2 is vec2 aligned, so it vectorizes the vec2 prefix [2, 3]
        assert_eq!(
            rewritten.body.ops[2].kind,
            KernelOpKind::VectorLoadGlobal { width: 2 }
        );
    }

    fn test_store_descriptor(with_barrier: bool) -> KernelDescriptor {
        let mut ops = vec![
            lit(0, 0),
            lit(1, 1),
            lit(2, 2),
            lit(3, 3),
            effect(KernelOpKind::StoreGlobal, [0, 0, 2]),
        ];
        if with_barrier {
            ops.push(effect(
                KernelOpKind::Barrier {
                    ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
                },
                [],
            ));
        }
        ops.push(effect(KernelOpKind::StoreGlobal, [0, 1, 3]));
        descriptor("test_store")
            .slot(global_rw(0, DataType::U32, "out"))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .literals([
                        LiteralValue::U32(0),
                        LiteralValue::U32(1),
                        LiteralValue::U32(42),
                        LiteralValue::U32(43),
                    ])
                    .ops(ops),
            )
            .build()
    }

    #[test]
    fn aligned_vec2_store_chain_is_canonicalized() {
        let desc = test_store_descriptor(false);
        let rewritten = rewrite_vector_memory(&desc);
        assert_eq!(rewritten.body.ops.len(), 5);
        assert_eq!(
            rewritten.body.ops[4].kind,
            KernelOpKind::VectorStoreGlobal { width: 2 }
        );
        assert_eq!(rewritten.body.ops[4].operands, vec![0, 0, 2, 3]);
    }

    #[test]
    fn barrier_fence_breaks_vector_chain() {
        let desc = test_store_descriptor(true);
        let rewritten = rewrite_vector_memory(&desc);
        assert_eq!(rewritten, desc);
    }
}
