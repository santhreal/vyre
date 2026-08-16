//! Vector memory-fusion chain detection for PTX.
//!
//! NVIDIA GPUs move 8 or 16 bytes per transaction with `ld.global.v2/v4` and
//! `st.global.v2/v4` instead of 2 or 4 scalar 4-byte accesses. This detects
//! the chains that qualify: 2 or 4 consecutive `LoadGlobal`/`LoadConstant` or
//! `StoreGlobal` ops that read or write one binding slot at indices
//! `i, i+1, i+2, [i+3]`, with no intervening op other than the
//! index-increment `Add`s. The emitter consumes the same chain shape and
//! binds every scalar result id to the vector instruction's registers.
//!
//! The load and store sides differ only in which operand carries the index
//! and whether there is a value operand, so [`MemoryFusionKind`] carries that
//! difference and one detector serves both. A caller that wants only one side
//! passes only that kind.
//!
//! `alignment_bytes` is a requirement, not a guarantee: the host allocator
//! must satisfy it for the fused access to be valid.

use crate::emitter::schedule::{is_schedulable_pure_op, is_scheduling_fence};
use crate::index_facts::IndexFacts;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use vyre_foundation::ir::DataType;
use vyre_lower::{BindingSlot, KernelBody, KernelDescriptor, KernelOp, KernelOpKind};

/// Which side of memory a fusion chain accesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryFusionKind {
    /// Consecutive reads, fusible into `ld.global.v2/v4`.
    Load,
    /// Consecutive writes, fusible into `st.global.v2/v4`.
    Store,
}

impl MemoryFusionKind {
    fn matches(self, kind: &KernelOpKind) -> bool {
        match self {
            Self::Load => matches!(kind, KernelOpKind::LoadGlobal | KernelOpKind::LoadConstant),
            Self::Store => matches!(kind, KernelOpKind::StoreGlobal),
        }
    }

    fn slot_and_index(self, op: &KernelOp) -> Option<(u32, u32)> {
        let min_operands = match self {
            Self::Load => 2,
            Self::Store => 3,
        };
        if op.operands.len() < min_operands {
            return None;
        }
        Some((op.operands[0], op.operands[1]))
    }

    fn value_operand(self, op: &KernelOp) -> Option<u32> {
        match self {
            Self::Load => None,
            Self::Store => op.operands.get(2).copied(),
        }
    }
}

/// One chain of consecutive scalar accesses that could be merged into a
/// single PTX vector access.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryFusionCandidate {
    /// Op-index of the first access in the chain.
    pub first_op_idx: usize,
    /// Number of accesses in the chain. Only 2 and 4 occur; PTX has no `v3`.
    pub group_size: u8,
    /// Binding slot every access in the chain touches.
    pub binding_slot: u32,
    /// Element type taken from the binding.
    pub element_type: DataType,
    /// Base-pointer alignment the fused access requires, in bytes:
    /// `group_size * element_size`.
    pub alignment_bytes: u32,
}

/// Fusion opportunities of one kind for one kernel.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct MemoryFusionPlan {
    /// Chains eligible for fusion, in op order.
    pub candidates: Vec<MemoryFusionCandidate>,
}

/// Detect fusible chains of the given kind.
#[must_use]
pub fn analyze(desc: &KernelDescriptor, kind: MemoryFusionKind) -> MemoryFusionPlan {
    let binding_by_slot: FxHashMap<u32, &BindingSlot> = desc
        .bindings
        .slots
        .iter()
        .map(|binding| (binding.slot, binding))
        .collect();
    let mut candidates = Vec::new();
    walk(&desc.body, &binding_by_slot, kind, &mut candidates);
    MemoryFusionPlan { candidates }
}

fn walk(
    body: &KernelBody,
    binding_by_slot: &FxHashMap<u32, &BindingSlot>,
    kind: MemoryFusionKind,
    candidates: &mut Vec<MemoryFusionCandidate>,
) {
    let facts = IndexFacts::new(body);
    let mut i = 0;
    while i < body.ops.len() {
        let op = &body.ops[i];
        if !kind.matches(&op.kind) {
            i += 1;
            continue;
        }
        let Some((slot, base_idx_id)) = kind.slot_and_index(op) else {
            i += 1;
            continue;
        };
        let Some(binding) = binding_by_slot.get(&slot).copied() else {
            i += 1;
            continue;
        };

        let mut chain_len: u8 = 1;
        let mut prev_idx_id = base_idx_id;
        let mut j = i + 1;
        while j < body.ops.len() && chain_len < 4 {
            let gap_start = j;
            while j < body.ops.len() {
                let next = &body.ops[j];
                if kind.matches(&next.kind) {
                    break;
                }
                if is_scheduling_fence(next) || !is_schedulable_pure_op(next) {
                    break;
                }
                j += 1;
            }
            if j >= body.ops.len() {
                break;
            }
            let next = &body.ops[j];
            if !kind.matches(&next.kind) {
                break;
            }
            let Some((next_slot, next_idx_id)) = kind.slot_and_index(next) else {
                break;
            };
            if let Some(next_value_id) = kind.value_operand(next) {
                if body.ops[gap_start..j]
                    .iter()
                    .any(|gap_op| gap_op.result == Some(next_value_id))
                {
                    break;
                }
            }
            if next_slot != slot || !facts.is_index_plus_one(body, next_idx_id, prev_idx_id) {
                break;
            }
            chain_len += 1;
            prev_idx_id = next_idx_id;
            j += 1;
        }

        if chain_len >= 2 {
            let group_size = if chain_len >= 4 { 4 } else { 2 };
            let elem_size = binding.element_type.size_bytes().unwrap_or(0) as u32;
            candidates.push(MemoryFusionCandidate {
                first_op_idx: i,
                group_size,
                binding_slot: slot,
                element_type: binding.element_type.clone(),
                alignment_bytes: group_size as u32 * elem_size,
            });
            i += (group_size as usize) * 2 - 1;
        } else {
            i += 1;
        }
    }

    for child in &body.child_bodies {
        walk(child, binding_by_slot, kind, candidates);
    }
}
