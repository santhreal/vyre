//! PERF B7: ldmatrix / cp.async detection for async tile loads.
//!
//! `cp.async` (sm_80+) lets a thread issue a global-to-shared transfer
//! that completes asynchronously, freeing the thread for compute while
//! the load is in flight. Combined with `ldmatrix` for shared-to-register
//! tile loads, this hides memory latency on tiled ops.
//!
//! Phase-1 detection: identify load-then-store-to-shared op sequences
//! that match the cp.async pattern. The sequence is:
//!   `LoadGlobal(g, idx) → result_id`
//!   `StoreShared(s, idx, result_id)`
//! When found in adjacent positions on the same logical index, the
//! emitter can replace both with a single `cp.async.ca.shared.global`
//! issue + an `AsyncWait` later in the kernel.
//!
//! The body traversal is `vyre_lower::analyses::structured_walk`, not a copy
//! of it. What stays here is the PTX judgment: the adjacency and same-index
//! conditions above describe what one `cp.async` instruction can actually
//! issue, and the gate on them is a compute-capability question.

use serde::{Deserialize, Serialize};
use vyre_lower::analyses::structured_walk::{walk_structured, ArmDescent, StructuredVisitor};
use vyre_lower::{KernelBody, KernelDescriptor, KernelOpKind};

use crate::ComputeCapability;

/// Adjacent global-load and shared-store pair eligible for asynchronous copy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AsyncCopyCandidate {
    /// Op-index of the LoadGlobal op.
    pub load_op_index: usize,
    /// Op-index of the StoreShared op (must immediately follow).
    pub store_op_index: usize,
    /// Global binding slot read by the candidate load.
    pub global_binding_slot: u32,
    /// Shared-memory binding slot written by the candidate store.
    pub shared_binding_slot: u32,
}

/// Asynchronous-copy opportunities for one kernel and target.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AsyncCopyPlan {
    /// Descriptor id that was analyzed.
    pub kernel_id: String,
    /// True when the selected PTX target can emit native `cp.async`.
    pub target_supports_cp_async: bool,
    /// True when the selected PTX target can use `ldmatrix` for matrix fragments.
    pub target_supports_ldmatrix: bool,
    /// Detected global-load to shared-store pairs that can be staged asynchronously.
    pub candidates: Vec<AsyncCopyCandidate>,
}

impl AsyncCopyPlan {
    /// Return the number of asynchronous-copy candidates.
    #[must_use]
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }
}

/// Analyze global-to-shared transfers for asynchronous-copy eligibility.
#[must_use]
pub fn analyze(desc: &KernelDescriptor, target: ComputeCapability) -> AsyncCopyPlan {
    let cp_async_supported = target.supports_async_copy();
    let mut collector = CandidateCollector::default();
    if cp_async_supported {
        walk_structured(&desc.body, ArmDescent::Enter, &mut collector);
    }
    AsyncCopyPlan {
        kernel_id: desc.id.clone(),
        target_supports_cp_async: cp_async_supported,
        target_supports_ldmatrix: target.supports_ldmatrix(),
        candidates: collector.candidates,
    }
}

#[derive(Default)]
struct CandidateCollector {
    candidates: Vec<AsyncCopyCandidate>,
}

impl<'a> StructuredVisitor<'a> for CandidateCollector {
    fn enter_body(&mut self, body: &'a KernelBody, op_index_offset: usize) {
        for (local_index, pair) in body.ops.windows(2).enumerate() {
            let [load, store] = pair else {
                continue;
            };
            if !matches!(
                (&load.kind, &store.kind),
                (KernelOpKind::LoadGlobal, KernelOpKind::StoreShared)
            ) {
                continue;
            }
            // The load must feed the store, and both must address the same
            // logical index: that is the shape one `cp.async` can issue.
            if load.result.is_none() || load.result != store.operands.get(2).copied() {
                continue;
            }
            if load.operands.get(1) != store.operands.get(1) {
                continue;
            }
            let (Some(global_slot), Some(shared_slot)) = (
                load.operands.first().copied(),
                store.operands.first().copied(),
            ) else {
                continue;
            };
            self.candidates.push(AsyncCopyCandidate {
                load_op_index: op_index_offset + local_index,
                store_op_index: op_index_offset + local_index + 1,
                global_binding_slot: global_slot,
                shared_binding_slot: shared_slot,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::DataType;
    use vyre_lower::descriptor_builder::{
        body, descriptor, effect, global_ro, global_rw, lit, op, shared_rw,
    };
    use vyre_lower::{KernelDescriptor, LiteralValue};

    fn cp_async_kernel() -> KernelDescriptor {
        // load(global, 0) → r0; store(shared, 0, r0)
        descriptor("cp_async")
            .slots([
                global_ro(0, DataType::F32, "g"),
                shared_rw(1, DataType::F32, 64, "s"),
            ])
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([
                        lit(0, 0),
                        op(KernelOpKind::LoadGlobal, [0, 0], 1),
                        effect(KernelOpKind::StoreShared, [1, 0, 1]),
                    ])
                    .literal(LiteralValue::U32(0)),
            )
            .build()
    }

    #[test]
    fn cp_async_unsupported_on_volta() {
        let p = analyze(&cp_async_kernel(), ComputeCapability::SM_70);
        assert!(!p.target_supports_cp_async);
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn cp_async_supported_on_ampere() {
        let p = analyze(&cp_async_kernel(), ComputeCapability::SM_80);
        assert!(p.target_supports_cp_async);
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].load_op_index, 1);
        assert_eq!(p.candidates[0].store_op_index, 2);
        assert_eq!(p.candidates[0].global_binding_slot, 0);
        assert_eq!(p.candidates[0].shared_binding_slot, 1);
    }

    #[test]
    fn empty_kernel_yields_no_candidates() {
        let desc = descriptor("empty").dispatch(64, 1, 1).build();
        let p = analyze(&desc, ComputeCapability::SM_80);
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn load_without_immediate_store_no_candidate() {
        let desc = descriptor("load_only")
            .slot(global_ro(0, DataType::F32, "g"))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([lit(0, 0), op(KernelOpKind::LoadGlobal, [0, 0], 1)])
                    .literal(LiteralValue::U32(0)),
            )
            .build();
        let p = analyze(&desc, ComputeCapability::SM_80);
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn store_to_global_not_shared_no_candidate() {
        let desc = descriptor("store_global")
            .slot(global_rw(0, DataType::F32, "g"))
            .dispatch(64, 1, 1)
            .body(
                body()
                    .ops([
                        lit(0, 0),
                        op(KernelOpKind::LoadGlobal, [0, 0], 1),
                        effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                    ])
                    .literal(LiteralValue::U32(0)),
            )
            .build();
        let p = analyze(&desc, ComputeCapability::SM_80);
        assert!(
            p.candidates.is_empty(),
            "global→global not a cp.async candidate"
        );
    }

    #[test]
    fn mismatched_load_store_index_no_candidate() {
        let mut desc = cp_async_kernel();
        desc.id = "cp_async_mismatched_index".into();
        desc.body.ops[2].operands[1] = 99;
        let p = analyze(&desc, ComputeCapability::SM_80);
        assert!(
            p.candidates.is_empty(),
            "cp.async requires the global load and shared store to use the same logical index"
        );
    }
}
