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
