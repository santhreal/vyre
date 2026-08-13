//! PERF B1 (PTX-side): vector-store fusion candidate detection.
//!
//! Mirror of [`super::vec_load_fusion`] for `StoreGlobal`. NVIDIA
//! GPUs support `st.global.v2.u32` and `st.global.v4.u32` for packed
//! stores  -  same throughput benefits as the load side.
//!
//! Same chain shape: `Store(slot, base_idx, val0); Add(base, 1);
//! Store(slot, idx1, val1); Add(idx1, 1); Store(slot, idx2, val2); ...`
//! up to 4 stores. The PTX emitter lowers the same chain to packed
//! `st.global.v2/v4` instructions.
//!
//! Differences from the load-side analysis:
//! - Stores have no result-id (the chain check looks at the index
//!   operand instead of the result).
//! - The "value" operands of the chained stores are independent  -
//!   they go into the v2/v4 register the way they appear.
//! - Same alignment requirement: `group_size * elem_size` bytes.
//!
//! Chain detection itself lives in [`super::vec_memory_fusion`], which
//! serves both the load and the store side. This module is the
//! store-side facade over it.

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::DataType;
use vyre_lower::KernelDescriptor;

use super::vec_memory_fusion::{analyze_memory_fusion, MemoryFusionCandidate, MemoryFusionKind};

/// One consecutive scalar-store group eligible for vector fusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionCandidate {
    /// Op-index of the FIRST store in the group.
    pub first_store_idx: usize,
    /// Number of stores in the group (2 or 4  -  PTX has no v3).
    pub group_size: u8,
    /// Binding slot all stores share.
    pub binding_slot: u32,
    /// Element type from the binding.
    pub element_type: DataType,
    /// Required base-pointer alignment in bytes.
    pub alignment_bytes: u32,
}

/// Vector-store fusion opportunities for one kernel.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FusionPlan {
    /// Consecutive scalar-store groups eligible for fusion.
    pub candidates: Vec<FusionCandidate>,
}

/// Analyze consecutive global stores for vector fusion.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> FusionPlan {
    FusionPlan {
        candidates: analyze_memory_fusion(desc, MemoryFusionKind::Store)
            .into_iter()
            .map(FusionCandidate::from)
            .collect(),
    }
}

impl From<MemoryFusionCandidate> for FusionCandidate {
    fn from(candidate: MemoryFusionCandidate) -> Self {
        Self {
            first_store_idx: candidate.first_op_idx,
            group_size: candidate.group_size,
            binding_slot: candidate.binding_slot,
            element_type: candidate.element_type,
            alignment_bytes: candidate.alignment_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::vec_memory_fusion::tests::mixed_load_and_store_chains;

    /// Chain detection itself is covered once, over both kinds, in
    /// `vec_memory_fusion`. This pins the facade: the store kind and the
    /// `first_store_idx` field name.
    #[test]
    fn facade_reports_only_the_store_chain() {
        let plan = analyze(&mixed_load_and_store_chains());
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].first_store_idx, 5);
        assert_eq!(plan.candidates[0].group_size, 2);
        assert_eq!(plan.candidates[0].binding_slot, 1);
        assert_eq!(plan.candidates[0].alignment_bytes, 8);
    }
}
