//! PERF B1 (PTX-side): vector-load fusion candidate detection.
//!
//! NVIDIA GPUs support packed vector loads: `ld.global.v2.u32` and
//! `ld.global.v4.u32` move 8 or 16 bytes per transaction with one
//! memory request, instead of 2 or 4 scalar 4-byte loads. On
//! memory-bound kernels this is up to 4× throughput AND reduces
//! per-load address-arithmetic instructions (mul.wide / add.u64).
//!
//! This pattern detects fusion candidates: groups of 2 or 4
//! consecutive `LoadGlobal` ops in the body's flat op stream that:
//!
//! 1. Read from the same `binding_slot`.
//! 2. Have indices `i, i+1, i+2, [i+3]` for the same base  -  detected
//!    when consecutive load's index_id is the result of an `Add(prev_index_id, Lit(1))`
//!    op present in the body.
//! 3. Have no intervening op (other than the index-increment Adds).
//! 4. The base index is naturally aligned for the vector width
//!    (alignment_required is reported; the host may need to verify
//!    this against the runtime allocation alignment).
//!
//! The PTX emitter consumes the same chain shape directly and emits a
//! packed vector load while binding every scalar result id to the
//! registers returned by the vector instruction.
//!
//! Same shape as `vyre-emit-naga::patterns::vec_pack` but PTX-aware:
//! reports vector widths PTX supports (`v2`, `v4`), alignment in
//! bytes, and the expected register class.
//!
//! Chain detection itself lives in [`super::vec_memory_fusion`], which
//! serves both the load and the store side. This module is the
//! load-side facade over it.

use serde::{Deserialize, Serialize};
use vyre_foundation::ir::DataType;
use vyre_lower::KernelDescriptor;

use super::vec_memory_fusion::{analyze_memory_fusion, MemoryFusionCandidate, MemoryFusionKind};

/// One fusion candidate: a group of consecutive scalar loads that
/// could be merged into a single PTX vector load.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionCandidate {
    /// Op-index of the FIRST load in the group.
    pub first_load_idx: usize,
    /// Number of loads in the group (2 or 4 only  -  PTX doesn't have
    /// `v3` loads).
    pub group_size: u8,
    /// Binding slot all loads share.
    pub binding_slot: u32,
    /// Element type all loads share  -  must be same.
    pub element_type: DataType,
    /// Required base-pointer alignment in bytes for the fused load
    /// to be valid: `group_size * element_size`. Host-side allocator
    /// must guarantee this.
    pub alignment_bytes: u32,
}

/// Vector-load fusion opportunities for one kernel.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct FusionPlan {
    /// Consecutive scalar-load groups eligible for fusion.
    pub candidates: Vec<FusionCandidate>,
}

/// Analyze consecutive global loads for vector fusion.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> FusionPlan {
    FusionPlan {
        candidates: analyze_memory_fusion(desc, MemoryFusionKind::Load)
            .into_iter()
            .map(FusionCandidate::from)
            .collect(),
    }
}

impl From<MemoryFusionCandidate> for FusionCandidate {
    fn from(candidate: MemoryFusionCandidate) -> Self {
        Self {
            first_load_idx: candidate.first_op_idx,
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
    /// `vec_memory_fusion`. This pins the facade: the load kind and the
    /// `first_load_idx` field name.
    #[test]
    fn facade_reports_only_the_load_chain() {
        let plan = analyze(&mixed_load_and_store_chains());
        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].first_load_idx, 2);
        assert_eq!(plan.candidates[0].group_size, 2);
        assert_eq!(plan.candidates[0].binding_slot, 0);
        assert_eq!(plan.candidates[0].alignment_bytes, 8);
    }
}
