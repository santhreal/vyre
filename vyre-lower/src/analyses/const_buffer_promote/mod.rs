//! PERF B10: constant-buffer promotion candidate detection.
//!
//! Source-of-truth: `PERF_ROADMAP_2026-05-01.md` section B item B10.
//!
//! Small read-only data accessed many times across a workgroup
//! benefits from being promoted from a Storage/SSBO buffer to a
//! Constant/Uniform buffer. Constant buffers are cached in dedicated
//! scalar-read hardware and serve reads in 1-2 cycles vs 100s for
//! global memory.
//!
//! Eligibility (phase 1):
//! - `memory_class == Global` and `visibility == ReadOnly`
//! - `element_count.is_some()` (fixed size  -  constant buffers have a
//!   compile-time size limit, typically 64 KiB)
//! - Total bytes ≤ const-buffer budget (default 64 KiB)
//! - Multiple loads against the binding (single-load doesn't repay
//!   the cache-line preload)
//!
//! Rewrite consumers change `binding.memory_class` to `Constant` and
//! let each emitter map the descriptor class to its native artifact.

use super::load_counts::count_global_loads_by_slot;
use crate::{BindingSlot, BindingVisibility, KernelDescriptor, MemoryClass};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use vyre_foundation::ir::DataType;

/// One read-only binding eligible for constant-buffer promotion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstBufferCandidate {
    /// Binding slot to promote.
    pub binding_slot: u32,
    /// Static binding size in bytes.
    pub bytes: u32,
    /// Number of loads that reuse the binding.
    pub load_count: u32,
    /// Estimated speedup: roughly `1.0 + load_count * 0.4` capped at 8x.
    pub estimated_speedup_factor: f32,
}

/// Constant-buffer promotion plan for one kernel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstBufferPlan {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Eligible bindings in slot order.
    pub candidates: Vec<ConstBufferCandidate>,
    /// Combined size of all eligible bindings.
    pub total_bytes: u32,
    /// Maximum constant-buffer bytes available.
    pub budget_bytes: u32,
}

impl ConstBufferPlan {
    /// Return whether all candidates fit within the configured budget.
    #[must_use]
    pub fn fits_in_budget(&self) -> bool {
        self.total_bytes <= self.budget_bytes
    }
}

/// Default const-buffer budget: 64 KiB. Callers with tighter backend
/// limits should pass their real budget into the analysis entry point.
pub const DEFAULT_CONST_BUFFER_BUDGET_BYTES: u32 = 64 * 1024;

/// Analyze constant-buffer candidates using the default budget.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> ConstBufferPlan {
    analyze_with_budget(desc, DEFAULT_CONST_BUFFER_BUDGET_BYTES)
}

/// Analyze constant-buffer candidates using an explicit byte budget.
#[must_use]
pub fn analyze_with_budget(desc: &KernelDescriptor, budget_bytes: u32) -> ConstBufferPlan {
    // Eligible bindings.
    let eligible: FxHashMap<u32, &BindingSlot> = desc
        .bindings
        .slots
        .iter()
        .filter(|b| {
            matches!(b.memory_class, MemoryClass::Global)
                && matches!(b.visibility, BindingVisibility::ReadOnly)
                && b.element_count.is_some()
        })
        .map(|b| (b.slot, b))
        .collect();

    // Count loads per slot.
    let mut load_counts =
        FxHashMap::<u32, u32>::with_capacity_and_hasher(eligible.len(), Default::default());
    count_global_loads_by_slot(
        &desc.body,
        &|slot| eligible.contains_key(&slot),
        &mut load_counts,
    );

    let mut candidates = Vec::new();
    let mut total: u32 = 0;
    for (slot, count) in load_counts {
        if count < 2 {
            continue;
        }
        let binding = eligible[&slot];
        let bytes_per_elem = match binding.element_type.size_bytes() {
            Some(b) => b as u32,
            None => continue,
        };
        let elem_count = binding.element_count.unwrap_or(0);
        let bytes = bytes_per_elem.saturating_mul(elem_count);
        if bytes == 0 || bytes > budget_bytes {
            continue;
        }
        let speedup = (1.0 + count as f32 * 0.4).min(8.0);
        candidates.push(ConstBufferCandidate {
            binding_slot: slot,
            bytes,
            load_count: count,
            estimated_speedup_factor: speedup,
        });
        total = total.saturating_add(bytes);
    }
    candidates.sort_unstable_by_key(|candidate| candidate.binding_slot);
    ConstBufferPlan {
        kernel_id: desc.id.clone(),
        candidates,
        total_bytes: total,
        budget_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyses::load_counts::fixtures;
    use crate::descriptor_builder::{descriptor, global_ro, SlotCount};
    use crate::KernelDescriptor;

    /// A read-only global of `count` elements, which is the only binding shape
    /// the eligibility rule admits.
    fn ro_global_with_size(slot: u32, count: u32, dtype: DataType) -> BindingSlot {
        global_ro(slot, dtype, &format!("ro{slot}")).with_count(count)
    }

    /// One binding read `load_count` times, the shape every case below varies.
    /// Owned by `load_counts`, whose traversal defines the op shape read here.
    fn loads_kernel(load_count: u32, binding: BindingSlot) -> KernelDescriptor {
        fixtures::kernel(binding, load_count)
    }

    #[test]
    fn empty_kernel_no_candidates() {
        let p = analyze(&descriptor("k").dispatch(64, 1, 1).build());
        assert!(p.candidates.is_empty());
        assert!(p.fits_in_budget());
    }

    #[test]
    fn fixed_size_ro_with_two_loads_is_candidate() {
        let p = analyze(&loads_kernel(2, ro_global_with_size(0, 16, DataType::F32)));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].bytes, 64); // 16 * 4
        assert_eq!(p.candidates[0].load_count, 2);
    }

    #[test]
    fn runtime_sized_binding_not_candidate() {
        let mut binding = ro_global_with_size(0, 16, DataType::F32);
        binding.element_count = None;
        let p = analyze(&loads_kernel(2, binding));
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn read_write_binding_not_candidate() {
        let mut binding = ro_global_with_size(0, 16, DataType::F32);
        binding.visibility = BindingVisibility::ReadWrite;
        let p = analyze(&loads_kernel(2, binding));
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn single_load_not_candidate() {
        let p = analyze(&loads_kernel(1, ro_global_with_size(0, 16, DataType::F32)));
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn over_budget_binding_not_candidate() {
        // 1M elements * 4 bytes = 4 MiB >> 64 KiB budget.
        let p = analyze(&loads_kernel(
            2,
            ro_global_with_size(0, 1_000_000, DataType::F32),
        ));
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn speedup_capped_at_8x() {
        let p = analyze(&loads_kernel(
            100,
            ro_global_with_size(0, 16, DataType::F32),
        ));
        assert_eq!(p.candidates[0].load_count, 100);
        // Without cap: 1 + 100*0.4 = 41. With cap: 8.0.
        assert!((p.candidates[0].estimated_speedup_factor - 8.0).abs() < 1e-5);
    }

    #[test]
    fn custom_budget_changes_eligibility() {
        let p = analyze_with_budget(
            &loads_kernel(2, ro_global_with_size(0, 16, DataType::F32)),
            32, // 32 byte budget  -  64-byte binding doesn't fit
        );
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn default_budget_is_64_kib() {
        assert_eq!(DEFAULT_CONST_BUFFER_BUDGET_BYTES, 65536);
    }
}
