//! Detect read-only bindings with multiple LoadGlobal sites  -  the
//! basic precondition for texture-memory promotion.

use super::plan::{TextureCandidate, TexturePromotionPlan};
use crate::analyses::load_counts::count_global_loads_by_slot;
use crate::{BindingVisibility, KernelDescriptor, MemoryClass};
use rustc_hash::{FxHashMap, FxHashSet};

/// Analyze read-only global bindings for texture-memory promotion.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> TexturePromotionPlan {
    // Eligible bindings: Global memory class, ReadOnly visibility.
    let eligible: FxHashSet<u32> = desc
        .bindings
        .slots
        .iter()
        .filter(|b| {
            matches!(b.memory_class, MemoryClass::Global)
                && matches!(b.visibility, BindingVisibility::ReadOnly)
        })
        .map(|b| b.slot)
        .collect();

    let mut load_counts: FxHashMap<u32, u32> =
        FxHashMap::with_capacity_and_hasher(eligible.len(), Default::default());
    count_global_loads_by_slot(
        &desc.body,
        &|slot| eligible.contains(&slot),
        &mut load_counts,
    );

    let mut candidates = Vec::new();
    for (slot, count) in load_counts {
        if count >= 2 {
            let speedup = 1.5 + (count as f32).log2();
            candidates.push(TextureCandidate {
                binding_slot: slot,
                load_count: count,
                estimated_speedup_factor: speedup,
            });
        }
    }
    candidates.sort_unstable_by_key(|candidate| candidate.binding_slot);

    TexturePromotionPlan {
        kernel_id: desc.id.clone(),
        candidates,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyses::load_counts::fixtures::kernel;
    use crate::descriptor_builder::{
        body, descriptor, global_ro, global_rw, lit, load_global, slot, SlotCount,
    };
    use crate::{BindingSlot, LiteralValue};
    use vyre_foundation::ir::DataType;

    fn ro_binding(index: u32) -> BindingSlot {
        global_ro(index, DataType::F32, &format!("ro{index}"))
    }

    #[test]
    fn empty_kernel_has_no_candidates() {
        let desc = descriptor("k").dispatch(64, 1, 1).build();
        let p = analyze(&desc);
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn read_only_binding_with_two_loads_is_candidate() {
        let p = analyze(&kernel(ro_binding(0), 2));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].binding_slot, 0);
        assert_eq!(p.candidates[0].load_count, 2);
    }

    #[test]
    fn read_write_binding_is_not_candidate() {
        let binding = global_rw(0, DataType::F32, "rw0");
        let p = analyze(&kernel(binding, 2));
        assert!(
            p.candidates.is_empty(),
            "RW bindings can't be promoted to texture"
        );
    }

    #[test]
    fn read_only_binding_with_one_load_is_not_candidate() {
        let p = analyze(&kernel(ro_binding(0), 1));
        assert!(
            p.candidates.is_empty(),
            "single-load bindings don't gain enough"
        );
    }

    #[test]
    fn shared_memory_binding_is_not_candidate() {
        let binding = slot(
            0,
            DataType::F32,
            MemoryClass::Shared,
            BindingVisibility::ReadOnly,
            "shared",
        )
        .with_count(64);
        let p = analyze(&kernel(binding, 2));
        assert!(
            p.candidates.is_empty(),
            "shared memory isn't promotable to texture"
        );
    }

    #[test]
    fn speedup_grows_with_load_count_log2() {
        let p = analyze(&kernel(ro_binding(0), 8));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].load_count, 8);
        // 1.5 + log2(8) = 4.5
        assert!((p.candidates[0].estimated_speedup_factor - 4.5).abs() < 1e-5);
    }

    #[test]
    fn loop_bounds_are_not_treated_as_child_body_indices() {
        let desc = descriptor("k")
            .slot(ro_binding(0))
            .dispatch(32, 1, 1)
            .body(
                body()
                    .op(lit(0, 0))
                    .op(crate::descriptor_builder::effect(
                        crate::KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        [0, 0, 1],
                    ))
                    .child(body().op(load_global(0, 0, 1)))
                    .child(body())
                    .literal(LiteralValue::U32(0)),
            )
            .build();

        let p = analyze(&desc);
        assert!(
            p.candidates.is_empty(),
            "loop bound operands must not cause traversal into unrelated child bodies"
        );
    }
}
