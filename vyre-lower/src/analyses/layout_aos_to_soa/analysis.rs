//! Walk every binding; if its dtype is a compound type (`Vec`,
//! `TensorShaped`, fixed-size `Array`) AND it has multiple loads,
//! flag it as a layout-transform candidate.

use super::plan::{LayoutCandidate, LayoutTransformPlan};
use crate::analyses::load_counts::count_global_loads_by_slot;
use crate::KernelDescriptor;
use rustc_hash::FxHashMap;
use vyre_foundation::ir::DataType;

/// Analyze compound bindings for array-of-structures to structure-of-arrays conversion.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> LayoutTransformPlan {
    let compound: FxHashMap<u32, u32> = desc
        .bindings
        .slots
        .iter()
        .filter_map(|b| compound_lane_count(&b.element_type).map(|c| (b.slot, c)))
        .collect();

    let mut load_counts: FxHashMap<u32, u32> =
        FxHashMap::with_capacity_and_hasher(compound.len(), Default::default());
    count_global_loads_by_slot(
        &desc.body,
        &|slot| compound.contains_key(&slot),
        &mut load_counts,
    );

    let mut candidates = Vec::new();
    for (slot, count) in load_counts {
        if count >= 2 {
            let component_count = *compound.get(&slot).unwrap_or(&1);
            let speedup = 1.0 + (component_count.saturating_sub(1) as f32) * 0.3;
            candidates.push(LayoutCandidate {
                binding_slot: slot,
                load_count: count,
                component_count,
                estimated_speedup_factor: speedup,
            });
        }
    }
    candidates.sort_unstable_by_key(|candidate| candidate.binding_slot);

    LayoutTransformPlan {
        kernel_id: desc.id.clone(),
        candidates,
    }
}

/// Return the lane / component count for a compound dtype, or `None`
/// for scalars (which are already SoA-friendly).
fn compound_lane_count(dtype: &DataType) -> Option<u32> {
    match dtype {
        DataType::Vec { count, .. } => Some(*count as u32),
        DataType::Vec2U32 => Some(2),
        DataType::Vec4U32 => Some(4),
        DataType::TensorShaped { shape, .. } if !shape.is_empty() => {
            // Use the innermost dimension as the lane count for AoS→SoA.
            shape.last().copied()
        }
        DataType::Array { .. } => Some(2), // Array is the AoS shape itself; conservative split count.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyses::load_counts::fixtures::kernel;
    use crate::descriptor_builder::{body, descriptor, effect, global_ro, load_global};
    use crate::{BindingSlot, KernelOpKind};

    fn vec4_binding(index: u32) -> BindingSlot {
        global_ro(
            index,
            DataType::Vec {
                element: Box::new(DataType::F32),
                count: 4,
            },
            &format!("v{index}"),
        )
    }

    #[test]
    fn empty_kernel_has_no_candidates() {
        let desc = descriptor("k").dispatch(64, 1, 1).build();
        assert!(analyze(&desc).candidates.is_empty());
    }

    #[test]
    fn scalar_binding_is_not_candidate() {
        let binding = global_ro(0, DataType::F32, "s0");
        assert!(analyze(&kernel(binding, 2)).candidates.is_empty());
    }

    #[test]
    fn vec4_binding_with_two_loads_is_candidate() {
        let p = analyze(&kernel(vec4_binding(0), 2));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].component_count, 4);
        assert_eq!(p.candidates[0].load_count, 2);
        // 1.0 + (4-1)*0.3 = 1.9
        assert!((p.candidates[0].estimated_speedup_factor - 1.9).abs() < 1e-5);
    }

    #[test]
    fn vec4_binding_with_one_load_is_not_candidate() {
        assert!(analyze(&kernel(vec4_binding(0), 1)).candidates.is_empty());
    }

    #[test]
    fn structured_if_else_counts_both_load_branches() {
        let desc = descriptor("k")
            .slot(vec4_binding(0))
            .dispatch(32, 1, 1)
            .body(
                body()
                    .op(effect(KernelOpKind::StructuredIfThenElse, [99, 0, 1]))
                    .child(body().op(load_global(0, 0, 1)))
                    .child(body().op(load_global(0, 0, 2))),
            )
            .build();

        let plan = analyze(&desc);

        assert_eq!(plan.candidates.len(), 1);
        assert_eq!(plan.candidates[0].load_count, 2);
    }

    #[test]
    fn vec2u32_recognized_as_compound() {
        assert_eq!(compound_lane_count(&DataType::Vec2U32), Some(2));
    }

    #[test]
    fn vec4u32_recognized_as_compound() {
        assert_eq!(compound_lane_count(&DataType::Vec4U32), Some(4));
    }

    #[test]
    fn scalar_types_return_none() {
        assert_eq!(compound_lane_count(&DataType::F32), None);
        assert_eq!(compound_lane_count(&DataType::U32), None);
        assert_eq!(compound_lane_count(&DataType::Bool), None);
    }
}
