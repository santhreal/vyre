//! Analysis pass: walk a `KernelDescriptor`, count per-binding access
//! frequency, identify shared-memory promotion candidates.
//!
//! Phase 1 algorithm:
//!
//! 1. Count `LoadGlobal` ops per binding slot. Stores don't qualify
//!    for the read-side promotion (a store's destination is the final
//!    output; promoting writes needs different infrastructure).
//! 2. Sum up: a binding with `access_count >= 2` is a candidate.
//! 3. Estimate tile size as `distinct_indices * bytes_per_element`.
//!    For phase 1, `distinct_indices = workgroup_size_x` (each thread
//!    accesses one element from the tile). Phase 2 refines this for
//!    multi-element-per-thread access patterns.
//! 4. Compute total tile bytes; flag whether it fits in the budget.
//!
//! Budget defaults to `DEFAULT_SHARED_BUDGET_BYTES`. Caller can pass
//! a smaller budget via `analyze_with_budget` for substrate-specific
//! limits.

use super::plan::{PromotionCandidate, PromotionPlan};
use super::DEFAULT_SHARED_BUDGET_BYTES;
use crate::analyses::load_counts::count_global_loads_by_slot;
use crate::{KernelBody, KernelDescriptor};
use rustc_hash::FxHashMap;
use vyre_foundation::ir::DataType;

/// Run promotion analysis with the default 48 KiB budget.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> PromotionPlan {
    analyze_with_budget(desc, DEFAULT_SHARED_BUDGET_BYTES)
}

/// Run promotion analysis with an explicit per-workgroup
/// shared-memory budget (in bytes).
#[must_use]
pub fn analyze_with_budget(desc: &KernelDescriptor, budget_bytes: u32) -> PromotionPlan {
    let access_counts = access_counts(&desc.body);

    let workgroup_size = desc.dispatch.workgroup_size[0].max(1);
    let mut candidates = Vec::new();
    let mut total_tile_bytes: u32 = 0;
    for (slot, count) in &access_counts {
        if *count < 2 {
            continue;
        }
        let binding = match desc.bindings.slots.iter().find(|b| b.slot == *slot) {
            Some(b) => b,
            None => continue,
        };
        let bpe = bytes_per_element(&binding.element_type);
        let distinct = workgroup_size;
        let tile_bytes = bpe.saturating_mul(distinct);
        let speedup = 5.0 + ((*count as f32 - 1.0) * 2.0);
        candidates.push(PromotionCandidate {
            binding_slot: *slot,
            access_count: *count,
            bytes_per_element: bpe,
            distinct_indices_per_workgroup: distinct,
            tile_bytes,
            estimated_speedup_factor: speedup,
        });
        total_tile_bytes = total_tile_bytes.saturating_add(tile_bytes);
    }
    PromotionPlan {
        kernel_id: desc.id.clone(),
        candidates,
        total_tile_bytes,
        budget_bytes,
    }
}

/// Load counts for every slot the body reads.
///
/// No eligibility filter: a slot the layout does not declare is dropped by the
/// candidate loop below, which keeps the count and the binding lookup in one
/// place instead of two.
fn access_counts(body: &KernelBody) -> FxHashMap<u32, u32> {
    let mut counts = FxHashMap::default();
    count_global_loads_by_slot(body, &|_| true, &mut counts);
    counts
}

fn bytes_per_element(t: &DataType) -> u32 {
    match t {
        DataType::Bool | DataType::U8 | DataType::I8 => 1,
        DataType::U16 | DataType::I16 | DataType::F16 | DataType::BF16 => 2,
        DataType::U32 | DataType::I32 | DataType::F32 | DataType::Handle(_) => 4,
        DataType::U64 | DataType::I64 | DataType::F64 | DataType::Vec2U32 => 8,
        DataType::Vec4U32 => 16,
        DataType::Bytes => 1,
        DataType::Array { element_size } => (*element_size).try_into().unwrap_or(u32::MAX),
        DataType::Vec { element, count } => {
            bytes_per_element(element).saturating_mul(u32::from(*count))
        }
        DataType::TensorShaped { element, .. }
        | DataType::SparseCsr { element }
        | DataType::SparseCoo { element }
        | DataType::SparseBsr { element, .. } => bytes_per_element(element),
        DataType::F8E4M3 | DataType::F8E5M2 | DataType::I4 | DataType::FP4 | DataType::NF4 => 1,
        DataType::Tensor | DataType::DeviceMesh { .. } | DataType::Opaque(_) => 4,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyses::load_counts::fixtures::literal_then_loads;
    use crate::descriptor_builder::{
        body, descriptor, effect, for_loop, if_then, lit, load_global, op, store_global,
    };
    use crate::{BindingSlot, KernelBody, KernelDescriptor, KernelOpKind, LiteralValue};
    use rustc_hash::FxHashMap;

    /// A read-only global with a runtime element count, the shape the
    /// precondition reads.
    fn binding(slot: u32, element_type: DataType) -> BindingSlot {
        crate::descriptor_builder::global_ro(slot, element_type, &format!("b{slot}"))
    }

    /// A kernel over one binding read `loads` times. The op shape is owned by
    /// `load_counts`, whose traversal defines what a load site looks like; only
    /// the workgroup width and the element type vary between cases.
    fn reads(workgroup_x: u32, element_type: DataType, loads: u32) -> KernelDescriptor {
        kernel(
            workgroup_x,
            vec![binding(0, element_type)],
            literal_then_loads(loads),
        )
    }

    /// A kernel over explicit bindings and an explicit body.
    fn kernel(
        workgroup_x: u32,
        slots: Vec<BindingSlot>,
        body: impl Into<KernelBody>,
    ) -> KernelDescriptor {
        descriptor("k")
            .slots(slots)
            .dispatch(workgroup_x, 1, 1)
            .body(body)
            .build()
    }

    // Positive truth (candidate detected)

    #[test]
    fn positive_buffer_read_twice_is_candidate() {
        let p = analyze(&reads(64, DataType::F32, 2));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].binding_slot, 0);
        assert_eq!(p.candidates[0].access_count, 2);
        assert_eq!(p.candidates[0].bytes_per_element, 4);
        assert_eq!(p.candidates[0].distinct_indices_per_workgroup, 64);
        assert_eq!(p.candidates[0].tile_bytes, 256);
    }

    #[test]
    fn positive_two_buffers_each_read_multiple_times_both_candidates() {
        let p = analyze(&kernel(
            32,
            vec![binding(0, DataType::F32), binding(1, DataType::U32)],
            body()
                .op(lit(0, 0))
                .op(load_global(0, 0, 1))
                .op(load_global(0, 0, 2))
                .op(load_global(1, 0, 3))
                .op(load_global(1, 0, 4))
                .op(load_global(1, 0, 5))
                .literal(LiteralValue::U32(0)),
        ));
        assert_eq!(p.candidates.len(), 2);
        let by_slot: FxHashMap<_, _> = p.candidates.iter().map(|c| (c.binding_slot, c)).collect();
        assert_eq!(by_slot[&0].access_count, 2);
        assert_eq!(by_slot[&1].access_count, 3);
    }

    #[test]
    fn positive_speedup_grows_with_access_count() {
        let p = analyze(&reads(32, DataType::F32, 5));
        // 5 + (5 - 1) * 2 = 13
        assert!((p.candidates[0].estimated_speedup_factor - 13.0).abs() < 1e-5);
    }

    // Negative precision (rule does NOT fire)

    #[test]
    fn negative_buffer_read_only_once_not_a_candidate() {
        let p = analyze(&reads(64, DataType::F32, 1));
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn negative_store_only_buffer_not_a_candidate() {
        let p = analyze(&kernel(
            64,
            vec![binding(0, DataType::F32)],
            body()
                .op(lit(0, 0))
                .op(store_global(0, 0, 0))
                .op(store_global(0, 0, 0))
                .literal(LiteralValue::U32(0)),
        ));
        assert!(p.candidates.is_empty());
    }

    #[test]
    fn negative_no_global_accesses_yields_empty_plan() {
        let p = analyze(&kernel(
            64,
            vec![binding(0, DataType::F32)],
            body()
                .op(op(KernelOpKind::LocalInvocationId, [], 0))
                .op(lit(0, 1))
                .op(crate::descriptor_builder::binop(
                    vyre_foundation::ir::BinOp::Add,
                    0,
                    1,
                    2,
                ))
                .literal(LiteralValue::U32(0)),
        ));
        assert!(p.candidates.is_empty());
        assert_eq!(p.total_tile_bytes, 0);
    }

    // Adversarial / boundary

    /// Two loads of one binding, in a body a structured op names as its child.
    fn nested_reads(workgroup_x: u32, parent: impl Into<KernelBody>) -> KernelDescriptor {
        kernel(workgroup_x, vec![binding(0, DataType::F32)], parent)
    }

    #[test]
    fn adversarial_load_inside_if_body_counted() {
        let p = analyze(&nested_reads(
            32,
            body()
                .op(lit(0, 0))
                .op(if_then(0, 0))
                .child(literal_then_loads(2))
                .literal(LiteralValue::Bool(true)),
        ));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].access_count, 2);
    }

    #[test]
    fn adversarial_load_inside_loop_body_counted() {
        let p = analyze(&nested_reads(
            16,
            body()
                .op(lit(0, 0))
                .op(lit(0, 1))
                .op(for_loop("", 0, 1, 0))
                .child(literal_then_loads(2))
                .literal(LiteralValue::U32(0)),
        ));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].access_count, 2);
    }

    #[test]
    fn adversarial_zero_workgroup_size_clamped_to_one() {
        // Defensive: workgroup_size_x = 0 should not crash and not
        // produce zero-byte tile sizes.
        let p = analyze(&reads(0, DataType::F32, 2));
        assert_eq!(p.candidates.len(), 1);
        assert_eq!(p.candidates[0].distinct_indices_per_workgroup, 1);
        assert_eq!(p.candidates[0].tile_bytes, 4);
    }

    #[test]
    fn adversarial_load_with_no_operands_skipped_safely() {
        let p = analyze(&kernel(
            32,
            vec![binding(0, DataType::F32)],
            body().op(effect(KernelOpKind::LoadGlobal, [])),
        ));
        // No operand → skipped, not counted.
        assert!(p.candidates.is_empty());
    }

    // Budget behavior

    #[test]
    fn fits_in_budget_when_sum_below_limit() {
        let p = analyze_with_budget(&reads(32, DataType::F32, 2), 4096);
        assert!(p.fits_in_budget());
    }

    #[test]
    fn does_not_fit_when_sum_exceeds_budget() {
        // 8 bpe * 1024 wg = 8192
        let p = analyze_with_budget(&reads(1024, DataType::F64, 2), 4096);
        assert!(!p.fits_in_budget());
    }

    // Element-type bytes

    #[test]
    fn bytes_per_element_for_each_scalar_type() {
        // Pin the table  -  if anyone changes DataType variants, this
        // test forces them to update bytes_per_element too.
        for (ty, expected) in [
            (DataType::Bool, 1u32),
            (DataType::U8, 1),
            (DataType::I8, 1),
            (DataType::U16, 2),
            (DataType::I16, 2),
            (DataType::F16, 2),
            (DataType::U32, 4),
            (DataType::I32, 4),
            (DataType::F32, 4),
            (DataType::U64, 8),
            (DataType::I64, 8),
            (DataType::F64, 8),
        ] {
            assert_eq!(super::bytes_per_element(&ty), expected, "for {ty:?}");
        }
    }

    #[test]
    fn report_kernel_id_echoes_descriptor_id() {
        let p = analyze(&kernel(32, vec![], body()));
        assert_eq!(p.kernel_id, "k");
    }
}
