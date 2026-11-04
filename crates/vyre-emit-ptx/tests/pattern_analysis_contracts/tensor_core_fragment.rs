//! `tensor_core_fragment` pattern analysis contracts.

use vyre_emit_ptx::patterns::tensor_core_fragment::*;
use vyre_emit_ptx::ComputeCapability;
use vyre_lower::descriptor_builder::{body, descriptor, lit, op};
use vyre_lower::KernelOpKind;
use vyre_lower::{KernelDescriptor, LiteralValue};

fn fma_kernel(fma_count: u32, workgroup_x: u32) -> KernelDescriptor {
    let mut ops = vec![lit(0, 0), lit(1, 1), lit(2, 2)];
    for i in 0..fma_count {
        ops.push(op(KernelOpKind::Fma, [0, 1, 2], 3 + i));
    }
    descriptor("fma_chain")
        .dispatch(workgroup_x, 1, 1)
        .body(body().ops(ops).literals([
            LiteralValue::F32(1.0),
            LiteralValue::F32(2.0),
            LiteralValue::F32(3.0),
        ]))
        .build()
}

#[test]
fn empty_kernel_has_no_candidates() {
    let desc = descriptor("empty").dispatch(64, 1, 1).build();
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(p.candidates.is_empty());
}

#[test]
fn fma_chain_aligned_workgroup_yields_candidates_on_sm_80() {
    let desc = fma_kernel(8, 64);
    let p = analyze(&desc, ComputeCapability::SM_80);
    // sm_80 supports both F16 and BF16 fragments.
    assert_eq!(p.candidates.len(), 3); // F16_16, BF16_16, F16_8
    assert_eq!(p.target_sm, "sm_80");
}

#[test]
fn fma_chain_on_sm_70_only_offers_f16_fragments() {
    let desc = fma_kernel(8, 64);
    let p = analyze(&desc, ComputeCapability::SM_70);
    // sm_70 supports F16 fragments only, not BF16.
    let bf16_count = p
        .candidates
        .iter()
        .filter(|c| matches!(c.fragment, FragmentTile::Bf16_16x16x16))
        .count();
    assert_eq!(bf16_count, 0);
    let f16_count = p
        .candidates
        .iter()
        .filter(|c| {
            matches!(
                c.fragment,
                FragmentTile::F16_16x16x16 | FragmentTile::F16_8x8x16
            )
        })
        .count();
    assert_eq!(f16_count, 2);
}

#[test]
fn small_fma_count_below_threshold_no_candidates() {
    let desc = fma_kernel(2, 64);
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(
        p.candidates.is_empty(),
        "fewer than 4 FMAs not worth promoting"
    );
}

#[test]
fn unaligned_workgroup_no_candidates() {
    let desc = fma_kernel(8, 33); // 33 doesn't divide 16
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(p.candidates.is_empty());
}

#[test]
fn small_workgroup_no_candidates() {
    let desc = fma_kernel(8, 16); // <32, below warp size
    let p = analyze(&desc, ComputeCapability::SM_80);
    assert!(p.candidates.is_empty());
}

#[test]
fn fragment_dims_match_documented_shapes() {
    assert_eq!(FragmentTile::F16_16x16x16.dims(), (16, 16, 16));
    assert_eq!(FragmentTile::Bf16_16x16x16.dims(), (16, 16, 16));
    assert_eq!(FragmentTile::F16_8x8x16.dims(), (8, 8, 16));
}

#[test]
fn f16_fragment_supported_on_sm_70_plus() {
    assert!(FragmentTile::F16_16x16x16.supported_on(ComputeCapability::SM_70));
    assert!(FragmentTile::F16_16x16x16.supported_on(ComputeCapability::SM_90));
}

#[test]
fn bf16_fragment_only_on_sm_80_plus() {
    assert!(!FragmentTile::Bf16_16x16x16.supported_on(ComputeCapability::SM_70));
    assert!(!FragmentTile::Bf16_16x16x16.supported_on(ComputeCapability::SM_75));
    assert!(FragmentTile::Bf16_16x16x16.supported_on(ComputeCapability::SM_80));
}

#[test]
fn speedup_grows_with_log_fma_count() {
    let desc = fma_kernel(16, 64);
    let p = analyze(&desc, ComputeCapability::SM_80);
    // 5.0 + log2(16) = 9.0
    assert!((p.candidates[0].estimated_speedup_factor - 9.0).abs() < 1e-5);
}

#[test]
fn target_sm_string_formatted_correctly() {
    let desc = fma_kernel(8, 64);
    for (target, expected) in [
        (ComputeCapability::SM_70, "sm_70"),
        (ComputeCapability::SM_75, "sm_75"),
        (ComputeCapability::SM_80, "sm_80"),
        (ComputeCapability::SM_89, "sm_89"),
        (ComputeCapability::SM_90, "sm_90"),
        (ComputeCapability::SM_100, "sm_100"),
        (ComputeCapability::SM_120, "sm_120"),
    ] {
        let p = analyze(&desc, target);
        assert_eq!(p.target_sm, expected);
    }
}
