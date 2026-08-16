//! Contracts for `vyre_driver::numeric`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::numeric::{
    align_up_u64, align_up_usize, checked_ceil_div_u64, checked_compose_basis_points_u64,
    checked_dim_product_u32, checked_dim_product_u64, compose_basis_points_u32,
    finite_f64_ratio_basis_points_round, finite_f64_ratio_basis_points_trunc,
    finite_f64_to_u32_round, finite_f64_to_u32_trunc, finite_f64_unit_basis_points_trunc,
    ratio_basis_points_u64, ratio_basis_points_u64_wide, ratio_parts_per_million_u64,
    rounded_f64_to_u64, scale_u64_by_basis_points_floor_min,
    scale_u64_by_basis_points_round_clamped, u128_to_u64, usize_to_u64,
    weighted_u64_by_basis_points_u128, BackendNumericPolicy,
};

#[test]
fn usize_boundary_accepts_fit_values() {
    assert_eq!(usize_to_u64(17, "bytes", "test").unwrap(), 17);
}

#[test]
fn backend_numeric_policy_carries_backend_label_without_local_wrappers() {
    let policy = BackendNumericPolicy::new("CUDA");
    assert_eq!(policy.backend(), "CUDA");
    assert_eq!(policy.usize_to_u64(17, "bytes").unwrap(), 17);
    assert_eq!(policy.ratio_basis_points_u64(1, 4, 0, "pressure"), 2_500);
    assert_eq!(
        policy.finite_f64_ratio_basis_points_round(1.0, 6.0, 99, 77, "ratio"),
        1_667
    );
    assert_eq!(policy.checked_ceil_div_u64(65_537, 65_536), Some(2));
    assert_eq!(
        policy.checked_dim_product_u64([65_535, 2, 3]),
        Some(393_210)
    );
    assert_eq!(
        policy.checked_dim_product_u32([65_535, 2, 3]),
        Some(393_210)
    );

    let err = policy
        .u128_to_u64(u128::from(u64::MAX) + 1, "resident bytes")
        .unwrap_err();
    let rendered = err.to_string();
    assert!(
        rendered.contains("CUDA resident bytes"),
        "backend policy diagnostics must carry the backend label and boundary name: {rendered}"
    );
}

#[test]
fn u128_boundary_rejects_overflow_with_backend_label() {
    let err = u128_to_u64(u128::from(u64::MAX) + 1, "counter", "test").unwrap_err();
    let rendered = err.to_string();
    assert!(
        rendered.contains("test counter"),
        "numeric boundary diagnostics must identify the backend and label: {rendered}"
    );
}

#[test]
fn rounded_f64_rejects_non_finite_values() {
    let err = rounded_f64_to_u64(f64::NAN, "timestamp", "test").unwrap_err();
    let rendered = err.to_string();
    assert!(
        rendered.contains("timestamp"),
        "rounded timestamp diagnostics must include the failing label: {rendered}"
    );
}

#[test]
fn ratio_basis_points_preserves_zero_denominator_policy() {
    assert_eq!(
        ratio_basis_points_u64(1, 0, u32::MAX, "pressure", "test"),
        u32::MAX
    );
    assert_eq!(ratio_basis_points_u64(0, 0, 0, "savings", "test"), 0);
}

#[test]
fn ratio_basis_points_uses_wide_arithmetic_before_clamping() {
    assert_eq!(
        ratio_basis_points_u64(u64::MAX, u64::MAX / 2, 0, "wide", "test"),
        20_000
    );
    assert_eq!(
        ratio_basis_points_u64(u64::MAX, 1, 0, "overflow", "test"),
        u32::MAX
    );
}

#[test]
fn wide_ratio_basis_points_retains_u64_telemetry_domain() {
    assert_eq!(ratio_basis_points_u64_wide(3, 2, 0, "wide", "test"), 15_000);
    assert_eq!(
        ratio_basis_points_u64_wide(u64::MAX, u64::MAX / 4, 0, "wide", "test"),
        40_000
    );
    assert_eq!(
        ratio_basis_points_u64_wide(u64::MAX, 1, 0, "overflow", "test"),
        u64::MAX
    );
}

#[test]
fn finite_f64_to_u32_helpers_pin_invalid_values() {
    assert_eq!(finite_f64_to_u32_trunc(12.9, "value", "test"), 12);
    assert_eq!(finite_f64_to_u32_round(12.5, "value", "test"), 13);
    assert_eq!(finite_f64_to_u32_trunc(-1.0, "value", "test"), 0);
    assert_eq!(
        finite_f64_to_u32_round(f64::INFINITY, "value", "test"),
        u32::MAX
    );
    assert_eq!(
        finite_f64_to_u32_trunc(f64::from(u32::MAX) * 2.0, "value", "test"),
        u32::MAX
    );
}

#[test]
fn finite_f64_basis_point_helpers_pin_invalid_policies() {
    assert_eq!(
        finite_f64_ratio_basis_points_trunc(1.0, 4.0, 99, 77, "ratio", "test"),
        2_500
    );
    assert_eq!(
        finite_f64_ratio_basis_points_round(1.0, 6.0, 99, 77, "ratio", "test"),
        1_667
    );
    assert_eq!(
        finite_f64_ratio_basis_points_trunc(f64::NAN, 1.0, 99, 77, "ratio", "test"),
        99
    );
    assert_eq!(
        finite_f64_ratio_basis_points_trunc(1.0, 0.0, 99, 77, "ratio", "test"),
        77
    );
    assert_eq!(
        finite_f64_ratio_basis_points_round(-1.0, 1.0, 99, 77, "ratio", "test"),
        0
    );
    assert_eq!(
        finite_f64_unit_basis_points_trunc(0.25, 33, "unit", "test"),
        2_500
    );
    assert_eq!(
        finite_f64_unit_basis_points_trunc(f64::INFINITY, 33, "unit", "test"),
        33
    );
}

#[test]
fn alignment_helpers_pad_minimums_and_reject_overflow() {
    assert_eq!(align_up_u64(0, 4, 4, "copy", "test").unwrap(), 4);
    assert_eq!(align_up_u64(5, 4, 0, "copy", "test").unwrap(), 8);
    assert_eq!(align_up_usize(0, 4, 4, "copy", "test").unwrap(), 4);
    assert_eq!(align_up_usize(5, 4, 0, "copy", "test").unwrap(), 8);

    let zero_alignment = align_up_u64(1, 0, 0, "copy", "test").unwrap_err();
    assert!(
        zero_alignment
            .to_string()
            .contains("alignment must be non-zero"),
        "zero-alignment diagnostics must be actionable: {zero_alignment}"
    );

    let overflow_u64 = align_up_u64(u64::MAX, 4, 0, "copy", "test").unwrap_err();
    assert!(
        overflow_u64.to_string().contains("overflows u64"),
        "u64 alignment overflow diagnostics must name the target type: {overflow_u64}"
    );

    let overflow_usize = align_up_usize(usize::MAX, 4, 0, "copy", "test").unwrap_err();
    assert!(
        overflow_usize.to_string().contains("overflows usize"),
        "usize alignment overflow diagnostics must name the target type: {overflow_usize}"
    );
}

#[test]
fn checked_ceil_div_u64_handles_cuda_queue_boundaries() {
    assert_eq!(checked_ceil_div_u64(0, 64), Some(0));
    assert_eq!(checked_ceil_div_u64(1, 64), Some(1));
    assert_eq!(checked_ceil_div_u64(65_537, 65_536), Some(2));
    assert_eq!(
        checked_ceil_div_u64(u64::MAX, 65_536),
        Some(281_474_976_710_656)
    );
    assert_eq!(checked_ceil_div_u64(u64::MAX, 1), Some(u64::MAX));
    assert_eq!(checked_ceil_div_u64(1, 0), None);
}

#[test]
fn checked_dim_product_helpers_cover_cuda_launch_boundaries() {
    assert_eq!(checked_dim_product_u64([1, 1, 1]), Some(1));
    assert_eq!(checked_dim_product_u64([0, 999, 999]), Some(0));
    assert_eq!(checked_dim_product_u64([65_535, 2, 3]), Some(393_210));
    assert_eq!(checked_dim_product_u32([65_535, 2, 3]), Some(393_210));
    assert_eq!(
        checked_dim_product_u64([u32::MAX, u32::MAX, u32::MAX]),
        None
    );
    assert_eq!(checked_dim_product_u32([u32::MAX, 2, 1]), None);
}

#[test]
fn generated_dim_product_matrix_matches_wide_integer_reference() {
    const VALUES: [u32; 9] = [0, 1, 2, 3, 7, 32, 255, 65_535, u32::MAX];
    for x in VALUES {
        for y in VALUES {
            for z in VALUES {
                let wide = u128::from(x) * u128::from(y) * u128::from(z);
                let expected_u64 = u64::try_from(wide).ok();
                let expected_u32 = u32::try_from(wide).ok();
                assert_eq!(checked_dim_product_u64([x, y, z]), expected_u64);
                assert_eq!(checked_dim_product_u32([x, y, z]), expected_u32);
            }
        }
    }
}

#[test]
fn ratio_parts_per_million_uses_wide_arithmetic_and_pins_overflow() {
    assert_eq!(
        ratio_parts_per_million_u64(1, 4, 0, "commit-rate", "test"),
        250_000
    );
    assert_eq!(
        ratio_parts_per_million_u64(1, 0, 7, "commit-rate", "test"),
        7
    );
    assert_eq!(
        ratio_parts_per_million_u64(u64::MAX, 1, 0, "commit-rate", "test"),
        u32::MAX
    );
}

#[test]
fn basis_point_composition_and_scaling_helpers_are_widened() {
    assert_eq!(
        compose_basis_points_u32(15_000, 2_500, "compose", "test"),
        3_750
    );
    assert_eq!(
        compose_basis_points_u32(u32::MAX, u32::MAX, "compose", "test"),
        u32::MAX
    );
    assert_eq!(
        checked_compose_basis_points_u64(50_000, 20_000),
        Some(100_000)
    );
    assert_eq!(checked_compose_basis_points_u64(u64::MAX, u64::MAX), None);
    assert_eq!(
        scale_u64_by_basis_points_round_clamped(10, 1_000_000, 10, 40_000, "scale", "test"),
        40
    );
    assert_eq!(
        scale_u64_by_basis_points_round_clamped(7, 0, 7, 40_000, "scale", "test"),
        7
    );
    assert_eq!(
        scale_u64_by_basis_points_floor_min(1, 1, 1, "scale", "test"),
        1
    );
    assert_eq!(
        weighted_u64_by_basis_points_u128(u64::MAX, 10_000),
        u128::from(u64::MAX)
    );
}
