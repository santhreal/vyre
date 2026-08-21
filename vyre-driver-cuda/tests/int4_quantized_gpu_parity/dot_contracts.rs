#![cfg(feature = "device-tests")]

use super::*;

/// Fixed-pattern packed operands for one dot lane count.
fn patterned_dot_operands(lane_count: u32) -> (Vec<u32>, Vec<u32>) {
    let lhs = cycled_rows(&WEIGHT_PATTERN, 1, lane_count, 0);
    let rhs = cycled_rows(&ACTIVATION_PATTERN, 1, lane_count, 0);
    (pack_i4x8_cpu(&lhs[0]), pack_i4x8_cpu(&rhs[0]))
}

#[test]
fn cuda_dispatch_matches_packed_int4_dot_i32_oracle() {
    let backend = cuda_backend();

    for lane_count in DOT_LANE_COUNTS {
        let (lhs_packed, rhs_packed) = patterned_dot_operands(lane_count);
        let actual = dispatch_i4_dot_i32(&backend, &lhs_packed, &rhs_packed, lane_count);
        let expected = i4x8_dot_i32_cpu(&lhs_packed, &rhs_packed, lane_count);

        assert_eq!(actual, expected, "lane_count={lane_count}");
    }
}

#[test]
fn cuda_dispatch_matches_packed_int4_scaled_dot_oracle() {
    let backend = cuda_backend();

    for lane_count in DOT_LANE_COUNTS {
        let (lhs_packed, rhs_packed) = patterned_dot_operands(lane_count);
        let lhs_scale = 0.125_f32 + (lane_count % 4) as f32 * 0.0625;
        let rhs_scale = 0.25_f32 + (lane_count % 3) as f32 * 0.125;
        assert_dot_f32_scaled_parity(
            &backend,
            &lhs_packed,
            &rhs_packed,
            lhs_scale,
            rhs_scale,
            lane_count,
            "patterned scaled dot",
        );
    }
}
