use super::*;

use vyre_libs::math::quantized::{i4x8_dot_f32_scaled, i4x8_dot_i32};
use vyre_reference::composition_witness::{
    i4x8_dot_f32_scaled_witness as i4x8_dot_f32_scaled_cpu,
    i4x8_dot_i32_witness as i4x8_dot_i32_cpu,
};

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
        let program = i4x8_dot_i32("lhs", "rhs", "out", lane_count);
        let outputs = backend
            .dispatch(
                &program,
                &[pack_u32_slice(&lhs_packed), pack_u32_slice(&rhs_packed)],
                &DispatchConfig::default(),
            )
            .expect("Fix: CUDA must execute packed INT4 dot without CPU fallback.");
        let expected = i4x8_dot_i32_cpu(&lhs_packed, &rhs_packed, lane_count);
        let actual = read_i32(&outputs[0]);

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
        let program =
            i4x8_dot_f32_scaled("lhs", "rhs", "lhs_scale", "rhs_scale", "out", lane_count);
        let outputs = backend
            .dispatch(
                &program,
                &[
                    pack_u32_slice(&lhs_packed),
                    pack_u32_slice(&rhs_packed),
                    pack_f32_slice(&[lhs_scale]),
                    pack_f32_slice(&[rhs_scale]),
                ],
                &DispatchConfig::default(),
            )
            .expect("Fix: CUDA must execute fused packed INT4 scaled dot without CPU fallback.");
        let expected =
            i4x8_dot_f32_scaled_cpu(&lhs_packed, &rhs_packed, lhs_scale, rhs_scale, lane_count);
        let actual = read_f32(&outputs[0]);

        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "lane_count={lane_count}"
        );
    }
}
