use super::generated_contracts::generated_i4_values;
use super::*;

#[test]
fn i4x8_dot_f32_scaled_via_dispatches_signed_boundary_accumulators() {
    let lhs_values = [-8, -7, -1, 0, 1, 2, 6, 7];
    let rhs_values = [7, 6, 2, 1, 0, -1, -7, -8];
    let lhs = pack_i4x8_cpu(&lhs_values);
    let rhs = pack_i4x8_cpu(&rhs_values);
    let lhs_scale = 0.125;
    let rhs_scale = 0.25;

    let out = i4x8_dot_f32_scaled_via(
        &QuantizedDotDispatcher, &crate::test_parity_oracles::policy(),
        &lhs,
        &rhs,
        lhs_scale,
        rhs_scale,
        lhs_values.len() as u32,
    )
    .expect("Fix: CUDA parity tests require backend dispatch; skip test if GPU unavailable, do not panic - fake dispatcher computes scaled INT4 dot");
    let expected =
        i4x8_dot_f32_scaled_cpu(&lhs, &rhs, lhs_scale, rhs_scale, lhs_values.len() as u32);

    assert_eq!(out.to_bits(), expected.to_bits());
}

#[test]
fn i4x8_dot_f32_scaled_via_reuses_cached_program_for_same_lane_shape() {
    let lhs8 = pack_i4x8_cpu(&[-8, -1, 0, 7, 3, -3, 6, -6]);
    let rhs8 = pack_i4x8_cpu(&[7, 1, -1, -8, 2, -2, 5, -5]);
    let lhs9 = pack_i4x8_cpu(&[-8, -1, 0, 7, 3, -3, 6, -6, 4]);
    let rhs9 = pack_i4x8_cpu(&[7, 1, -1, -8, 2, -2, 5, -5, 3]);
    let mut out = Vec::with_capacity(1);

    assert_program_cache_keys_on_shape(
        "INT4 dot",
        "lane_count",
        |scratch: &QuantizedDotGpuScratch| scratch.program_cache.builds(),
        |scratch, changed| {
            let (lhs, rhs, lane_count) = if changed {
                (&lhs9[..], &rhs9[..], 9)
            } else {
                (&lhs8[..], &rhs8[..], 8)
            };
            i4x8_dot_f32_scaled_via_with_scratch_into(
                &QuantizedDotDispatcher,
                &crate::test_parity_oracles::policy(),
                lhs,
                rhs,
                0.5,
                0.25,
                lane_count,
                scratch,
                &mut out,
            )
            .expect(
                "Fix: fake dot dispatcher must complete every lane shape this cache test drives",
            );
        },
    );
}

#[test]
fn generated_i4x8_dot_hot_warm_cache_survives_alternating_shapes() {
    let mut scratch = QuantizedDotGpuScratch::default();
    let mut out = Vec::with_capacity(1);

    for seed in 0..8192u32 {
        let lane_count = if seed % 2 == 0 { 8 } else { 16 };
        let lhs_values = generated_i4_values(lane_count as usize, seed ^ 0x9e37_79b9);
        let rhs_values = generated_i4_values(lane_count as usize, seed ^ 0x85eb_ca6b);
        let lhs = pack_i4x8_cpu(&lhs_values);
        let rhs = pack_i4x8_cpu(&rhs_values);
        let lhs_scale = 0.03125 * f32::from((seed % 7) as u8 + 1);
        let rhs_scale = 0.015625 * f32::from((seed % 5) as u8 + 1);

        i4x8_dot_f32_scaled_via_with_scratch_into(
            &QuantizedDotDispatcher, &crate::test_parity_oracles::policy(),
            &lhs,
            &rhs,
            lhs_scale,
            rhs_scale,
            lane_count,
            &mut scratch,
            &mut out,
        )
        .expect("Fix: CUDA parity tests require backend dispatch; skip test if GPU unavailable, do not panic - generated alternating INT4 dot dispatch should succeed");

        let expected = i4x8_dot_f32_scaled_cpu(&lhs, &rhs, lhs_scale, rhs_scale, lane_count);
        assert_eq!(
            out[0].to_bits(),
            expected.to_bits(),
            "seed={seed} lane_count={lane_count}"
        );
    }

    assert_eq!(
            scratch.program_cache.builds(),
            2,
            "Fix: alternating two INT4 dot lane shapes must stay in the hot/warm ProgramCache instead of rebuilding every dispatch."
        );
}

#[test]
fn i4x8_dot_f32_scaled_via_rejects_bad_shape_before_dispatch() {
    let err = i4x8_dot_f32_scaled_via(
        &QuantizedDotDispatcher,
        &crate::test_parity_oracles::policy(),
        &[0],
        &[0],
        1.0,
        1.0,
        0,
    )
    .expect_err("zero lanes must fail");
    assert!(err.to_string().contains("lane_count > 0"));

    let err = i4x8_dot_f32_scaled_via(
        &QuantizedDotDispatcher,
        &crate::test_parity_oracles::policy(),
        &[],
        &[0],
        1.0,
        1.0,
        8,
    )
    .expect_err("missing lhs packed word must fail");
    assert!(err.to_string().contains("packed lengths"));
}
