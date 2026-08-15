use super::*;

#[test]
fn i4x8_matvec_f32_scaled_via_dispatches_signed_boundary_rows() {
    let rows = 3_u32;
    let cols = 9_u32;
    let row0 = [-8, -7, -1, 0, 1, 2, 6, 7, -3];
    let row1 = [7, 6, 2, 1, 0, -1, -7, -8, 3];
    let row2 = [-4, 5, -6, 4, -2, 3, -5, 2, 1];
    let mut weights = Vec::new();
    for row in [&row0[..], &row1[..], &row2[..]] {
        weights.extend(pack_i4x8_cpu(row));
    }
    let x = [0.5, -1.0, 2.0, -0.25, 0.75, -1.5, 1.25, 0.125, -0.875];
    let row_scales = [0.125, 0.25, 0.5];

    let out = i4x8_matvec_f32_scaled_via(
        &QuantizedMatvecDispatcher,
        &weights,
        &x,
        &row_scales,
        rows,
        cols,
    )
    .expect("Fix: CUDA parity tests require backend dispatch; skip test if GPU unavailable, do not panic - fake dispatcher computes scaled INT4 matvec");
    let expected = i4x8_matvec_f32_scaled_cpu(&weights, &x, &row_scales, rows, cols);

    assert_eq!(out.len(), rows as usize);
    assert_f32_bits_eq(&out, &expected, "INT4 matvec");
}

#[test]
fn i4x8_matvec_f32_scaled_via_reuses_cached_program_for_same_shape() {
    let rows = 2_u32;
    let weights = pack_i4x8_cpu(&[-8, -1, 0, 7, 3, -3, 6, -6, 7, 1, -1, -8, 2, -2, 5, -5]);
    let x = [1.0, -1.0, 0.5, 0.25, -0.75, 1.5, -0.5, 2.0];
    let row_scales = [0.25, 0.5];
    let mut changed_weights = Vec::new();
    changed_weights.extend(pack_i4x8_cpu(&[-8, -1, 0, 7, 3, -3, 6, -6, 4]));
    changed_weights.extend(pack_i4x8_cpu(&[7, 1, -1, -8, 2, -2, 5, -5, 3]));
    let changed_x = [1.0, -1.0, 0.5, 0.25, -0.75, 1.5, -0.5, 2.0, 0.125];
    let mut out = Vec::new();

    assert_program_cache_keys_on_shape(
        "INT4 matvec",
        "rows/cols",
        |scratch: &QuantizedMatvecGpuScratch| scratch.program_cache.builds(),
        |scratch, changed| {
            let (weights, x, cols) = if changed {
                (&changed_weights[..], &changed_x[..], 9)
            } else {
                (&weights[..], &x[..], 8)
            };
            i4x8_matvec_f32_scaled_via_with_scratch_into(
                &QuantizedMatvecDispatcher,
                weights,
                x,
                &row_scales,
                rows,
                cols,
                scratch,
                &mut out,
            )
            .expect("Fix: fake matvec dispatcher must complete every shape this cache test drives");
        },
    );
}

#[test]
fn i4x8_matvec_f32_scaled_via_rejects_shape_errors_before_dispatch() {
    let weights = pack_i4x8_cpu(&[-1, 2, 3, -4, 5, -6, 7, -8]);
    let x = [1.0; 8];
    let row_scales = [0.5];

    let err =
        i4x8_matvec_f32_scaled_via(&QuantizedMatvecDispatcher, &weights, &x, &row_scales, 0, 8)
            .expect_err("zero rows must fail");
    assert!(err.to_string().contains("rows > 0 and cols > 0"));

    let err = i4x8_matvec_f32_scaled_via(&QuantizedMatvecDispatcher, &[], &x, &row_scales, 1, 8)
        .expect_err("missing weights must fail");
    assert!(err.to_string().contains("weights_packed.len()"));

    let err = i4x8_matvec_f32_scaled_via(
        &QuantizedMatvecDispatcher,
        &weights,
        &x[..7],
        &row_scales,
        1,
        8,
    )
    .expect_err("short x must fail");
    assert!(err.to_string().contains("x.len() == cols"));

    let err = i4x8_matvec_f32_scaled_via(&QuantizedMatvecDispatcher, &weights, &x, &[], 1, 8)
        .expect_err("missing scale must fail");
    assert!(err.to_string().contains("row_scales.len() == rows"));
}

