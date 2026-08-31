//! Integration test crate for the containing Vyre package.

use super::*;

#[test]
fn parity_rms_norm_linear_matches_reference_three_sizes() {
    for (n, in_dim, out_dim) in [(4_u32, 4_u32, 4_u32), (16, 64, 16), (64, 128, 64)] {
        parity_case(n, in_dim, out_dim);
    }
}

#[test]
fn try_rms_norm_linear_rejects_bad_dimensions_without_panic() {
    assert!(matches!(
        try_rms_norm_linear("input", "w", "b", "out", 0, 4, 4, 1e-5),
        Err(crate::plumbing::operand::tensor_ref::TensorRefError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        try_rms_norm_linear("input", "w", "b", "out", 8, 4, 4, 1e-5),
        Err(crate::plumbing::operand::tensor_ref::TensorRefError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        try_rms_norm_linear("input", "w", "b", "out", 1, u32::MAX, 2, 1e-5),
        Err(crate::plumbing::operand::tensor_ref::TensorRefError::ElementCountOverflow { .. })
    ));
}

#[test]
fn rms_norm_linear_very_small_variance_eps_dominates() {
    let n = 4u32;
    let in_dim = 4u32;
    let out_dim = 2u32;
    let eps = 1e-5_f32;
    let input = [3.0f32; 4];
    let weights: Vec<f32> = (0..(in_dim * out_dim)).map(|i| i as f32 * 0.1).collect();
    let bias = [0.0f32, 0.0];
    let fused = rms_norm_linear("input", "w", "b", "out", n, in_dim, out_dim, eps);
    let fused_inputs = vec![
        Value::from(to_f32_bytes(&input)),
        Value::from(to_f32_bytes(&weights)),
        Value::from(to_f32_bytes(&bias)),
    ];
    let fused_outputs = vyre_reference::reference_eval(&fused, &fused_inputs)
        .expect("Fix: rms_norm_linear must execute on zero-variance input");
    let fused_out = bytes_to_f32(&fused_outputs[0].to_bytes());
    for (i, &v) in fused_out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "rms_norm_linear zero-variance output at {i} must be finite, got {v}"
        );
    }
}

#[test]
fn rms_norm_linear_very_large_variance() {
    let n = 4u32;
    let in_dim = 4u32;
    let out_dim = 2u32;
    let eps = 1e-5_f32;
    let input = [1e20f32, -1e20, 1e20, -1e20];
    let weights: Vec<f32> = (0..(in_dim * out_dim)).map(|i| i as f32 * 0.1).collect();
    let bias = [0.0f32, 0.0];
    let fused = rms_norm_linear("input", "w", "b", "out", n, in_dim, out_dim, eps);
    let fused_inputs = vec![
        Value::from(to_f32_bytes(&input)),
        Value::from(to_f32_bytes(&weights)),
        Value::from(to_f32_bytes(&bias)),
    ];
    let fused_outputs = vyre_reference::reference_eval(&fused, &fused_inputs)
        .expect("Fix: rms_norm_linear must execute on large-variance input");
    let fused_out = bytes_to_f32(&fused_outputs[0].to_bytes());
    for (i, &v) in fused_out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "rms_norm_linear large-variance output at {i} must be finite, got {v}"
        );
    }
}

#[test]
fn rms_norm_linear_single_element() {
    let n = 1u32;
    let in_dim = 4u32;
    let out_dim = 2u32;
    let eps = 1e-5_f32;
    let input = [5.0f32, 0.0, 0.0, 0.0];
    let weights: Vec<f32> = (0..(in_dim * out_dim)).map(|i| i as f32 * 0.1).collect();
    let bias = [0.0f32, 0.0];
    let fused = rms_norm_linear("input", "w", "b", "out", n, in_dim, out_dim, eps);
    let fused_inputs = vec![
        Value::from(to_f32_bytes(&input)),
        Value::from(to_f32_bytes(&weights)),
        Value::from(to_f32_bytes(&bias)),
    ];
    let fused_outputs = vyre_reference::reference_eval(&fused, &fused_inputs)
        .expect("Fix: rms_norm_linear single element must execute");
    let fused_out = bytes_to_f32(&fused_outputs[0].to_bytes());
    let expected = linear_reference(
        &input,
        &input[0..1],
        &weights,
        &bias,
        out_dim,
        in_dim,
        n,
        eps,
    );
    compare_ulp(&fused_out, &expected, n, in_dim, out_dim);
}

#[test]
fn rms_norm_linear_empty_tensor_traps() {
    let result = try_rms_norm_linear("input", "w", "b", "out", 0, 4, 4, 1e-5);
    assert!(
        matches!(
            result,
            Err(crate::plumbing::operand::tensor_ref::TensorRefError::ShapeMismatch { .. })
        ),
        "rms_norm_linear n=0 must be rejected by the builder"
    );
}

#[test]
fn parity_rms_norm_linear_matches_reference_sibling_sizes() {
    for (n, in_dim, out_dim) in [
        (1_u32, 1_u32, 1_u32),
        (2, 4, 2),
        (8, 8, 8),
        (32, 32, 32),
        (32, 128, 32),
        (128, 256, 128),
    ] {
        parity_case(n, in_dim, out_dim);
    }
}

#[test]
fn parity_rms_norm_linear_matches_reference_adversarial_inputs() {
    let n = 8_u32;
    let in_dim = 16_u32;
    let out_dim = 8_u32;

    let test_epsilons = [1e-6_f32, 1e-5, 1e-4, 1e-2, 1.0];
    for &eps in &test_epsilons {
        // Subnormal-adjacent small magnitudes
        let small_input: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 1e-4).collect();
        let weights: Vec<f32> = (0..(in_dim * out_dim))
            .map(|i| (i as f32) * 0.05 - 0.5)
            .collect();
        let bias: Vec<f32> = (0..out_dim).map(|i| (i as f32) * 0.2 - 0.8).collect();

        let fused = rms_norm_linear("input", "w", "b", "out", n, in_dim, out_dim, eps);
        let fused_inputs = vec![
            Value::from(to_f32_bytes(&small_input)),
            Value::from(to_f32_bytes(&weights)),
            Value::from(to_f32_bytes(&bias)),
        ];
        let fused_outputs = vyre_reference::reference_eval(&fused, &fused_inputs)
            .expect("Fix: fused rms_norm_linear must execute on small inputs");
        let fused_out = bytes_to_f32(&fused_outputs[0].to_bytes());
        let expected = linear_reference(
            &small_input,
            &small_input[0..n as usize],
            &weights,
            &bias,
            out_dim,
            in_dim,
            n,
            eps,
        );
        compare_ulp(&fused_out, &expected, n, in_dim, out_dim);

        // Alternating high dynamic range
        let alt_input: Vec<f32> = (0..in_dim)
            .map(|i| {
                if i % 2 == 0 {
                    100.0 * (i as f32 + 1.0)
                } else {
                    -0.01 * (i as f32 + 1.0)
                }
            })
            .collect();
        let fused_alt = rms_norm_linear("input", "w", "b", "out", n, in_dim, out_dim, eps);
        let fused_alt_inputs = vec![
            Value::from(to_f32_bytes(&alt_input)),
            Value::from(to_f32_bytes(&weights)),
            Value::from(to_f32_bytes(&bias)),
        ];
        let fused_alt_outputs = vyre_reference::reference_eval(&fused_alt, &fused_alt_inputs)
            .expect("Fix: fused rms_norm_linear must execute on alternating dynamic range inputs");
        let fused_alt_out = bytes_to_f32(&fused_alt_outputs[0].to_bytes());
        let expected_alt = linear_reference(
            &alt_input,
            &alt_input[0..n as usize],
            &weights,
            &bias,
            out_dim,
            in_dim,
            n,
            eps,
        );
        compare_ulp(&fused_alt_out, &expected_alt, n, in_dim, out_dim);
    }
}
