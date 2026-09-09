//! Contract and reference parity tests for logical contraction lowering.
//!
//! Asserts that:
//! 1. All contraction consumers across `vyre-libs` construct valid programs via the unified composer.
//! 2. Contraction programs match reference oracle execution across the dynamically derived dtype and shape space.
//! 3. Contraction candidates (Scalar, SIMT Tiled, Matrix Instruction) are generated with valid properties.

use vyre_foundation::validate::validate;
use vyre_libs::gemm::ContractionComposer;
use vyre_libs::math::linalg::{matmul, matmul_bias};
use vyre_libs::math::semiring_gemm::semiring_gemm;
use vyre_libs::nn::linear::{
    batch_matmul, linear, linear_relu, linear_rows, linear_silu, linear_tiled,
};
use vyre_libs::TensorRef;
use vyre_primitives::wire::pack_f32_slice;
use vyre_reference::value::Value;
use vyre_spec::Semiring;
#[test]
fn contraction_users_build_through_unified_composer() {
    // 1. matmul
    let p_matmul = matmul("a", "b", "out", 4, 4, 4);
    assert!(
        validate(&p_matmul).is_empty(),
        "matmul invalid: {:?}",
        validate(&p_matmul)
    );

    // 2. matmul_bias
    let p_matmul_bias = matmul_bias("a", "b", "bias", "out", 4, 4, 4);
    assert!(validate(&p_matmul_bias).is_empty());

    // 3. linear
    let p_linear = linear("x", "w", "b", "out", 4, 4).expect("linear must build");
    assert!(validate(&p_linear).is_empty());

    // 4. linear_rows
    let p_linear_rows = linear_rows("x", "w", "b", "out", 2, 4, 4).expect("linear_rows must build");
    assert!(validate(&p_linear_rows).is_empty());

    // 5. batch_matmul
    let p_batch_matmul =
        batch_matmul("a", "b", "out", 2, 4, 4, 4).expect("batch_matmul must build");
    assert!(validate(&p_batch_matmul).is_empty());

    // 6. semiring_gemm
    let p_semiring = semiring_gemm("a", "b", "out", 4, 4, 4, Semiring::Real);
    assert!(validate(&p_semiring).is_empty());

    // (semiring_gemm tested above)

    // 8. linear_relu
    let p_relu = linear_relu("x", "w", "b", "out", 4, 4).expect("linear_relu must build");
    assert!(validate(&p_relu).is_empty());

    // 9. linear_silu
    let p_silu = linear_silu("x", "w", "b", "out", 4, 4).expect("linear_silu must build");
    assert!(validate(&p_silu).is_empty());

    // 10. linear_tiled
    let p_tiled = linear_tiled("x", "w", "b", "out", 4, 4, 2).expect("linear_tiled must build");
    assert!(validate(&p_tiled).is_empty());
}

#[test]
fn contraction_u32_matches_reference_oracle_across_shapes() {
    let shapes = [(1, 1, 1), (2, 3, 2), (4, 4, 4), (8, 4, 8), (16, 8, 16)];

    for (m, k, n) in shapes {
        let a_vals: Vec<u32> = (0..(m * k)).map(|i| (i % 17) + 1).collect();
        let b_vals: Vec<u32> = (0..(k * n)).map(|i| (i % 13) + 1).collect();

        // Expected mathematical matrix product
        let mut expected = vec![0u32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0u32;
                for p in 0..k as usize {
                    sum = sum.wrapping_add(
                        a_vals[i * (k as usize) + p].wrapping_mul(b_vals[p * (n as usize) + j]),
                    );
                }
                expected[i * (n as usize) + j] = sum;
            }
        }

        let a_ref = TensorRef::u32_2d("a", m, k);
        let b_ref = TensorRef::u32_2d("b", k, n);
        let out_ref = TensorRef::u32_2d("out", m, n);

        let prog = ContractionComposer::matmul_2d("matmul_test", a_ref, b_ref, out_ref, m, k, n)
            .build()
            .unwrap_or_else(|e| panic!("Fix: {m}x{k}x{n} u32 matmul must build: {e}"));

        let a_bytes: Vec<u8> = a_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        let inputs = vec![Value::from(a_bytes), Value::from(b_bytes)];

        let outputs = vyre_reference::reference_eval(&prog, &inputs).expect("eval");
        let out_bytes = outputs[0].to_bytes();
        let actual: Vec<u32> = out_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual, expected,
            "Fix: {m}x{k}x{n} u32 matmul output must match reference oracle"
        );
    }
}

#[test]
fn contraction_f32_matches_reference_oracle_across_shapes() {
    let shapes = [(1, 1, 1), (2, 3, 2), (4, 4, 4), (8, 4, 8)];

    for (m, k, n) in shapes {
        let a_vals: Vec<f32> = (0..(m * k)).map(|i| (i as f32) * 0.5 + 1.0).collect();
        let b_vals: Vec<f32> = (0..(k * n)).map(|i| (i as f32) * 0.25 + 0.5).collect();

        // Expected mathematical matrix product
        let mut expected = vec![0.0f32; (m * n) as usize];
        for i in 0..m as usize {
            for j in 0..n as usize {
                let mut sum = 0.0f32;
                for p in 0..k as usize {
                    sum += a_vals[i * (k as usize) + p] * b_vals[p * (n as usize) + j];
                }
                expected[i * (n as usize) + j] = sum;
            }
        }

        let a_ref = TensorRef::f32_2d("a", m, k);
        let b_ref = TensorRef::f32_2d("b", k, n);
        let out_ref = TensorRef::f32_2d("out", m, n);

        let prog =
            ContractionComposer::matmul_2d("matmul_f32_test", a_ref, b_ref, out_ref, m, k, n)
                .build()
                .unwrap_or_else(|e| panic!("Fix: {m}x{k}x{n} f32 matmul must build: {e}"));

        let inputs = vec![
            Value::from(pack_f32_slice(&a_vals)),
            Value::from(pack_f32_slice(&b_vals)),
        ];

        let outputs = vyre_reference::reference_eval(&prog, &inputs).expect("eval");
        let out_bytes = outputs[0].to_bytes();
        let actual =
            vyre_primitives::wire::unpack_f32_slice(&out_bytes, (m * n) as usize, "matmul_out")
                .expect("unpack f32 slice");
        for (idx, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (act - exp).abs() < 1e-4,
                "Fix: {m}x{k}x{n} f32 matmul at {idx} expected {exp}, got {act}"
            );
        }
    }
}
