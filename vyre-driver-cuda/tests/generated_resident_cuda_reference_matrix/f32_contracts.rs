#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn generated_resident_f32_comparison_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = f32_bytes(&generated_f32_values(0x55aa_1234));
    let rhs = f32_bytes(&generated_f32_values(0xaa55_4321));

    assert_resident_u32_sweep(
        &backend,
        "Fix: resident f32 comparison matrix must compare every output lane, including NaN comparison lanes.",
        F32_COMPARE_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_f32_compare_program(case),
            inputs: vec![lhs.clone(), rhs.clone()],
        }),
    );
}

#[test]
fn generated_resident_f32_binary_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = f32_bytes(&generated_f32_values(0x1357_9bdf));
    let mixed_rhs = f32_bytes(&generated_f32_values(0x2468_ace0));
    let nonzero_rhs = f32_bytes(&generated_f32_nonzero_values(0x0bad_f00d));

    assert_resident_f32_sweep(
        &backend,
        "Fix: resident f32 binary matrix must compare every output lane.",
        MAX_F32_ULP,
        F32_BINARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_f32_binary_program(case),
            inputs: vec![
                lhs.clone(),
                match case.rhs {
                    F32RhsKind::Mixed => mixed_rhs.clone(),
                    F32RhsKind::NonZero => nonzero_rhs.clone(),
                },
            ],
        }),
    );
}

#[test]
fn generated_resident_f32_unary_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let mixed = f32_bytes(&generated_f32_values(0xfeed_beef));
    let nonzero = f32_bytes(&generated_f32_nonzero_values(0xabcd_1234));
    let sqrt_domain = f32_bytes(&generated_f32_sqrt_domain_values(0xdec0_ded1));

    assert_resident_f32_sweep(
        &backend,
        "Fix: resident f32 unary matrix must compare every output lane.",
        MAX_F32_ULP,
        F32_UNARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_f32_unary_program(case),
            inputs: vec![match case.inputs {
                F32InputKind::Mixed => mixed.clone(),
                F32InputKind::NonZero => nonzero.clone(),
                F32InputKind::SqrtDomain => sqrt_domain.clone(),
            }],
        }),
    );
}

#[test]
fn generated_resident_f32_classification_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let input = f32_bytes(&generated_f32_classification_values());

    assert_resident_u32_sweep(
        &backend,
        "Fix: resident f32 classification matrix must compare every output lane.",
        F32_CLASSIFY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_f32_classify_program(case),
            inputs: vec![input.clone()],
        }),
    );
}

#[test]
fn generated_resident_f32_fma_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let a = generated_f32_fma_values(LANE_COUNT, 0x1234_5678);
    let b = generated_f32_fma_values(LANE_COUNT, 0x9abc_def0);
    let c = generated_f32_fma_values(LANE_COUNT, 0x0fed_cba9);

    assert_resident_f32_sweep(
        &backend,
        "Fix: resident FMA generated matrix must compare every output lane.",
        MAX_F32_ULP,
        [ResidentMatrixCase {
            name: "resident_f32_fma",
            program: resident_fma_program(),
            inputs: vec![f32_bytes(&a), f32_bytes(&b), f32_bytes(&c)],
        }]
        .into_iter(),
    );
}
