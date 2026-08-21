#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn generated_resident_bool_scalar_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = bool_bytes(&generated_bool_values(0x1020_3040));
    let rhs = bool_bytes(&generated_bool_values(0xa5a5_5a5a));

    // The binary and unary tables are swept separately so each proves its own
    // lane coverage; one combined total lets either hide the other's shortfall.
    assert_resident_u32_sweep(
        &backend,
        "Fix: resident Bool generated matrix must compare every output lane.",
        BOOL_BINARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_bool_binary_program(case),
            inputs: vec![lhs.clone(), rhs.clone()],
        }),
    );
    assert_resident_u32_sweep(
        &backend,
        "Fix: resident Bool generated matrix must compare every output lane.",
        BOOL_UNARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_bool_unary_program(case),
            inputs: vec![lhs.clone()],
        }),
    );
}

#[test]
fn generated_resident_bool_select_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let flag = generated_bool_values(0x3333_cccc);
    let lhs = generated_bool_values(0x1234_abcd);
    let rhs = generated_bool_values(0xdcba_4321);

    assert_resident_u32_sweep(
        &backend,
        "Fix: resident Bool select generated matrix must compare every output lane.",
        [ResidentMatrixCase {
            name: "resident_bool_select",
            program: resident_bool_select_program(),
            inputs: vec![bool_bytes(&flag), bool_bytes(&lhs), bool_bytes(&rhs)],
        }]
        .into_iter(),
    );
}
