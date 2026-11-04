use super::*;

#[test]
fn generated_resident_u32_scalar_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = u32_bytes(&generated_atomic_values(0x1020_3040));
    let rhs = u32_bytes(&generated_atomic_values(0xa5a5_5a5a));

    // The binary and unary tables are swept separately so each proves its own
    // lane coverage; one combined total lets either hide the other's shortfall.
    assert_resident_u32_sweep(
        &backend,
        "Fix: resident u32 scalar generated matrix must compare every output lane.",
        U32_BINARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_u32_binary_program(case),
            inputs: vec![lhs.clone(), rhs.clone()],
        }),
    );
    assert_resident_u32_sweep(
        &backend,
        "Fix: resident u32 scalar generated matrix must compare every output lane.",
        U32_UNARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_u32_unary_program(case),
            inputs: vec![lhs.clone()],
        }),
    );
}

#[test]
fn generated_resident_i32_scalar_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let lhs = i32_bytes(&generated_i32_cast_values(LANE_COUNT));
    let rhs = i32_bytes(&defined_i32_divisors());

    assert_resident_u32_sweep(
        &backend,
        "Fix: resident i32 scalar generated matrix must compare every output lane.",
        I32_BINARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_i32_binary_program(case),
            inputs: vec![lhs.clone(), rhs.clone()],
        }),
    );
    assert_resident_u32_sweep(
        &backend,
        "Fix: resident i32 scalar generated matrix must compare every output lane.",
        I32_UNARY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_i32_unary_program(case),
            inputs: vec![lhs.clone()],
        }),
    );
}

/// The i32 right operand, shaped so no lane divides by zero and none hits the
/// `i32::MIN / -1` overflow the division cases would otherwise trap on.
fn defined_i32_divisors() -> Vec<i32> {
    generated_i32_cast_values(LANE_COUNT)
        .into_iter()
        .enumerate()
        .map(|(lane, value)| {
            let mixed = value ^ ((lane as i32).wrapping_mul(0x1f1f_0101));
            if mixed == 0 || mixed == -1 {
                ((lane as i32) & 0x3ff) + 1
            } else {
                mixed
            }
        })
        .collect()
}
