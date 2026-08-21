#![cfg(feature = "device-tests")]

use super::*;

#[test]
fn generated_resident_memory_permutation_matrix_matches_reference_on_live_cuda() {
    let backend = live_backend();
    let u32_input = u32_bytes(&generated_atomic_values(0x3141_5926));
    let bool_input = bool_bytes(&generated_bool_values(0x2718_2818));
    let f32_input = f32_bytes(&generated_f32_values(0x1234_abcd));
    const FIX: &str = "Fix: resident memory generated matrix must compare every output lane.";

    // One sweep per storage type: the f32 permutations are compared under the
    // arithmetic ULP bound, and each table proves its own lane coverage rather
    // than contributing to a total that hides a short table.
    assert_resident_u32_sweep(
        &backend,
        FIX,
        U32_MEMORY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_memory_program(case),
            inputs: vec![u32_input.clone()],
        }),
    );
    assert_resident_u32_sweep(
        &backend,
        FIX,
        BOOL_MEMORY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_memory_program(case),
            inputs: vec![bool_input.clone()],
        }),
    );
    assert_resident_f32_sweep(
        &backend,
        FIX,
        MAX_F32_ULP,
        F32_MEMORY_CASES.iter().map(|case| ResidentMatrixCase {
            name: case.name,
            program: resident_memory_program(case),
            inputs: vec![f32_input.clone()],
        }),
    );
}
