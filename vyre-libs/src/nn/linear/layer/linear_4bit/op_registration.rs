//! Operation-catalog registrations for the INT4 linear builders.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::DataType;

use super::affine_grouped::linear_4bit_affine_grouped;
use super::unpack_on_demand::linear_4bit;

const EXPECTED_LINEAR_4BIT_OUTPUT_BYTES: [u8; 16] = [
    0x00, 0x00, 0x0C, 0x43, 0x00, 0x00, 0xB6, 0x43, 0x00, 0x00, 0xE0, 0x41, 0x00, 0x00, 0x00, 0x00,
];
const EXPECTED_LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_BYTES: [u8; 8] =
    [0x00, 0x00, 0x16, 0x43, 0x00, 0x00, 0x40, 0x40];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        "vyre-libs::nn::linear_4bit",
        || {
            linear_4bit("x", "w", "b", "out", 8, 4).unwrap_or_else(|error| {
                trap_program(
                    "vyre-libs::nn::linear_4bit",
                    Some(("out", DataType::F32)),
                    error,
                )
            })
        },
        Some(|| {
            let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
            let w: Vec<u32> = vec![0x7654_3210, 0xFEDC_BA98, 0x1111_1111, 0x0000_0000];
            let b: Vec<f32> = vec![0.0; 4];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&x),
                vyre_primitives::wire::pack_u32_slice(&w),
                vyre_primitives::wire::pack_f32_slice(&b),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LINEAR_4BIT_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        "vyre-libs::nn::linear_4bit_affine_grouped",
        || {
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 8, 2, 4)
                .unwrap_or_else(|error| {
                    trap_program(
                        "vyre-libs::nn::linear_4bit_affine_grouped",
                        Some(("out", DataType::F32)),
                        error,
                    )
                })
        },
        Some(|| {
            let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let w = [0x8765_4321u32, 0x0000_0000u32];
            let scale = [0.5f32, 1.0, 2.0, 1.0];
            let zp = [1u32, 0, 4, 0];
            let b = [0.0f32, 3.0];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&x),
                vyre_primitives::wire::pack_u32_slice(&w),
                vyre_primitives::wire::pack_f32_slice(&scale),
                vyre_primitives::wire::pack_u32_slice(&zp),
                vyre_primitives::wire::pack_f32_slice(&b),
            ]]
        }),
        Some(|| {
            vec![vec![EXPECTED_LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_BYTES.to_vec()]]
        }),
    )
    .with_category("nn")
}
