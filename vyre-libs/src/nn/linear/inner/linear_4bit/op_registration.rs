//! Operation-catalog registrations for the INT4 linear builders.

use vyre_foundation::ir::DataType;

use super::affine_grouped::linear_4bit_affine_grouped;
use super::unpack_on_demand::linear_4bit;

inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        semantic_version: 1,
        signature: None,
        tier: vyre_foundation::operation::OperationTier::Library,
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
        id: "vyre-libs::nn::linear_4bit",
        build: Some(|| {
            linear_4bit("x", "w", "b", "out", 8, 4).unwrap_or_else(|error| {
                crate::builder::invalid_builder_trap_program(
                    "vyre-libs::nn::linear_4bit",
                    "out",
                    DataType::F32,
                    error,
                )
            })
        }),
        test_inputs: Some(|| {
            let x: Vec<f32> = (0..8).map(|i| i as f32).collect();
            let w: Vec<u32> = vec![0x7654_3210, 0xFEDC_BA98, 0x1111_1111, 0x0000_0000];
            let b: Vec<f32> = vec![0.0; 4];
            vec![vec![
                vyre_primitives::wire::pack_f32_slice(&x),
                vyre_primitives::wire::pack_u32_slice(&w),
                vyre_primitives::wire::pack_f32_slice(&b),
            ]]
        }),
        expected_output: Some(|| {
            let out = [140.0f32, 364.0, 28.0, 0.0];
            vec![vec![vyre_primitives::wire::pack_f32_slice(&out)]]
        }),
        category: Some("nn"),
    }
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration {
        semantic_version: 1,
        signature: None,
        tier: vyre_foundation::operation::OperationTier::Library,
        laws: &[],
        tolerance: vyre_foundation::operation::TolerancePolicy::EXACT,
        id: "vyre-libs::nn::linear_4bit_affine_grouped",
        build: Some(|| {
            linear_4bit_affine_grouped("x", "w", "scale", "zp", "b", "out", 8, 2, 4)
                .unwrap_or_else(|error| {
                    crate::builder::invalid_builder_trap_program(
                        "vyre-libs::nn::linear_4bit_affine_grouped",
                        "out",
                        DataType::F32,
                        error,
                    )
                })
        }),
        test_inputs: Some(|| {
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
        expected_output: Some(|| {
            let out = [150.0f32, 3.0];
            vec![vec![vyre_primitives::wire::pack_f32_slice(&out)]]
        }),
        category: Some("nn"),
    }
}
