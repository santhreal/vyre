//! Quantized packing primitives.
//!
//! This module gives the spec-level `I4`/quantized type family executable
//! behavior: signed INT4 lanes are packed two per byte in a u32 word stream.
//! GPU kernels operate on u32 storage words so each word carries eight signed
//! 4-bit values.

mod i4_expressions;
mod programs;
#[cfg(test)]
#[path = "../../../tests/internal/math/quantized/mod.rs"]
mod tests;

pub use programs::{
    i4x8_batched_matmul_f32_scaled, i4x8_batched_matmul_top1_f32_scaled,
    i4x8_batched_matvec_f32_scaled, i4x8_dot_f32_scaled, i4x8_dot_i32, i4x8_matvec_f32_scaled,
    unpack_i4x8,
};

#[cfg(test)]
pub(crate) use vyre_reference::composition_witness::{
    i4x8_batched_matmul_f32_scaled_witness as i4x8_batched_matmul_f32_scaled_cpu,
    i4x8_batched_matmul_top1_f32_scaled_witness as i4x8_batched_matmul_top1_f32_scaled_cpu,
    i4x8_batched_matvec_f32_scaled_witness as i4x8_batched_matvec_f32_scaled_cpu,
    i4x8_dot_f32_scaled_witness as i4x8_dot_f32_scaled_cpu,
    i4x8_dot_i32_witness as i4x8_dot_i32_cpu,
    i4x8_matvec_f32_scaled_witness as i4x8_matvec_f32_scaled_cpu,
    pack_i4x8_witness as pack_i4x8_cpu, unpack_i4x8_witness as unpack_i4x8_cpu,
};

#[cfg(test)]
pub(crate) fn pack_i4x8_cpu_into(values: &[i32], out: &mut Vec<u32>) {
    if let Err(error) = try_pack_i4x8_cpu_into(values, out) {
        panic!("vyre-primitives pack_i4x8 CPU reference failed: {error}");
    }
}

#[cfg(test)]
pub(crate) fn try_pack_i4x8_cpu_into(values: &[i32], out: &mut Vec<u32>) -> Result<(), String> {
    let lane_count = u32::try_from(values.len()).map_err(|_| {
        format!(
            "pack_i4x8 CPU oracle received {} lanes, exceeding u32 lane count. Fix: shard quantized activations before parity evaluation.",
            values.len()
        )
    })?;
    let word_count = i4_packed_words(lane_count) as usize;
    if word_count > out.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            out,
            word_count - out.len(),
            "quantized INT4 CPU oracle",
            "pack_i4x8 output words",
        )?;
    }
    vyre_reference::composition_witness::pack_i4x8_witness_into(values, out);
    Ok(())
}

#[cfg(test)]
pub(crate) fn unpack_i4x8_cpu_into(packed: &[u32], lane_count: u32, out: &mut Vec<i32>) {
    try_unpack_i4x8_cpu_into(packed, lane_count, out).unwrap();
}

#[cfg(test)]
pub(crate) fn try_unpack_i4x8_cpu_into(
    packed: &[u32],
    lane_count: u32,
    out: &mut Vec<i32>,
) -> Result<(), String> {
    let count = usize::try_from(lane_count)
        .map_err(|_| format!("unpack_i4x8 CPU oracle received invalid lane count {lane_count}"))?;
    if count > out.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            out,
            count - out.len(),
            "quantized INT4 CPU oracle",
            "unpack_i4x8 output lanes",
        )?;
    }
    vyre_reference::composition_witness::unpack_i4x8_witness_into(packed, lane_count, out);
    Ok(())
}
/// Canonical op id for packed signed INT4 unpacking.
pub const UNPACK_I4_OP_ID: &str = "vyre-libs::math::quantized::unpack_i4x8";

/// Canonical op id for packed signed INT4 dot products.
pub const I4_DOT_I32_OP_ID: &str = "vyre-libs::math::quantized::i4x8_dot_i32";

/// Canonical op id for fused scaled packed signed INT4 dot products.
pub const I4_DOT_F32_SCALED_OP_ID: &str = "vyre-libs::math::quantized::i4x8_dot_f32_scaled";

/// Canonical op id for fused scaled packed signed INT4 matrix-vector products.
pub const I4_MATVEC_F32_SCALED_OP_ID: &str = "vyre-libs::math::quantized::i4x8_matvec_f32_scaled";

/// Canonical op id for batched fused scaled packed signed INT4 matvec.
pub const I4_BATCHED_MATVEC_F32_SCALED_OP_ID: &str =
    "vyre-libs::math::quantized::i4x8_batched_matvec_f32_scaled";

/// Canonical op id for batched fused scaled packed signed INT4 matmul.
pub const I4_BATCHED_MATMUL_F32_SCALED_OP_ID: &str =
    "vyre-libs::math::quantized::i4x8_batched_matmul_f32_scaled";

/// Canonical op id for fused packed signed INT4 batched matmul top-1 routing.
pub const I4_BATCHED_MATMUL_TOP1_F32_SCALED_OP_ID: &str =
    "vyre-libs::math::quantized::i4x8_batched_matmul_top1_f32_scaled";

/// Number of signed 4-bit lanes per packed u32 word.
pub const I4_LANES_PER_WORD: u32 = 8;

/// Number of packed signed INT4 words required for `lane_count` lanes.
#[must_use]
pub const fn i4_packed_words(lane_count: u32) -> u32 {
    lane_count.div_ceil(I4_LANES_PER_WORD)
}

fn u32s(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

fn i32s(lanes: &[i32]) -> Vec<u8> {
    vyre_primitives::wire::pack_i32_slice(lanes)
}

fn f32s(floats: &[f32]) -> Vec<u8> {
    vyre_primitives::wire::pack_f32_slice(floats)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        UNPACK_I4_OP_ID,
        || unpack_i4x8("packed_words", "out_lanes", 8),
        // `out_lanes` is a ReadWrite in/out buffer (not a backend-allocated `output`),
        // so, like byte_histogram's histogram and persistent_bfs's frontier_out, the
        // fixture must seed it (zeros) as its own input Value. Omitting it makes the
        // reference interpreter reject the fixture ("missing input for buffer
        // `out_lanes`"), silently dropping this op from every registry parity gate.
        Some(|| vec![vec![u32s(&[0x7621_0F98]), i32s(&[0; 8])]]),
        Some(|| {
            vec![vec![vec![
                0xf8, 0xff, 0xff, 0xff, // -8
                0xf9, 0xff, 0xff, 0xff, // -7
                0xff, 0xff, 0xff, 0xff, // -1
                0x00, 0x00, 0x00, 0x00, // 0
                0x01, 0x00, 0x00, 0x00, // 1
                0x02, 0x00, 0x00, 0x00, // 2
                0x06, 0x00, 0x00, 0x00, // 6
                0x07, 0x00, 0x00, 0x00, // 7
            ]]]
        }),
    ).with_category("math")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        I4_DOT_I32_OP_ID,
        || i4x8_dot_i32("lhs_packed", "rhs_packed", "out", 8),
        Some(|| vec![vec![u32s(&[0x7621_0F98]), u32s(&[0x7621_0F98])]]),
        Some(|| vec![vec![vec![0xcc, 0x00, 0x00, 0x00]]]),
    ).with_category("math")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        I4_DOT_F32_SCALED_OP_ID,
        || i4x8_dot_f32_scaled("lhs_packed", "rhs_packed", "lhs_scale", "rhs_scale", "out", 8),
        Some(|| vec![vec![u32s(&[0x7621_0F98]), u32s(&[0x7621_0F98]), f32s(&[1.0]), f32s(&[1.0])]]),
        Some(|| vec![vec![vec![0x00, 0x00, 0x4c, 0x43]]]),
    ).with_category("math")
}

inventory::submit! {
        vyre_foundation::operation::OperationRegistration::library(
            I4_MATVEC_F32_SCALED_OP_ID,
            || i4x8_matvec_f32_scaled("matrix_packed", "vector_packed", "matrix_scale", "out", 4, 8),
            Some(|| vec![vec![
                u32s(&[0x7621_0F98; 4]),
                f32s(&[-8.0, -7.0, -1.0, 0.0, 1.0, 2.0, 6.0, 7.0]),
                f32s(&[1.0; 4]),
            ]]),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x4c, 0x43,
                0x00, 0x00, 0x4c, 0x43,
                0x00, 0x00, 0x4c, 0x43,
                0x00, 0x00, 0x4c, 0x43,
            ]]]
        }),
    ).with_category("math")
}

inventory::submit! {
        vyre_foundation::operation::OperationRegistration::library(
            I4_BATCHED_MATVEC_F32_SCALED_OP_ID,
            || i4x8_batched_matvec_f32_scaled("matrix_packed", "vector_packed", "matrix_scale", "out", 2, 4, 8),
            Some(|| vec![vec![
                u32s(&[0x7621_0F98; 4]),
                f32s(&[-8.0, -7.0, -1.0, 0.0, 1.0, 2.0, 6.0, 7.0, -8.0, -7.0, -1.0, 0.0, 1.0, 2.0, 6.0, 7.0]),
                f32s(&[1.0; 4]),
            ]]),
            Some(|| {
                vec![vec![vec![
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                ]]]
            }),
    ).with_category("math")
}

inventory::submit! {
        vyre_foundation::operation::OperationRegistration::library(
            I4_BATCHED_MATMUL_F32_SCALED_OP_ID,
            || i4x8_batched_matmul_f32_scaled("lhs_packed", "rhs_packed", "lhs_scale", "rhs_scale", "out", 2, 4, 8),
            Some(|| vec![vec![
                u32s(&[0x7621_0F98; 4]),
                u32s(&[0x7621_0F98; 2]),
                f32s(&[1.0; 4]),
                f32s(&[1.0; 2]),
            ]]),
            Some(|| {
                vec![vec![vec![
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                    0x00, 0x00, 0x4c, 0x43,
                ]]]
            }),
    ).with_category("math")
}

inventory::submit! {
        vyre_foundation::operation::OperationRegistration::library(
            I4_BATCHED_MATMUL_TOP1_F32_SCALED_OP_ID,
            || i4x8_batched_matmul_top1_f32_scaled("lhs_packed", "rhs_packed", "lhs_scale", "rhs_scale", "out_scores", 2, 4, 8),
            Some(|| vec![vec![
                u32s(&[0x7621_0F98; 4]),
                u32s(&[0x7621_0F98; 2]),
                f32s(&[1.0; 4]),
                f32s(&[1.0; 2]),
            ]]),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x4c, 0x43,
                0x00, 0x00, 0x4c, 0x43,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ]]]
        }),
    ).with_category("math")
}
