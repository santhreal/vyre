//! Self-substrate dispatch wrappers for quantized packing primitives.
//!
//! Quantized low-bit layouts are now executable primitives, not just spec
//! variants. This wrapper lets optimizer/self-hosted paths unpack packed INT4
//! tensors through the same backend seam as every other primitive.

mod shapes;

use crate::math::quantized::{
    i4_packed_words, i4x8_batched_matmul_f32_scaled, i4x8_batched_matmul_top1_f32_scaled,
    i4x8_batched_matvec_f32_scaled, i4x8_dot_f32_scaled, i4x8_matvec_f32_scaled, unpack_i4x8,
};
use crate::plumbing::host::dispatch_buffers::{
    ensure_input_slots, execute_program_sole_f32_into, execute_program_sole_i32_into,
    write_f32_slice_le_bytes, write_u32_slice_le_bytes, write_zero_bytes,
};
use crate::plumbing::host::program_cache::ProgramCache;
use shapes::dispatch_packed_batched_matmul;
pub use shapes::PackedI4BatchedMatmul;
use vyre_foundation::ir::Program;

/// Caller-owned dispatch scratch for quantized INT4 unpacking.
#[derive(Debug, Default)]
pub struct QuantizedUnpackGpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<u32, Program>,
}

/// Caller-owned dispatch scratch for packed INT4 scaled dot products.
#[derive(Debug, Default)]
pub struct QuantizedDotGpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<u32, Program>,
}

/// Caller-owned dispatch scratch for packed INT4 row-scaled matvecs.
#[derive(Debug, Default)]
pub struct QuantizedMatvecGpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<(u32, u32), Program>,
}

/// Caller-owned dispatch scratch for packed INT4 batched row-scaled matvecs.
#[derive(Debug, Default)]
pub struct QuantizedBatchedMatvecGpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<(u32, u32, u32), Program>,
}

/// Caller-owned dispatch scratch for packed INT4 batched packed-activation matmuls.
#[derive(Debug, Default)]
pub struct QuantizedBatchedMatmulGpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<(u32, u32, u32), Program>,
}

/// Caller-owned dispatch scratch for packed INT4 batched matmul top-1 routing.
#[derive(Debug, Default)]
pub struct QuantizedBatchedMatmulTop1GpuScratch {
    inputs: Vec<Vec<u8>>,
    program_cache: ProgramCache<(u32, u32, u32), Program>,
}

mod batched_matmul;
mod batched_matvec;
mod dot;
mod matvec;
mod top1;
mod unpack;

pub use batched_matmul::{
    i4x8_batched_matmul_f32_scaled_via, i4x8_batched_matmul_f32_scaled_via_with_scratch_into,
};
pub use batched_matvec::{
    i4x8_batched_matvec_f32_scaled_via, i4x8_batched_matvec_f32_scaled_via_with_scratch_into,
};
pub use dot::{i4x8_dot_f32_scaled_via, i4x8_dot_f32_scaled_via_with_scratch_into};
pub use matvec::{i4x8_matvec_f32_scaled_via, i4x8_matvec_f32_scaled_via_with_scratch_into};
pub use top1::{
    i4x8_batched_matmul_top1_f32_scaled_via,
    i4x8_batched_matmul_top1_f32_scaled_via_with_scratch_into,
};
pub use unpack::{unpack_i4x8_via, unpack_i4x8_via_with_scratch_into};
