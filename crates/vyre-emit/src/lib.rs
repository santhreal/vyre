//! Target emitters for a lowered vyre `KernelDescriptor`.
//!
//! Each target is one feature-gated module:
//! - `naga`: `naga::Module` for wgpu (WGSL) and the shared input of `spirv` and `metal`
//! - `ptx`: NVRTC-compatible PTX text for CUDA
//! - `spirv`: SPIR-V words through `naga::back::spv`
//! - `metal`: Metal Shading Language through `naga::back::msl`

#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "naga")]
pub mod naga;
#[cfg(feature = "ptx")]
pub mod ptx;
#[cfg(feature = "spirv")]
pub mod spirv;

/// Digest of this crate's source and manifest, stamped in by `build.rs`.
///
/// A pipeline or module cache keyed on the program alone serves an artifact
/// the previous emitter produced after an emitter fix. Every target cache mixes
/// this value into its key, so any emitter edit invalidates them.
pub const LOWERING_DIGEST: &str = env!("VYRE_EMIT_LOWERING_DIGEST");
