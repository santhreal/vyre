//! # vyre-driver-cuda  -  CUDA/PTX backend for vyre
//!
//! Implements [`VyreBackend`](vyre_driver::VyreBackend) via the CUDA driver API through `cudarc`.
//! Translates vyre `Program` IR into PTX kernels, loads them through
//! the CUDA driver JIT, and dispatches on NVIDIA GPUs.
//!
//! The backend registers itself as `"cuda"` in the vyre backend registry
//! via `inventory::submit!` so `vyre::registered_backends()` enumerates
//! it alongside `wgpu`, `spirv`, etc.
//!
//! ## Architecture
//!
//! ```text
//!    Program ─► PTX emitter ─► cuModuleLoadData ─► cuLaunchKernel
//! ```
//!
// CUDA driver bindings (`cudarc::driver::sys::cu*`) are inherently unsafe FFI;
// every call site is the boundary between safe vyre code and the CUDA driver
// API. Allow `unsafe` here so the rest of the workspace can keep
// `unsafe_code = "deny"` while this backend wraps cudarc properly with
// per-call SAFETY comments the `lint-unsafe-justification` gate enforces.
#![allow(unsafe_code)]

mod aot_launcher;
/// CUDA backend core: device management and dispatch.
pub(crate) mod backend;
/// PTX code generation from vyre IR.
pub mod codegen;
/// CUDA device capability probing.
pub(crate) mod device;
/// Live CUDA clock, thermal and power state for measurement provenance.
pub(crate) mod device_state;
#[cfg(test)]
mod device_state_tests;
mod instrumentation;
/// Cross-process persistent CUDA JIT cache wiring (E4 + E5): configures
/// the NVIDIA driver's built-in disk cache at backend bring-up so the
/// JIT-compiled cuBINs persist across runs and are shared across every
/// vyre process on the host.
pub mod jit_cache;
/// Actionable CUDA kernel capability diagnostics.
pub(crate) mod kernel_failure_diagnostics;
mod materializer;
mod numeric;
/// Occupancy facts the device reports for one candidate launch shape: resident
/// blocks, warps and occupancy per compute unit, and the cooperative residency
/// ceiling. Every function here states what the hardware admits; the compiler
/// selects the shape.
pub mod occupancy;
pub(crate) mod pending_dispatch;
mod pipeline;
/// CUDA profiler range integration for Nsight/NVTX without mandatory NVTX linkage.
pub mod profiler;
/// CUDA backend registration and device-buffer substrate adapter.
pub(crate) mod registration;
mod stream;
// Neutral policies are imported from `vyre-driver`; CUDA exports only concrete behavior.
/// A fixed synthetic device envelope for context-free estimator tests. Not a
/// live probe: never derive a hardware decision from it.
pub mod synthetic_device_caps;
mod target_compiler;

pub use backend::CachedCudaGraph;
pub use backend::{
    CudaBackend, CudaPtxSourceCacheSnapshot, CudaResidentBuffer, CudaTelemetrySnapshot,
};
pub use device::{CudaDeviceCaps, CudaDeviceHandle};
pub use kernel_failure_diagnostics::{
    diagnose_cuda_kernel_launch_shape, CudaKernelDeviceEnvelope, CudaKernelLaunchEnvelope,
    CudaKernelLaunchShape,
};
pub use stream::CudaLaunchResourceCounts;

pub use registration::{
    cuda_factory, cuda_supported_ops, registered_backend_id, registered_device,
    CudaBackendRegistration,
};
use vyre_foundation::operation::TargetId;

/// Stable backend identifier for registration and conform certificates.
pub const CUDA_BACKEND_ID: &str = "cuda";
/// Validated target identity owned by the CUDA driver.
pub const CUDA_TARGET_ID: TargetId = TargetId::expect_valid(CUDA_BACKEND_ID);
