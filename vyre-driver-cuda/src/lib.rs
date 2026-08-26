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
/// CUDA upload planning for GPU e-graph device images.
pub(crate) mod egraph_device_image;
/// CUDA launch-wave planning for resident e-graph device images.
pub(crate) mod egraph_kernel_plan;
mod egraph_readback;
/// Adapter from frontier-typed IR plans to CUDA frontier wave envelopes.
pub(crate) mod frontier_typed_ir_adapter;
mod instrumentation;
/// Cross-process persistent CUDA JIT cache wiring (E4 + E5): configures
/// the NVIDIA driver's built-in disk cache at backend bring-up so the
/// JIT-compiled cuBINs persist across runs and are shared across every
/// vyre process on the host.
pub mod jit_cache;
/// Actionable CUDA kernel capability diagnostics.
pub(crate) mod kernel_failure_diagnostics;
mod materializer;
/// Bounded CUDA megakernel plan cache keyed by graph, analysis, device, and
/// runtime pressure buckets.
pub(crate) mod megakernel_plan_cache;
pub(crate) mod megakernel_plan_cache_records;
#[cfg(test)]
mod megakernel_plan_cache_tests;
mod numeric;
/// Occupancy-aware empirical autotuning (I4): pure estimator that picks
/// the workgroup size with the highest predicted hardware occupancy from
/// `(CudaDeviceCaps, KernelResourceUsage)`. The runtime feeds the result
/// into `AutotuneStore` (I3) so subsequent dispatches reuse the choice.
pub mod occupancy;
pub(crate) mod pending_dispatch;
mod pipeline;
/// CUDA profiler range integration for Nsight/NVTX without mandatory NVTX linkage.
pub mod profiler;
/// CUDA regex hardware-comparison evidence.
pub(crate) mod regex_hardware_comparison;
/// CUDA backend registration and device-buffer substrate adapter.
pub(crate) mod registration;
/// Repeated execution over persistent CUDA-resident graph state.
pub(crate) mod resident_graph_session;
mod stream;
// Neutral policies are imported from `vyre-driver`; CUDA exports only concrete behavior.
/// A fixed synthetic device envelope for context-free estimator tests. Not a
/// live probe: never derive a hardware decision from it.
pub mod synthetic_device_caps;
mod target_compiler;
/// CUDA execution planning for unified token/fact graph frontier waves.
pub(crate) mod token_fact_frontier_execution;
#[cfg(test)]
mod token_fact_frontier_execution_tests;
/// CUDA warp-word bit-parallel automata layout evidence.
pub(crate) mod warp_word_automata;

pub use backend::CachedCudaGraph;
pub use backend::{
    CudaBackend, CudaPtxSourceCacheSnapshot, CudaResidentBuffer, CudaStreamOrderedPool,
    CudaTelemetrySnapshot,
};
pub use stream::CudaLaunchResourceCounts;
/// CUDA megakernel global-barrier minimization for dependency-typed waves.
pub(crate) mod megakernel_barrier_planner;
pub(crate) mod megakernel_scheduler;
/// Release gate for steady-state CUDA megakernel speedup claims.
pub(crate) mod megakernel_speedup_gate;
pub use device::{CudaDeviceCaps, CudaDeviceHandle};
pub use egraph_device_image::{
    plan_cuda_egraph_device_upload, plan_cuda_egraph_device_upload_from_image,
    plan_cuda_egraph_device_upload_from_image_ref, CudaEGraphDeviceBorrowedUploadPlan,
    CudaEGraphDeviceByteLayout, CudaEGraphDeviceByteSpan, CudaEGraphDeviceKernelView,
    CudaEGraphDeviceUploadError, CudaEGraphDeviceUploadPlan, CudaResidentEGraphDeviceImage,
};
pub use egraph_kernel_plan::plan_cuda_egraph_structural_equivalence_launch_artifact_from_plan;
pub use egraph_kernel_plan::{
    collect_cuda_egraph_structural_equivalences, cuda_egraph_canonical_rewrite_kernel_ptx,
    cuda_egraph_signature_pair_rows, cuda_egraph_signature_refresh_kernel_ptx,
    cuda_egraph_structural_equivalence_kernel_ptx, pack_cuda_egraph_canonical_rewrite_device_image,
    pack_cuda_egraph_signature_bucket_device_image, plan_cuda_egraph_kernel_work,
    plan_cuda_egraph_signature_buckets, plan_cuda_egraph_signature_buckets_from_resident_snapshot,
    plan_cuda_egraph_signature_buckets_from_signature_snapshot,
    plan_cuda_egraph_structural_equivalence_launch_artifact,
    plan_cuda_egraph_structural_equivalence_output, plan_cuda_egraph_structural_equivalences,
    plan_cuda_egraph_union_compaction, CudaEGraphCanonicalRewrite,
    CudaEGraphCanonicalRewriteDeviceImage, CudaEGraphCanonicalRewriteKernelPtx,
    CudaEGraphCanonicalRewriteKernelResult, CudaEGraphFixedPointReadback,
    CudaEGraphKernelLaunchConfig, CudaEGraphKernelPass, CudaEGraphKernelPlanError,
    CudaEGraphKernelWave, CudaEGraphKernelWorkPlan, CudaEGraphResidentColumnSnapshot,
    CudaEGraphResidentSignatureSnapshot, CudaEGraphSignatureBucket,
    CudaEGraphSignatureBucketDeviceImage, CudaEGraphSignatureBucketPlan,
    CudaEGraphSignaturePairWave, CudaEGraphSignatureRefreshKernelPtx,
    CudaEGraphSignatureRefreshKernelResult, CudaEGraphStructuralCanonicalizationFixedPointReport,
    CudaEGraphStructuralCanonicalizationFixedPointResult,
    CudaEGraphStructuralCanonicalizationRoundResult, CudaEGraphStructuralEquivalenceKernelPtx,
    CudaEGraphStructuralEquivalenceKernelResult, CudaEGraphStructuralEquivalenceLaunchArtifact,
    CudaEGraphStructuralEquivalenceOutputPlan, CudaEGraphStructuralEquivalencePlan,
    CudaEGraphUnionCompactionPass, CudaEGraphUnionCompactionPlan, CudaEGraphUnionCompactionWave,
    CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_ENTRY, CUDA_EGRAPH_CANONICAL_REWRITE_KERNEL_PARAM_COUNT,
    CUDA_EGRAPH_CANONICAL_REWRITE_RECORD_WORDS, CUDA_EGRAPH_SIGNATURE_BUCKET_RECORD_WORDS,
    CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_ENTRY, CUDA_EGRAPH_SIGNATURE_REFRESH_KERNEL_PARAM_COUNT,
    CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_ENTRY,
    CUDA_EGRAPH_STRUCTURAL_EQUIVALENCE_KERNEL_PARAM_COUNT,
};
pub use frontier_typed_ir_adapter::adapt_frontier_typed_ir_to_cuda_into;
pub use frontier_typed_ir_adapter::{
    adapt_frontier_typed_ir_to_cuda, CudaFrontierTypedIrAdapterError, CudaFrontierTypedIrInput,
};
pub use kernel_failure_diagnostics::{
    diagnose_cuda_kernel_launch, diagnose_cuda_kernel_launch_shape,
    diagnose_cuda_kernel_launch_with_scratch, CudaKernelCapabilityFailure,
    CudaKernelDeviceEnvelope, CudaKernelLaunchDiagnostic, CudaKernelLaunchDiagnosticRef,
    CudaKernelLaunchDiagnosticScratch, CudaKernelLaunchEnvelope, CudaKernelLaunchEnvelopeError,
    CudaKernelLaunchShape, CudaKernelRequirement,
};
pub use megakernel_barrier_planner::{
    plan_cuda_frontier_megakernel_execution, plan_cuda_frontier_megakernel_execution_with_scratch,
    CudaMegakernelFrontierExecutionPlan, CudaMegakernelFrontierExecutionPlanError,
};
pub use megakernel_plan_cache::{
    CudaMegakernelAnalysisKind, CudaMegakernelCachedPlan, CudaMegakernelDeviceKey,
    CudaMegakernelPlanCache, CudaMegakernelPlanCacheKey, CudaMegakernelPlanCacheStats,
};
pub use megakernel_scheduler::{
    plan_cuda_megakernel_execution, select_cuda_megakernel_topology,
    select_cuda_megakernel_topology_stable, CudaMegakernelScheduleSample,
};
pub use megakernel_speedup_gate::{
    format_validated_cuda_megakernel_speedup_evidence_csv,
    validate_cuda_megakernel_speedup_evidence_csv, validate_cuda_megakernel_speedup_gate,
    CudaMegakernelSpeedupGateError, CudaMegakernelSpeedupProof, CudaMegakernelSpeedupSample,
    MEGAKERNEL_SPEEDUP_EVIDENCE_CSV_HEADER,
};
pub use regex_hardware_comparison::{
    cuda_regex_hardware_comparison_evidence, cuda_regex_software_fallback_comparison_evidence,
    CudaRegexHardwareComparisonEvidence, CUDA_REGEX_HARDWARE_COMPARISON_SCHEMA_VERSION,
};
pub use resident_graph_session::{
    format_validated_cuda_resident_graph_session_evidence_csv, plan_cuda_resident_graph_session,
    resident_graph_session_speedup_sample, CudaResidentGraphReadback,
    CudaResidentGraphSessionError, CudaResidentGraphSessionEvidence,
    CudaResidentGraphSessionEvidenceError, CudaResidentGraphSessionPlan,
    CudaResidentGraphSessionProfile,
};
pub use token_fact_frontier_execution::{
    plan_cuda_token_fact_frontier_execution, plan_cuda_token_fact_frontier_execution_with_scratch,
    CudaTokenFactFrontierExecutionError, CudaTokenFactFrontierExecutionPlan,
};
pub use token_fact_frontier_execution::{
    plan_cuda_token_fact_frontier_execution_envelope,
    plan_cuda_token_fact_frontier_execution_envelope_with_scratch,
    CudaTokenFactFrontierExecutionEnvelope, CudaTokenFactGraphResidency,
};
pub use warp_word_automata::{
    plan_cuda_warp_word_automata_layout, CudaWarpWordAutomataLayoutError,
    CudaWarpWordAutomataLayoutEvidence, CudaWarpWordAutomataLayoutRequest,
    CudaWarpWordInstructionClass, CUDA_WARP_WORD_AUTOMATA_LAYOUT_SCHEMA_VERSION,
};

pub use registration::{
    cuda_factory, cuda_supported_ops, registered_backend_id, CudaBackendRegistration,
    CudaDeviceBuffer,
};
use vyre_foundation::operation::TargetId;

/// Stable backend identifier for registration and conform certificates.
pub const CUDA_BACKEND_ID: &str = "cuda";
/// Validated target identity owned by the CUDA driver.
pub const CUDA_TARGET_ID: TargetId = TargetId::expect_valid(CUDA_BACKEND_ID);
