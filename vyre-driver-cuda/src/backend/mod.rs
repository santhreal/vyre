//! CUDA backend module: device lifecycle, allocation pools, and kernel dispatch.
//!
//! `allocations` owns transient device and pinned-host pools plus the
//! `cuda_check` error wrapper. `module_cache` owns loaded PTX modules.
//! `resident` owns long-lived CUDA allocations and in-flight handle guards.
//! `dispatch` owns the `CudaBackend` struct, launch geometry, and
//! kernel-launch orchestration. The public surface is re-exported below.

/// Shared non-wrapping atomic accounting primitives.
pub(crate) mod accounting;
/// Device-side allocation pools, pinned-host pools, and `cuda_check`.
pub mod allocations;
/// Capability, feature-flag, and validation-cache policy.
pub(crate) mod capabilities;
/// Checked CUDA copy primitives shared by host, resident, and graph paths.
pub(crate) mod copy;
/// cudaGraph capture-and-replay path. Records one full Program dispatch into
/// a `CUgraph` then replays it on demand to reduce hot-path launch overhead.
pub mod cuda_graph;
/// cudaGraph replay path.
pub(crate) mod cuda_graph_replay;
/// CUDA backend handle, launch geometry, and kernel-launch orchestration  -
/// including the cooperative-launch path that routes through
/// `cuLaunchCooperativeKernel` when the caller opts in via
/// `DispatchConfig::cooperative`.
pub mod dispatch;
/// Per-dispatch host and device phase attribution for the timed dispatch path.
pub(crate) mod dispatch_phase_probe;
/// Host-borrowed buffer dispatch path.
pub(crate) mod host_dispatch;
/// Checked CUDA host-memory registration boundary.
pub(crate) mod host_memory;
/// Raw CUDA kernel launch boundary.
pub(crate) mod launch;
/// Checked launch-parameter byte sizing.
pub(crate) mod launch_params;
/// Loaded PTX module cache and submodular eviction policy.
pub(crate) mod module_cache;
/// Shared monotonic ordering helpers for staging hot paths.
pub(crate) mod ordering;
/// CUDA output readback range handling.
pub(crate) mod output_range;
/// Shared dispatch-plan assembly helpers.
pub(crate) mod plan;
/// PTX target probing against the live CUDA driver.
pub(crate) mod ptx_target;
/// Resident buffer management  -  long-lived device allocations.
pub(crate) mod resident;
/// Resident-buffer dispatch path.
pub(crate) mod resident_dispatch;
/// Shared resident-dispatch contracts and checked accounting.
pub(crate) mod resident_dispatch_support;
/// Host and device copies for resident buffers.
pub(crate) mod resident_io;
/// Shared resident readback interval fusion.
pub(crate) mod resident_readback_fusion;
/// Shared resident upload interval fusion.
pub(crate) mod resident_upload_fusion;
/// Shared fallible staging reservation helpers for backend hot paths.
pub(crate) mod staging_reserve;
/// Stream-ordered device allocator over the driver's default memory pool.
pub mod stream_ordered_pool;
/// Atomic CUDA runtime telemetry counters.
pub(crate) mod telemetry;

fn required_input<'a>(
    inputs: &'a [&[u8]],
    input_index: usize,
    binding_name: &str,
    context: &'static str,
    prefix: &str,
    input_kind: &str,
    fix: &str,
) -> Result<&'a [u8], vyre_driver::BackendError> {
    inputs
        .get(input_index)
        .copied()
        .ok_or_else(|| vyre_driver::BackendError::InvalidProgram {
            fix: format!(
                "Fix: {prefix} {context} expected {input_kind} index {input_index} for `{binding_name}` but only {} {input_kind}(s) were supplied. {fix}",
                inputs.len()
            ),
        })
}

macro_rules! define_required_input {
    ($name:ident, $prefix:literal, $input_kind:literal, $fix:literal) => {
        fn $name<'a>(
            inputs: &'a [&[u8]],
            input_index: usize,
            binding_name: &str,
            context: &'static str,
        ) -> Result<&'a [u8], vyre_driver::BackendError> {
            super::required_input(
                inputs,
                input_index,
                binding_name,
                context,
                $prefix,
                $input_kind,
                $fix,
            )
        }
    };
}
pub(crate) use define_required_input;

pub(crate) use allocations::*;
pub(crate) use module_cache::ModuleCacheKey;
pub(crate) use plan::CudaDispatchPlan;
pub(crate) use resident::{resident_bindings_from_handles, ResidentUseGuard};
pub(crate) use resident_dispatch_support::CudaResidentDispatchStep;
// Public surface  -  these names appear on the crate root.
pub use cuda_graph::CachedCudaGraph;
pub use dispatch::CudaBackend;
pub use module_cache::CudaPtxSourceCacheSnapshot;
pub use resident::CudaResidentBuffer;
pub use stream_ordered_pool::CudaStreamOrderedPool;
pub use telemetry::CudaTelemetrySnapshot;
