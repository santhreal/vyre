//! CUDA capability, feature-flag, and validation policy.

use std::sync::Arc;
use vyre_driver::validation::{LaunchGeometryLimits, ProgramValidationCaps};
use vyre_driver::BindingRole;
use vyre_driver::{BackendError, DispatchConfig, LaunchPlan};
use vyre_foundation::ir::Program;
use vyre_foundation::validate::ValidationOptions;

use super::dispatch::CudaBackend;
use super::module_cache::PtxSourceCacheKey;
use super::plan::CudaDispatchPlan;
use super::resident::CudaDispatchBinding;
use super::resident_dispatch::next_dispatch_binding;
use crate::kernel_failure_diagnostics::{
    diagnose_cuda_kernel_launch_shape, CudaKernelDeviceEnvelope, CudaKernelLaunchEnvelope,
    CudaKernelLaunchShape,
};
use crate::numeric::CUDA_NUMERIC;
use crate::occupancy::cooperative_thread_residency_block_limit;

const CUDA_TRANSIENT_DISPATCH_BUDGET_NUMERATOR: u64 = 9;
const CUDA_TRANSIENT_DISPATCH_BUDGET_DENOMINATOR: u64 = 10;

/// Pick which of two value-equal programs the PTX cache key is derived from,
/// preferring the one that still carries its memos.
///
/// `lower_subgroup_reductions` takes its program BY VALUE, so every caller
/// hands it a `Program::clone`. That clone shares every `Arc` and so costs
/// nothing in memory, but it deliberately resets `normalized_cache_digest`,
/// `fingerprint` and `hash` to empty `OnceLock`s, because a memo copied onto a
/// value that is about to be rewritten would be a stale digest, which is worse
/// than a slow one. The consequence on this path is that the digest was
/// recomputed from scratch on a throwaway value on EVERY dispatch, so the
/// per-program memo could never be hit even once: measured at 79 ns per IR
/// node, which is 92 percent of the PTX phase and the single largest host term
/// on the encode path for programs of 12,000 nodes.
///
/// The lowering is a no-op for most programs, and when it is, it hands the
/// input straight back rather than rebuilding it: of its three returns, two
/// return the program untouched and only the third calls
/// `with_rewritten_entry`. So `entry` pointer equality is a sound and O(1)
/// witness that nothing was rewritten, and in that case `original` and
/// `lowered` are the same program VALUE, differing only in which memos are
/// warm. Deriving the key from `original` therefore produces byte-identical
/// key input while letting the memo live as long as the caller's program.
///
/// Returning `original` when the entries are NOT pointer-equal would be a
/// correctness bug, not a slow path: the key would describe the unlowered
/// program while the emitted PTX came from the lowered one, so a program whose
/// subgroup reductions lower differently under two adapters would serve the
/// wrong cached PTX. The comparison is therefore the whole guard, and the
/// fallback is the lowered program rather than a guess.
///
/// The opposite failure, a MISSED no-op returning `lowered` for a program that
/// did not change, is silent in a nastier way: it simply restores the full
/// per-node digest cost with every gate still green. That is why the companion
/// test counts digest computations rather than timing them, and why the
/// comparison covers every field the digest reads instead of `entry` alone.
fn cache_identity_program<'a>(original: &'a Program, lowered: &'a Program) -> &'a Program {
    if Arc::ptr_eq(&original.entry, &lowered.entry)
        && Arc::ptr_eq(&original.buffers, &lowered.buffers)
        && original.workgroup_size == lowered.workgroup_size
        && original.non_composable_with_self == lowered.non_composable_with_self
        && original.entry_op_id == lowered.entry_op_id
    {
        original
    } else {
        lowered
    }
}

impl CudaBackend {
    /// Compute capability as (major, minor).
    #[must_use]
    pub fn compute_capability(&self) -> (u32, u32) {
        self.caps.compute_capability
    }

    /// CUDA SM target number used by PTX emission.
    #[must_use]
    pub fn target_sm(&self) -> u32 {
        self.caps.native_sm()
    }

    /// CUDA SM target used by the current PTX ISA emitter.
    #[must_use]
    pub fn ptx_target_sm(&self) -> u32 {
        self.ptx_target_sm
    }

    /// Total device memory in bytes.
    #[must_use]
    pub fn device_memory_bytes(&self) -> u64 {
        self.caps.total_memory
    }

    /// Maximum number of threads per CUDA block.
    #[must_use]
    pub fn max_threads_per_block(&self) -> u32 {
        self.caps.max_threads_per_block_u32()
    }

    /// Maximum CUDA block dimensions.
    #[must_use]
    pub fn max_block_dim(&self) -> [u32; 3] {
        self.caps.max_block_dim_u32()
    }

    /// Maximum CUDA grid dimensions.
    #[must_use]
    pub fn max_grid_dim(&self) -> [u32; 3] {
        self.caps.max_grid_dim_u32()
    }

    /// Maximum threads resident on one streaming multiprocessor.
    ///
    /// Blocks per SM is `this / block threads` with an integral division, so a
    /// width that does not divide it strands the remainder on every SM.
    #[must_use]
    pub fn max_threads_per_sm(&self) -> u32 {
        self.caps.max_threads_per_sm_u32()
    }

    /// Shared memory available per CUDA thread block in bytes.
    #[must_use]
    pub fn max_shared_memory_per_block_bytes(&self) -> u32 {
        self.caps.shared_memory_per_block_bytes()
    }

    /// CUDA warp size used by subgroup-style execution.
    #[must_use]
    pub fn warp_size(&self) -> Option<u32> {
        self.caps.warp_size_u32()
    }

    /// Whether the device has hardware subgroup/warp execution.
    #[must_use]
    pub fn hardware_supports_subgroup_ops(&self) -> bool {
        self.warp_size()
            .map(vyre_driver::SubgroupCaps::native)
            .is_some_and(|caps| caps.supports_subgroup)
    }

    /// Whether the device can execute asynchronous CUDA work concurrently.
    #[must_use]
    pub fn hardware_supports_async_compute(&self) -> bool {
        self.caps.concurrent_kernels || self.caps.async_engine_count > 0
    }

    /// Whether this device can run a cooperative whole-grid barrier.
    #[must_use]
    pub fn hardware_supports_grid_sync(&self) -> bool {
        self.caps.compute_capability >= (6, 0) && self.caps.cooperative_launch
    }

    /// Whether the device generation has native fp16 arithmetic support.
    #[must_use]
    pub fn hardware_supports_f16(&self) -> bool {
        self.caps.hardware_supports_f16()
    }

    /// Whether the device generation has native bf16 arithmetic support.
    #[must_use]
    pub fn hardware_supports_bf16(&self) -> bool {
        self.caps.hardware_supports_bf16()
    }

    /// Whether the device generation has NVIDIA tensor-core instructions.
    #[must_use]
    pub fn hardware_supports_tensor_cores(&self) -> bool {
        self.caps.hardware_supports_tensor_cores()
    }

    /// Whether this backend launches grid-sync kernels through the cooperative ABI.
    ///
    /// The PTX emitter lowers each `MemoryOrdering::GridSync` barrier to a
    /// monotonic-counter cooperative grid barrier
    /// (`vyre_emit_ptx` `emit_grid_sync_barrier`) backed by the module-scope
    /// `_vyre_grid_barrier` counter, and the host dispatch path launches such
    /// kernels with `cuLaunchCooperativeKernel`, zeroing the counter before
    /// each launch. `supports_grid_sync()` additionally gates on
    /// `hardware_supports_grid_sync()` so non-cooperative-capable devices still
    /// route to the kernel-split path; when the cooperative grid cannot be made
    /// fully resident the launch returns `CooperativeResidencyExceeded` and the
    /// orchestrator falls back to the resident-fixpoint path.
    #[must_use]
    pub fn lowers_grid_sync(&self) -> bool {
        true
    }

    /// Whether CUDA can execute `MemoryOrdering::GridSync` inside one dispatch.
    pub fn supports_grid_sync(&self) -> bool {
        self.hardware_supports_grid_sync() && self.lowers_grid_sync()
    }

    /// Whether CUDA PTX lowering emits tensor-core instructions.
    #[must_use]
    pub fn lowers_tensor_core_ops(&self) -> bool {
        true
    }

    /// Pipeline feature flags that participate in shared cache identity.
    #[must_use]
    pub fn pipeline_feature_flags(&self) -> vyre_driver::PipelineFeatureFlags {
        let mut flags = vyre_driver::PipelineFeatureFlags::empty();
        if self.hardware_supports_subgroup_ops() {
            flags = flags.union(vyre_driver::PipelineFeatureFlags::SUBGROUP_OPS);
        }
        if self.hardware_supports_f16() {
            flags = flags.union(vyre_driver::PipelineFeatureFlags::F16);
        }
        if self.hardware_supports_bf16() {
            flags = flags.union(vyre_driver::PipelineFeatureFlags::BF16);
        }
        if self.hardware_supports_tensor_cores() && self.lowers_tensor_core_ops() {
            flags = flags.union(vyre_driver::PipelineFeatureFlags::TENSOR_CORES);
        }
        if self.hardware_supports_async_compute() {
            flags = flags.union(vyre_driver::PipelineFeatureFlags::ASYNC_COMPUTE);
        }
        flags
    }

    pub(crate) fn ptx_for_program_cached(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<Arc<str>, BackendError> {
        self.ptx_for_program_cached_with_key(program, config)
            .map(|(ptx, _)| ptx)
    }

    pub(crate) fn ptx_for_program_cached_with_key(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<(Arc<str>, PtxSourceCacheKey), BackendError> {
        let subgroup_size = self.warp_size().ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: CUDA device probe reported no hardware warp size on a GPU-required host; fix the CUDA capability probe before lowering."
                .to_string(),
        })?;
        let lowered_program = vyre_foundation::lower::lower_subgroup_reductions(
            program.clone(),
            &self.caps.to_adapter_caps(),
        );
        let key = self.ptx_source_cache.key_for_program(
            cache_identity_program(program, &lowered_program),
            config,
            self.ptx_target_sm(),
            subgroup_size,
            self.pipeline_feature_flags(),
        )?;
        let ptx = self.ptx_source_cache.get_or_lower(key, || {
            crate::codegen::program_to_ptx_for_sm_and_subgroup(
                &lowered_program,
                config,
                self.ptx_target_sm(),
                subgroup_size,
            )
            .map_err(|compiler_message| BackendError::KernelCompileFailed {
                backend: crate::CUDA_BACKEND_ID.to_string(),
                compiler_message,
            })
        })?;
        Ok((ptx, key))
    }

    pub(crate) fn launch_limits(&self) -> LaunchGeometryLimits {
        LaunchGeometryLimits {
            backend: "CUDA",
            max_threads_per_block: self.max_threads_per_block(),
            max_block_dim: self.max_block_dim(),
            max_grid_dim: self.max_grid_dim(),
            max_threads_per_sm: self.caps.max_threads_per_sm_u32(),
        }
    }

    /// Device capability envelope used by release launch diagnostics.
    #[must_use]
    pub fn kernel_device_envelope(&self) -> CudaKernelDeviceEnvelope {
        let sm_major = if self.caps.compute_capability.0 > u32::from(u16::MAX) {
            tracing::error!(
                "CUDA compute capability major value {} cannot fit u16. Fix: widen CudaKernelDeviceEnvelope before release diagnostics.",
                self.caps.compute_capability.0
            );
            u16::MAX
        } else {
            self.caps.compute_capability.0 as u16
        };
        let sm_minor = if self.caps.compute_capability.1 > u32::from(u16::MAX) {
            tracing::error!(
                "CUDA compute capability minor value {} cannot fit u16. Fix: widen CudaKernelDeviceEnvelope before release diagnostics.",
                self.caps.compute_capability.1
            );
            u16::MAX
        } else {
            self.caps.compute_capability.1 as u16
        };
        CudaKernelDeviceEnvelope {
            sm_major,
            sm_minor,
            max_threads_per_block: self.max_threads_per_block(),
            shared_memory_per_block_bytes: u64::from(self.max_shared_memory_per_block_bytes()),
            supports_cooperative_launch: self.hardware_supports_grid_sync(),
            supports_tensor_cores: self.hardware_supports_tensor_cores()
                && self.lowers_tensor_core_ops(),
        }
    }

    /// Build the release launch envelope for a prepared CUDA launch plan.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when launch shape products overflow the
    /// diagnostic envelope.
    pub fn diagnose_launch_plan(
        &self,
        kernel: &'static str,
        launch: &LaunchPlan,
        cooperative: bool,
        requires_tensor_cores: bool,
    ) -> Result<CudaKernelLaunchEnvelope, BackendError> {
        let threads_per_block = CUDA_NUMERIC
            .checked_dim_product_u64(launch.workgroup)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA launch diagnostic workgroup product overflowed u64 for {:?}. Lower the workgroup before dispatch.",
                    launch.workgroup
                ),
            })?;
        let cooperative_resident_block_limit = if cooperative {
            let threads_per_block = u32::try_from(threads_per_block).map_err(|source| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA launch diagnostic workgroup {:?} has {threads_per_block} thread slots, which does not fit u32: {source}. Lower the workgroup before dispatch.",
                        launch.workgroup
                    ),
                }
            })?;
            Some(cooperative_thread_residency_block_limit(
                &self.caps,
                threads_per_block,
            ))
        } else {
            None
        };
        diagnose_cuda_kernel_launch_shape(
            kernel,
            self.kernel_device_envelope(),
            CudaKernelLaunchShape {
                grid: launch.grid,
                block: launch.workgroup,
                dynamic_shared_memory_bytes: 0,
                cooperative,
                requires_tensor_cores,
            },
            cooperative_resident_block_limit,
        )
        .map_err(|error| BackendError::InvalidProgram { fix: error.fix })
    }

    pub(crate) fn program_validation_caps(&self) -> ProgramValidationCaps {
        ProgramValidationCaps {
            backend_id: crate::CUDA_BACKEND_ID,
            supports_subgroup_ops: self.hardware_supports_subgroup_ops(),
            supports_f16: self.hardware_supports_f16(),
            supports_bf16: self.hardware_supports_bf16(),
            supports_indirect_dispatch: false,
            // True because a trapping lane now leaves a record the host reads: the
            // emitter writes address, tag code, and lane into the module-scope trap
            // sidecar under a compare-and-swap, and the launch path zeroes the
            // record before the sequence and reads it back after synchronizing, so
            // a trapped launch refuses instead of returning wrong data. This and
            // the device profile must report the same answer.
            supports_trap_propagation: true,
            supports_distributed_collectives: false,
            max_workgroup_size: self.max_block_dim(),
        }
    }

    pub(crate) fn validation_options(&self) -> ValidationOptions<'_> {
        ValidationOptions::default()
            .with_backend_capabilities(self.caps.to_device_profile().validation_capabilities())
            .with_shadowing(true)
    }

    pub(crate) fn validate_transient_dispatch_memory_budget(
        &self,
        prepared: &CudaDispatchPlan,
        inputs: &[&[u8]],
        context: &'static str,
    ) -> Result<(), BackendError> {
        let required_bytes = cuda_transient_dispatch_required_bytes(prepared, inputs)?;
        let budget_bytes = cuda_transient_dispatch_live_available_budget_bytes(
            self.caps.total_memory,
            cuda_live_free_memory_bytes()?,
            self.resident_store.allocated_bytes(),
            cuda_usize_bytes_to_u64(
                self.transient_pool.allocated_bytes()?,
                "transient pool allocated bytes",
            )?,
        );
        let budget_bytes = self
            .reclaim_cached_transient_allocations_when_over_budget(required_bytes, budget_bytes)?;
        validate_cuda_transient_dispatch_budget(required_bytes, budget_bytes, context)
    }

    /// Preflight the transient device memory a mixed resident/borrowed
    /// dispatch will stage.
    ///
    /// Only borrowed bindings allocate, so an all-resident dispatch returns
    /// without querying the driver and keeps the resident hot path free of
    /// added FFI.
    pub(crate) fn validate_mixed_dispatch_staging_budget(
        &self,
        prepared: &CudaDispatchPlan,
        bindings: &[CudaDispatchBinding<'_>],
        context: &'static str,
    ) -> Result<(), BackendError> {
        let required_bytes = cuda_mixed_dispatch_staging_bytes(prepared, bindings)?;
        if required_bytes == 0 {
            return Ok(());
        }
        let budget_bytes = cuda_transient_dispatch_live_available_budget_bytes(
            self.caps.total_memory,
            cuda_live_free_memory_bytes()?,
            self.resident_store.allocated_bytes(),
            cuda_usize_bytes_to_u64(
                self.transient_pool.allocated_bytes()?,
                "transient pool allocated bytes",
            )?,
        );
        let budget_bytes = self
            .reclaim_cached_transient_allocations_when_over_budget(required_bytes, budget_bytes)?;
        validate_cuda_transient_dispatch_budget(required_bytes, budget_bytes, context)
    }

    pub(crate) fn validate_transient_allocation_memory_budget(
        &self,
        byte_len: usize,
        label: &str,
        context: &str,
    ) -> Result<(), BackendError> {
        let required_bytes = cuda_dispatch_allocation_bucket(byte_len, label)?;
        let budget_bytes = cuda_transient_dispatch_live_available_budget_bytes(
            self.caps.total_memory,
            cuda_live_free_memory_bytes()?,
            self.resident_store.allocated_bytes(),
            cuda_usize_bytes_to_u64(
                self.transient_pool.allocated_bytes()?,
                "transient pool allocated bytes",
            )?,
        );
        let budget_bytes = self
            .reclaim_cached_transient_allocations_when_over_budget(required_bytes, budget_bytes)?;
        validate_cuda_transient_dispatch_budget(required_bytes, budget_bytes, context)
    }

    fn reclaim_cached_transient_allocations_when_over_budget(
        &self,
        required_bytes: u64,
        budget_bytes: u64,
    ) -> Result<u64, BackendError> {
        if required_bytes <= budget_bytes {
            return Ok(budget_bytes);
        }
        self.transient_pool.clear()?;
        Ok(cuda_transient_dispatch_live_available_budget_bytes(
            self.caps.total_memory,
            cuda_live_free_memory_bytes()?,
            self.resident_store.allocated_bytes(),
            cuda_usize_bytes_to_u64(
                self.transient_pool.allocated_bytes()?,
                "transient pool allocated bytes after reclaim",
            )?,
        ))
    }

    pub(crate) fn validate_program_cached(&self, program: &Program) -> Result<(), BackendError> {
        if !crate::instrumentation::cuda_dispatch_validation_enabled() {
            return Ok(());
        }
        self.validation_cache.get_or_validate(
            program,
            self.validation_options(),
            crate::cuda_supported_ops(),
            self.program_validation_caps(),
        )
    }
}

pub(crate) fn cuda_transient_dispatch_budget_bytes(total_memory: u64) -> u64 {
    let numerator = u128::from(total_memory) * u128::from(CUDA_TRANSIENT_DISPATCH_BUDGET_NUMERATOR);
    (numerator / u128::from(CUDA_TRANSIENT_DISPATCH_BUDGET_DENOMINATOR)) as u64
}

pub(crate) fn cuda_live_free_memory_bytes() -> Result<u64, BackendError> {
    let (free, _total) = cudarc::driver::result::mem_get_info().map_err(|error| {
        BackendError::DispatchFailed {
            code: None,
            message: format!(
                "CUDA live-memory query failed: {error}. Fix: keep the CUDA context bound before memory preflight and treat query failure as a GPU release-path configuration error, not a CPU escape."
            ),
        }
    })?;
    cuda_usize_bytes_to_u64(free, "CUDA live free memory bytes")
}

pub(crate) fn cuda_transient_dispatch_available_budget_bytes(
    total_memory: u64,
    resident_bytes: u64,
    transient_pool_bytes: u64,
) -> u64 {
    let budget = u128::from(cuda_transient_dispatch_budget_bytes(total_memory));
    let used = u128::from(resident_bytes) + u128::from(transient_pool_bytes);
    if used >= budget {
        0
    } else {
        (budget - used) as u64
    }
}

pub(crate) fn cuda_transient_dispatch_live_available_budget_bytes(
    total_memory: u64,
    live_free_memory: u64,
    resident_bytes: u64,
    transient_pool_bytes: u64,
) -> u64 {
    let accounted = cuda_transient_dispatch_available_budget_bytes(
        total_memory,
        resident_bytes,
        transient_pool_bytes,
    );
    let live = cuda_transient_dispatch_budget_bytes(live_free_memory);
    accounted.min(live)
}

pub(crate) fn cuda_transient_dispatch_required_bytes(
    prepared: &CudaDispatchPlan,
    inputs: &[&[u8]],
) -> Result<u64, BackendError> {
    let mut required_bytes = 0u64;
    for binding in &prepared.bindings.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let byte_len = match binding.input_index {
            Some(input_index) => inputs
                .get(input_index)
                .map(|input| input.len())
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA dispatch memory preflight expected input index {input_index} for `{}` but only {} input(s) were supplied.",
                        binding.name,
                        inputs.len()
                    ),
                })?,
            None => binding.static_byte_len.ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch memory preflight needs a static byte length for output `{}`; set BufferDecl::with_count or output_byte_range before launch.",
                    binding.name
                ),
            })?,
        };
        required_bytes = checked_dispatch_bytes_add(
            required_bytes,
            cuda_dispatch_allocation_bucket(byte_len, "CUDA dispatch buffer bytes")?,
            "CUDA dispatch buffer bytes",
        )?;
    }
    let param_bytes = super::launch_params::launch_param_byte_len(
        &prepared.launch.param_words,
        "dispatch memory preflight",
    )?;
    let param_allocation_bytes = if param_bytes == 0 {
        0
    } else {
        cuda_dispatch_allocation_bucket(param_bytes, "CUDA dispatch parameter bytes")?
    };
    checked_dispatch_bytes_add(
        required_bytes,
        param_allocation_bytes,
        "CUDA dispatch parameter bytes",
    )
}

/// Transient device bytes a mixed dispatch stages for its borrowed bindings.
///
/// Resident bindings already own their device memory and contribute nothing,
/// which is the whole point of leaving them resident.
fn cuda_mixed_dispatch_staging_bytes(
    prepared: &CudaDispatchPlan,
    bindings: &[CudaDispatchBinding<'_>],
) -> Result<u64, BackendError> {
    let mut required_bytes = 0u64;
    let mut next_binding = 0usize;
    for binding in &prepared.bindings.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let source = next_dispatch_binding(
            bindings,
            &mut next_binding,
            "mixed dispatch memory preflight",
        )?;
        let CudaDispatchBinding::Borrowed(bytes) = source else {
            continue;
        };
        let byte_len = match binding.input_index {
            Some(_) => bytes.len(),
            None => binding.static_byte_len.ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch memory preflight needs a static byte length for borrowed output `{}`; set BufferDecl::with_count or output_byte_range before launch.",
                    binding.name
                ),
            })?,
        };
        required_bytes = checked_dispatch_bytes_add(
            required_bytes,
            cuda_dispatch_allocation_bucket(byte_len, "CUDA mixed dispatch staging bytes")?,
            "CUDA mixed dispatch staging bytes",
        )?;
    }
    Ok(required_bytes)
}

pub(crate) fn validate_cuda_transient_dispatch_budget(
    required_bytes: u64,
    budget_bytes: u64,
    context: &str,
) -> Result<(), BackendError> {
    if required_bytes > budget_bytes {
        return Err(BackendError::InvalidProgram {
        fix: format!(
            "Fix: {context} requires {required_bytes} transient CUDA device bytes but the live-device preflight budget is {budget_bytes} bytes. Reduce input/output size, shard the dispatch, use resident handles with explicit reuse, or raise the CUDA memory budget deliberately."
        ),
    });
    }
    Ok(())
}

fn checked_dispatch_bytes_add(
    left: u64,
    right: u64,
    field: &'static str,
) -> Result<u64, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: {field} overflowed u64 during CUDA memory preflight. Shard the dispatch before CUDA allocation."
            ),
        })
}

fn cuda_dispatch_allocation_bucket(byte_len: usize, field: &str) -> Result<u64, BackendError> {
    let bucket = byte_len
    .max(1)
    .checked_next_power_of_two()
    .ok_or_else(|| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {field} request of {byte_len} bytes cannot be rounded to the CUDA allocation bucket. Shard the dispatch before CUDA allocation."
        ),
    })?;
    cuda_usize_bytes_to_u64(bucket, field)
}

fn cuda_usize_bytes_to_u64(byte_len: usize, field: &str) -> Result<u64, BackendError> {
    u64::try_from(byte_len).map_err(|_| BackendError::InvalidProgram {
    fix: format!(
        "Fix: {field} value of {byte_len} bytes cannot fit u64 CUDA memory telemetry. Shard the dispatch or widen budget accounting."
    ),
})
}

// Inline: covers `cache_identity_program`, `cuda_transient_dispatch_budget_bytes`,
// `cuda_transient_dispatch_live_available_budget_bytes`, `cuda_transient_dispatch_required_bytes`
// and 1 more item this module keeps private, which no integration test can name.
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use smallvec::smallvec;
    use vyre_driver::{BackendError, LaunchPlan};
    use vyre_driver::{Binding, BindingPlan, BindingRole};

    use super::{
        cuda_transient_dispatch_budget_bytes, cuda_transient_dispatch_live_available_budget_bytes,
        cuda_transient_dispatch_required_bytes, validate_cuda_transient_dispatch_budget,
    };
    use crate::backend::CudaDispatchPlan;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
    use vyre_foundation::lower::lower_subgroup_reductions;
    use vyre_foundation::optimizer::AdapterCaps;

    use super::cache_identity_program;

    fn plan(static_output_bytes: usize) -> CudaDispatchPlan {
        CudaDispatchPlan {
            bindings: BindingPlan {
                bindings: vec![
                    Binding {
                        name: Arc::from("input"),
                        binding: 0,
                        buffer_index: 0,
                        role: BindingRole::Input,
                        element_size: 1,
                        preferred_alignment: 1,
                        element_count: 8,
                        static_byte_len: Some(8),
                        input_index: Some(0),
                        output_index: None,
                    },
                    Binding {
                        name: Arc::from("output"),
                        binding: 1,
                        buffer_index: 1,
                        role: BindingRole::Output,
                        element_size: 1,
                        preferred_alignment: 1,
                        element_count: u32::try_from(static_output_bytes).expect("Fix: CUDA parity tests require backend dispatch; skip test if GPU unavailable, do not panic - test CUDA dispatch plan static output bytes must fit u32 element count",
                        ),
                        static_byte_len: Some(static_output_bytes),
                        input_index: None,
                        output_index: Some(0),
                    },
                ],
                input_indices: vec![0],
                output_indices: vec![1],
                shared_indices: vec![],
            },
            output_binding_indices: smallvec![1],
            launch: LaunchPlan {
                grid: [1, 1, 1],
                workgroup: [32, 1, 1],
                element_count: 8,
                param_words: vec![1, 2],
                max_binding_alignment: 1,
            },
            cooperative: false,
            fixpoint_iterations: 1,
        }
    }

    #[test]
    fn transient_dispatch_memory_preflight_sums_buffers_and_params() {
        let input = [0u8; 8];
        let required = cuda_transient_dispatch_required_bytes(&plan(16), &[input.as_slice()])
            .expect("Fix: valid dispatch memory plan should sum");

        assert_eq!(required, 8 + 16 + 8);
    }

    #[test]
    fn transient_dispatch_memory_preflight_does_not_charge_empty_params() {
        let input = [0u8; 8];
        let mut plan = plan(16);
        plan.launch.param_words.clear();
        let required = cuda_transient_dispatch_required_bytes(&plan, &[input.as_slice()])
            .expect("Fix: valid zero-param dispatch memory plan should sum");

        assert_eq!(
            required,
            8 + 16,
            "Fix: CUDA memory preflight must not charge a rounded one-byte allocation for empty launch params."
        );
    }

    #[test]
    fn transient_dispatch_memory_preflight_counts_bucketed_allocation_pressure() {
        let input = [0u8; 9];
        let required = cuda_transient_dispatch_required_bytes(&plan(17), &[input.as_slice()])
            .expect("Fix: valid dispatch memory plan should sum bucketed allocation pressure");

        assert_eq!(required, 16 + 32 + 8);
    }

    #[test]
    fn transient_dispatch_memory_preflight_rejects_over_budget_before_allocation() {
        let error = validate_cuda_transient_dispatch_budget(1025, 1024, "CUDA test dispatch")
            .expect_err("over-budget dispatch must fail before CUDA allocation");

        match error {
            BackendError::InvalidProgram { fix } => {
                assert!(fix.contains("CUDA test dispatch requires 1025"));
                assert!(fix.contains("preflight budget is 1024"));
                assert!(fix.contains("Shard") || fix.contains("shard"));
            }
            other => panic!("expected InvalidProgram, got {other:?}"),
        }
    }

    #[test]
    fn transient_dispatch_budget_uses_conservative_live_vram_fraction() {
        assert_eq!(cuda_transient_dispatch_budget_bytes(1000), 900);
        assert_eq!(
            cuda_transient_dispatch_budget_bytes(u64::MAX),
            16_602_069_666_338_596_453,
            "Fix: CUDA transient budget must widen before multiplying so huge live-memory probes do not saturate before division."
        );
    }

    #[test]
    fn transient_dispatch_live_available_budget_caps_against_free_vram() {
        assert_eq!(
            cuda_transient_dispatch_live_available_budget_bytes(10_000, 1_000, 0, 0),
            900,
            "Fix: CUDA preflight must cap dispatch pressure against live free VRAM, not just total board memory."
        );
        assert_eq!(
            cuda_transient_dispatch_live_available_budget_bytes(10_000, 8_000, 2_000, 1_000),
            6_000,
            "Fix: CUDA preflight must still subtract resident and transient allocations from the total-device budget."
        );
        assert_eq!(
            cuda_transient_dispatch_live_available_budget_bytes(10_000, 0, 0, 0),
            0,
            "Fix: zero live free VRAM must produce zero preflight budget instead of allowing optimistic allocation."
        );
    }

    /// Fixture whose region body the subgroup lowering pass rewrites.
    ///
    /// The pass only fires on a `Region` whose generator carries a canonical
    /// `vyre-libs::reduce::workgroup_*` prefix and whose body yields both
    /// a scratch buffer and a reduction scope, so a plain program is a no-op
    /// and cannot exercise the rewriting half of the discriminator.
    fn lowering_program() -> Program {
        Program::wrapped(
            vec![BufferDecl::output("scratch", 0, DataType::U32).with_count(64)],
            [64, 1, 1],
            vec![Node::Region {
                generator: "vyre-libs::reduce::workgroup_sum_u32".into(),
                source_region: None,
                body: Arc::new(vec![Node::store("scratch", Expr::u32(0), Expr::u32(7))]),
            }],
        )
    }

    fn plain_program() -> Program {
        Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        )
    }

    fn subgroup_caps() -> AdapterCaps {
        AdapterCaps {
            supports_subgroup_ops: true,
            subgroup_size: 32,
            ..AdapterCaps::default()
        }
    }

    /// A no-op lowering MUST hand the key back the caller's own program, so the
    /// digest memo lands on the long-lived value instead of on a clone that
    /// dies at the end of the dispatch.
    ///
    /// This is the MISSED NO-OP guard. Before the fix the key was always
    /// derived from the lowering pass's return value, and because
    /// `Program::clone` starts every memo empty, the normalized digest was
    /// recomputed from scratch on every dispatch forever: 79 ns per IR node,
    /// 92 percent of the measured PTX phase, and the largest host term on the
    /// encode path. If this regresses nothing fails and nothing is wrong, the
    /// encode simply pays a full whole-program hash per dispatch again, which
    /// is exactly why the assertion is pointer identity and not a duration.
    #[test]
    fn a_no_op_lowering_keys_from_the_callers_own_program() {
        let program = plain_program();
        let lowered = lower_subgroup_reductions(program.clone(), &subgroup_caps());

        assert!(
            Arc::ptr_eq(&program.entry, &lowered.entry),
            "Fix: fixture must not be rewritten by subgroup lowering, or this \
             test proves nothing about the no-op path."
        );
        assert!(
            std::ptr::eq(cache_identity_program(&program, &lowered), &program),
            "Fix: a no-op lowering must key from the caller's program so the \
             normalized digest memo outlives the dispatch."
        );
    }

    /// A REWRITING lowering MUST key from the lowered program, never the
    /// caller's.
    ///
    /// This is the FALSE NO-OP guard and it is a correctness test, not a
    /// performance one. Keying from the pre-lowering program while emitting PTX
    /// from the post-lowering one would file that PTX under the unlowered
    /// program's identity, so a later dispatch of the unlowered form would be
    /// served PTX containing subgroup reductions it never asked for. That is a
    /// wrong-kernel bug and it would not show up as a slowdown.
    #[test]
    fn a_rewriting_lowering_keys_from_the_lowered_program() {
        let program = lowering_program();
        let lowered = lower_subgroup_reductions(program.clone(), &subgroup_caps());

        assert!(
            !Arc::ptr_eq(&program.entry, &lowered.entry),
            "Fix: fixture must actually be rewritten by subgroup lowering, or \
             this test cannot see a false no-op."
        );
        assert!(
            std::ptr::eq(cache_identity_program(&program, &lowered), &lowered),
            "Fix: a rewriting lowering must key from the lowered program or the \
             PTX cache files lowered PTX under the unlowered program's identity."
        );
    }

    /// The two programs the no-op path treats as interchangeable MUST produce
    /// byte-identical digests.
    ///
    /// Pointer identity is the mechanism, but this is the property that makes
    /// substituting one for the other sound: the digest is a pure function of
    /// the program value, so equal values give equal key input and the swap
    /// cannot move a cache key. Asserting the exact 32 bytes rather than "the
    /// keys agree" is what makes this fail if a future field is added to the
    /// digest that `cache_identity_program` does not compare.
    #[test]
    fn the_no_op_substitution_does_not_move_the_digest() {
        let program = plain_program();
        let lowered = lower_subgroup_reductions(program.clone(), &subgroup_caps());

        let from_original = vyre_driver::try_normalized_program_cache_digest(&program)
            .expect("Fix: fixture program must produce a normalized cache digest");
        let from_lowered = vyre_driver::try_normalized_program_cache_digest(&lowered)
            .expect("Fix: lowered fixture must produce a normalized cache digest");

        assert_eq!(
            from_original, from_lowered,
            "Fix: the no-op substitution changed the digest, so deriving the \
             PTX key from the caller's program would move the cache key."
        );
    }

    /// A rewritten program MUST NOT share the unlowered program's digest.
    ///
    /// The negative control for the test above. If lowering could produce a
    /// program whose digest matched its input, the FALSE NO-OP guard would be
    /// defending nothing, because keying from either program would be
    /// indistinguishable and a real rewrite could be served the wrong PTX
    /// without any assertion noticing.
    #[test]
    fn a_rewritten_program_gets_a_different_digest() {
        let program = lowering_program();
        let lowered = lower_subgroup_reductions(program.clone(), &subgroup_caps());

        let from_original = vyre_driver::try_normalized_program_cache_digest(&program)
            .expect("Fix: fixture program must produce a normalized cache digest");
        let from_lowered = vyre_driver::try_normalized_program_cache_digest(&lowered)
            .expect("Fix: lowered fixture must produce a normalized cache digest");

        assert_ne!(
            from_original, from_lowered,
            "Fix: subgroup lowering rewrote the program without moving its \
             digest, so the PTX cache cannot tell the two forms apart."
        );
    }

    /// Every field the comparison reads MUST be able to veto the substitution
    /// on its own.
    ///
    /// `cache_identity_program` returns the caller's program only when all five
    /// compared fields agree. A future edit that drops one of them, or reorders
    /// the `&&` chain into something short-circuiting incorrectly, would let a
    /// genuinely different program pass as a no-op. Each case below differs
    /// from the base in exactly ONE field, so a dropped comparison fails here
    /// and names itself rather than surfacing as a wrong cached kernel.
    #[test]
    fn any_single_differing_field_vetoes_the_substitution() {
        let base = plain_program();

        let other_entry = Program::wrapped(
            base.buffers.to_vec(),
            base.workgroup_size,
            vec![Node::store("out", Expr::u32(0), Expr::u32(2))],
        );
        assert!(
            std::ptr::eq(cache_identity_program(&base, &other_entry), &other_entry),
            "Fix: a different entry body must veto the no-op substitution."
        );

        let other_workgroup =
            Program::wrapped(base.buffers.to_vec(), [32, 1, 1], base.entry.to_vec());
        assert!(
            std::ptr::eq(
                cache_identity_program(&base, &other_workgroup),
                &other_workgroup
            ),
            "Fix: a different workgroup size must veto the no-op substitution."
        );

        let other_buffers = Program::wrapped(
            vec![BufferDecl::output("out", 1, DataType::U32).with_count(4)],
            base.workgroup_size,
            base.entry.to_vec(),
        );
        assert!(
            std::ptr::eq(
                cache_identity_program(&base, &other_buffers),
                &other_buffers
            ),
            "Fix: a different buffer table must veto the no-op substitution."
        );

        let other_op_id = base.clone().with_entry_op_id("some::other::op");
        assert!(
            std::ptr::eq(cache_identity_program(&base, &other_op_id), &other_op_id),
            "Fix: a different entry op id must veto the no-op substitution."
        );
    }

    /// Negative control: each plausible simplification of
    /// `cache_identity_program` MUST be visible to one of the gates above.
    ///
    /// A gate is worth nothing until it has been shown to fail on the exact
    /// defect it exists to catch, and the three defects below are the ones a
    /// future reader is most likely to introduce while "cleaning up" a
    /// five-clause `&&`. Rather than patch production and leave the shared tree
    /// broken for the length of a test run, each defect is reproduced here as a
    /// local twin and checked against the SAME fixture pairs the gates use. If
    /// a twin agreed with the correct implementation on those fixtures, the
    /// corresponding gate would be blind and this test says so.
    ///
    /// Mapping, so a failure here names its own gate:
    ///   always_original  ->  a_rewriting_lowering_keys_from_the_lowered_program
    ///   always_lowered   ->  a_no_op_lowering_keys_from_the_callers_own_program
    ///   entry_only       ->  any_single_differing_field_vetoes_the_substitution
    #[test]
    fn each_discriminator_defect_is_visible_to_one_gate() {
        fn always_original<'a>(original: &'a Program, _lowered: &'a Program) -> &'a Program {
            original
        }
        fn always_lowered<'a>(_original: &'a Program, lowered: &'a Program) -> &'a Program {
            lowered
        }
        fn entry_only<'a>(original: &'a Program, lowered: &'a Program) -> &'a Program {
            if Arc::ptr_eq(&original.entry, &lowered.entry) {
                original
            } else {
                lowered
            }
        }

        let rewriting = lowering_program();
        let rewritten = lower_subgroup_reductions(rewriting.clone(), &subgroup_caps());
        assert!(
            std::ptr::eq(always_original(&rewriting, &rewritten), &rewriting)
                && !std::ptr::eq(cache_identity_program(&rewriting, &rewritten), &rewriting),
            "Fix: the rewriting fixture no longer separates a false no-op from \
             correct behaviour, so the false-no-op gate is blind."
        );

        let plain = plain_program();
        let unchanged = lower_subgroup_reductions(plain.clone(), &subgroup_caps());
        assert!(
            std::ptr::eq(always_lowered(&plain, &unchanged), &unchanged)
                && std::ptr::eq(cache_identity_program(&plain, &unchanged), &plain),
            "Fix: the no-op fixture no longer separates a missed no-op from \
             correct behaviour, so the missed-no-op gate is blind."
        );

        let base = plain_program();
        // `Program::clone` shares the entry `Arc`, which is exactly the state
        // an entry-only comparison cannot distinguish: same entry pointer,
        // different workgroup size.
        let mut same_entry_other_workgroup = base.clone();
        same_entry_other_workgroup.workgroup_size = [32, 1, 1];
        assert!(
            Arc::ptr_eq(&base.entry, &same_entry_other_workgroup.entry),
            "Fix: this control needs two programs that SHARE an entry and differ \
             elsewhere, or it cannot see an entry-only comparison."
        );
        assert!(
            std::ptr::eq(entry_only(&base, &same_entry_other_workgroup), &base)
                && std::ptr::eq(
                    cache_identity_program(&base, &same_entry_other_workgroup),
                    &same_entry_other_workgroup
                ),
            "Fix: an entry-only comparison is indistinguishable from the full \
             one on this fixture, so dropping four field comparisons would pass \
             every gate above."
        );
    }
}
