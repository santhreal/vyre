//! CUDA capability, feature-flag, and validation policy.

use std::sync::Arc;
use vyre_driver::validation::{LaunchGeometryLimits, ProgramValidationCaps};
use vyre_driver::{BackendError, DispatchConfig, LaunchPlan};
use vyre_foundation::ir::Program;
use vyre_foundation::validate::ValidationOptions;

use super::dispatch::CudaBackend;
use super::module_cache::PtxSourceCacheKey;
use super::plan::CudaDispatchPlan;
use super::resident::CudaDispatchBinding;
use crate::kernel_failure_diagnostics::{
    diagnose_cuda_kernel_launch_shape, CudaKernelDeviceEnvelope, CudaKernelLaunchEnvelope,
    CudaKernelLaunchShape,
};
use crate::numeric::CUDA_NUMERIC;
use crate::occupancy::cooperative_thread_residency_block_limit;

use super::transient_memory_budget::{
    cuda_dispatch_allocation_bucket, cuda_live_free_memory_bytes,
    cuda_mixed_dispatch_staging_bytes, cuda_transient_dispatch_live_available_budget_bytes,
    cuda_transient_dispatch_required_bytes, cuda_usize_bytes_to_u64,
    validate_cuda_transient_dispatch_budget,
};

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
    /// (`vyre_emit::ptx` `emit_grid_sync_barrier`) backed by the module-scope
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

    pub(crate) fn ptx_for_program_cached_with_key(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<(Arc<str>, PtxSourceCacheKey), BackendError> {
        let subgroup_size = self.warp_size().ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: CUDA device probe reported no hardware warp size on a GPU-required host; fix the CUDA capability probe before lowering."
                .to_string(),
        })?;
        // A borrowed result is the caller's program, so the key reads its warm
        // digest memo instead of recomputing on a per-dispatch value.
        let lowered_program = vyre_foundation::lower::lower_subgroup_reductions(
            program,
            &self.caps.to_adapter_caps(),
        );
        let key = self.ptx_source_cache.key_for_program(
            &lowered_program,
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
            // Asked of the device, not asserted: cooperative launch needs
            // compute capability 6.0 or later, and a device without it cannot
            // run a whole-grid barrier at any geometry. Whether a specific
            // dispatch fits co-resident is the separate question the residency
            // check answers.
            supports_grid_sync: self.supports_grid_sync(),
            // CUDA refuses to emulate a whole-grid barrier by splitting into
            // host launches, so callers get an unsupported-feature refusal
            // instead of a silently slower path.
            allows_host_grid_sync_split: false,
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
    use vyre_foundation::lower::lower_subgroup_reductions;
    use vyre_foundation::optimizer::AdapterCaps;

    fn subgroup_caps() -> AdapterCaps {
        AdapterCaps {
            supports_subgroup_ops: true,
            subgroup_size: 32,
            ..AdapterCaps::default()
        }
    }

    /// A no-op lowering MUST return the caller's own program, so the PTX key
    /// digest memo lands on the long-lived value instead of on a per-dispatch
    /// clone whose memo starts empty: 79 ns per IR node on every dispatch.
    #[test]
    fn a_no_op_lowering_borrows_the_callers_program() {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
            [64, 1, 1],
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
        );
        let lowered = lower_subgroup_reductions(&program, &subgroup_caps());
        assert!(
            matches!(lowered, Cow::Borrowed(borrowed) if std::ptr::eq(borrowed, &program)),
            "Fix: a no-op subgroup lowering must hand back the caller's program."
        );
    }

    /// A rewriting lowering MUST return a new program whose digest differs, or
    /// the PTX cache files lowered PTX under the unlowered program's identity.
    #[test]
    fn a_rewriting_lowering_owns_a_program_with_a_new_digest() {
        let program = Program::wrapped(
            vec![BufferDecl::output("scratch", 0, DataType::U32).with_count(64)],
            [64, 1, 1],
            vec![Node::Region {
                generator: "vyre-libs::reduce::workgroup_sum_u32".into(),
                source_region: None,
                body: Arc::new(vec![Node::store("scratch", Expr::u32(0), Expr::u32(7))]),
            }],
        );
        let lowered = lower_subgroup_reductions(&program, &subgroup_caps());
        assert!(
            matches!(lowered, Cow::Owned(_)),
            "Fix: fixture must be rewritten by subgroup lowering."
        );
        assert_ne!(
            vyre_driver::try_normalized_program_cache_digest(&program)
                .expect("Fix: fixture program must produce a normalized cache digest"),
            vyre_driver::try_normalized_program_cache_digest(&lowered)
                .expect("Fix: lowered fixture must produce a normalized cache digest"),
            "Fix: subgroup lowering rewrote the program without moving its digest."
        );
    }
}
