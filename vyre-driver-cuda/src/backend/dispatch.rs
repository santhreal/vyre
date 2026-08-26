//! CUDA backend: device lifecycle, buffer management, and kernel dispatch.

use std::sync::{Arc, Mutex};

use dashmap::DashMap;

use cudarc::driver::CudaContext;
use smallvec::SmallVec;
use vyre_driver::validation::ValidationCache;
use vyre_driver::SpeculationMode;
use vyre_driver::{resolve_fixpoint_iterations, BackendError, DispatchConfig, LaunchPlan};
use vyre_driver::{BindingPlan, BindingRole};
use vyre_foundation::ir::Program;
use vyre_megakernel::EmittedResources;

use super::allocations::DeviceAllocationPool;
use super::module_cache::{
    CudaModuleCache, CudaPtxSourceCache, CudaPtxSourceCacheSnapshot, ModuleCacheKey, ModuleGlobals,
    PtxSourceCacheKey,
};
use super::module_globals::*;
use super::pinned_allocations::PinnedHostAllocationPool;
use super::plan::{compute_ordered_output_indices, CudaDispatchPlan};
use super::ptx_target::select_loadable_ptx_target_sm;
use super::resident::{
    CudaDispatchBinding, CudaResidentBuffer, CudaResidentStore, ResidentBufferView,
};
use super::resident_dispatch::next_dispatch_binding;
use super::staging_reserve::reserve_smallvec;
use super::telemetry::{CudaTelemetry, CudaTelemetrySnapshot};
use crate::device::{CudaDeviceCaps, CudaDeviceHandle};

const TRANSIENT_ALLOCATION_POOL_BYTES: usize = 256 * 1024 * 1024;
const PINNED_HOST_POOL_BYTES: usize = 128 * 1024 * 1024;
const CUDA_LAUNCH_RESOURCE_CACHE: usize = 128;
/// A live CUDA backend handle bound to a specific device.
#[derive(Debug, Clone)]
pub struct CudaBackend {
    /// Probed device capabilities over the hardware limit.
    pub caps: CudaDeviceCaps,
    pub(crate) ptx_target_sm: u32,
    pub(crate) launch_resources: Arc<crate::stream::CudaLaunchResourcePool>,
    pub(crate) transient_pool: Arc<DeviceAllocationPool>,
    pub(crate) host_pool: Arc<PinnedHostAllocationPool>,
    pub(crate) ptx_source_cache: Arc<CudaPtxSourceCache>,
    module_cache: Arc<CudaModuleCache>,
    pub(crate) resident_store: Arc<CudaResidentStore>,
    pub(crate) validation_cache: Arc<ValidationCache>,
    pub(crate) graph_capture_lock: Arc<Mutex<()>>,
    pub(crate) async_upload_stream: Arc<Mutex<Option<crate::stream::CudaStream>>>,
    pub(crate) telemetry: Arc<CudaTelemetry>,
    /// Cache of driver-measured active-blocks-per-SM keyed by
    /// `(CUfunction as usize, threads_per_block)`: occupancy is constant per
    /// kernel shape, so this makes the per-launch occupancy-evidence query a map
    /// lookup after the first launch instead of repeated FFI (Law 7).
    pub(crate) occupancy_blocks_cache: Arc<DashMap<(usize, u32), u32>>,
    /// Serializing gate per module-cache key for launches that hold a
    /// module-scope global, created on first use. Keyed exactly like the module
    /// cache because that is the aliasing set: one key means one loaded CUmodule
    /// and therefore one `_vyre_grid_barrier` counter and one trap record. See
    /// [`ModuleGlobalsGate`].
    module_globals_gates: Arc<DashMap<ModuleCacheKey, Arc<ModuleGlobalsGate>>>,
    pub(crate) ctx: Arc<CudaContext>,
}

impl CudaBackend {
    /// Acquire the default CUDA device (ordinal 0).
    pub fn acquire() -> Result<Self, String> {
        Self::acquire_ordinal(0)
    }

    /// Acquire a specific CUDA device by ordinal.
    ///
    /// # Errors
    ///
    /// Returns an error when the CUDA driver cannot initialize, the ordinal is
    /// out of range, or required device attributes cannot be queried.
    pub fn acquire_ordinal(ordinal: usize) -> Result<Self, String> {
        // E4 + E5: enable the CUDA driver's persistent disk JIT cache
        // before any module load so the first dispatch this process
        // does on a previously-seen kernel hits the on-disk cuBIN
        // instead of re-JITing. Idempotent and respectful of operator
        // overrides via the CUDA_CACHE_* env vars.
        crate::jit_cache::configure_jit_cache_default()?;
        let device = CudaDeviceHandle::acquire_ordinal(ordinal)?;
        let caps = device.caps;
        let ptx_target_sm = select_loadable_ptx_target_sm(caps.ptx_target_sm())?;
        let ctx = device.ctx;
        let resident_store = CudaResidentStore::new().map_err(|error| error.to_string())?;
        Ok(Self {
            caps,
            ptx_target_sm,
            launch_resources: Arc::new(crate::stream::CudaLaunchResourcePool::new(
                CUDA_LAUNCH_RESOURCE_CACHE,
            )),
            transient_pool: Arc::new(DeviceAllocationPool::new(TRANSIENT_ALLOCATION_POOL_BYTES)),
            host_pool: Arc::new(PinnedHostAllocationPool::new(PINNED_HOST_POOL_BYTES)),
            ptx_source_cache: Arc::new(CudaPtxSourceCache::new()),
            module_cache: Arc::new(CudaModuleCache::new()),
            resident_store: Arc::new(resident_store),
            validation_cache: Arc::new(ValidationCache::default()),
            graph_capture_lock: Arc::new(Mutex::new(())),
            async_upload_stream: Arc::new(Mutex::new(None)),
            telemetry: Arc::new(CudaTelemetry::default()),
            occupancy_blocks_cache: Arc::new(DashMap::new()),
            module_globals_gates: Arc::new(DashMap::new()),
            ctx,
        })
    }

    pub(crate) fn prepare_launch_plan(
        &self,
        program: &Program,
        bindings: &BindingPlan,
        config: &DispatchConfig,
    ) -> Result<LaunchPlan, BackendError> {
        self.enforce_config_caps(config)?;
        LaunchPlan::from_bindings(program, &bindings.bindings, config, self.launch_limits())
    }

    pub(crate) fn prepare_host_dispatch(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<CudaDispatchPlan, BackendError> {
        let bindings = BindingPlan::from_borrowed_inputs(program, inputs)?;
        let launch = self.prepare_launch_plan(program, &bindings, config)?;
        self.validate_program_cached(program)?;
        let cooperative = self.resolve_cooperative_flag_for_program(program, config)?;
        let output_binding_indices = compute_ordered_output_indices(&bindings)?;
        let fixpoint_iterations = resolve_fixpoint_iterations(config, "CUDA")?;
        Ok(CudaDispatchPlan {
            bindings,
            output_binding_indices,
            launch,
            cooperative,
            fixpoint_iterations,
        })
    }

    /// Cooperative-launch flag for `program`, accounting for its grid-sync
    /// content.
    ///
    /// A program that still contains `MemoryOrdering::GridSync` barriers when it
    /// reaches a launch path has been lowered to PTX with native in-kernel grid
    /// barriers (the resident-fixpoint and host-split paths split the barriers
    /// out before lowering, so they never arrive here). Such a kernel MUST be
    /// launched cooperatively, every CTA co-resident, or the in-kernel grid
    /// barrier deadlocks. Force cooperative and fail closed when the device
    /// cannot run cooperative launch, rather than silently launching a kernel
    /// that would hang.
    ///
    /// Every prepare entrypoint routes through this, not only the borrowed-host
    /// one. A compiled pipeline plans through `prepare_static_dispatch` and its
    /// persistent-handle routes plan through `prepare_resident_dispatch`, so a
    /// grid-sync program compiled without `DispatchConfig::cooperative` set would
    /// otherwise plan a plain `cuLaunchKernel` for a kernel whose barriers
    /// require every CTA to be co-resident.
    fn resolve_cooperative_flag_for_program(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        if vyre_driver::grid_sync::contains_grid_sync(program) {
            if !self.hardware_supports_grid_sync() {
                return Err(BackendError::UnsupportedFeature {
                    name: format!(
                        "cuda_native_grid_sync (compute_capability={:?}, cooperative_launch={})",
                        self.caps.compute_capability, self.caps.cooperative_launch
                    ),
                    backend: crate::CUDA_BACKEND_ID.to_string(),
                });
            }
            return Ok(true);
        }
        self.resolve_cooperative_flag(config)
    }

    pub(crate) fn prepare_static_dispatch(
        &self,
        program: &Program,
        config: &DispatchConfig,
    ) -> Result<CudaDispatchPlan, BackendError> {
        let bindings = BindingPlan::build(program)?;
        let launch = self.prepare_launch_plan(program, &bindings, config)?;
        self.validate_program_cached(program)?;
        let cooperative = self.resolve_cooperative_flag_for_program(program, config)?;
        let output_binding_indices = compute_ordered_output_indices(&bindings)?;
        let fixpoint_iterations = resolve_fixpoint_iterations(config, "CUDA")?;
        Ok(CudaDispatchPlan {
            bindings,
            output_binding_indices,
            launch,
            cooperative,
            fixpoint_iterations,
        })
    }

    pub(crate) fn prepare_resident_dispatch(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
    ) -> Result<CudaDispatchPlan, BackendError> {
        let static_bindings = BindingPlan::build(program)?;
        let required_bindings = static_bindings
            .bindings
            .len()
            .checked_sub(static_bindings.shared_indices.len())
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident binding plan has {} binding(s) but {} shared binding index(es). Rebuild the dispatch plan before launching.",
                    static_bindings.bindings.len(),
                    static_bindings.shared_indices.len()
                ),
            })?;
        if bindings.len() != required_bindings {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident dispatch expected {required_bindings} bound resource(s) but received {}.",
                    bindings.len()
                ),
            });
        }

        let mut input_lengths = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut input_lengths,
            static_bindings.input_indices.len(),
            "resident dispatch input lengths",
        )?;
        input_lengths.extend(std::iter::repeat_n(0, static_bindings.input_indices.len()));
        let mut next_binding = 0usize;
        for binding in &static_bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let source = next_dispatch_binding(
                bindings,
                &mut next_binding,
                "resident dispatch input-length derivation",
            )?;
            let byte_len = match source {
                CudaDispatchBinding::Resident(handle) => self.resident_store.view(handle)?.byte_len,
                CudaDispatchBinding::Borrowed(bytes) => bytes.len(),
            };
            if let Some(input_index) = binding.input_index {
                let Some(input_len) = input_lengths.get_mut(input_index) else {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident dispatch input binding index {input_index} has no matching input-length slot after deriving {} resident input length(s). Rebuild the binding plan before resident launch.",
                            input_lengths.len()
                        ),
                    });
                };
                *input_len = byte_len;
            }
        }

        let bindings = BindingPlan::from_input_lengths(program, &input_lengths)?;
        let launch = self.prepare_launch_plan(program, &bindings, config)?;
        self.validate_program_cached(program)?;
        let cooperative = self.resolve_cooperative_flag_for_program(program, config)?;
        let output_binding_indices = compute_ordered_output_indices(&bindings)?;
        let fixpoint_iterations = resolve_fixpoint_iterations(config, "CUDA")?;
        Ok(CudaDispatchPlan {
            bindings,
            output_binding_indices,
            launch,
            cooperative,
            fixpoint_iterations,
        })
    }

    /// Validate that the caller's cooperative-launch request is consistent
    /// with the device's reported capabilities. Returns the resolved flag
    /// (always `false` when the caller didn't ask) or an `UnsupportedFeature`
    /// error when the caller asked for cooperative launch on a device that
    /// can't run it.
    ///
    /// This method gates *only* the host-side launch API, NOT the codegen
    /// emission of in-kernel grid-sync barriers. The barrier emission is
    /// still controlled by `lowers_grid_sync()`. Callers that opt into
    /// cooperative launch but whose program does not contain any GridSync
    /// barriers get the cooperative API call (resident grid) but no
    /// in-kernel sync sequence  -  the launcher still runs faster on programs
    /// that benefit from a resident grid even without explicit grid-sync.
    fn resolve_cooperative_flag(&self, config: &DispatchConfig) -> Result<bool, BackendError> {
        if !config.cooperative {
            return Ok(false);
        }
        if !self.hardware_supports_grid_sync() {
            return Err(BackendError::UnsupportedFeature {
                name: format!(
                    "cuda_cooperative_launch (compute_capability={:?}, cooperative_launch={})",
                    self.caps.compute_capability, self.caps.cooperative_launch
                ),
                backend: crate::CUDA_BACKEND_ID.to_string(),
            });
        }
        Ok(true)
    }

    fn enforce_config_caps(&self, config: &DispatchConfig) -> Result<(), BackendError> {
        if matches!(config.speculation, Some(SpeculationMode::Force)) {
            return Err(BackendError::UnsupportedFeature {
                name: "speculative dispatch".to_string(),
                backend: crate::CUDA_BACKEND_ID.to_string(),
            });
        }
        Ok(())
    }

    /// Pre-warmup: ensures the CUDA context is active.
    pub fn warmup(&self) -> Result<(), BackendError> {
        self.ctx
            .bind_to_thread()
            .map_err(|e| BackendError::DispatchFailed {
                code: None,
                message: format!("CUDA context bind failed: {e}"),
            })
    }

    /// Cleanup: sync and release cached modules.
    pub fn cleanup(&self) -> Result<(), BackendError> {
        self.warmup()?;
        self.ptx_source_cache.clear();
        self.module_cache.clear();
        self.resident_store.clear()?;
        self.transient_pool.clear()?;
        self.host_pool.clear()?;
        self.launch_resources.clear()?;
        Ok(())
    }

    pub(crate) fn with_resident<T>(
        &self,
        handle: CudaResidentBuffer,
        f: impl FnOnce(ResidentBufferView) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        self.warmup()?;
        let buffer = self.resident_store.view(handle)?;
        f(buffer)
    }

    pub(crate) fn resident_handles_from_resources(
        &self,
        resources: &[vyre_driver::Resource],
    ) -> Result<SmallVec<[CudaResidentBuffer; 8]>, BackendError> {
        self.resident_store.handles_from_resources(resources)
    }

    /// Resolve a dispatch resource list into mixed resident/borrowed bindings.
    pub(crate) fn resident_bindings_from_resources<'a>(
        &self,
        resources: &'a [vyre_driver::Resource],
    ) -> Result<SmallVec<[CudaDispatchBinding<'a>; 8]>, BackendError> {
        self.resident_store.bindings_from_resources(resources)
    }

    pub(crate) fn resident_handle_from_resource(
        &self,
        resource: &vyre_driver::Resource,
    ) -> Result<CudaResidentBuffer, BackendError> {
        self.resident_store.handle_from_resource(resource)
    }

    pub(crate) fn module_cache_key_for_ptx_source_key(
        &self,
        ptx_source_key: PtxSourceCacheKey,
    ) -> Result<ModuleCacheKey, BackendError> {
        self.module_cache
            .key_for_ptx_source_key(ptx_source_key, self.caps.compute_capability)
    }

    pub(crate) fn module_cache_key_for_raw_ptx_artifact(
        &self,
        raw_ptx_source: &str,
    ) -> Result<ModuleCacheKey, BackendError> {
        self.module_cache
            .key_for_raw_ptx_artifact(raw_ptx_source, self.caps.compute_capability)
    }

    pub(crate) fn module_for_ptx_with_key(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
    ) -> Result<cudarc::driver::sys::CUfunction, BackendError> {
        self.module_cache
            .function_for_ptx(ptx_src, key, self.ptx_target_sm())
    }

    /// Registers, spill bytes and static shared bytes the driver assigned to
    /// this PTX module's entry point.
    pub(crate) fn module_resources_with_key(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
    ) -> Result<EmittedResources, BackendError> {
        let func = self.module_for_ptx_with_key(ptx_src, key)?;
        let (registers_per_invocation, spill_bytes_per_invocation, shared_memory_bytes) =
            super::module_cache::cuda_function_resources(func).map_err(|res| {
                BackendError::DispatchFailed {
                    code: None,
                    message: format!(
                        "cuFuncGetAttribute failed with {res:?} for a loaded CUDA entry point. Fix: verify the module is still resident on the acquired device."
                    ),
                }
            })?;
        Ok(EmittedResources {
            registers_per_invocation,
            spill_bytes_per_invocation,
            shared_memory_bytes,
        })
    }

    /// The module-scope globals this PTX module exposes to the host.
    pub(crate) fn module_globals_with_key(
        &self,
        ptx_src: &str,
        key: ModuleCacheKey,
    ) -> Result<ModuleGlobals, BackendError> {
        self.module_cache
            .module_globals_for_ptx(ptx_src, key, self.ptx_target_sm())
    }

    /// The module-scope `_vyre_grid_barrier` counter this launch must start from
    /// zero, or `None` when the launch needs no reset.
    ///
    /// A native grid-sync kernel drives the counter up to `N * gridSize` for `N`
    /// in-kernel barriers, and each barrier's release target is a compile-time
    /// multiple of `gridSize`. A launch that starts from a stale value therefore
    /// releases its first barrier before every CTA has arrived. Every cooperative
    /// launch site takes a [`ModuleGlobalsLease`] over it, which zeroes it before
    /// each launch, so the borrowed-host path, the resident paths, and the
    /// compiled pipeline that reuses them share ONE reset instead of drifting
    /// copies.
    ///
    /// A grid-sync program whose loaded module declares no counter is a codegen
    /// failure, not a launch to attempt quietly.
    fn grid_barrier_reset_target(
        &self,
        program: &Program,
        prepared: &CudaDispatchPlan,
        globals: &ModuleGlobals,
    ) -> Result<Option<(u64, usize)>, BackendError> {
        if !prepared.cooperative || !vyre_driver::grid_sync::contains_grid_sync(program) {
            return Ok(None);
        }
        match globals.grid_barrier {
            Some(global) => Ok(Some(global)),
            None => Err(BackendError::InvalidProgram {
                fix:
                    "Fix: CUDA cooperative grid-sync launch found no `_vyre_grid_barrier` counter in the loaded module although the program contains grid-sync barriers. Ensure the PTX emitter declares the module-scope counter for grid-sync kernels."
                        .to_string(),
            }),
        }
    }

    /// Exclusive lease on this module's host-visible module-scope globals for one
    /// launch sequence, or an inert lease when the module has none.
    ///
    /// Acquiring the lease BLOCKS while another launch sequence on the same module
    /// is still in flight, which is what makes a per-module global safe to share
    /// across concurrent dispatches. Hold it across the resets and launches, then
    /// end it with [`ModuleGlobalsLease::launch_then_release`].
    ///
    /// A trap-declaring module takes the lease whether or not the launch is
    /// cooperative, because the trap record is per-module exactly as the counter
    /// is: two overlapping launches would zero each other's record and the second
    /// one's trap would be reported against the first one's launch, or lost.
    pub(crate) fn lease_module_globals(
        &self,
        program: &Program,
        prepared: &CudaDispatchPlan,
        ptx_src: &str,
        module_key: ModuleCacheKey,
    ) -> Result<ModuleGlobalsLease, BackendError> {
        let globals = self.module_globals_with_key(ptx_src, module_key)?;
        let barrier = self.grid_barrier_reset_target(program, prepared, &globals)?;
        let trap = globals.trap;
        if barrier.is_none() && trap.is_none() {
            return Ok(ModuleGlobalsLease {
                barrier: None,
                trap: None,
                guard: None,
                arrival_ceiling: 0,
            });
        }
        let arrival_ceiling = if barrier.is_some() {
            grid_barrier_arrival_ceiling(ptx_src, prepared.launch.grid)?
        } else {
            0
        };
        let gate = Arc::clone(
            self.module_globals_gates
                .entry(module_key)
                .or_insert_with(|| Arc::new(ModuleGlobalsGate::default()))
                .value(),
        );
        let guard = ModuleGlobalsGate::acquire(&gate)?;
        Ok(ModuleGlobalsLease {
            barrier,
            trap,
            guard: Some(guard),
            arrival_ceiling,
        })
    }

    /// Number of loaded CUDA modules currently held in the warm cache.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the cache lock is poisoned.
    pub fn cached_module_count(&self) -> Result<usize, BackendError> {
        Ok(self.module_cache.len())
    }

    /// Compiled module cache counters for honest compile telemetry.
    #[must_use]
    pub fn pipeline_cache_snapshot(&self) -> vyre_driver::PipelineCacheSnapshot {
        self.module_cache.snapshot()
    }

    /// PTX source cache counters for pre-module-load lowering telemetry.
    #[must_use]
    pub fn ptx_source_cache_snapshot(&self) -> CudaPtxSourceCacheSnapshot {
        self.ptx_source_cache.snapshot()
    }

    /// Runtime CUDA telemetry counters for launches, copies, readbacks, and syncs.
    ///
    /// The transient device-allocation-pool hit/miss counters are overlaid here
    /// from the pool itself (their source of truth). `CudaTelemetry` does not hold
    /// the pool, so a bare `CudaTelemetry::snapshot` reports them as zero and this
    /// boundary fills in the real values (ONE PLACE for the count, read once here).
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CudaTelemetrySnapshot {
        let mut snapshot = self.telemetry.snapshot();
        snapshot.device_pool_hits = self.transient_pool.hits();
        snapshot.device_pool_misses = self.transient_pool.misses();
        snapshot
    }

    /// Reset runtime CUDA telemetry counters without clearing caches or resident buffers.
    pub fn reset_telemetry(&self) {
        self.telemetry.reset();
        // Reset the pool hit/miss counters into the same epoch so the hit rate
        // reflects the window measured after the reset, not lifetime-of-process.
        self.transient_pool.reset_hit_counters();
    }

    /// Bytes of transient CUDA device memory retained for dispatch reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the allocation-pool lock is poisoned.
    pub fn cached_transient_allocation_bytes(&self) -> Result<usize, BackendError> {
        self.transient_pool.cached_bytes()
    }

    /// Bytes of transient CUDA device memory currently owned by the transient pool.
    ///
    /// This includes both checked-out allocations and cached allocations retained for reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if allocation accounting cannot be read.
    pub fn allocated_transient_allocation_bytes(&self) -> Result<usize, BackendError> {
        self.transient_pool.allocated_bytes()
    }

    /// Cached CUDA streams/events retained for dispatch reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if a launch-resource pool lock is poisoned.
    pub fn cached_launch_resource_counts(&self) -> Result<(usize, usize), BackendError> {
        self.launch_resources.cached_counts()
    }

    /// Detailed cached CUDA launch resources retained for dispatch reuse,
    /// including timing-enabled events used by CUDA graph replay telemetry.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if launch-resource accounting cannot be read.
    pub fn cached_launch_resource_counts_detailed(
        &self,
    ) -> Result<crate::CudaLaunchResourceCounts, BackendError> {
        self.launch_resources.cached_counts_detailed()
    }

    /// Snapshot the driver-tier observability surface
    /// ([`vyre_driver::observability::DriverObservability`]) plus the
    /// cuda module-cache count as a single backend metric.
    ///
    /// Operators scrape this in addition to per-substrate Prometheus
    /// counters when correlating substrate activity with backend
    /// resource usage.
    #[must_use]
    pub fn observability_snapshot(&self) -> vyre_driver::observability::DriverObservability {
        vyre_driver::observability::DriverObservability::snapshot()
    }

    /// PTX disk-cache directory path. Reuses the shared on-disk pipeline-cache
    /// layout, keyed by the VSA fingerprint.
    ///
    /// P-CUDA-2: PTX/CUBIN blobs persist across runs in this directory
    /// so first-run compile cost amortizes over the cluster.
    pub fn ptx_disk_cache_dir() -> Result<std::path::PathBuf, BackendError> {
        if let Some(path) = std::env::var_os("VYRE_PTX_CACHE_DIR") {
            let path = std::path::PathBuf::from(path);
            if path.as_os_str().is_empty() {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: VYRE_PTX_CACHE_DIR is empty. Set it to a writable persistent directory or unset it so XDG/HOME cache discovery can run."
                        .to_string(),
                });
            }
            return Ok(path);
        }
        if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
            return Ok(std::path::PathBuf::from(xdg).join("vyre").join("ptx-cache"));
        }
        if let Some(home) = std::env::var_os("HOME") {
            return Ok(std::path::PathBuf::from(home)
                .join(".cache")
                .join("vyre")
                .join("ptx-cache"));
        }
        Err(BackendError::InvalidProgram {
            fix: "Fix: CUDA PTX disk cache has no VYRE_PTX_CACHE_DIR, XDG_CACHE_HOME, or HOME. Configure a writable persistent cache root; temporary fallback is forbidden for production compile performance."
                .to_string(),
        })
    }
}
