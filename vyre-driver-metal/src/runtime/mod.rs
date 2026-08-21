//! Apple Metal.framework runtime implementation.

mod buffer_plan;
mod dispatch;
mod metrics;
mod pipeline;
mod resident;
mod scan_resource;

use std::collections::{BTreeMap, HashMap};
use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    Arc, Mutex, MutexGuard,
};
use std::time::Instant;

use metal::Device;
use vyre_driver::resident_transfer_fusion::fuse_resident_transfer_intervals;
use vyre_driver::resident_transfer_fusion::ResidentTransferInterval;
use vyre_driver::{
    output_binding_layouts, BindingPlan, DeviceProfile, DeviceTimingQuality, DispatchConfig,
    PipelineCacheSnapshot, ResidentHandle, ResidentOwner, Resource, TimedDispatchResult,
    VyreBackend,
};
use vyre_driver::{sealed, BackendError, PendingDispatch};
use vyre_foundation::ir::{OpId, Program};

use self::buffer_plan::{
    metal_slot_map, output_layout_map, plan_buffers, plan_resident_buffers, resident_input_lengths,
    PlannedBuffer,
};
use self::dispatch::{
    dispatch_planned_buffers_with_queue, submit_planned_buffers_with_queue,
    validate_metal_dispatch_config, MetalDispatchResult, MetalPendingDispatch,
};
pub(crate) use self::metrics::push_resident_table_metrics;
use self::metrics::{
    bytes_per_second_to_gbps, elapsed_ns, record_buffer_allocation, record_device_to_host_copy,
    record_host_to_device_copy, record_output_readback_metrics, record_planned_buffer_metrics,
    MetalMetricCounters, MetalMetrics, METAL_COUNTERS,
};
use self::pipeline::MetalCompiledPipeline;
pub(crate) use self::pipeline::MetalTargetModule;
pub(crate) use self::resident::MetalResidentBufferTable;
use self::resident::{
    copy_fused_resident_view_into, copy_shared_buffer_range_into, copy_to_shared_buffer_range,
    lock_resident_buffer_table, new_zero_buffer, next_resident_id, ns_uint_to_u32_saturating,
    reserve_fused_resident_view_outputs, resolve_resident_resources_from_table,
    validate_resident_range, zero_shared_buffer_range, MetalResidentBuffer, ResolvedMetalResource,
};
pub use self::scan_resource::{
    metal_resident_scan_resource_table, MetalResidentScanResourceEntry,
    MetalResidentScanResourceError, MetalResidentScanResourceLifetime,
    MetalResidentScanResourceTableEvidence, METAL_RESIDENT_SCAN_RESOURCE_TABLE_SCHEMA_VERSION,
};
use crate::METAL_BACKEND_ID;

/// Native Metal implementation of [`VyreBackend`].
pub struct MetalBackend {
    pub(super) device: Device,
    pub(super) queue: metal::CommandQueue,
    /// Crate-visible because the only way to observe the poison contract in
    /// `backend_metric_snapshot` is to poison this lock.
    pub(crate) resident_buffers: MetalResidentBufferTable,
    /// Identity of this instance's resident-buffer namespace.
    ///
    /// `next_resident` restarts at 1 in every fresh backend, so a bare id is
    /// only meaningful next to the owner that minted it. Carrying the owner in
    /// the handle makes a stale handle a refusal instead of a silent hit on an
    /// unrelated buffer of the same id.
    pub(super) resident_owner: ResidentOwner,
    pub(super) next_resident: AtomicU64,
    pub(super) pipeline_cache: Mutex<BTreeMap<[u8; 32], MetalCompiledPipeline>>,
    pub(super) metrics: MetalMetricCounters,
    /// SIMD-group width as reported by the first compiled `ComputePipelineState`.
    /// `0` means "not yet probed". Metal does not expose this at the Device level;
    /// it is pipeline-state-dependent and queried via
    /// `MTLComputePipelineState::threadExecutionWidth` after the first compile.
    pub(super) cached_simd_width: AtomicU32,
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalBackend")
            .field("backend", &METAL_BACKEND_ID)
            .finish_non_exhaustive()
    }
}

impl sealed::Sealed for MetalBackend {}

// SAFETY: `MTLDevice` and `MTLCommandQueue` are Objective-C Metal objects whose
// public API is designed for cross-thread command creation and submission. This
// backend does not expose interior raw pointers or share command encoders across
// calls; each dispatch creates its own command buffer and encoder. Resident
// handle state is protected by a Mutex and Metal buffer handles are cloned
// Objective-C object references.
unsafe impl Send for MetalBackend {}

// SAFETY: See the `Send` rationale above. Shared access only reaches Metal's
// thread-safe device/queue handles, and per-dispatch mutable state is local
// except for the resident table guarded by `resident_buffers`.
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// Acquire the system default Metal device and command queue.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when no Metal device is available.
    pub fn acquire() -> Result<Self, BackendError> {
        let device = Device::system_default().ok_or_else(|| BackendError::UnsupportedFeature {
            name: "system default Metal device".to_string(),
            backend: METAL_BACKEND_ID.to_string(),
        })?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            resident_buffers: Arc::new(Mutex::new(HashMap::new())),
            resident_owner: ResidentOwner::new()?,
            next_resident: AtomicU64::new(1),
            pipeline_cache: Mutex::new(BTreeMap::new()),
            metrics: Arc::new(MetalMetrics::default()),
            cached_simd_width: AtomicU32::new(0),
        })
    }

    pub(crate) fn artifact_device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(super) fn dispatch_planned_buffers(
        &self,
        program: &Program,
        binding_plan: &BindingPlan,
        config: &DispatchConfig,
        artifact: &vyre_emit_metal::MetalArtifact,
        pipeline: &metal::ComputePipelineState,
        buffers: Vec<PlannedBuffer>,
    ) -> Result<MetalDispatchResult, BackendError> {
        self.record_planned_buffer_metrics(&buffers);
        let result = dispatch_planned_buffers_with_queue(
            &self.device,
            &self.queue,
            program,
            binding_plan,
            config,
            artifact,
            pipeline,
            buffers,
        )?;
        self.record_output_readback_metrics(&result.outputs);
        Ok(result)
    }

    fn record_planned_buffer_metrics(&self, buffers: &[PlannedBuffer]) {
        record_planned_buffer_metrics(&self.metrics, buffers);
    }

    fn record_output_readback_metrics(&self, outputs: &[Vec<u8>]) {
        record_output_readback_metrics(&self.metrics, outputs);
    }

    fn record_host_to_device_copy(&self, byte_len: usize) {
        record_host_to_device_copy(&self.metrics, byte_len);
    }

    fn record_device_to_host_copy(&self, byte_len: usize) {
        record_device_to_host_copy(&self.metrics, byte_len);
    }

    fn record_buffer_allocation(&self, byte_len: usize) {
        record_buffer_allocation(&self.metrics, byte_len);
    }

    fn lock_resident_buffers(
        &self,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, HashMap<ResidentHandle, MetalResidentBuffer>>, BackendError> {
        lock_resident_buffer_table(&self.resident_buffers, operation)
    }

    pub(super) fn lock_pipeline_cache(
        &self,
        operation: &'static str,
    ) -> Result<MutexGuard<'_, BTreeMap<[u8; 32], MetalCompiledPipeline>>, BackendError> {
        self.pipeline_cache
            .lock()
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal pipeline cache was poisoned during {operation}: {error}. Drop and reacquire the Metal backend before dispatch."
                ),
            })
    }

    fn resident_buffer(
        &self,
        resource: &Resource,
        operation: &'static str,
    ) -> Result<(ResidentHandle, MetalResidentBuffer), BackendError> {
        let Resource::Resident(id) = resource else {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal {operation} expected a resident resource handle, but received a borrowed host buffer. Allocate with allocate_resident first."
                ),
            });
        };
        self.resident_owner.resolve(*id, operation)?;
        let table = self.lock_resident_buffers(operation)?;
        let resident = table.get(id).cloned().ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {operation} received stale resident handle {id}. Keep the resource allocated until all resident operations finish and free each handle exactly once."
            ),
        })?;
        Ok((*id, resident))
    }

    fn resolve_resident_resources<'a>(
        &self,
        binding_plan: &BindingPlan,
        resources: &'a [Resource],
    ) -> Result<Vec<ResolvedMetalResource<'a>>, BackendError> {
        resolve_resident_resources_from_table(
            self.resident_owner,
            &self.resident_buffers,
            binding_plan,
            resources,
        )
    }
}

impl VyreBackend for MetalBackend {
    fn id(&self) -> &'static str {
        METAL_BACKEND_ID
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn supported_ops(&self) -> &std::collections::HashSet<OpId> {
        vyre_driver::core_supported_ops()
    }

    fn max_workgroup_size(&self) -> [u32; 3] {
        let size = self.device.max_threads_per_threadgroup();
        [
            ns_uint_to_u32_saturating(size.width),
            ns_uint_to_u32_saturating(size.height).max(1),
            ns_uint_to_u32_saturating(size.depth).max(1),
        ]
    }

    fn supports_subgroup_ops(&self) -> bool {
        true
    }

    fn subgroup_size(&self) -> Option<u32> {
        // Metal's SIMD-group width is pipeline-state-dependent, not device-level.
        // We probe it from `ComputePipelineState::threadExecutionWidth` on the
        // first successful `compile_pipeline` call and cache the result here.
        // `0` means "not yet probed", return `None` so callers can handle
        // the unknown case rather than receiving a potentially-wrong constant.
        // In practice the first dispatch fills the cache so `subgroup_size()`
        // always returns `Some` after the first kernel is compiled on this backend.
        let cached = self.cached_simd_width.load(Ordering::Relaxed);
        if cached > 0 {
            Some(cached)
        } else {
            None
        }
    }

    fn max_compute_workgroups_per_dimension(&self) -> u32 {
        u32::MAX
    }

    fn max_compute_invocations_per_workgroup(&self) -> u32 {
        ns_uint_to_u32_saturating(self.device.max_threads_per_threadgroup().width).max(1)
    }

    fn max_storage_buffer_bytes(&self) -> u64 {
        self.device
            .max_buffer_length()
            .try_into()
            .unwrap_or(u64::MAX)
    }

    fn device_profile(&self) -> DeviceProfile {
        let max_shared_memory_bytes =
            ns_uint_to_u32_saturating(self.device.max_threadgroup_memory_length());
        DeviceProfile {
            has_shared_memory: max_shared_memory_bytes > 0,
            max_shared_memory_bytes,
            mem_bw_gbps: bytes_per_second_to_gbps(self.device.max_transfer_rate()),
            timing_quality: DeviceTimingQuality::HostEnqueueWait,
            ..DeviceProfile::from_backend(self)
        }
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.dispatch_borrowed_timed(program, inputs, config)
            .map(|timed| timed.outputs)
    }

    fn dispatch_borrowed_timed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let started = Instant::now();
        validate_metal_dispatch_config(
            config,
            "Metal cooperative grid dispatch",
            "Metal non-resident repeated dispatch",
            "Metal dispatch",
        )?;
        let binding_plan = BindingPlan::from_borrowed_inputs(program, inputs)?;
        let output_layouts = output_binding_layouts(program)?;
        let output_by_binding = output_layout_map(output_layouts)?;
        let (_, artifact, pipeline) = self.compile_pipeline(program, config)?;
        let metal_slots = metal_slot_map(&artifact)?;
        let buffers = plan_buffers(
            &self.device,
            &binding_plan,
            inputs,
            &output_by_binding,
            &metal_slots,
            &artifact.bindings,
        )?;
        let result = self.dispatch_planned_buffers(
            program,
            &binding_plan,
            config,
            &artifact,
            &pipeline,
            buffers,
        )?;
        Ok(TimedDispatchResult::split_timed(
            result.outputs,
            elapsed_ns(started, "Metal borrowed timed dispatch")?,
            None,
            result.enqueue_ns,
            result.wait_ns,
        ))
    }

    fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
        let buffer = new_zero_buffer(&self.device, byte_len)?;
        let id = self
            .resident_owner
            .handle(next_resident_id(&self.next_resident)?);
        let mut table = self.lock_resident_buffers("resident allocation")?;
        if table
            .insert(id, MetalResidentBuffer { buffer, byte_len })
            .is_some()
        {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident allocation generated duplicate handle {id}. Drop and reacquire the backend before allocating more resident buffers."
                ),
            });
        }
        self.record_buffer_allocation(byte_len);
        Ok(Resource::Resident(id))
    }

    fn upload_resident(&self, resource: &Resource, bytes: &[u8]) -> Result<(), BackendError> {
        self.upload_resident_many(&[(resource, bytes)])
    }

    fn upload_resident_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError> {
        let mut resolved = Vec::new();
        resolved
            .try_reserve(uploads.len())
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident batch upload could not reserve {} upload descriptor(s): {error}. Split the resident upload batch.",
                    uploads.len()
                ),
            })?;
        for &(resource, bytes) in uploads {
            let (id, resident) = self.resident_buffer(resource, "resident batch upload")?;
            validate_resident_range(
                id,
                resident.byte_len,
                0,
                bytes.len(),
                "resident batch upload",
            )?;
            resolved.push((resident, bytes));
        }
        for (resident, bytes) in resolved {
            copy_to_shared_buffer_range(&resident.buffer, 0, bytes, "resident batch upload")?;
            self.record_host_to_device_copy(bytes.len());
            if bytes.len() < resident.byte_len {
                zero_shared_buffer_range(
                    &resident.buffer,
                    bytes.len(),
                    resident.byte_len - bytes.len(),
                    "resident batch upload padding",
                )?;
            }
        }
        Ok(())
    }

    fn upload_resident_at(
        &self,
        resource: &Resource,
        dst_offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        self.upload_resident_at_many(&[(resource, dst_offset_bytes, bytes)])
    }

    fn upload_resident_at_many(
        &self,
        uploads: &[(&Resource, usize, &[u8])],
    ) -> Result<(), BackendError> {
        let mut resolved = Vec::new();
        resolved
            .try_reserve(uploads.len())
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch upload could not reserve {} upload descriptor(s): {error}. Split the resident upload batch.",
                    uploads.len()
                ),
            })?;
        for &(resource, dst_offset_bytes, bytes) in uploads {
            let (id, resident) = self.resident_buffer(resource, "resident ranged batch upload")?;
            validate_resident_range(
                id,
                resident.byte_len,
                dst_offset_bytes,
                bytes.len(),
                "resident ranged batch upload",
            )?;
            resolved.push((resident, dst_offset_bytes, bytes));
        }
        for (resident, dst_offset_bytes, bytes) in resolved {
            copy_to_shared_buffer_range(
                &resident.buffer,
                dst_offset_bytes,
                bytes,
                "resident ranged batch upload",
            )?;
            self.record_host_to_device_copy(bytes.len());
        }
        Ok(())
    }

    fn download_resident_into(
        &self,
        resource: &Resource,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        let (id, resident) = self.resident_buffer(resource, "resident download")?;
        validate_resident_range(
            id,
            resident.byte_len,
            0,
            resident.byte_len,
            "resident download",
        )?;
        copy_shared_buffer_range_into(
            &resident.buffer,
            0,
            resident.byte_len,
            out,
            "resident download",
        )?;
        self.record_device_to_host_copy(resident.byte_len);
        Ok(())
    }

    fn download_resident_range_into(
        &self,
        resource: &Resource,
        byte_offset: usize,
        byte_len: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        let (id, resident) = self.resident_buffer(resource, "resident ranged download")?;
        validate_resident_range(
            id,
            resident.byte_len,
            byte_offset,
            byte_len,
            "resident ranged download",
        )?;
        copy_shared_buffer_range_into(
            &resident.buffer,
            byte_offset,
            byte_len,
            out,
            "resident ranged download",
        )?;
        self.record_device_to_host_copy(byte_len);
        Ok(())
    }

    fn download_resident_ranges_into(
        &self,
        ranges: &[(&Resource, usize, usize)],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        if ranges.len() != outputs.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download expected matching range/output counts but got {} range(s) and {} output buffer(s).",
                    ranges.len(),
                    outputs.len()
                ),
            });
        }

        let mut copies = Vec::new();
        copies
            .try_reserve(ranges.len())
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download could not reserve {} validated transfer interval(s): {error}. Split the resident readback batch.",
                    ranges.len()
                ),
            })?;
        // Keyed by local id to match the fusion plan. Every entry below is
        // owner-checked first, so local ids are exact keys within this batch.
        let mut buffers = BTreeMap::new();
        for &(resource, byte_offset, byte_len) in ranges {
            let (id, resident) =
                self.resident_buffer(resource, "resident ranged batch download")?;
            validate_resident_range(
                id,
                resident.byte_len,
                byte_offset,
                byte_len,
                "resident ranged batch download",
            )?;
            let src = u64::try_from(byte_offset).map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download offset {byte_offset} for handle {id} cannot fit u64 transfer fusion coordinates: {error}. Split the readback range."
                ),
            })?;
            buffers.entry(id.id()).or_insert(resident.buffer);
            copies.push(ResidentTransferInterval {
                handle_id: id.id(),
                src,
                byte_len,
            });
        }

        let fused = fuse_resident_transfer_intervals(&copies)?;
        reserve_fused_resident_view_outputs(&fused.copies, &fused.views, outputs)?;
        let mut fused_outputs = Vec::new();
        fused_outputs
            .try_reserve(fused.copies.len())
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download could not reserve {} fused readback output slot(s): {error}. Split the resident readback batch.",
                    fused.copies.len()
                ),
            })?;
        for copy in fused.copies.iter().copied() {
            let buffer = buffers.get(&copy.handle_id).ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download fused copy references unknown handle {}. Rebuild the resident transfer fusion plan after validation.",
                    copy.handle_id
                ),
            })?;
            let src = usize::try_from(copy.src).map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident fused ranged batch download offset {} for handle {} cannot fit usize readback coordinates: {error}. Split the readback range.",
                    copy.src, copy.handle_id
                ),
            })?;
            let mut fused_output = Vec::new();
            copy_shared_buffer_range_into(
                buffer,
                src,
                copy.byte_len,
                &mut fused_output,
                "resident fused ranged batch download",
            )?;
            self.record_device_to_host_copy(copy.byte_len);
            fused_outputs.push(fused_output);
        }
        for (view, output) in fused.views.iter().copied().zip(outputs.iter_mut()) {
            copy_fused_resident_view_into(&fused_outputs, view, output)?;
        }
        Ok(())
    }

    fn free_resident(&self, resource: Resource) -> Result<(), BackendError> {
        let Resource::Resident(id) = resource else {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: Metal resident free expected a handle returned by allocate_resident, but received a borrowed host buffer.".to_string(),
            });
        };
        self.resident_owner.resolve(id, "resident free")?;
        let mut table = self.lock_resident_buffers("resident free")?;
        table.remove(&id).ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident free received stale handle {id}. Free each resident resource exactly once."
            ),
        })?;
        Ok(())
    }

    fn shutdown(&self) -> Result<(), BackendError> {
        self.lock_resident_buffers("shutdown")?.clear();
        self.lock_pipeline_cache("shutdown")?.clear();
        Ok(())
    }

    fn pipeline_cache_snapshot(&self) -> Option<PipelineCacheSnapshot> {
        Some(PipelineCacheSnapshot {
            hits: self.metrics.pipeline_cache_hits.load(Ordering::Relaxed),
            misses: self.metrics.pipeline_cache_misses.load(Ordering::Relaxed),
        })
    }

    fn backend_metric_snapshot(&self) -> Vec<(&'static str, u64)> {
        let mut metrics = Vec::with_capacity(METAL_COUNTERS.len() + 3);
        for (name, counter) in METAL_COUNTERS {
            metrics.push((name, counter(&self.metrics).load(Ordering::Relaxed)));
        }
        push_resident_table_metrics(&self.resident_buffers, &mut metrics);
        metrics
    }

    fn dispatch_resident_timed(
        &self,
        program: &Program,
        resources: &[Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        self.dispatch_resident_async(program, resources, config)?
            .await_timed_result()
    }

    fn dispatch_resident_async(
        &self,
        program: &Program,
        resources: &[Resource],
        config: &DispatchConfig,
    ) -> Result<Box<dyn PendingDispatch>, BackendError> {
        let started = Instant::now();
        validate_metal_dispatch_config(
            config,
            "Metal cooperative grid resident dispatch",
            "Metal repeated resident dispatch",
            "Metal resident dispatch",
        )?;

        let base_plan = BindingPlan::build(program)?;
        let resolved = self.resolve_resident_resources(&base_plan, resources)?;
        let input_lengths = resident_input_lengths(&base_plan, &resolved)?;
        let binding_plan = BindingPlan::from_input_lengths(program, &input_lengths)?;
        let output_by_binding = output_layout_map(output_binding_layouts(program)?)?;
        let (_, artifact, pipeline) = self.compile_pipeline(program, config)?;
        let metal_slots = metal_slot_map(&artifact)?;
        let buffers = plan_resident_buffers(
            &self.device,
            &binding_plan,
            &resolved,
            &output_by_binding,
            &metal_slots,
            &artifact.bindings,
        )?;
        self.record_planned_buffer_metrics(&buffers);
        let command = submit_planned_buffers_with_queue(
            &self.device,
            &self.queue,
            program,
            &binding_plan,
            config,
            &artifact,
            &pipeline,
            buffers,
            output_by_binding,
        )?;
        Ok(Box::new(MetalPendingDispatch {
            command,
            metrics: Arc::clone(&self.metrics),
            started,
        }))
    }
}
