use smallvec::SmallVec;
use vyre_driver::{BackendError, BackendRegistration, DispatchConfig, Resource, VyreBackend};
use vyre_foundation::ir::Program;

use crate::aot_launcher;
use crate::backend::staging_reserve::reserve_smallvec;
use crate::backend::{
    self, CudaBackend, CudaPtxSourceCacheSnapshot, CudaResidentBuffer, CudaTelemetrySnapshot,
};
use crate::materializer;
use crate::target_compiler;
use crate::{CUDA_BACKEND_ID, CUDA_TARGET_ID};

/// CUDA implementation of [`vyre_driver::DeviceBuffer`]. Wraps a
/// [`backend::CudaResidentBuffer`] handle so consumers can hold a
/// `Box<dyn DeviceBuffer>` against the CUDA backend without naming
/// `CudaResidentBuffer` directly.
///
/// Lifecycle is explicit-free  -  call
/// `VyreBackend::free_device_buffer(boxed_buffer)` when done. This
/// matches the existing CUDA-resident contract and keeps the substrate
/// free of reference-counted backend handles. A future RAII variant
/// (Drop-managed via `Arc<CudaBackend>`) can ship as a drop-in
/// replacement when the backend ownership model accommodates it.
#[derive(Debug)]
pub struct CudaDeviceBuffer {
    backend_id: &'static str,
    handle: backend::CudaResidentBuffer,
}

impl vyre_driver::DeviceBuffer for CudaDeviceBuffer {
    fn backend_id(&self) -> &'static str {
        self.backend_id
    }

    fn byte_len(&self) -> usize {
        self.handle.byte_len
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Factory wrapper for the inventory registration path.
///
/// Unlike the SPIR-V backend, the CUDA backend owns a live device handle
/// and can dispatch programs directly.
#[derive(Debug)]
pub struct CudaBackendRegistration {
    pub(crate) inner: CudaBackend,
}

type ResolvedUploads<'a> = SmallVec<[(CudaResidentBuffer, &'a [u8]); 8]>;
type ResolvedOffsetUploads<'a> = SmallVec<[(CudaResidentBuffer, usize, &'a [u8]); 8]>;
type ResolvedDownloadRanges = SmallVec<[(CudaResidentBuffer, usize, usize); 8]>;
type ResolvedReadRanges = (
    SmallVec<[CudaResidentBuffer; 8]>,
    SmallVec<[crate::backend::output_range::CudaOutputReadback; 8]>,
);

impl CudaBackendRegistration {
    /// Wrap an already-acquired [`CudaBackend`] as a [`VyreBackend`] trait object.
    ///
    /// The inventory-driven path uses [`cuda_factory`] which acquires its own
    /// device handle. Callers that already own a [`CudaBackend`] (e.g. so they
    /// can keep the live device handle for direct API access while also handing
    /// it to a megakernel) use this constructor instead.
    #[must_use]
    pub fn new(inner: CudaBackend) -> Self {
        Self { inner }
    }

    /// Borrow the inner [`CudaBackend`] for direct device-API access.
    #[must_use]
    pub fn inner(&self) -> &CudaBackend {
        &self.inner
    }

    /// Snapshot the CUDA PTX-source cache used before driver module loading.
    #[must_use]
    pub fn ptx_source_cache_snapshot(&self) -> CudaPtxSourceCacheSnapshot {
        self.inner.ptx_source_cache_snapshot()
    }

    /// Runtime CUDA telemetry counters for release-path performance gates.
    #[must_use]
    pub fn telemetry_snapshot(&self) -> CudaTelemetrySnapshot {
        self.inner.telemetry_snapshot()
    }

    /// Reset runtime CUDA telemetry counters without clearing backend caches.
    pub fn reset_telemetry(&self) {
        self.inner.reset_telemetry();
    }

    fn resolve_uploads<'a>(
        &self,
        uploads: &[(&Resource, &'a [u8])],
    ) -> Result<ResolvedUploads<'a>, BackendError> {
        let mut concrete = SmallVec::<[(CudaResidentBuffer, &'a [u8]); 8]>::new();
        reserve_smallvec(&mut concrete, uploads.len(), "CUDA resident upload handles")?;
        for (resource, bytes) in uploads {
            let handle = self.inner.resident_handle_from_resource(resource)?;
            concrete.push((handle, *bytes));
        }
        Ok(concrete)
    }

    fn resolve_offset_uploads<'a>(
        &self,
        uploads: &[(&Resource, usize, &'a [u8])],
    ) -> Result<ResolvedOffsetUploads<'a>, BackendError> {
        let mut concrete = SmallVec::<[(CudaResidentBuffer, usize, &'a [u8]); 8]>::new();
        reserve_smallvec(
            &mut concrete,
            uploads.len(),
            "CUDA resident offset upload handles",
        )?;
        for (resource, dst_offset_bytes, bytes) in uploads {
            let handle = self.inner.resident_handle_from_resource(resource)?;
            concrete.push((handle, *dst_offset_bytes, *bytes));
        }
        Ok(concrete)
    }

    fn resolve_download_ranges(
        &self,
        ranges: &[(&Resource, usize, usize)],
    ) -> Result<ResolvedDownloadRanges, BackendError> {
        let mut concrete = SmallVec::<[(CudaResidentBuffer, usize, usize); 8]>::new();
        reserve_smallvec(
            &mut concrete,
            ranges.len(),
            "CUDA resident download range handles",
        )?;
        for (resource, byte_offset, byte_len) in ranges {
            let handle = self.inner.resident_handle_from_resource(resource)?;
            concrete.push((handle, *byte_offset, *byte_len));
        }
        Ok(concrete)
    }

    fn resolve_read_ranges(
        &self,
        read_ranges: &[vyre_driver::ResidentReadRange<'_>],
    ) -> Result<ResolvedReadRanges, BackendError> {
        let mut handles = SmallVec::<[CudaResidentBuffer; 8]>::new();
        let mut concrete_readbacks =
            SmallVec::<[crate::backend::output_range::CudaOutputReadback; 8]>::new();
        reserve_smallvec(
            &mut handles,
            read_ranges.len(),
            "CUDA resident read handles",
        )?;
        reserve_smallvec(
            &mut concrete_readbacks,
            read_ranges.len(),
            "CUDA resident readback ranges",
        )?;
        for range in read_ranges {
            handles.push(self.inner.resident_handle_from_resource(range.resource)?);
            concrete_readbacks.push(crate::backend::output_range::CudaOutputReadback {
                device_offset: range.byte_offset,
                byte_len: range.byte_len,
            });
        }
        Ok((handles, concrete_readbacks))
    }

    fn resolve_step_handle_sets(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
        field: &'static str,
    ) -> Result<SmallVec<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>, BackendError>
    {
        let mut handle_sets =
            SmallVec::<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>::new();
        reserve_smallvec(&mut handle_sets, steps.len(), field)?;
        for step in steps {
            handle_sets.push(self.inner.resident_handles_from_resources(step.resources)?);
        }
        Ok(handle_sets)
    }

    fn resolve_repeated_step_handle_sets(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
        repeat_count: usize,
    ) -> Result<SmallVec<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>, BackendError>
    {
        let mut handle_sets =
            SmallVec::<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>::new();
        let capacity = if repeat_count == 0 { 0 } else { steps.len() };
        reserve_smallvec(
            &mut handle_sets,
            capacity,
            "CUDA repeated resident repeated handle sets",
        )?;
        if repeat_count != 0 {
            for step in steps {
                handle_sets.push(self.inner.resident_handles_from_resources(step.resources)?);
            }
        }
        Ok(handle_sets)
    }

    fn concrete_resident_steps<'program: 'handles, 'handles>(
        steps: &[vyre_driver::ResidentDispatchStep<'program>],
        handle_sets: &'handles [SmallVec<[crate::backend::CudaResidentBuffer; 8]>],
        field: &'static str,
    ) -> Result<SmallVec<[crate::backend::CudaResidentDispatchStep<'handles>; 8]>, BackendError>
    {
        let mut concrete_steps =
            SmallVec::<[crate::backend::CudaResidentDispatchStep<'handles>; 8]>::new();
        reserve_smallvec(&mut concrete_steps, handle_sets.len(), field)?;
        for (step, handles) in steps.iter().zip(handle_sets.iter()) {
            let mut config = DispatchConfig::default();
            config.grid_override = step.grid_override;
            config.workgroup_override = step.workgroup_override;
            concrete_steps.push(crate::backend::CudaResidentDispatchStep {
                program: step.program,
                handles,
                config,
            });
        }
        Ok(concrete_steps)
    }

    /// Bytes of transient CUDA device memory currently owned by the transient pool.
    ///
    /// This includes checked-out dispatch allocations, compiled-pipeline static parameter
    /// allocations, and cached transient blocks retained for reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if allocation accounting cannot be read.
    pub fn allocated_transient_allocation_bytes(&self) -> Result<usize, BackendError> {
        self.inner.allocated_transient_allocation_bytes()
    }

    fn reject_grid_sync_without_native_lowering(
        &self,
        program: &Program,
    ) -> Result<(), BackendError> {
        if vyre_driver::grid_sync::contains_grid_sync(program) && !self.supports_grid_sync() {
            return Err(BackendError::UnsupportedFeature {
                name: "cuda_native_grid_sync_lowering (MemoryOrdering::GridSync requires explicit split routing or native cooperative-grid barrier lowering)"
                    .to_string(),
                backend: CUDA_BACKEND_ID.to_string(),
            });
        }
        Ok(())
    }

    fn validate_program_for_dispatch(&self, program: &Program) -> Result<(), BackendError> {
        // Before the generic capability check, so a whole-grid barrier this
        // device cannot launch is refused by name, with the compute
        // capability and the three ways out, rather than as a bare missing
        // `grid_sync` bit. The generic check still covers every backend that
        // has no such diagnostic of its own.
        self.reject_grid_sync_without_native_lowering(program)?;
        let required = vyre_foundation::program_caps::scan(program);
        vyre_foundation::program_caps::check_backend_capabilities(
            CUDA_BACKEND_ID,
            &vyre_driver::validation::ProgramValidationCaps::from_backend(self).support(),
            &required,
        )
        .map_err(|error| BackendError::InvalidProgram {
            fix: error.to_string(),
        })
    }

    fn validate_resident_steps_for_dispatch(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
    ) -> Result<(), BackendError> {
        for step in steps {
            self.validate_program_for_dispatch(step.program)?;
        }
        Ok(())
    }
}

impl vyre_driver::sealed::Sealed for CudaBackendRegistration {}

impl VyreBackend for CudaBackendRegistration {
    fn id(&self) -> &'static str {
        CUDA_BACKEND_ID
    }

    fn version(&self) -> &'static str {
        env!("CARGO_PKG_VERSION")
    }

    fn dispatch_async(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        config: &DispatchConfig,
    ) -> Result<Box<dyn vyre_driver::PendingDispatch>, BackendError> {
        self.validate_program_for_dispatch(program)?;
        self.inner.dispatch_async(program, inputs, config)
    }

    fn dispatch_borrowed_async(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Box<dyn vyre_driver::PendingDispatch>, BackendError> {
        self.validate_program_for_dispatch(program)?;
        self.inner.dispatch_borrowed_async(program, inputs, config)
    }

    fn dispatch_borrowed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.validate_program_for_dispatch(program)?;
        self.inner
            .dispatch_borrowed_async(program, inputs, config)?
            .await_result()
    }

    fn dispatch_borrowed_into(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
        outputs: &mut vyre_driver::OutputBuffers,
    ) -> Result<(), BackendError> {
        self.validate_program_for_dispatch(program)?;
        self.inner
            .dispatch_borrowed_async(program, inputs, config)?
            .await_result_into(outputs)
    }

    fn dispatch_borrowed_timed(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        self.validate_program_for_dispatch(program)?;
        self.inner.dispatch_borrowed_timed(program, inputs, config)
    }

    fn allocate_resident(&self, byte_len: usize) -> Result<Resource, BackendError> {
        self.inner
            .allocate_resident(byte_len)
            .map(|handle| Resource::Resident(handle.handle))
    }

    fn allocate_device_buffer(
        &self,
        byte_len: usize,
    ) -> Result<Box<dyn vyre_driver::DeviceBuffer>, BackendError> {
        let handle = self.inner.allocate_resident(byte_len)?;
        Ok(Box::new(CudaDeviceBuffer {
            backend_id: CUDA_BACKEND_ID,
            handle,
        }))
    }

    fn upload_device_buffer(
        &self,
        buffer: &mut dyn vyre_driver::DeviceBuffer,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        let backend_id = buffer.backend_id().to_string();
        let handle = buffer
            .as_any_mut()
            .downcast_mut::<CudaDeviceBuffer>()
            .map(|cuda_buf| cuda_buf.handle)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: upload_device_buffer expected a CudaDeviceBuffer (allocated by `cuda` backend) but got buffer owned by `{backend_id}`."
                ),
            })?;
        self.inner.upload_resident(handle, bytes)
    }

    fn download_device_buffer(
        &self,
        buffer: &dyn vyre_driver::DeviceBuffer,
    ) -> Result<Vec<u8>, BackendError> {
        let cuda_buf = buffer
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: download_device_buffer expected a CudaDeviceBuffer (allocated by `cuda` backend) but got buffer owned by `{}`.",
                    buffer.backend_id()
                ),
            })?;
        self.inner.download_resident(cuda_buf.handle)
    }

    fn free_device_buffer(
        &self,
        buffer: Box<dyn vyre_driver::DeviceBuffer>,
    ) -> Result<(), BackendError> {
        let backend_id = buffer.backend_id().to_string();
        let handle = buffer
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .map(|cuda_buf| cuda_buf.handle)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: free_device_buffer expected a CudaDeviceBuffer but got buffer owned by `{backend_id}`."
                ),
            })?;
        // Drop the Box (releases the wrapper allocation) before freeing
        // the underlying CUDA-resident allocation. CudaResidentBuffer is
        // Copy so we already captured the handle.
        drop(buffer);
        self.inner.free_resident(handle)
    }

    fn dispatch_with_device_buffers(
        &self,
        program: &Program,
        inputs: &[&dyn vyre_driver::DeviceBuffer],
        outputs: &mut [&mut dyn vyre_driver::DeviceBuffer],
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        self.validate_program_for_dispatch(program)?;
        // Convert &[&dyn DeviceBuffer] into &[Resource::Resident(id)]
        // so we can re-use the existing dispatch_resident_timed path.
        // Outputs are bound by Resource::Resident as well  -  the kernel
        // writes results in-place into the device-resident buffers; the
        // caller reads them via download_device_buffer afterwards.
        vyre_driver::validate_buffer_ownership(self.id(), inputs.iter().copied())?;
        vyre_driver::validate_buffer_ownership(
            self.id(),
            outputs
                .iter()
                .map(|b| &**b as &dyn vyre_driver::DeviceBuffer),
        )?;
        let resource_capacity =
            inputs
                .len()
                .checked_add(outputs.len())
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA borrowed dispatch resource capacity overflowed usize for {} input buffer(s) plus {} output buffer(s); split the dispatch.",
                        inputs.len(),
                        outputs.len()
                    ),
                })?;
        let mut handles = SmallVec::<[CudaResidentBuffer; 8]>::new();
        reserve_smallvec(
            &mut handles,
            resource_capacity,
            "CUDA borrowed dispatch resource handles",
        )?;
        for buffer in inputs {
            let handle = buffer
                .as_any()
                .downcast_ref::<CudaDeviceBuffer>()
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: dispatch_with_device_buffers expected CudaDeviceBuffer inputs but got buffer owned by `{}`.",
                        buffer.backend_id()
                    ),
                })?
                .handle;
            handles.push(handle);
        }
        for buffer in outputs.iter() {
            let backend_id = buffer.backend_id().to_string();
            let handle = buffer
                .as_any()
                .downcast_ref::<CudaDeviceBuffer>()
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: dispatch_with_device_buffers expected CudaDeviceBuffer outputs but got buffer owned by `{backend_id}`."
                    ),
                })?
                .handle;
            handles.push(handle);
        }
        let _timed = self
            .inner
            .dispatch_resident_timed(program, &handles, config)?;
        Ok(())
    }

    fn upload_resident(&self, resource: &Resource, bytes: &[u8]) -> Result<(), BackendError> {
        let handle = self.inner.resident_handle_from_resource(resource)?;
        self.inner.upload_resident(handle, bytes)
    }

    fn upload_resident_many(&self, uploads: &[(&Resource, &[u8])]) -> Result<(), BackendError> {
        let concrete = self.resolve_uploads(uploads)?;
        self.inner.upload_resident_many(&concrete)
    }

    fn upload_resident_at(
        &self,
        resource: &Resource,
        dst_offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        let handle = self.inner.resident_handle_from_resource(resource)?;
        self.inner
            .upload_resident_at(handle, dst_offset_bytes, bytes)
    }

    fn upload_resident_at_many(
        &self,
        uploads: &[(&Resource, usize, &[u8])],
    ) -> Result<(), BackendError> {
        let concrete = self.resolve_offset_uploads(uploads)?;
        self.inner.upload_resident_at_many(&concrete)
    }

    fn download_resident(&self, resource: &Resource) -> Result<Vec<u8>, BackendError> {
        let handle = self.inner.resident_handle_from_resource(resource)?;
        self.inner.download_resident(handle)
    }

    fn download_resident_into(
        &self,
        resource: &Resource,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        let handle = self.inner.resident_handle_from_resource(resource)?;
        self.inner.download_resident_into(handle, out)
    }

    fn download_resident_range(
        &self,
        resource: &Resource,
        byte_offset: usize,
        byte_len: usize,
    ) -> Result<Vec<u8>, BackendError> {
        let handle = self.inner.resident_handle_from_resource(resource)?;
        self.inner
            .download_resident_range(handle, byte_offset, byte_len)
    }

    fn download_resident_range_into(
        &self,
        resource: &Resource,
        byte_offset: usize,
        byte_len: usize,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        let handle = self.inner.resident_handle_from_resource(resource)?;
        self.inner
            .download_resident_range_into(handle, byte_offset, byte_len, out)
    }

    fn download_resident_ranges_into(
        &self,
        ranges: &[(&Resource, usize, usize)],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        let concrete = self.resolve_download_ranges(ranges)?;
        self.inner.download_resident_ranges_into(&concrete, outputs)
    }

    fn free_resident(&self, resource: Resource) -> Result<(), BackendError> {
        let handle = self.inner.resident_handle_from_resource(&resource)?;
        self.inner.free_resident(handle)
    }

    fn dispatch_resident_timed(
        &self,
        program: &Program,
        resources: &[Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        self.validate_program_for_dispatch(program)?;
        let handles = self.inner.resident_handles_from_resources(resources)?;
        self.inner
            .dispatch_resident_timed(program, &handles, config)
    }

    fn dispatch_resident_async(
        &self,
        program: &Program,
        resources: &[Resource],
        config: &DispatchConfig,
    ) -> Result<Box<dyn vyre_driver::PendingDispatch>, BackendError> {
        self.validate_program_for_dispatch(program)?;
        let handles = self.inner.resident_handles_from_resources(resources)?;
        self.inner
            .dispatch_resident_async(program, &handles, config)
    }

    fn dispatch_resident_sequence_read_ranges_into(
        &self,
        steps: &[vyre_driver::ResidentDispatchStep<'_>],
        read_ranges: &[vyre_driver::ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.validate_resident_steps_for_dispatch(steps)?;
        if read_ranges.len() != outputs.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident sequence ranged readback expected matching range/output counts but got {} range(s) and {} output(s).",
                    read_ranges.len(),
                    outputs.len()
                ),
            });
        }
        let handle_sets =
            self.resolve_step_handle_sets(steps, "CUDA resident sequence handle sets")?;
        let concrete_steps =
            Self::concrete_resident_steps(steps, &handle_sets, "CUDA resident sequence steps")?;

        let (read_handles, concrete_readbacks) = self.resolve_read_ranges(read_ranges)?;

        let uploads: [(crate::backend::CudaResidentBuffer, &[u8]); 0] = [];
        self.inner
            .upload_resident_many_sequence_read_ranges_borrowed_into(
                &uploads,
                &concrete_steps,
                &read_handles,
                &concrete_readbacks,
                outputs,
            )
    }

    fn dispatch_resident_repeated_sequence_read_ranges_into(
        &self,
        prefix_steps: &[vyre_driver::ResidentDispatchStep<'_>],
        repeated_steps: &[vyre_driver::ResidentDispatchStep<'_>],
        repeat_count: u32,
        read_ranges: &[vyre_driver::ResidentReadRange<'_>],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.validate_resident_steps_for_dispatch(prefix_steps)?;
        self.validate_resident_steps_for_dispatch(repeated_steps)?;
        let repeat_count =
            usize::try_from(repeat_count).map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA repeated resident sequence count does not fit usize: {error}."
                ),
            })?;
        if read_ranges.len() != outputs.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA repeated resident sequence ranged readback expected matching range/output counts but got {} range(s) and {} output(s).",
                    read_ranges.len(),
                    outputs.len()
                ),
            });
        }

        let prefix_handle_sets = self
            .resolve_step_handle_sets(prefix_steps, "CUDA repeated resident prefix handle sets")?;
        let repeated_handle_sets =
            self.resolve_repeated_step_handle_sets(repeated_steps, repeat_count)?;
        let concrete_prefix = Self::concrete_resident_steps(
            prefix_steps,
            &prefix_handle_sets,
            "CUDA repeated resident prefix steps",
        )?;
        let concrete_repeated = Self::concrete_resident_steps(
            repeated_steps,
            &repeated_handle_sets,
            "CUDA repeated resident repeated steps",
        )?;

        let (read_handles, concrete_readbacks) = self.resolve_read_ranges(read_ranges)?;
        let uploads: [(crate::backend::CudaResidentBuffer, &[u8]); 0] = [];
        self.inner
            .upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
                &uploads,
                &concrete_prefix,
                &concrete_repeated,
                repeat_count,
                &read_handles,
                &concrete_readbacks,
                outputs,
            )
    }

    fn pipeline_cache_snapshot(&self) -> Option<vyre_driver::PipelineCacheSnapshot> {
        Some(self.inner.pipeline_cache_snapshot())
    }

    fn backend_metric_snapshot(&self) -> Vec<(&'static str, u64)> {
        let source_cache = self.inner.ptx_source_cache_snapshot();
        let mut metrics = Vec::new();
        match u64::try_from(source_cache.entries) {
            Ok(entries) => metrics.push(("cuda_ptx_source_cache_entries", entries)),
            Err(source) => {
                tracing::error!(
                    "CUDA PTX source cache entry count cannot fit u64: {source}. Fix: shard backend metrics before source-cache cardinality exceeds u64."
                );
                metrics.push(("cuda_ptx_source_cache_entries_unrepresentable", 1));
            }
        }
        metrics.push(("cuda_ptx_source_cache_hits", source_cache.hits));
        metrics.push(("cuda_ptx_source_cache_misses", source_cache.misses));
        let telemetry = self.inner.telemetry_snapshot();
        metrics.push(("cuda_graph_launches", telemetry.cuda_graph_launches));
        metrics.push((
            "cuda_graph_materialized_cache_hits",
            telemetry.cuda_graph_materialized_cache_hits,
        ));
        metrics.push((
            "cuda_graph_batched_replay_chunks",
            telemetry.cuda_graph_batched_replay_chunks,
        ));
        metrics.push((
            "cuda_graph_batched_replay_lanes",
            telemetry.cuda_graph_batched_replay_lanes,
        ));
        metrics.push(("cuda_host_to_device_bytes", telemetry.host_to_device_bytes));
        metrics.push(("cuda_device_to_host_bytes", telemetry.device_to_host_bytes));
        metrics.push(("cuda_readback_bytes", telemetry.readback_bytes));
        metrics.push(("cuda_param_upload_bytes", telemetry.param_upload_bytes));
        metrics.push(("cuda_kernel_launches", telemetry.kernel_launches));
        metrics.push(("cuda_sync_points", telemetry.sync_points));
        metrics.push((
            "cuda_host_upload_operations",
            telemetry.host_upload_operations,
        ));
        metrics.push((
            "cuda_device_readback_operations",
            telemetry.device_readback_operations,
        ));
        metrics.push((
            "cuda_resident_borrowed_fallback_dispatches",
            telemetry.resident_borrowed_fallback_dispatches,
        ));
        metrics.push(("cuda_launched_elements", telemetry.launched_elements));
        metrics.push(("cuda_wasted_thread_slots", telemetry.wasted_thread_slots));
        metrics.push((
            "cuda_logical_thread_utilization_bps",
            u64::from(telemetry.logical_thread_utilization_bps),
        ));
        metrics.push((
            "cuda_logical_thread_waste_bps",
            u64::from(telemetry.logical_thread_waste_bps),
        ));
        metrics.push((
            "cuda_logical_elements_per_thread_slot_bps",
            telemetry.logical_elements_per_thread_slot_bps,
        ));
        metrics.push(("cuda_timed_dispatches", telemetry.timed_dispatches));
        metrics.push((
            "cuda_timed_device_measurements",
            telemetry.timed_device_measurements,
        ));
        metrics.push((
            "cuda_timed_dispatches_missing_device_time",
            telemetry.timed_dispatches_missing_device_time,
        ));
        metrics.push(("cuda_timed_wall_ns_total", telemetry.timed_wall_ns_total));
        metrics.push((
            "cuda_timed_device_ns_total",
            telemetry.timed_device_ns_total,
        ));
        metrics.push(("cuda_timed_device_ns_max", telemetry.timed_device_ns_max));
        metrics.push((
            "cuda_timed_enqueue_ns_total",
            telemetry.timed_enqueue_ns_total,
        ));
        metrics.push(("cuda_timed_wait_ns_total", telemetry.timed_wait_ns_total));
        metrics
    }

    fn supports_subgroup_ops(&self) -> bool {
        self.inner.hardware_supports_subgroup_ops()
    }

    fn supports_f16(&self) -> bool {
        self.inner.hardware_supports_f16()
    }

    fn supports_bf16(&self) -> bool {
        self.inner.hardware_supports_bf16()
    }

    fn supports_tensor_cores(&self) -> bool {
        self.inner.hardware_supports_tensor_cores() && self.inner.lowers_tensor_core_ops()
    }

    fn supports_async_compute(&self) -> bool {
        self.inner.hardware_supports_async_compute()
    }

    fn supports_grid_sync(&self) -> bool {
        self.inner.supports_grid_sync()
    }

    fn cooperative_grid_sync_fits(
        &self,
        program: &Program,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        self.inner
            .cooperative_grid_sync_launch_fits(program, inputs, config)
    }

    fn allows_host_grid_sync_split(&self) -> bool {
        false
    }

    fn supports_resident_dispatch(&self) -> bool {
        true
    }

    fn supports_speculation(&self) -> bool {
        false
    }

    fn max_workgroup_size(&self) -> [u32; 3] {
        self.inner.max_block_dim()
    }

    fn max_compute_workgroups_per_dimension(&self) -> u32 {
        self.inner.max_grid_dim()[0]
    }

    fn max_compute_invocations_per_workgroup(&self) -> u32 {
        self.inner.max_threads_per_block()
    }

    fn subgroup_size(&self) -> Option<u32> {
        self.inner.warp_size()
    }

    fn max_storage_buffer_bytes(&self) -> u64 {
        self.inner.device_memory_bytes()
    }

    fn device_profile(&self) -> vyre_driver::DeviceProfile {
        let mut profile = self.inner.caps.to_device_profile();
        profile.supports_tensor_cores = self.supports_tensor_cores();
        profile.supports_indirect_dispatch = self.supports_indirect_dispatch();
        profile
    }

    fn prepare(&self) -> Result<(), BackendError> {
        self.inner.warmup()
    }

    fn shutdown(&self) -> Result<(), BackendError> {
        self.inner.cleanup()
    }
}

/// Factory function for inventory registration.
pub fn cuda_factory() -> Result<Box<dyn VyreBackend>, BackendError> {
    let backend = CudaBackend::acquire().map_err(|e| BackendError::DispatchFailed {
        code: None,
        message: format!("CUDA backend acquisition failed: {e}"),
    })?;
    Ok(Box::new(CudaBackendRegistration { inner: backend }))
}

/// Op-support set  -  CUDA supports every op the foundation IR defines
/// plus hardware intrinsics. Populated at runtime by the conform runner.
pub fn cuda_supported_ops() -> &'static std::collections::HashSet<vyre_foundation::ir::OpId> {
    vyre_driver::default_supported_ops_with_trap()
}

pub(crate) fn cuda_semantic_operations(
) -> &'static std::collections::HashSet<vyre_foundation::ir::OpId> {
    vyre_driver::dialect_only_supported_ops()
}

/// Backend id this crate submits into the backend registry on this target.
///
/// WHY: the registration below lives in this crate's object file, and a linker
/// keeps that object only when a symbol inside it is referenced. Naming the
/// crate with `use vyre_driver_cuda as _;` references nothing, and reading
/// [`CUDA_BACKEND_ID`] is a `const` that inlines at the use site, so neither
/// keeps the registration. Calling this function does, which is why the backend
/// registry owner calls it instead of importing the crate for effect.
#[must_use]
pub fn registered_backend_id() -> Option<&'static str> {
    Some(CUDA_BACKEND_ID)
}

inventory::submit! {
    BackendRegistration {
        id: CUDA_BACKEND_ID,
        target_id: CUDA_TARGET_ID,
        payload_format: Some(target_compiler::CUDA_TARGET_FORMAT),
        reference_oracle: false,
        factory: cuda_factory,
        supported_ops: cuda_supported_ops,
        semantic_operations: cuda_semantic_operations,
        target_compiler: Some(target_compiler::target_compiler_factory),
        materializer: Some(materializer::materializer_factory),
    }
}

// rank 5 - CUDA is the canonical release dispatch backend when linked.
inventory::submit! {
    vyre_driver::BackendPrecedence {
        id: CUDA_BACKEND_ID,
        rank: 5,
    }
}

// CUDA owns a live dispatch stack, so conform can prove against it.
inventory::submit! {
    vyre_driver::BackendCapability {
        id: CUDA_BACKEND_ID,
        dispatches: true,
    }
}

inventory::submit! {
    vyre_driver::AotLauncherEmitter {
        target: CUDA_TARGET_ID,
        emit: aot_launcher::emit_launcher,
    }
}
