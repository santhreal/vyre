use vyre_driver::accounting::checked_add_u64_lazy;
use vyre_driver::DispatchConfig;
use vyre_foundation::ir::Program;
use vyre_foundation::program_dispatch::{
    DispatchError, ProgramDispatcher, ResidentDispatchStep, ResidentReadRange,
    ResidentStaticBufferSet,
};

use crate::numeric::CUDA_NUMERIC;
use crate::resident_dispatcher::{
    reserve_resident_vec, CudaProgramDispatcher, StaticUploadCacheEntry,
};

impl<'a> ProgramDispatcher for CudaProgramDispatcher<'a> {
    fn dispatch(
        &self,
        program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let mut config = DispatchConfig::default();
        config.grid_override = grid_override;
        // CudaBackend's borrowed-dispatch path is what `dispatch` was
        // routing through previously. Keep parity for callers that
        // don't want the persistent fast-path.
        self.backend
            .dispatch(program, inputs, &config)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn supports_persistent(&self) -> bool {
        true
    }

    fn device_feature_cache_key(&self) -> u64 {
        (u64::from(self.backend.ptx_target_sm()) << 32)
            | u64::from(self.backend.pipeline_feature_flags().bits())
    }

    fn alloc_resident(&self, byte_len: usize) -> Result<u64, DispatchError> {
        // Try the pool first. The size-class lookup is exact: a
        // handle of `byte_len = 4096` is NOT pulled for a request of
        // `byte_len = 2048` even though it would fit, because the
        // backend's static-size verifier checks
        // `resident.byte_len >= expected` and the kernel's binding
        // contract assumes the buffer is of the declared length, not
        // larger. Exact-match keeps the pool semantics safe.
        if let Some(handles) = self.free_pool.borrow_mut().get_mut(&byte_len) {
            if let Some(handle) = handles.pop() {
                {
                    let mut pooled_bytes = self.pooled_bytes.borrow_mut();
                    let handle_bytes = resident_usize_to_u64(
                        handle.byte_len,
                        "resident pool reused handle bytes",
                    )?;
                    *pooled_bytes = pooled_bytes.checked_sub(handle_bytes).ok_or_else(|| {
                        DispatchError::BackendError(
                            "CUDA optimizer resident pool byte accounting underflowed during reuse"
                                .to_string(),
                        )
                    })?;
                }
                self.sizes.borrow_mut().insert(handle.handle.id(), handle);
                return Ok(handle.handle.id());
            }
        }
        let handle = self
            .backend
            .allocate_resident(byte_len)
            .map_err(|e| DispatchError::BackendError(e.to_string()))?;
        self.sizes.borrow_mut().insert(handle.handle.id(), handle);
        Ok(handle.handle.id())
    }

    fn upload_resident(&self, id: u64, bytes: &[u8]) -> Result<(), DispatchError> {
        let handle = self.resolve(id)?;
        self.backend
            .upload_resident(handle, bytes)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn upload_resident_many(&self, uploads: &[(u64, &[u8])]) -> Result<(), DispatchError> {
        let concrete = self.resolve_uploads(uploads)?;
        self.backend
            .upload_resident_many(&concrete)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn acquire_resident_static_uploads(
        &self,
        cache_domain: u64,
        payloads: &[&[u8]],
    ) -> Result<ResidentStaticBufferSet, DispatchError> {
        let key = self.static_upload_cache_key(cache_domain, payloads)?;
        if let Some(entry) = self.static_upload_cache.borrow().get(&key) {
            let mut handles = Vec::new();
            reserve_resident_vec(
                &mut handles,
                entry.handles.len(),
                "optimizer cached static handles",
            )?;
            for handle in &entry.handles {
                handles.push(handle.handle.id());
            }
            return Ok(ResidentStaticBufferSet {
                handles,
                cache_hit: true,
                retained_by_dispatcher: true,
            });
        }

        let mut handles = Vec::new();
        reserve_resident_vec(&mut handles, payloads.len(), "optimizer static handles")?;
        for payload in payloads {
            match self.alloc_resident(payload.len()) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    for handle in handles.iter().copied() {
                        let _ = self.free_resident(handle);
                    }
                    return Err(error);
                }
            }
        }

        let mut uploads = Vec::new();
        reserve_resident_vec(&mut uploads, payloads.len(), "optimizer static uploads")?;
        for (&handle, &payload) in handles.iter().zip(payloads.iter()) {
            uploads.push((handle, payload));
        }
        if let Err(error) = self.upload_resident_many(&uploads) {
            for handle in handles.iter().copied() {
                let _ = self.free_resident(handle);
            }
            return Err(error);
        }

        let bytes = self.static_payload_bytes(payloads)?;
        if !self.evict_until_static_upload_cache_has_room(bytes)? {
            return Ok(ResidentStaticBufferSet {
                handles,
                cache_hit: false,
                retained_by_dispatcher: false,
            });
        }

        let mut cached_handles = Vec::new();
        reserve_resident_vec(
            &mut cached_handles,
            handles.len(),
            "optimizer static cached concrete handles",
        )?;
        for &handle in &handles {
            cached_handles.push(self.resolve(handle)?);
        }
        self.static_upload_cache.borrow_mut().insert(
            key,
            StaticUploadCacheEntry {
                handles: cached_handles,
                bytes,
            },
        );
        {
            let mut cached_bytes = self.static_cached_bytes.borrow_mut();
            *cached_bytes = checked_add_u64_lazy(*cached_bytes, bytes, || {
                DispatchError::BackendError(
                    "CUDA optimizer static cache byte accounting overflowed while inserting"
                        .to_string(),
                )
            })?;
        }
        Ok(ResidentStaticBufferSet {
            handles,
            cache_hit: false,
            retained_by_dispatcher: true,
        })
    }

    fn read_resident(&self, id: u64) -> Result<Vec<u8>, DispatchError> {
        let handle = self.resolve(id)?;
        self.backend
            .download_resident(handle)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn read_resident_many(&self, ids: &[u64]) -> Result<Vec<Vec<u8>>, DispatchError> {
        let handles = self.resolve_many(ids)?;
        self.backend
            .download_resident_many(&handles)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn read_resident_ranges(
        &self,
        ranges: &[ResidentReadRange],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let (handles, readbacks) = self.resolve_read_ranges(ranges)?;
        self.backend
            .download_resident_readbacks_many(&handles, &readbacks)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn free_resident(&self, id: u64) -> Result<(), DispatchError> {
        let handle = self.resolve(id)?;
        // Don't actually free; return the handle to the pool. Exact
        // size-class push so the next `alloc_resident(byte_len)` of
        // the same size can pop in O(1). The handle id stays in
        // `free_pool` until exact-size reuse or budget eviction.
        self.sizes.borrow_mut().remove(&id);
        let handle_bytes =
            resident_usize_to_u64(handle.byte_len, "resident pool freed handle bytes")?;
        if !self.evict_until_resident_pool_has_room(handle_bytes)? {
            self.backend
                .free_resident(handle)
                .map_err(|e| DispatchError::BackendError(e.to_string()))?;
            return Ok(());
        }
        self.free_pool
            .borrow_mut()
            .entry(handle.byte_len)
            .or_default()
            .push(handle);
        {
            let mut pooled_bytes = self.pooled_bytes.borrow_mut();
            *pooled_bytes = checked_add_u64_lazy(*pooled_bytes, handle_bytes, || {
                DispatchError::BackendError(
                    "CUDA optimizer resident pool byte accounting overflowed while pooling a handle"
                        .to_string(),
                )
            })?;
        }
        Ok(())
    }

    fn dispatch_resident(
        &self,
        program: &Program,
        handle_ids: &[u64],
        grid_override: Option<[u32; 3]>,
    ) -> Result<(), DispatchError> {
        let handles = self.resolve_many(handle_ids)?;
        let mut config = DispatchConfig::default();
        config.grid_override = grid_override;
        // `CudaBackend::dispatch_resident` does NOT auto-readback; that
        // is what makes the persistent path fast. Caller invokes
        // `read_resident` only at the end of the pipeline.
        self.backend
            .dispatch_resident(program, &handles, &config)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn dispatch_resident_sequence(
        &self,
        steps: &[ResidentDispatchStep<'_>],
    ) -> Result<(), DispatchError> {
        let resolved_handles = self.resolve_step_handles(steps, "optimizer sequence handles")?;
        let cuda_steps =
            self.build_cuda_steps(steps, &resolved_handles, "optimizer sequence step")?;
        self.backend
            .dispatch_resident_sequence(&cuda_steps)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn dispatch_resident_sequence_read_many(
        &self,
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer read sequence handles")?;
        let resolved_reads = self.resolve_many(read_handles)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer read sequence step",
        )?;
        self.backend
            .dispatch_resident_sequence_read_many(&cuda_steps, &resolved_reads)
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn upload_resident_many_sequence_read_many(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        let concrete_uploads = self.resolve_uploads(uploads)?;
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer upload-read sequence handles")?;
        let resolved_reads = self.resolve_many(read_handles)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer upload-read sequence step",
        )?;
        self.backend
            .upload_resident_many_sequence_read_many(
                &concrete_uploads,
                &cuda_steps,
                &resolved_reads,
            )
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn upload_resident_many_sequence_read_many_into(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        let concrete_uploads = self.resolve_uploads(uploads)?;
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer upload-read-into sequence handles")?;
        let resolved_reads = self.resolve_many(read_handles)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer upload-read-into sequence step",
        )?;
        self.backend
            .upload_resident_many_sequence_read_many_into(
                &concrete_uploads,
                &cuda_steps,
                &resolved_reads,
                outputs,
            )
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn clear_upload_resident_many_sequence_read_many_into(
        &self,
        clears: &[(u64, usize)],
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        let concrete_clears = self.resolve_clears(clears)?;
        let concrete_uploads = self.resolve_uploads(uploads)?;
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer clear-upload-read sequence handles")?;
        let resolved_reads = self.resolve_many(read_handles)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer clear-upload-read sequence step",
        )?;
        self.backend
            .clear_upload_resident_many_sequence_read_many_into(
                &concrete_clears,
                &concrete_uploads,
                &cuda_steps,
                &resolved_reads,
                outputs,
            )
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn fill_upload_resident_many_sequence_read_many_into(
        &self,
        fills: &[(u64, usize, u8)],
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_handles: &[u64],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        let concrete_fills = self.resolve_fills(fills)?;
        let concrete_uploads = self.resolve_uploads(uploads)?;
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer fill-upload-read sequence handles")?;
        let resolved_reads = self.resolve_many(read_handles)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer fill-upload-read sequence step",
        )?;
        self.backend
            .fill_upload_resident_many_sequence_read_many_into(
                &concrete_fills,
                &concrete_uploads,
                &cuda_steps,
                &resolved_reads,
                outputs,
            )
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn fill_upload_resident_many_sequence_read_ranges_into(
        &self,
        fills: &[(u64, usize, u8)],
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        let concrete_fills = self.resolve_fills(fills)?;
        let concrete_uploads = self.resolve_uploads(uploads)?;
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer fill-upload-range sequence handles")?;
        let (resolved_reads, concrete_readbacks) = self.resolve_read_ranges(read_ranges)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer fill-upload-range sequence step",
        )?;
        if outputs.len() < read_ranges.len() {
            outputs.resize_with(read_ranges.len(), Vec::new);
        } else {
            outputs.truncate(read_ranges.len());
        }
        let mut borrowed_outputs = Vec::new();
        reserve_resident_vec(
            &mut borrowed_outputs,
            outputs.len(),
            "optimizer fill-upload-range borrowed output",
        )?;
        borrowed_outputs.extend(outputs.iter_mut());
        self.backend
            .fill_upload_resident_many_sequence_read_ranges_borrowed_into(
                &concrete_fills,
                &concrete_uploads,
                &cuda_steps,
                &resolved_reads,
                &concrete_readbacks,
                borrowed_outputs.as_mut_slice(),
            )
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }

    fn upload_resident_many_sequence_read_ranges_into(
        &self,
        uploads: &[(u64, &[u8])],
        steps: &[ResidentDispatchStep<'_>],
        read_ranges: &[ResidentReadRange],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), DispatchError> {
        let concrete_uploads = self.resolve_uploads(uploads)?;
        let resolved_step_handles =
            self.resolve_step_handles(steps, "optimizer upload-range sequence handles")?;
        let (resolved_reads, concrete_readbacks) = self.resolve_read_ranges(read_ranges)?;
        let cuda_steps = self.build_cuda_steps(
            steps,
            &resolved_step_handles,
            "optimizer upload-range sequence step",
        )?;
        self.backend
            .upload_resident_many_sequence_read_ranges_into(
                &concrete_uploads,
                &cuda_steps,
                &resolved_reads,
                &concrete_readbacks,
                outputs,
            )
            .map_err(|e| DispatchError::BackendError(e.to_string()))
    }
}

pub(crate) fn resident_usize_to_u64(
    value: usize,
    label: &'static str,
) -> Result<u64, DispatchError> {
    CUDA_NUMERIC
        .usize_to_u64(value, label)
        .map_err(|error| DispatchError::BackendError(error.to_string()))
}
