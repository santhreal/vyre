//! CUDA dispatch path for long-lived resident buffers.

use std::ffi::c_void;
use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use vyre_driver::binding::BindingRole;
use vyre_driver::{BackendError, BindingPlan, DispatchConfig, PendingDispatch};
use vyre_foundation::ir::Program;

use super::allocations::{DispatchAllocations, HostTransferAllocations};
use super::dispatch::CudaBackend;
use super::launch_params::launch_param_byte_len;
use super::module_cache::ModuleCacheKey;
use super::ordering::sort_unstable_by_key_if_needed;
use super::output_range::{cuda_output_readback, CudaOutputReadback};
use super::plan::CudaDispatchPlan;
use super::resident::{CudaResidentBuffer, ResidentViewCache};
use super::resident_dispatch_support::{
    add_resident_dispatch_bytes, add_resident_dispatch_u64_count,
    checked_resident_dispatch_capacity_add, checked_resident_dispatch_capacity_mul,
};
pub(crate) use super::resident_dispatch_support::{
    CudaResidentBatchDispatch, CudaResidentDispatch, CudaResidentDispatchStep,
};
use super::resident_readback_fusion::{fuse_resident_readback_copies, ResidentReadbackCopy};
use super::resident_upload_fusion::{
    fuse_resident_upload_copies, push_resident_upload_copy, ResidentUploadCopy,
};
use super::staging_reserve::{
    reserve_hash_set, reserve_smallvec, reserve_vec, reserved_vec, resize_vec_slots,
};

fn resident_required_handles(prepared: &CudaDispatchPlan) -> Result<usize, BackendError> {
    prepared
        .bindings
        .bindings
        .len()
        .checked_sub(prepared.bindings.shared_indices.len())
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident binding plan has {} binding(s) but {} shared binding index(es). Rebuild the dispatch plan before launching.",
                prepared.bindings.bindings.len(),
                prepared.bindings.shared_indices.len()
            ),
        })
}

fn stage_resident_fill_payload(
    payload: &mut Vec<u8>,
    value: u8,
    byte_len: usize,
) -> Result<&[u8], BackendError> {
    reserve_vec(payload, byte_len, "resident fallback fill byte")?;
    payload.clear();
    payload.resize(byte_len, value);
    Ok(payload.as_slice())
}

fn enqueue_resident_h2d_copy(
    dst_ptr: u64,
    host_ptr: *const c_void,
    byte_len: usize,
    stream_raw: cudarc::driver::sys::CUstream,
) -> Result<(), BackendError> {
    // SAFETY: The caller owns the stream ordering and guarantees that the
    // pinned host allocation and resident destination remain live until the
    // stream reaches this copy. The shared copy helper validates null pointers
    // for non-empty copies and treats zero-byte copies as no-ops.
    unsafe { super::copy::h2d_async_checked(dst_ptr, host_ptr, byte_len, stream_raw) }
}

fn enqueue_optional_resident_h2d_copy(
    upload: Option<(u64, *const c_void, usize)>,
    stream_raw: cudarc::driver::sys::CUstream,
) -> Result<(), BackendError> {
    if let Some((dst_ptr, host_ptr, byte_len)) = upload {
        enqueue_resident_h2d_copy(dst_ptr, host_ptr, byte_len, stream_raw)?;
    }
    Ok(())
}

fn enqueue_resident_upload_copies_on_stream(
    copies: &[ResidentUploadCopy<'_>],
    host_transfers: &mut HostTransferAllocations,
    stream_raw: cudarc::driver::sys::CUstream,
) -> Result<(), BackendError> {
    for copy in copies {
        let bytes = copy.bytes.as_slice();
        let host_ptr = host_transfers.push_upload(bytes)?;
        enqueue_resident_h2d_copy(copy.dst_ptr, host_ptr, bytes.len(), stream_raw)?;
    }
    Ok(())
}

fn borrow_resident_sequence_output_slots(
    outputs: &mut Vec<Vec<u8>>,
    slot_count: usize,
) -> Result<SmallVec<[&mut Vec<u8>; 8]>, BackendError> {
    resize_vec_slots(outputs, slot_count, "resident sequence output slots")?;
    let mut borrowed_outputs = SmallVec::<[&mut Vec<u8>; 8]>::new();
    reserve_smallvec(
        &mut borrowed_outputs,
        outputs.len(),
        "resident sequence borrowed output slots",
    )?;
    borrowed_outputs.extend(outputs.iter_mut());
    Ok(borrowed_outputs)
}

fn prepare_resident_sequence_fills(
    fills: &[(CudaResidentBuffer, u8)],
    uploads: &[(CudaResidentBuffer, &[u8])],
) -> Result<SmallVec<[(CudaResidentBuffer, u8); 8]>, BackendError> {
    let mut uploaded_handles = FxHashSet::<CudaResidentBuffer>::default();
    if !uploads.is_empty() {
        reserve_hash_set(
            &mut uploaded_handles,
            uploads.len(),
            "resident sequence upload handle set",
        )?;
        uploaded_handles.extend(uploads.iter().map(|&(handle, _)| handle));
    }

    let mut effective = SmallVec::<[(CudaResidentBuffer, u8); 8]>::new();
    reserve_smallvec(
        &mut effective,
        fills.len(),
        "resident sequence effective fills",
    )?;

    let mut effective_indices = FxHashMap::<CudaResidentBuffer, usize>::default();
    effective_indices
        .try_reserve(fills.len())
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident sequence fill index could not reserve {} handle slot(s): {error}.",
                fills.len()
            ),
        })?;

    for &(handle, value) in fills {
        if !uploaded_handles.is_empty() && uploaded_handles.contains(&handle) {
            continue;
        }
        if let Some(&index) = effective_indices.get(&handle) {
            effective[index].1 = value;
            continue;
        }
        effective_indices.insert(handle, effective.len());
        effective.push((handle, value));
    }

    Ok(effective)
}

struct PreparedStep<'a> {
    program: &'a Program,
    handles: SmallVec<[CudaResidentBuffer; 8]>,
    config: &'a DispatchConfig,
    ptx_src: Arc<str>,
    module_key: ModuleCacheKey,
    prepared: CudaDispatchPlan,
}

impl CudaBackend {
    fn resolve_resident_sequence_launch_ptrs(
        &self,
        step: &PreparedStep<'_>,
        resident_view_cache: &mut ResidentViewCache,
    ) -> Result<SmallVec<[u64; 8]>, BackendError> {
        let mut launch_ptrs = SmallVec::<[u64; 8]>::new();
        reserve_smallvec(
            &mut launch_ptrs,
            step.prepared.bindings.bindings.len(),
            "resident sequence launch pointers",
        )?;
        let mut next_handle = 0usize;
        for binding in &step.prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let handle = step.handles[next_handle];
            next_handle += 1;
            let resident = self.resident_store.view_cached(
                handle,
                resident_view_cache,
                "resident sequence view cache",
            )?;
            if let Some(expected) = binding.static_byte_len {
                if resident.byte_len < expected {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident sequence binding `{}` expected at least {expected} bytes but handle {} has {} bytes.",
                            binding.name, handle.id, resident.byte_len
                        ),
                    });
                }
            }
            if resident.ptr == 0 {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident sequence binding `{}` resolved to a null device pointer; resident launch arguments must preserve descriptor order.",
                        binding.name
                    ),
                });
            }
            launch_ptrs.push(resident.ptr);
        }
        Ok(launch_ptrs)
    }

    pub(crate) fn dispatch_resident_via_borrowed_into(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        let plan = BindingPlan::build(program)?;
        let required_handles = plan
            .bindings
            .len()
            .checked_sub(plan.shared_indices.len())
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident fallback binding plan has {} binding(s) but {} shared binding index(es). Rebuild the dispatch plan before launching.",
                    plan.bindings.len(),
                    plan.shared_indices.len()
                ),
            })?;
        if handles.len() != required_handles {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident fallback expected {required_handles} resident buffer handle(s) but received {}.",
                    handles.len()
                ),
            });
        }
        let mut input_storage =
            reserved_vec(plan.input_indices.len(), "resident fallback input storage")?;
        let mut output_handles =
            reserved_vec(plan.output_indices.len(), "resident fallback output handle")?;
        let mut next_handle = 0usize;
        for binding in &plan.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let handle = handles[next_handle];
            next_handle += 1;
            if binding.input_index.is_some() {
                input_storage.push(self.download_resident(handle)?);
            }
            if let Some(output_index) = binding.output_index {
                output_handles.push((output_index, handle));
            }
        }
        let mut input_refs = SmallVec::<[&[u8]; 8]>::new();
        reserve_smallvec(
            &mut input_refs,
            input_storage.len(),
            "resident fallback input reference",
        )?;
        input_refs.extend(input_storage.iter().map(Vec::as_slice));
        let dispatch_outputs = self.dispatch_borrowed(program, &input_refs, config)?;
        let mut output_uploads = SmallVec::<[(CudaResidentBuffer, &[u8]); 8]>::new();
        reserve_smallvec(
            &mut output_uploads,
            output_handles.len(),
            "resident fallback output upload",
        )?;
        for &(output_index, handle) in &output_handles {
            let output =
                dispatch_outputs
                    .get(output_index)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident fallback missing output slot {output_index}; keep borrowed dispatch output ordering aligned with BindingPlan."
                        ),
                    })?;
            if !output.is_empty() {
                output_uploads.push((handle, output.as_slice()));
            }
        }
        self.upload_resident_many(&output_uploads)?;
        drop(output_uploads);
        vyre_driver::replace_output_buffers_preserving_slots(dispatch_outputs, outputs);
        Ok(())
    }

    pub(crate) fn dispatch_resident_via_borrowed(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = reserved_vec(0, "borrowed resident dispatch outputs")?;
        self.dispatch_resident_via_borrowed_into(program, handles, config, &mut outputs)?;
        Ok(outputs)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_resident_param_upload(
        &self,
        param_words: &[u32],
        param_bytes: usize,
        allocation_budget_label: &'static str,
        upload_budget_label: &'static str,
        allocation_metric_label: &'static str,
        upload_metric_label: &'static str,
        allocations: &mut DispatchAllocations,
        host_transfers: &mut HostTransferAllocations,
    ) -> Result<(u64, Option<(u64, *const c_void, usize)>), BackendError> {
        self.validate_transient_allocation_memory_budget(
            param_bytes,
            allocation_budget_label,
            upload_budget_label,
        )?;
        let params_allocation = self.transient_pool.acquire(param_bytes)?;
        self.telemetry.record_transient_allocation_bytes(
            crate::numeric::CUDA_NUMERIC
                .usize_to_u64(params_allocation.byte_len, allocation_metric_label)?,
        );
        let params_ptr = params_allocation.ptr;
        let param_host_ptr = host_transfers.push_u32_words(param_words)?;
        self.telemetry.record_host_to_device_bytes(
            crate::numeric::CUDA_NUMERIC.usize_to_u64(param_bytes, upload_metric_label)?,
        );
        self.telemetry.record_host_upload_operations(1);
        self.telemetry.record_param_upload_bytes(
            crate::numeric::CUDA_NUMERIC.usize_to_u64(param_bytes, upload_metric_label)?,
        );
        allocations.set_params(params_allocation);
        Ok((params_ptr, Some((params_ptr, param_host_ptr, param_bytes))))
    }

    fn prepare_resident_sequence_upload_copies<'a>(
        &self,
        uploads: &[(CudaResidentBuffer, &'a [u8])],
    ) -> Result<(SmallVec<[ResidentUploadCopy<'a>; 8]>, u64), BackendError> {
        let mut upload_copies = SmallVec::<[ResidentUploadCopy<'a>; 8]>::new();
        reserve_smallvec(
            &mut upload_copies,
            uploads.len(),
            "resident sequence upload copies",
        )?;
        let mut uploaded_bytes = 0_u64;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            uploads.len(),
            "resident sequence upload view cache",
        )?;
        for &(handle, bytes) in uploads {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident sequence upload view cache",
            )?;
            if bytes.len() != buffer.byte_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident sequence upload for handle {} expected {} bytes but received {}.",
                        handle.id,
                        buffer.byte_len,
                        bytes.len()
                    ),
                });
            }
            push_resident_upload_copy(
                &mut upload_copies,
                &mut uploaded_bytes,
                handle.id,
                buffer.ptr,
                bytes,
                "sequence upload",
            )?;
        }
        fuse_resident_upload_copies(upload_copies)
    }

    fn push_prepared_resident_sequence_step<'a>(
        &self,
        step: &'a CudaResidentDispatchStep<'a>,
        prepared_steps: &mut SmallVec<[PreparedStep<'a>; 8]>,
        target_indices: &mut SmallVec<[usize; 16]>,
        all_handles: &mut SmallVec<[CudaResidentBuffer; 32]>,
    ) -> Result<(), BackendError> {
        all_handles.extend(step.handles.iter().copied());
        if let Some(index) = prepared_steps.iter().position(|cached| {
            std::ptr::addr_eq(cached.program, step.program)
                && cached.handles.as_slice() == step.handles
                && cached.config == &step.config
        }) {
            target_indices.push(index);
            return Ok(());
        }
        let prepared = self.prepare_resident_dispatch(step.program, step.handles, &step.config)?;
        let (ptx_src, ptx_source_key) =
            self.ptx_for_program_cached_with_key(step.program, &step.config)?;
        let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
        let step_index = prepared_steps.len();
        prepared_steps.push(PreparedStep {
            program: step.program,
            handles: SmallVec::<[CudaResidentBuffer; 8]>::from_slice(step.handles),
            config: &step.config,
            ptx_src,
            module_key,
            prepared,
        });
        target_indices.push(step_index);
        Ok(())
    }

    /// Dispatch a Program asynchronously using caller-provided CUDA-resident buffers.
    pub fn dispatch_resident_async(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<Box<dyn PendingDispatch>, BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            let outputs = self.dispatch_resident_via_borrowed(program, handles, config)?;
            return Ok(Box::new(crate::stream::CudaPendingDispatch::new_ready(
                Arc::clone(&self.ctx),
                Arc::clone(&self.launch_resources),
                outputs,
                Arc::clone(&self.telemetry),
            )));
        }
        {
            let prepared = self.prepare_resident_dispatch(program, handles, config)?;
            let (ptx_src, ptx_source_key) =
                self.ptx_for_program_cached_with_key(program, config)?;
            let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
            let native = self.dispatch_resident_async_concrete_with_ptx_key(
                program, handles, config, &ptx_src, module_key, false, None, true, &prepared,
            )?;
            return Ok(Box::new(native.pending));
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_resident_async_concrete_with_ptx_key(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        _config: &DispatchConfig,
        ptx_src: &str,
        module_key: ModuleCacheKey,
        capture_timing: bool,
        static_params_ptr: Option<u64>,
        capture_outputs: bool,
        prepared: &CudaDispatchPlan,
    ) -> Result<CudaResidentDispatch, BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_RESIDENT_DISPATCH_RANGE);
        let trace = crate::instrumentation::cuda_stage_trace_enabled();
        let start = std::time::Instant::now();
        if trace {
            tracing::debug!(
                "[cuda-trace] resident dispatch start buffers={} handles={}",
                program.buffers().len(),
                handles.len()
            );
        }
        self.warmup()?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident warmup",
                start.elapsed().as_millis()
            );
        }
        let required_handles = resident_required_handles(prepared)?;
        if handles.len() != required_handles {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident dispatch expected {required_handles} resident buffer handle(s) but received {}.",
                    handles.len()
                ),
            });
        }
        let mut allocations =
            DispatchAllocations::new(program.buffers().len(), Arc::clone(&self.transient_pool))?;
        let mut launch_ptrs = SmallVec::<[u64; 8]>::new();
        reserve_smallvec(
            &mut launch_ptrs,
            prepared.bindings.bindings.len(),
            "resident dispatch launch pointers",
        )?;
        let mut output_stage_readbacks = SmallVec::<[(u64, CudaOutputReadback); 8]>::new();
        reserve_smallvec(
            &mut output_stage_readbacks,
            if capture_outputs {
                prepared.output_binding_indices.len()
            } else {
                0
            },
            "resident dispatch output staged readbacks",
        )?;
        let mut next_handle = 0usize;
        let mut output_handles_by_index =
            SmallVec::<[(usize, CudaResidentBuffer, CudaOutputReadback, u64); 8]>::new();
        reserve_smallvec(
            &mut output_handles_by_index,
            prepared.output_binding_indices.len(),
            "resident dispatch output handles by index",
        )?;
        let mut output_clears = SmallVec::<[(u64, usize); 8]>::new();
        reserve_smallvec(
            &mut output_clears,
            prepared.output_binding_indices.len(),
            "resident dispatch output clears",
        )?;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            handles.len(),
            "resident dispatch view cache",
        )?;
        for binding in &prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let handle = handles[next_handle];
            next_handle += 1;
            let resident = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident dispatch view cache",
            )?;
            if let Some(expected) = binding.static_byte_len {
                if resident.byte_len < expected {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident buffer `{}` expected at least {expected} bytes but handle {} has {} bytes.",
                            binding.name, handle.id, resident.byte_len
                        ),
                    });
                }
            }
            if resident.ptr == 0 {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident binding `{}` resolved to a null device pointer; resident launch arguments must preserve descriptor order.",
                        binding.name
                    ),
                });
            }
            let launch_ptr = resident.ptr;
            launch_ptrs.push(launch_ptr);
            if let Some(output_index) = binding.output_index {
                let full_byte_len = match binding.static_byte_len {
                    Some(len) => len,
                    None => resident.byte_len,
                };
                let readback =
                    cuda_output_readback(&program.buffers()[binding.buffer_index], full_byte_len)?;
                output_handles_by_index.push((output_index, handle, readback, launch_ptr));
                if binding.input_index.is_none() && full_byte_len != 0 {
                    output_clears.push((launch_ptr, full_byte_len));
                }
            }
        }
        if output_handles_by_index.len() != prepared.output_binding_indices.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident dispatch expected {} output handle(s) but resolved {}.",
                    prepared.output_binding_indices.len(),
                    output_handles_by_index.len()
                ),
            });
        }
        sort_unstable_by_key_if_needed(
            output_handles_by_index.as_mut_slice(),
            |(output_index, _, _, _)| *output_index,
        );
        let mut output_handles = SmallVec::<[CudaResidentBuffer; 8]>::new();
        reserve_smallvec(
            &mut output_handles,
            output_handles_by_index.len(),
            "resident dispatch output handles",
        )?;
        let mut output_readbacks = SmallVec::<[CudaOutputReadback; 8]>::new();
        reserve_smallvec(
            &mut output_readbacks,
            output_handles_by_index.len(),
            "resident dispatch output readbacks",
        )?;
        for (_, handle, readback, launch_ptr) in output_handles_by_index {
            output_handles.push(handle);
            output_readbacks.push(readback);
            if capture_outputs {
                output_stage_readbacks.push((launch_ptr, readback));
            }
        }
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident args/readbacks launch_ptrs={:x?} output_clears={} output_stage_readbacks={}",
                start.elapsed().as_millis(),
                launch_ptrs,
                output_clears.len(),
                output_stage_readbacks.len()
            );
        }

        let param_bytes = launch_param_byte_len(&prepared.launch.param_words, "resident dispatch")?;
        let mut host_transfers = HostTransferAllocations::with_capacity(
            Arc::clone(&self.host_pool),
            usize::from(static_params_ptr.is_none() && param_bytes != 0),
            output_stage_readbacks.len(),
        )?;
        let mut param_upload: Option<(u64, *const c_void, usize)> = None;
        let params_ptr = match static_params_ptr {
            Some(ptr) => ptr,
            None if param_bytes == 0 => 0,
            None => {
                let (params_ptr, upload) = self.prepare_resident_param_upload(
                    &prepared.launch.param_words,
                    param_bytes,
                    "CUDA resident dispatch parameter bytes",
                    "CUDA resident dispatch parameter upload",
                    "resident dispatch parameter allocation byte count",
                    "resident dispatch parameter upload byte count",
                    &mut allocations,
                    &mut host_transfers,
                )?;
                param_upload = upload;
                params_ptr
            }
        };
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident params ptr=0x{params_ptr:x} words={:?} grid={:?} workgroup={:?} element_count={}",
                start.elapsed().as_millis(),
                prepared.launch.param_words,
                prepared.launch.grid,
                prepared.launch.workgroup,
                prepared.launch.element_count
            );
        }

        let resident_use = self.resident_store.mark_inflight(handles)?;
        let launch_resources = crate::stream::CudaLaunchResourceLease::acquire(
            Arc::clone(&self.launch_resources),
            capture_timing,
        )?;
        let stream_raw = launch_resources.stream_raw()?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident allocations/stream",
                start.elapsed().as_millis()
            );
        }
        enqueue_optional_resident_h2d_copy(param_upload, stream_raw)?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident param upload enqueued",
                start.elapsed().as_millis()
            );
        }
        for &(dst_ptr, byte_len) in &output_clears {
            // SAFETY: FFI to libcuda.so. Resident output pointers were
            // validated above and byte lengths come from the binding/readback
            // plan. The memset is enqueued on the same stream before launch,
            // matching the borrowed CUDA dispatch output-zeroing contract.
            unsafe {
                super::copy::memset_d8_async_checked(dst_ptr, 0, byte_len, stream_raw)?;
            }
        }
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident output clears enqueued",
                start.elapsed().as_millis()
            );
        }
        if crate::instrumentation::cuda_resident_sync_before_launch_enabled() {
            // SAFETY: stream_raw is owned by launch_resources for the
            // duration of this dispatch. This opt-in diagnostic fence isolates
            // setup copies/memsets from kernel execution without changing the
            // release default.
            crate::stream::synchronize_raw_stream(
                stream_raw,
                "cuStreamSynchronize (resident prelaunch)",
            )?;
            self.telemetry.record_sync_point();
            if trace {
                tracing::debug!(
                    "[cuda-trace] +{}ms resident prelaunch sync complete",
                    start.elapsed().as_millis()
                );
            }
        }

        if let Some((start_event, _)) = launch_resources.timing_events()? {
            start_event.record(stream_raw)?;
        }
        // Fixpoint loop  -  see dispatch_borrowed_async_with_ptx_concrete
        // for the contract. Resolve the CUDA function and argument vector
        // once; fixpoint iterations are kernel replays, not relowering or
        // module-cache lookups.
        let func = self.resolve_launch_function(
            ptx_src,
            module_key,
            &prepared.launch,
            prepared.cooperative,
        )?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident resolve_launch_function",
                start.elapsed().as_millis()
            );
        }
        let mut params_ref = params_ptr;
        let mut kernel_args = Self::kernel_args(&mut launch_ptrs, &mut params_ref)?;
        for _ in 0..prepared.fixpoint_iterations {
            self.launch_prevalidated_function(
                func,
                &mut kernel_args,
                &prepared.launch,
                stream_raw,
                false,
                prepared.cooperative,
            )?;
        }
        if let Some((_, end_event)) = launch_resources.timing_events()? {
            end_event.record(stream_raw)?;
        }
        // SAFETY: stream_raw is the live CUDA stream used for the launches
        // above. Native resident dispatch intentionally fences after the
        // kernel before host-visible output staging. The direct async DtoH/DtoD
        // path after a resident-staged launch can leave the completion event
        // unsignaled on current CUDA drivers, while an explicit post-kernel
        // fence followed by synchronous readback preserves correctness and
        // keeps the actual Program execution on CUDA instead of falling back
        // to host-buffer dispatch.
        crate::stream::synchronize_raw_stream(
            stream_raw,
            "cuStreamSynchronize (resident post-kernel)",
        )?;
        self.telemetry.record_sync_point();
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident post-kernel sync complete",
                start.elapsed().as_millis()
            );
        }
        let mut staged_readback_bytes = 0_u64;
        let mut staged_readback_ops = 0_u64;
        for &(src_base_ptr, readback) in &output_stage_readbacks {
            let dst = host_transfers.push_output(readback.byte_len)?;
            if readback.byte_len != 0 {
                add_resident_dispatch_bytes(
                    &mut staged_readback_bytes,
                    readback.byte_len,
                    "resident staged output readback",
                )?;
                add_resident_dispatch_u64_count(
                    &mut staged_readback_ops,
                    "resident staged output readback operation",
                )?;
                let src_ptr = vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
                    src_base_ptr,
                    readback.device_offset,
                    || {
                        BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident staged output readback offset {} does not fit CUdeviceptr arithmetic.",
                            readback.device_offset
                        ),
                    }
                    },
                    || BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident staged output pointer overflowed at offset {}.",
                            readback.device_offset
                        ),
                    },
                )?;
                // SAFETY: The source is the transient launch output buffer and
                // the destination is pinned host staging owned by
                // host_transfers. The stream was explicitly synchronized after
                // the kernel above, so a synchronous copy is ordered and
                // cannot strand the completion event behind an async copy that
                // the driver never completes.
                unsafe {
                    super::copy::d2h_sync_checked_with_label(
                        dst,
                        src_ptr,
                        readback.byte_len,
                        "cuMemcpyDtoH_v2",
                    )?;
                }
            }
        }
        self.telemetry
            .record_device_to_host_readback(staged_readback_bytes);
        self.telemetry
            .record_device_readback_operations(staged_readback_ops);
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident launch/output readbacks",
                start.elapsed().as_millis()
            );
        }
        let (stream, timing_events) = launch_resources.into_parts()?;
        let mut outputs = reserved_vec(output_stage_readbacks.len(), "resident staged output")?;
        host_transfers.collect_outputs_into(&mut outputs)?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident output collection complete",
                start.elapsed().as_millis()
            );
        }
        self.launch_resources.release_stream(stream);
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident stream released",
                start.elapsed().as_millis()
            );
        }
        let device_ns = match timing_events.as_ref() {
            Some((start_event, end_event)) => Some(start_event.elapsed_time_ns(end_event)?),
            None => None,
        };
        if let Some((start_event, end_event)) = timing_events {
            self.launch_resources.release_timing_event(start_event);
            self.launch_resources.release_timing_event(end_event);
        }
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident timing events released",
                start.elapsed().as_millis()
            );
        }
        drop(resident_use);
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident use released",
                start.elapsed().as_millis()
            );
        }
        drop(allocations);
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident allocations released",
                start.elapsed().as_millis()
            );
        }
        drop(host_transfers);
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident host transfers released",
                start.elapsed().as_millis()
            );
        }
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident synchronous completion",
                start.elapsed().as_millis()
            );
        }
        Ok(CudaResidentDispatch {
            pending: crate::stream::CudaPendingDispatch::new_ready_timed(
                Arc::clone(&self.ctx),
                Arc::clone(&self.launch_resources),
                outputs,
                device_ns,
                Arc::clone(&self.telemetry),
            ),
            output_handles,
            output_readbacks,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_resident_batch_async_concrete_with_ptx_key(
        &self,
        program: &Program,
        batches: &[SmallVec<[CudaResidentBuffer; 8]>],
        _config: &DispatchConfig,
        ptx_src: &str,
        module_key: ModuleCacheKey,
        static_params_ptr: Option<u64>,
        prepared: &CudaDispatchPlan,
    ) -> Result<CudaResidentBatchDispatch, BackendError> {
        if batches.is_empty() {
            return Err(BackendError::InvalidProgram {
                fix:
                    "Fix: CUDA resident batch dispatch requires at least one resident handle tuple."
                        .into(),
            });
        }
        self.warmup()?;
        let required_handles = resident_required_handles(prepared)?;
        let batch_handle_capacity = checked_resident_dispatch_capacity_mul(
            batches.len(),
            required_handles,
            "batch handle",
        )?;
        let mut all_handles = SmallVec::<[CudaResidentBuffer; 32]>::new();
        reserve_smallvec(
            &mut all_handles,
            batch_handle_capacity,
            "resident batch all handles",
        )?;
        for (batch_index, handles) in batches.iter().enumerate() {
            if handles.len() != required_handles {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident batch dispatch item {batch_index} expected {required_handles} resident buffer handle(s) but received {}.",
                        handles.len()
                    ),
                });
            }
            all_handles.extend(handles.iter().copied());
        }

        let param_bytes =
            launch_param_byte_len(&prepared.launch.param_words, "resident batch dispatch")?;
        let mut allocations =
            DispatchAllocations::new(program.buffers().len(), Arc::clone(&self.transient_pool))?;
        let mut host_transfers = HostTransferAllocations::with_capacity(
            Arc::clone(&self.host_pool),
            usize::from(static_params_ptr.is_none() && param_bytes != 0),
            0,
        )?;
        let mut param_upload: Option<(u64, *const c_void, usize)> = None;
        let params_ptr = match static_params_ptr {
            Some(ptr) => ptr,
            None if param_bytes == 0 => 0,
            None => {
                let (params_ptr, upload) = self.prepare_resident_param_upload(
                    &prepared.launch.param_words,
                    param_bytes,
                    "CUDA resident batch dispatch parameter bytes",
                    "CUDA resident batch dispatch parameter upload",
                    "resident batch dispatch parameter allocation byte count",
                    "resident batch dispatch parameter upload byte count",
                    &mut allocations,
                    &mut host_transfers,
                )?;
                param_upload = upload;
                params_ptr
            }
        };

        let func = self.resolve_launch_function(
            ptx_src,
            module_key,
            &prepared.launch,
            prepared.cooperative,
        )?;
        let mut output_handles_by_batch = SmallVec::<[SmallVec<[CudaResidentBuffer; 8]>; 8]>::new();
        reserve_smallvec(
            &mut output_handles_by_batch,
            batches.len(),
            "resident batch output handles",
        )?;
        let mut output_readbacks_by_batch =
            SmallVec::<[SmallVec<[CudaOutputReadback; 8]>; 8]>::new();
        reserve_smallvec(
            &mut output_readbacks_by_batch,
            batches.len(),
            "resident batch output readbacks",
        )?;
        let mut launch_ptrs_by_batch = SmallVec::<[SmallVec<[u64; 8]>; 8]>::new();
        reserve_smallvec(
            &mut launch_ptrs_by_batch,
            batches.len(),
            "resident batch launch pointer groups",
        )?;
        let output_binding_count = prepared.output_binding_indices.len();
        let total_output_entries = if output_binding_count == 0 {
            0usize
        } else {
            checked_resident_dispatch_capacity_mul(
                batches.len(),
                output_binding_count,
                "batch output-handle set",
            )?
        };
        let seen_outputs_small = total_output_entries <= 8 && total_output_entries != 0;
        let mut seen_output_handles_small = SmallVec::<[u64; 8]>::new();
        reserve_smallvec(
            &mut seen_output_handles_small,
            total_output_entries.min(8),
            "resident batch small output duplicate set",
        )?;
        let mut seen_output_handles = if !seen_outputs_small && total_output_entries != 0 {
            let mut set = FxHashSet::default();
            reserve_hash_set(
                &mut set,
                total_output_entries,
                "resident batch output duplicate set",
            )?;
            Some(set)
        } else {
            None
        };

        for (batch_index, handles) in batches.iter().enumerate() {
            let mut launch_ptrs = SmallVec::<[u64; 8]>::new();
            reserve_smallvec(
                &mut launch_ptrs,
                prepared.bindings.bindings.len(),
                "resident batch launch pointers",
            )?;
            let mut next_handle = 0usize;
            let mut output_handles_by_index =
                SmallVec::<[(usize, CudaResidentBuffer, CudaOutputReadback); 8]>::new();
            reserve_smallvec(
                &mut output_handles_by_index,
                prepared.output_binding_indices.len(),
                "resident batch output handles by index",
            )?;
            let mut resident_view_cache = ResidentViewCache::new();
            reserve_smallvec(
                &mut resident_view_cache,
                handles.len(),
                "resident batch dispatch view cache",
            )?;
            for binding in &prepared.bindings.bindings {
                if binding.role == BindingRole::Shared {
                    continue;
                }
                let handle = handles[next_handle];
                next_handle += 1;
                let resident = self.resident_store.view_cached(
                    handle,
                    &mut resident_view_cache,
                    "resident batch dispatch view cache",
                )?;
                if let Some(expected) = binding.static_byte_len {
                    if resident.byte_len < expected {
                        return Err(BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: CUDA resident batch dispatch item {batch_index} binding `{}` expected at least {expected} bytes but handle {} has {} bytes.",
                                binding.name, handle.id, resident.byte_len
                            ),
                        });
                    }
                }
                if resident.ptr == 0 {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident batch dispatch item {batch_index} binding `{}` resolved to a null device pointer; resident launch arguments must preserve descriptor order.",
                            binding.name
                        ),
                    });
                }
                launch_ptrs.push(resident.ptr);
                if let Some(output_index) = binding.output_index {
                    let full_byte_len = match binding.static_byte_len {
                        Some(len) => len,
                        None => resident.byte_len,
                    };
                    let readback = cuda_output_readback(
                        &program.buffers()[binding.buffer_index],
                        full_byte_len,
                    )?;
                    output_handles_by_index.push((output_index, handle, readback));
                }
            }
            if output_handles_by_index.len() != prepared.output_binding_indices.len() {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident batch dispatch item {batch_index} expected {} output handle(s) but resolved {}.",
                        prepared.output_binding_indices.len(),
                        output_handles_by_index.len()
                    ),
                });
            }
            sort_unstable_by_key_if_needed(
                output_handles_by_index.as_mut_slice(),
                |(output_index, _, _)| *output_index,
            );
            let mut output_handles = SmallVec::<[CudaResidentBuffer; 8]>::new();
            reserve_smallvec(
                &mut output_handles,
                output_handles_by_index.len(),
                "resident batch output handles",
            )?;
            let mut output_readbacks = SmallVec::<[CudaOutputReadback; 8]>::new();
            reserve_smallvec(
                &mut output_readbacks,
                output_handles_by_index.len(),
                "resident batch output readbacks",
            )?;
            for (_, handle, readback) in output_handles_by_index {
                if !seen_outputs_small {
                    if let Some(seen_output_handles) = seen_output_handles.as_mut() {
                        if !seen_output_handles.insert(handle.id) {
                            return Err(BackendError::InvalidProgram {
                                fix: format!(
                                    "Fix: CUDA resident batch dispatch cannot reuse output handle {} across submitted items; allocate one output resident buffer tuple per in-flight batch item so batched readback observes every result instead of the final overwrite.",
                                    handle.id
                                ),
                            });
                        }
                    }
                } else {
                    if seen_output_handles_small.contains(&handle.id) {
                        return Err(BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: CUDA resident batch dispatch cannot reuse output handle {} across submitted items; allocate one output resident buffer tuple per in-flight batch item so batched readback observes every result instead of the final overwrite.",
                                handle.id
                            ),
                        });
                    }
                    seen_output_handles_small.push(handle.id);
                }
                output_handles.push(handle);
                output_readbacks.push(readback);
            }

            if output_handles.len() != prepared.output_binding_indices.len() {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident batch dispatch item {batch_index} expected {} output handle(s) but resolved {}.",
                        prepared.output_binding_indices.len(),
                        output_handles.len()
                    ),
                });
            }
            if output_handles.len() != output_readbacks.len() {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: CUDA resident batch dispatch output handle/readback stream mismatch after reordering outputs."
                        .into(),
                });
            }

            launch_ptrs_by_batch.push(launch_ptrs);
            output_handles_by_batch.push(output_handles);
            output_readbacks_by_batch.push(output_readbacks);
        }

        let resident_use = self.resident_store.mark_inflight(&all_handles)?;
        let launch_resources = crate::stream::CudaLaunchResourceLease::acquire(
            Arc::clone(&self.launch_resources),
            false,
        )?;
        let stream_raw = launch_resources.stream_raw()?;
        enqueue_optional_resident_h2d_copy(param_upload, stream_raw)?;

        let mut kernel_args = SmallVec::<[*mut c_void; 8]>::new();
        for mut launch_ptrs in launch_ptrs_by_batch {
            let mut params_ref = params_ptr;
            Self::kernel_args_into(&mut launch_ptrs, &mut params_ref, &mut kernel_args)?;
            for _ in 0..prepared.fixpoint_iterations {
                self.launch_prevalidated_function(
                    func,
                    &mut kernel_args,
                    &prepared.launch,
                    stream_raw,
                    false,
                    prepared.cooperative,
                )?;
            }
        }

        let event = self.launch_resources.acquire_event()?;
        if let Err(error) = event.record(stream_raw) {
            self.launch_resources.release_event(event);
            return Err(error);
        }
        let (stream, _) = launch_resources.into_parts()?;
        let pending = crate::stream::CudaPendingDispatch::new_resident_batch_pending(
            Arc::clone(&self.ctx),
            Arc::clone(&self.launch_resources),
            event,
            stream,
            allocations,
            resident_use,
            host_transfers,
            Arc::clone(&self.telemetry),
        );
        Ok(CudaResidentBatchDispatch {
            pending,
            output_handles: output_handles_by_batch,
            output_readbacks: output_readbacks_by_batch,
        })
    }

    /// Dispatch a Program using caller-provided CUDA-resident buffers.
    pub fn dispatch_resident(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            return self
                .dispatch_resident_via_borrowed(program, handles, config)
                .map(|_| ());
        }
        {
            let prepared = self.prepare_resident_dispatch(program, handles, config)?;
            let (ptx_src, ptx_source_key) =
                self.ptx_for_program_cached_with_key(program, config)?;
            let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
            self.dispatch_resident_async_concrete_with_ptx_key(
                program, handles, config, &ptx_src, module_key, false, None, false, &prepared,
            )?;
            return Ok(());
        }
    }

    pub(crate) fn dispatch_resident_sequence(
        &self,
        steps: &[CudaResidentDispatchStep<'_>],
    ) -> Result<(), BackendError> {
        self.dispatch_resident_sequence_read_many(steps, &[])
            .map(|_| ())
    }

    pub(crate) fn dispatch_resident_sequence_read_many(
        &self,
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        self.upload_resident_many_sequence_read_many(&[], steps, read_handles)
    }

    pub(crate) fn upload_resident_many_sequence_read_many(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = reserved_vec(read_handles.len(), "resident sequence read outputs")?;
        self.upload_resident_many_sequence_read_many_into(
            uploads,
            steps,
            read_handles,
            &mut outputs,
        )?;
        Ok(outputs)
    }

    pub(crate) fn upload_resident_many_sequence_read_many_into(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        let mut readbacks = SmallVec::<[CudaOutputReadback; 8]>::new();
        self.prepare_full_resident_readbacks(read_handles, &mut readbacks)?;
        self.upload_resident_many_sequence_read_ranges_into(
            uploads,
            steps,
            read_handles,
            &readbacks,
            outputs,
        )
    }

    pub(crate) fn clear_upload_resident_many_sequence_read_many_into(
        &self,
        clears: &[CudaResidentBuffer],
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        let mut fills = SmallVec::<[(CudaResidentBuffer, u8); 8]>::new();
        reserve_smallvec(&mut fills, clears.len(), "resident sequence clear fills")?;
        fills.extend(clears.iter().copied().map(|handle| (handle, 0)));
        self.fill_upload_resident_many_sequence_read_many_into(
            &fills,
            uploads,
            steps,
            read_handles,
            outputs,
        )
    }

    pub(crate) fn fill_upload_resident_many_sequence_read_many_into(
        &self,
        fills: &[(CudaResidentBuffer, u8)],
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        let mut readbacks = SmallVec::<[CudaOutputReadback; 8]>::new();
        self.prepare_full_resident_readbacks(read_handles, &mut readbacks)?;
        let mut borrowed_outputs =
            borrow_resident_sequence_output_slots(outputs, read_handles.len())?;
        self.fill_upload_resident_many_sequence_read_ranges_borrowed_into(
            fills,
            uploads,
            steps,
            read_handles,
            &readbacks,
            borrowed_outputs.as_mut_slice(),
        )
    }

    fn prepare_full_resident_readbacks(
        &self,
        read_handles: &[CudaResidentBuffer],
        readbacks: &mut SmallVec<[CudaOutputReadback; 8]>,
    ) -> Result<(), BackendError> {
        readbacks.clear();
        reserve_smallvec(
            readbacks,
            read_handles.len(),
            "resident sequence full readbacks",
        )?;
        let mut resident_view_cache = ResidentViewCache::new();
        reserve_smallvec(
            &mut resident_view_cache,
            read_handles.len(),
            "resident sequence full-readback view cache",
        )?;
        for &handle in read_handles {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident sequence full-readback view cache",
            )?;
            readbacks.push(CudaOutputReadback {
                device_offset: 0,
                byte_len: buffer.byte_len,
            });
        }
        Ok(())
    }

    pub(crate) fn upload_resident_many_sequence_read_ranges_into(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        let mut borrowed_outputs =
            borrow_resident_sequence_output_slots(outputs, read_handles.len())?;
        self.upload_resident_many_sequence_read_ranges_borrowed_into(
            uploads,
            steps,
            read_handles,
            readbacks,
            borrowed_outputs.as_mut_slice(),
        )
    }

    pub(crate) fn upload_resident_many_sequence_read_ranges_borrowed_into(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
            &[],
            uploads,
            steps,
            &[],
            0,
            read_handles,
            readbacks,
            outputs,
        )
    }

    pub(crate) fn fill_upload_resident_many_sequence_read_ranges_borrowed_into(
        &self,
        fills: &[(CudaResidentBuffer, u8)],
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
            fills,
            uploads,
            steps,
            &[],
            0,
            read_handles,
            readbacks,
            outputs,
        )
    }

    pub(crate) fn upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        prefix_steps: &[CudaResidentDispatchStep<'_>],
        repeated_steps: &[CudaResidentDispatchStep<'_>],
        repeat_count: usize,
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
            &[],
            uploads,
            prefix_steps,
            repeated_steps,
            repeat_count,
            read_handles,
            readbacks,
            outputs,
        )
    }

    pub(crate) fn fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
        &self,
        fills: &[(CudaResidentBuffer, u8)],
        uploads: &[(CudaResidentBuffer, &[u8])],
        prefix_steps: &[CudaResidentDispatchStep<'_>],
        repeated_steps: &[CudaResidentDispatchStep<'_>],
        repeat_count: usize,
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        if read_handles.len() != readbacks.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident sequence compact readback expected matching handle/range counts but got {} handle(s) and {} range(s).",
                    read_handles.len(),
                    readbacks.len()
                ),
            });
        }
        if outputs.len() != read_handles.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident sequence compact readback expected matching output/range counts but got {} output slot(s) and {} range(s).",
                    outputs.len(),
                    read_handles.len()
                ),
            });
        }
        if fills.is_empty()
            && uploads.is_empty()
            && prefix_steps.is_empty()
            && (repeated_steps.is_empty() || repeat_count == 0)
            && read_handles.is_empty()
        {
            return Ok(());
        }
        const RESIDENT_BORROWED_FALLBACK_ENV: &str = "VYRE_CUDA_RESIDENT_BORROWED_FALLBACK";
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            tracing::debug!(
                "[cuda-trace] resident repeated sequence using borrowed fallback enabled by {RESIDENT_BORROWED_FALLBACK_ENV}"
            );
            let mut fill_payload = Vec::new();
            for &(handle, value) in fills {
                let bytes = stage_resident_fill_payload(&mut fill_payload, value, handle.byte_len)?;
                self.upload_resident(handle, bytes)?;
            }
            if !uploads.is_empty() {
                self.upload_resident_many(uploads)?;
            }
            for step in prefix_steps {
                self.dispatch_resident(step.program, step.handles, &step.config)?;
            }
            if repeat_count != 0 {
                for _ in 0..repeat_count {
                    for step in repeated_steps {
                        self.dispatch_resident(step.program, step.handles, &step.config)?;
                    }
                }
            }
            for ((&handle, readback), output) in read_handles
                .iter()
                .zip(readbacks.iter())
                .zip(outputs.iter_mut())
            {
                self.download_resident_range_into(
                    handle,
                    readback.device_offset,
                    readback.byte_len,
                    output,
                )?;
            }
            return Ok(());
        }

        struct ResolvedStep {
            func: cudarc::driver::sys::CUfunction,
            launch_ptrs: SmallVec<[u64; 8]>,
            params_ptr: u64,
        }

        let effective_fills = prepare_resident_sequence_fills(fills, uploads)?;
        let (upload_copies, uploaded_bytes) =
            self.prepare_resident_sequence_upload_copies(uploads)?;
        let effective_repeated_steps_len = if repeat_count == 0 {
            0
        } else {
            repeated_steps.len()
        };
        let prepared_step_capacity = checked_resident_dispatch_capacity_add(
            prefix_steps.len(),
            effective_repeated_steps_len,
            "prepared step",
        )?;
        let mut prepared_steps = SmallVec::<[PreparedStep<'_>; 8]>::new();
        reserve_smallvec(
            &mut prepared_steps,
            prepared_step_capacity,
            "resident sequence prepared steps",
        )?;
        let mut prefix_step_indices = SmallVec::<[usize; 16]>::new();
        reserve_smallvec(
            &mut prefix_step_indices,
            prefix_steps.len(),
            "resident sequence prefix step indices",
        )?;
        let mut repeated_step_indices = SmallVec::<[usize; 16]>::new();
        reserve_smallvec(
            &mut repeated_step_indices,
            effective_repeated_steps_len,
            "resident sequence repeated step indices",
        )?;
        let prefix_step_handle_count = prefix_steps.iter().try_fold(0usize, |total, step| {
            vyre_driver::accounting::checked_add_usize_lazy(
                total,
                step.handles.len(),
                || BackendError::InvalidProgram {
                    fix: "Fix: CUDA resident sequence handle capacity overflowed usize while counting prefix step handles; split the resident sequence."
                        .into(),
                },
            )
        })?;
        let repeated_step_handle_count = if repeat_count == 0 {
            0
        } else {
            repeated_steps.iter().try_fold(0usize, |total, step| {
                vyre_driver::accounting::checked_add_usize_lazy(
                    total,
                    step.handles.len(),
                    || BackendError::InvalidProgram {
                        fix: "Fix: CUDA resident sequence handle capacity overflowed usize while counting repeated step handles; split the resident sequence."
                            .into(),
                    },
                )
            })?
        };
        let step_handle_count = checked_resident_dispatch_capacity_add(
            prefix_step_handle_count,
            repeated_step_handle_count,
            "sequence handle",
        )?;
        let all_handle_capacity = checked_resident_dispatch_capacity_add(
            checked_resident_dispatch_capacity_add(
                checked_resident_dispatch_capacity_add(
                    fills.len(),
                    uploads.len(),
                    "sequence clear/upload handle",
                )?,
                step_handle_count,
                "sequence handle",
            )?,
            read_handles.len(),
            "sequence read-handle",
        )?;
        let mut all_handles = SmallVec::<[CudaResidentBuffer; 32]>::new();
        reserve_smallvec(
            &mut all_handles,
            all_handle_capacity,
            "resident sequence all handles",
        )?;
        all_handles.extend(fills.iter().map(|(handle, _)| *handle));
        all_handles.extend(uploads.iter().map(|(handle, _)| *handle));
        for step in prefix_steps {
            self.push_prepared_resident_sequence_step(
                step,
                &mut prepared_steps,
                &mut prefix_step_indices,
                &mut all_handles,
            )?;
        }
        if repeat_count != 0 {
            for step in repeated_steps {
                self.push_prepared_resident_sequence_step(
                    step,
                    &mut prepared_steps,
                    &mut repeated_step_indices,
                    &mut all_handles,
                )?;
            }
        }
        all_handles.extend(read_handles.iter().copied());

        self.warmup()?;
        let resident_use = self.resident_store.mark_inflight(&all_handles)?;
        let stream = self.launch_resources.acquire_stream()?;
        let mut allocations = SmallVec::<[DispatchAllocations; 8]>::new();
        reserve_smallvec(
            &mut allocations,
            prepared_steps.len(),
            "resident sequence dispatch allocations",
        )?;
        let mut host_transfers = SmallVec::<[HostTransferAllocations; 8]>::new();
        reserve_smallvec(
            &mut host_transfers,
            prepared_steps.len(),
            "resident sequence host transfers",
        )?;
        let mut sequence_param_cache = SmallVec::<[(SmallVec<[u32; 8]>, u64); 8]>::new();
        let mut upload_host_transfers = HostTransferAllocations::with_capacity(
            Arc::clone(&self.host_pool),
            upload_copies.len(),
            0,
        )?;
        let result = (|| {
            let mut sequence_view_cache = ResidentViewCache::new();
            reserve_smallvec(
                &mut sequence_view_cache,
                all_handle_capacity,
                "resident sequence view cache",
            )?;
            for &(handle, value) in &effective_fills {
                let buffer = self.resident_store.view_cached(
                    handle,
                    &mut sequence_view_cache,
                    "resident sequence view cache",
                )?;
                if buffer.byte_len != 0 {
                    // SAFETY: FFI to libcuda.so. Resident pointers are
                    // validated through resident_store.view and marked
                    // in-flight for this sequence before the stream work is
                    // submitted.
                    unsafe {
                        super::copy::memset_d8_async_checked(
                            buffer.ptr,
                            value,
                            buffer.byte_len,
                            stream.raw(),
                        )?;
                    }
                }
            }
            enqueue_resident_upload_copies_on_stream(
                &upload_copies,
                &mut upload_host_transfers,
                stream.raw(),
            )?;
            let mut resolved_steps = SmallVec::<[ResolvedStep; 8]>::new();
            reserve_smallvec(
                &mut resolved_steps,
                prepared_steps.len(),
                "resident sequence resolved steps",
            )?;
            for step in &prepared_steps {
                let launch_ptrs =
                    self.resolve_resident_sequence_launch_ptrs(step, &mut sequence_view_cache)?;
                let func = self.resolve_launch_function(
                    &step.ptx_src,
                    step.module_key,
                    &step.prepared.launch,
                    step.prepared.cooperative,
                )?;
                let mut step_allocations = DispatchAllocations::new(
                    step.program.buffers().len(),
                    Arc::clone(&self.transient_pool),
                )?;
                let param_bytes = launch_param_byte_len(
                    &step.prepared.launch.param_words,
                    "resident sequence dispatch",
                )?;
                let params_ptr = if param_bytes == 0 {
                    0
                } else if let Some((_, params_ptr)) =
                    sequence_param_cache.iter().find(|(words, _)| {
                        words.as_slice() == step.prepared.launch.param_words.as_slice()
                    })
                {
                    *params_ptr
                } else {
                    self.validate_transient_allocation_memory_budget(
                        param_bytes,
                        "CUDA resident sequence dispatch parameter bytes",
                        "CUDA resident sequence dispatch parameter upload",
                    )?;
                    let mut step_host_transfers =
                        HostTransferAllocations::with_capacity(Arc::clone(&self.host_pool), 1, 0)?;
                    let params_allocation = self.transient_pool.acquire(param_bytes)?;
                    self.telemetry.record_transient_allocation_bytes(
                        crate::numeric::CUDA_NUMERIC.usize_to_u64(
                            params_allocation.byte_len,
                            "resident sequence parameter allocation byte count",
                        )?,
                    );
                    let params_ptr = params_allocation.ptr;
                    let param_host_ptr =
                        step_host_transfers.push_u32_words(&step.prepared.launch.param_words)?;
                    enqueue_resident_h2d_copy(
                        params_ptr,
                        param_host_ptr,
                        param_bytes,
                        stream.raw(),
                    )?;
                    self.telemetry.record_host_to_device_bytes(
                        crate::numeric::CUDA_NUMERIC.usize_to_u64(
                            param_bytes,
                            "resident sequence parameter upload byte count",
                        )?,
                    );
                    self.telemetry.record_host_upload_operations(1);
                    self.telemetry.record_param_upload_bytes(
                        crate::numeric::CUDA_NUMERIC.usize_to_u64(
                            param_bytes,
                            "resident sequence parameter upload byte count",
                        )?,
                    );
                    step_allocations.set_params(params_allocation);
                    let mut cached_param_words = SmallVec::<[u32; 8]>::new();
                    reserve_smallvec(
                        &mut cached_param_words,
                        step.prepared.launch.param_words.len(),
                        "resident sequence cached parameter words",
                    )?;
                    cached_param_words.extend_from_slice(&step.prepared.launch.param_words);
                    sequence_param_cache.push((cached_param_words, params_ptr));
                    allocations.push(step_allocations);
                    host_transfers.push(step_host_transfers);
                    params_ptr
                };
                resolved_steps.push(ResolvedStep {
                    func,
                    launch_ptrs,
                    params_ptr,
                });
            }

            let mut kernel_args = SmallVec::<[*mut c_void; 8]>::new();
            let mut launch_resolved_step = |step_index: usize| -> Result<(), BackendError> {
                let step = &prepared_steps[step_index];
                let resolved = &mut resolved_steps[step_index];
                let mut params_ref = resolved.params_ptr;
                Self::kernel_args_into(
                    &mut resolved.launch_ptrs,
                    &mut params_ref,
                    &mut kernel_args,
                )?;
                for _ in 0..step.prepared.fixpoint_iterations {
                    self.launch_prevalidated_function(
                        resolved.func,
                        &mut kernel_args,
                        &step.prepared.launch,
                        stream.raw(),
                        false,
                        step.prepared.cooperative,
                    )?;
                }
                Ok(())
            };

            for &step_index in &prefix_step_indices {
                launch_resolved_step(step_index)?;
            }
            for _ in 0..repeat_count {
                for &step_index in &repeated_step_indices {
                    launch_resolved_step(step_index)?;
                }
            }
            let mut requested_readbacks = SmallVec::<[ResidentReadbackCopy; 8]>::new();
            reserve_smallvec(
                &mut requested_readbacks,
                read_handles.len(),
                "resident sequence requested readbacks",
            )?;
            for (handle, readback) in read_handles.iter().copied().zip(readbacks.iter()) {
                let buffer = self.resident_store.view_cached(
                    handle,
                    &mut sequence_view_cache,
                    "resident sequence view cache",
                )?;
                let end = vyre_driver::accounting::checked_usize_byte_range_end_lazy(
                    readback.device_offset,
                    readback.byte_len,
                    buffer.byte_len,
                    || {
                        BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident sequence compact readback for handle {} overflows usize at offset {} len {}.",
                            handle.id, readback.device_offset, readback.byte_len
                        ),
                    }
                    },
                    |end| {
                        BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident sequence compact readback for handle {} requested bytes [{}..{}) but buffer has {} bytes.",
                            handle.id, readback.device_offset, end, buffer.byte_len
                        ),
                    }
                    },
                )?;
                let src = if readback.byte_len == 0 {
                    0
                } else {
                    vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
                        buffer.ptr,
                        readback.device_offset,
                        || {
                            BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: CUDA resident sequence compact readback device offset {} does not fit CUdeviceptr arithmetic for handle {}.",
                                readback.device_offset, handle.id
                            ),
                        }
                        },
                        || {
                            BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: CUDA resident sequence compact readback pointer arithmetic overflowed for handle {} at offset {}.",
                                handle.id, readback.device_offset
                            ),
                        }
                        },
                    )?
                };
                let copy = ResidentReadbackCopy {
                    handle_id: handle.id,
                    src,
                    byte_len: readback.byte_len,
                };
                requested_readbacks.push(copy);
            }

            let fused_readbacks = fuse_resident_readback_copies(&requested_readbacks)?;

            let mut readback_host_transfers = HostTransferAllocations::with_capacity(
                Arc::clone(&self.host_pool),
                fused_readbacks.non_empty_copy_count,
                fused_readbacks.copies.len(),
            )?;
            for copy in &fused_readbacks.copies {
                let dst = readback_host_transfers.push_output(copy.byte_len)?;
                if copy.byte_len != 0 {
                    // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
                    unsafe {
                        super::copy::d2h_async_checked(dst, copy.src, copy.byte_len, stream.raw())?;
                    }
                }
            }
            stream.synchronize()?;
            self.telemetry.record_sync_point();
            if fused_readbacks.non_empty_copy_count == 0 {
                for output in outputs.iter_mut() {
                    output.clear();
                }
            } else {
                for (output, view) in outputs.iter_mut().zip(fused_readbacks.views.iter()) {
                    readback_host_transfers.collect_output_range_into(
                        view.copy_slot,
                        view.byte_offset,
                        view.byte_len,
                        *output,
                    )?;
                }
            }
            self.telemetry.record_host_to_device_bytes(uploaded_bytes);
            self.telemetry.record_host_upload_operations(
                crate::numeric::CUDA_NUMERIC
                    .usize_to_u64(upload_copies.len(), "resident host upload operation count")?,
            );
            self.telemetry
                .record_device_to_host_readback(fused_readbacks.bytes);
            self.telemetry.record_device_readback_operations(
                crate::numeric::CUDA_NUMERIC.usize_to_u64(
                    fused_readbacks.non_empty_copy_count,
                    "resident sequence readback operation count",
                )?,
            );
            Ok(())
        })();
        if result.is_err() {
            let _ = stream.synchronize();
            self.telemetry.record_sync_point();
        }
        self.launch_resources.release_stream(stream);
        drop(resident_use);
        result
    }

    /// Dispatch with CUDA-resident buffers and return ordered output readbacks.
    pub fn dispatch_resident_timed(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            let started = std::time::Instant::now();
            let enqueue_started = std::time::Instant::now();
            let outputs = self.dispatch_resident_via_borrowed(program, handles, config)?;
            let enqueue_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(enqueue_started, "resident-dispatch enqueue latency")?;
            let wait_started = std::time::Instant::now();
            let wait_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(wait_started, "resident-dispatch wait latency")?;
            let wall_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(started, "resident-dispatch wall latency")?;
            self.telemetry
                .record_timed_dispatch(wall_ns, None, Some(enqueue_ns), Some(wait_ns));
            return Ok(vyre_driver::TimedDispatchResult {
                outputs,
                wall_ns,
                device_ns: None,
                enqueue_ns: Some(enqueue_ns),
                wait_ns: Some(wait_ns),
            });
        }
        let started = std::time::Instant::now();
        let enqueue_started = std::time::Instant::now();
        let prepared = self.prepare_resident_dispatch(program, handles, config)?;
        let (ptx_src, ptx_source_key) = self.ptx_for_program_cached_with_key(program, config)?;
        let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
        let resident_dispatch = self.dispatch_resident_async_concrete_with_ptx_key(
            program, handles, config, &ptx_src, module_key, true, None, true, &prepared,
        )?;
        let enqueue_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(enqueue_started, "native-resident-dispatch enqueue latency")?;
        let wait_started = std::time::Instant::now();
        let (outputs, device_ns) = resident_dispatch.pending.await_timed_result()?;
        let wait_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(wait_started, "native-resident-dispatch wait latency")?;
        let wall_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(started, "native-resident-dispatch wall latency")?;
        self.telemetry
            .record_timed_dispatch(wall_ns, device_ns, Some(enqueue_ns), Some(wait_ns));
        Ok(vyre_driver::TimedDispatchResult {
            outputs,
            wall_ns,
            device_ns,
            enqueue_ns: Some(enqueue_ns),
            wait_ns: Some(wait_ns),
        })
    }

    pub(crate) fn dispatch_resident_outputs_with_ptx_key_into(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
        ptx_src: &str,
        module_key: ModuleCacheKey,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            return self.dispatch_resident_via_borrowed_into(program, handles, config, outputs);
        }
        let prepared = self.prepare_resident_dispatch(program, handles, config)?;
        let dispatch = self.dispatch_resident_async_concrete_with_ptx_key(
            program, handles, config, ptx_src, module_key, false, None, true, &prepared,
        )?;
        let (dispatch_outputs, _) = dispatch.pending.await_timed_result()?;
        vyre_driver::replace_output_buffers_preserving_slots(dispatch_outputs, outputs);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{prepare_resident_sequence_fills, stage_resident_fill_payload};
    use crate::backend::resident::CudaResidentBuffer;

    #[test]
    fn resident_fallback_fill_payload_preserves_last_good_bytes_when_reservation_fails() {
        let mut payload = vec![0xC3, 0xC3, 0x7E, 0x11];

        let result = stage_resident_fill_payload(&mut payload, 0x5A, usize::MAX);

        assert!(
            result.is_err(),
            "oversized CUDA resident fill payload must fail preflight instead of mutating staging"
        );
        assert_eq!(
            payload,
            vec![0xC3, 0xC3, 0x7E, 0x11],
            "failed CUDA resident fill staging must preserve the last diagnostic payload"
        );
    }
    use super::borrow_resident_sequence_output_slots;

    #[test]
    fn resident_fallback_fill_payload_reuses_capacity_and_overwrites_values() {
        let mut payload = Vec::new();
        {
            let bytes = stage_resident_fill_payload(&mut payload, 0xA5, 16)
                .expect("Fix: reusable resident fallback fill staging should reserve bytes");
            assert_eq!(bytes, &[0xA5; 16]);
        }
        let initial_capacity = payload.capacity();

        {
            let bytes = stage_resident_fill_payload(&mut payload, 0x5A, 8)
                .expect("Fix: smaller resident fallback fill staging should reuse capacity");
            assert_eq!(bytes, &[0x5A; 8]);
        }
        assert_eq!(
            payload.capacity(),
            initial_capacity,
            "CUDA resident fallback fill staging must reuse capacity across fills instead of allocating one Vec per fill"
        );

        {
            let bytes = stage_resident_fill_payload(&mut payload, 0x11, 0)
                .expect("Fix: zero-byte resident fallback fill staging should be valid");
            assert!(bytes.is_empty());
        }
        assert_eq!(
            payload.capacity(),
            initial_capacity,
            "zero-byte fallback fills must not release reusable staging capacity"
        );
    }

    #[test]
    fn resident_borrowed_fallback_does_not_allocate_vec_per_fill() {
        let source = include_str!("resident_dispatch.rs");
        assert!(
            source.contains("stage_resident_fill_payload(&mut fill_payload")
                && source.contains("let mut fill_payload = Vec::new();")
                && !source.contains(concat!("vec![value; ", "handle.byte_len]")),
            "Fix: CUDA resident borrowed fallback must stage fills through one reusable Vec, not allocate a fresh Vec per resident clear/fill."
        );
    }

    #[test]
    fn resident_h2d_enqueues_are_single_sourced_without_stealing_stream_order() {
        let source = include_str!("resident_dispatch.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("Fix: resident_dispatch production source must precede tests.");
        assert!(
            production.contains("fn enqueue_resident_h2d_copy")
                && production.contains("fn enqueue_optional_resident_h2d_copy")
                && production.contains("fn enqueue_resident_upload_copies_on_stream")
                && production
                    .matches(concat!("super::copy::", "h2d_async_checked"))
                    .count()
                    == 1,
            "Fix: resident dispatch parameter uploads, sequence uploads, and per-step parameter uploads must share one local H2D enqueue helper while preserving the caller-owned CUDA stream."
        );
        assert!(
            production.contains("enqueue_resident_upload_copies_on_stream(\n                &upload_copies")
                && production.contains("enqueue_resident_h2d_copy(\n                        params_ptr")
                && production.contains("param_host_ptr,\n                        param_bytes")
                && production.contains("stream.raw(),"),
            "Fix: resident sequence uploads and per-step parameter uploads must use the shared stream-preserving enqueue helpers."
        );
    }

    #[test]
    fn resident_sequence_fills_coalesce_duplicates_and_skip_full_upload_overwrites() {
        let first = CudaResidentBuffer {
            id: 1,
            byte_len: 16,
        };
        let second = CudaResidentBuffer {
            id: 2,
            byte_len: 16,
        };
        let upload = [0xFE_u8; 16];

        let effective = prepare_resident_sequence_fills(
            &[(first, 0x11), (second, 0x22), (first, 0x33)],
            &[(second, upload.as_slice())],
        )
        .expect("Fix: generated resident sequence fill coalescing must succeed.");

        assert_eq!(
            effective.as_slice(),
            &[(first, 0x33)],
            "Fix: resident sequence fills must keep the last duplicate fill and drop fills fully overwritten by same-sequence uploads."
        );
    }

    #[test]
    fn resident_sequence_fills_handle_dense_duplicate_streams_without_changing_order() {
        let handles: Vec<_> = (0..256)
            .map(|id| CudaResidentBuffer { id, byte_len: 1 })
            .collect();
        let mut fills = Vec::new();
        for round in 0..8_u8 {
            fills.extend(handles.iter().copied().map(|handle| (handle, round)));
        }

        let upload = [0xAA_u8];
        let uploads: Vec<_> = handles
            .iter()
            .copied()
            .filter(|handle| handle.id % 2 == 0)
            .map(|handle| (handle, upload.as_slice()))
            .collect();

        let effective = prepare_resident_sequence_fills(&fills, &uploads)
            .expect("Fix: dense CUDA resident fill coalescing must reserve bounded indices.");

        assert_eq!(
            effective.len(),
            128,
            "Fix: uploaded handles must suppress same-sequence fills even under dense duplicate traffic."
        );
        for (position, (handle, value)) in effective.iter().copied().enumerate() {
            assert_eq!(
                handle.id % 2,
                1,
                "Fix: uploaded resident handle {} must not retain a redundant fill.",
                handle.id
            );
            assert_eq!(
                handle.id as usize,
                position * 2 + 1,
                "Fix: first-seen fill order must be stable after duplicate coalescing."
            );
            assert_eq!(
                value, 7,
                "Fix: duplicate resident fills must keep the final value for each handle."
            );
        }
    }

    #[test]
    fn resident_full_readback_preparation_is_single_sourced() {
        let source = include_str!("resident_dispatch.rs");
        assert!(
            source.contains("fn prepare_full_resident_readbacks")
                && source
                    .matches(concat!("self.", "prepare_full_resident_readbacks(read_handles"))
                    .count()
                    == 2,
            "Fix: CUDA resident full-handle readback preparation must be shared by read_many and fill_read_many paths."
        );
    }

    #[test]
    fn resident_sequence_output_slot_borrowing_is_single_sourced_and_reuses_slots() {
        let source = include_str!("resident_dispatch.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("Fix: resident_dispatch production source must precede tests.");
        assert!(
            production.contains("fn borrow_resident_sequence_output_slots")
                && production
                    .matches("borrow_resident_sequence_output_slots(outputs,")
                    .count()
                    == 2,
            "Fix: CUDA resident sequence read_many and fill_read_many must share output-slot borrowing."
        );

        let mut outputs = vec![vec![1, 2, 3], Vec::new(), vec![4]];
        let initial_first_capacity = outputs[0].capacity();
        {
            let borrowed = borrow_resident_sequence_output_slots(&mut outputs, 2)
                .expect("Fix: output-slot borrowing should resize existing slots.");
            assert_eq!(borrowed.len(), 2);
        }
        assert_eq!(outputs.len(), 2);
        assert!(
            outputs[0].capacity() >= initial_first_capacity,
            "Fix: resizing borrowed resident output slots must preserve existing slot allocation."
        );
    }

    #[test]
    fn resident_sequence_resolves_views_once_per_sequence() {
        let source = include_str!("resident_dispatch.rs");
        let production = source
            .split("#[cfg(test)]")
            .next()
            .expect("Fix: resident_dispatch production source must precede tests.");

        assert!(
            production.contains("fn resolve_resident_sequence_launch_ptrs")
                && production
                    .matches("resolve_resident_sequence_launch_ptrs(step,")
                    .count()
                    == 1,
            "Fix: CUDA resident sequence launch-pointer validation must be single-sourced."
        );
        assert!(
            production.contains("let mut sequence_view_cache = ResidentViewCache::new();")
                && production.contains("resident sequence view cache")
                && !production.contains("resident sequence fill view cache")
                && !production.contains("resident sequence step view cache")
                && !production.contains("resident sequence readback view cache")
                && !production.contains("struct ClearCopy"),
            "Fix: CUDA resident sequence dispatch must use one sequence-wide resident view cache instead of rebuilding fill, step, and readback caches."
        );
    }
}
