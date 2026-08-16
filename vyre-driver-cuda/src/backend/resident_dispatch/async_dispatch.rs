use std::ffi::c_void;
use std::sync::Arc;

use smallvec::SmallVec;
use vyre_driver::BindingRole;
use vyre_driver::{BackendError, DispatchConfig, PendingDispatch};
use vyre_foundation::ir::Program;

use crate::backend::allocations::{DispatchAllocations, HostTransferAllocations};
use crate::backend::copy::aligned_async_copy_len;
use crate::backend::dispatch::CudaBackend;
use crate::backend::dispatch_phase_probe as probe;
use crate::backend::enqueue_cleanup::EnqueueGuards;
use crate::backend::launch_params::launch_param_byte_len;
use crate::backend::module_cache::ModuleCacheKey;
use crate::backend::ordering::sort_unstable_by_key_if_needed;
use crate::backend::output_range::{cuda_output_readback_for_binding, CudaOutputReadback};
use crate::backend::plan::CudaDispatchPlan;
use crate::backend::resident::{
    resident_bindings_from_handles, CudaDispatchBinding, CudaResidentBuffer, ResidentViewCache,
};
use crate::backend::resident_dispatch::dense_index_validation::validate_dense_resident_output_indices;
use crate::backend::resident_dispatch::descriptor_cursor::{
    next_dispatch_binding, resident_required_handles,
};
use crate::backend::resident_dispatch::host_uploads::{
    enqueue_optional_resident_h2d_copy, enqueue_resident_h2d_copy,
};
use crate::backend::resident_dispatch_accounting::{
    add_resident_dispatch_bytes, add_resident_dispatch_u64_count, CudaResidentDispatch,
};
use crate::backend::staging_reserve::{reserve_smallvec, reserved_vec};
use crate::numeric::CUDA_NUMERIC;

pub(super) fn resident_output_clear_for_readback(
    base_ptr: u64,
    readback: CudaOutputReadback,
    binding_name: &str,
) -> Result<Option<(u64, usize)>, BackendError> {
    if readback.byte_len == 0 {
        return Ok(None);
    }
    let clear_ptr = vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
        base_ptr,
        readback.device_offset,
        || {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident output clear offset {} for binding `{binding_name}` does not fit CUdeviceptr arithmetic.",
                readback.device_offset
            ),
        }
        },
        || {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident output clear pointer for binding `{binding_name}` overflowed at offset {}.",
                readback.device_offset
            ),
        }
        },
    )?;
    Ok(Some((clear_ptr, readback.byte_len)))
}

impl CudaBackend {
    /// Dispatch a Program asynchronously using caller-provided CUDA-resident buffers.
    pub fn dispatch_resident_async(
        &self,
        program: &Program,
        handles: &[CudaResidentBuffer],
        config: &DispatchConfig,
    ) -> Result<Box<dyn PendingDispatch>, BackendError> {
        self.dispatch_bindings_async(program, &resident_bindings_from_handles(handles)?, config)
    }

    /// Dispatch a Program asynchronously against a mixed binding list.
    ///
    /// Residency is per binding: resident entries bind their existing device
    /// memory, borrowed entries are staged into the transient pool for this
    /// dispatch alone.
    pub(crate) fn dispatch_bindings_async(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
        config: &DispatchConfig,
    ) -> Result<Box<dyn PendingDispatch>, BackendError> {
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            let outputs = self.dispatch_resident_via_borrowed(program, bindings, config)?;
            return Ok(Box::new(crate::stream::CudaPendingDispatch::new_ready(
                Arc::clone(&self.ctx),
                Arc::clone(&self.launch_resources),
                outputs,
                Arc::clone(&self.telemetry),
            )));
        }
        let prepared = self.prepare_resident_dispatch(program, bindings, config)?;
        let (ptx_src, ptx_source_key) = self.ptx_for_program_cached_with_key(program, config)?;
        let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
        let native = self.dispatch_resident_async_concrete_with_ptx_key(
            program, bindings, config, &ptx_src, module_key, true, None, true, &prepared,
        )?;
        Ok(Box::new(native.pending))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch_resident_async_concrete_with_ptx_key(
        &self,
        program: &Program,
        bindings: &[CudaDispatchBinding<'_>],
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
                "[cuda-trace] resident dispatch start buffers={} bindings={}",
                program.buffers().len(),
                bindings.len()
            );
        }
        self.warmup()?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident warmup",
                start.elapsed().as_millis()
            );
        }
        let required_bindings = resident_required_handles(prepared)?;
        if bindings.len() != required_bindings {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident dispatch expected {required_bindings} bound resource(s) but received {}.",
                    bindings.len()
                ),
            });
        }
        // Borrowed bindings are staged out of the transient pool, so they have
        // to clear the same live-device budget a fully borrowed dispatch does.
        // An all-resident dispatch stages nothing and skips the preflight
        // entirely, so the existing hot path gains no driver calls.
        self.validate_mixed_dispatch_staging_budget(
            prepared,
            bindings,
            "CUDA mixed resident dispatch",
        )?;
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
        let mut next_binding = 0usize;
        let mut output_handles_by_index =
            SmallVec::<[(usize, Option<CudaResidentBuffer>, CudaOutputReadback, u64); 8]>::new();
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
            bindings.len(),
            "resident dispatch view cache",
        )?;
        let mut resident_handles = SmallVec::<[CudaResidentBuffer; 8]>::new();
        reserve_smallvec(
            &mut resident_handles,
            bindings.len(),
            "resident dispatch in-flight handles",
        )?;
        let mut borrowed_stages = SmallVec::<[(u64, &[u8]); 8]>::new();
        reserve_smallvec(
            &mut borrowed_stages,
            bindings.len(),
            "resident dispatch borrowed staging",
        )?;
        for binding in &prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let source =
                next_dispatch_binding(bindings, &mut next_binding, "resident dispatch launch")?;
            let (launch_ptr, bound_byte_len, resident_handle) = match source {
                CudaDispatchBinding::Resident(handle) => {
                    let resident = self.resident_store.view_cached(
                        handle,
                        &mut resident_view_cache,
                        "resident dispatch view cache",
                    )?;
                    resident.validate_binding(
                        "resident dispatch",
                        &binding.name,
                        binding.static_byte_len,
                        handle.handle,
                    )?;
                    resident_handles.push(handle);
                    (resident.ptr, resident.byte_len, Some(handle))
                }
                CudaDispatchBinding::Borrowed(bytes) => {
                    // An output-only binding carries no host payload to stage,
                    // so it is sized from the declared output extent exactly
                    // like the fully borrowed dispatch sizes it.
                    let staged_byte_len = match binding.input_index {
                        Some(_) => bytes.len(),
                        None => binding.static_byte_len.ok_or_else(|| {
                            BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: CUDA borrowed output `{}` needs a static byte length before launch; set BufferDecl::with_count or output_byte_range, or bind a resident buffer for that output.",
                                binding.name
                            ),
                        }
                        })?,
                    };
                    if let Some(expected) = binding.static_byte_len {
                        if staged_byte_len < expected {
                            return Err(BackendError::InvalidProgram {
                                fix: format!(
                                    "Fix: CUDA borrowed binding `{}` expected at least {expected} bytes but the supplied buffer has {staged_byte_len} bytes.",
                                    binding.name
                                ),
                            });
                        }
                    }
                    let allocation = self
                        .transient_pool
                        .acquire(aligned_async_copy_len(staged_byte_len)?)?;
                    self.telemetry
                        .record_transient_allocation_bytes(CUDA_NUMERIC.usize_to_u64(
                            allocation.byte_len,
                            "resident dispatch borrowed staging byte count",
                        )?);
                    let staged_ptr = allocation.ptr;
                    allocations.set_ptr(binding.buffer_index, allocation, &binding.name)?;
                    if binding.input_index.is_some() && !bytes.is_empty() {
                        borrowed_stages.push((staged_ptr, bytes));
                    }
                    (staged_ptr, staged_byte_len, None)
                }
            };
            launch_ptrs.push(launch_ptr);
            if let Some(output_index) = binding.output_index {
                let full_byte_len = match binding.static_byte_len {
                    Some(len) => len,
                    None => bound_byte_len,
                };
                let readback = cuda_output_readback_for_binding(
                    program.buffers(),
                    binding.buffer_index,
                    &binding.name,
                    full_byte_len,
                    "resident async output readback",
                )?;
                output_handles_by_index.push((output_index, resident_handle, readback, launch_ptr));
                if binding.input_index.is_none() {
                    output_clears.extend(resident_output_clear_for_readback(
                        launch_ptr,
                        readback,
                        &binding.name,
                    )?);
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
        validate_dense_resident_output_indices(
            output_handles_by_index
                .iter()
                .map(|(output_index, _, _, _)| *output_index),
            prepared.output_binding_indices.len(),
            "resident dispatch output handles",
        )?;
        let mut output_handles = SmallVec::<[Option<CudaResidentBuffer>; 8]>::new();
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
        let param_transfer_slots = usize::from(static_params_ptr.is_none() && param_bytes != 0);
        let transfer_capacity = borrowed_stages
            .len()
            .checked_add(param_transfer_slots)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: CUDA resident dispatch host staging slot count overflowed usize; shard the dispatch before launch.".to_string(),
            })?;
        let mut host_transfers = HostTransferAllocations::with_capacity(
            Arc::clone(&self.host_pool),
            transfer_capacity,
            output_stage_readbacks.len(),
        )?;
        let mut borrowed_upload_copies = SmallVec::<[(u64, *const c_void, usize); 8]>::new();
        reserve_smallvec(
            &mut borrowed_upload_copies,
            borrowed_stages.len(),
            "resident dispatch borrowed upload copies",
        )?;
        let mut borrowed_upload_bytes = 0_u64;
        let mut borrowed_upload_ops = 0_u64;
        for &(staged_ptr, bytes) in &borrowed_stages {
            let copy_byte_len = aligned_async_copy_len(bytes.len())?;
            let host_ptr = host_transfers.push_upload_padded(bytes, copy_byte_len)?;
            add_resident_dispatch_bytes(
                &mut borrowed_upload_bytes,
                bytes.len(),
                "resident dispatch borrowed upload",
            )?;
            add_resident_dispatch_u64_count(
                &mut borrowed_upload_ops,
                "resident dispatch borrowed upload operation",
            )?;
            borrowed_upload_copies.push((staged_ptr, host_ptr, copy_byte_len));
        }
        let (params_ptr, param_upload) = self.resolve_resident_params_ptr(
            &prepared.launch.param_words,
            param_bytes,
            static_params_ptr,
            "resident dispatch",
            &mut allocations,
            &mut host_transfers,
        )?;
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

        // Marked in-flight before the launch lease is taken: the lease blocks
        // while another launch holds it, and a handle must already be pinned
        // before this dispatch can wait behind one that reads it.
        let resident_use = self.resident_store.mark_inflight(&resident_handles)?;
        let mut guards = EnqueueGuards::new(
            "resident dispatch",
            crate::stream::CudaLaunchResourceLease::acquire(
                Arc::clone(&self.launch_resources),
                capture_timing,
            )?,
            allocations,
            host_transfers,
            Some(resident_use),
        );
        let stream_raw = guards.stream_raw()?;
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident allocations/stream",
                start.elapsed().as_millis()
            );
        }
        // Completion and timing events are transferred to the pending handle.
        // Do not allocate a second probe-only pair here: reading or recycling
        // it would require the host fence this asynchronous path removes.
        let enqueue_result = (|| {
            enqueue_optional_resident_h2d_copy(param_upload, stream_raw)?;
            for &(dst_ptr, host_ptr, byte_len) in &borrowed_upload_copies {
                enqueue_resident_h2d_copy(dst_ptr, host_ptr, byte_len, stream_raw)?;
            }
            if borrowed_upload_ops != 0 {
                self.telemetry
                    .record_host_to_device_bytes(borrowed_upload_bytes);
                self.telemetry
                    .record_host_upload_operations(borrowed_upload_ops);
            }
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
                    crate::backend::copy::memset_d8_async_checked(
                        dst_ptr, 0, byte_len, stream_raw,
                    )?;
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

            probe::charge_since(probe::Phase::Stage, start);
            if let Some((start_event, _)) = guards.timing_events()? {
                start_event.record(stream_raw)?;
            }
            // Fixpoint loop  -  see dispatch_borrowed_async_with_ptx_concrete
            // for the contract. Resolve the CUDA function and argument vector
            // once; fixpoint iterations are kernel replays, not relowering or
            // module-cache lookups.
            let resolve_started = probe::mark();
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
            probe::charge(probe::Phase::Resolve, resolve_started);
            // Hold this module's module-scope globals across the launch sequence.
            // The lease blocks while another launch of the same module is still in
            // flight; see `ModuleGlobalsGate`.
            let lease_started = probe::mark();
            let module_globals =
                self.lease_module_globals(program, prepared, ptx_src, module_key)?;
            probe::charge(probe::Phase::Lease, lease_started);
            // `launch_then_defer_release` runs the launches and hands the lease
            // back instead of ending it. The module-scope globals are live for the
            // kernel's whole EXECUTION, and this path returns before that, so
            // ending the lease here would synchronize the stream: the one thing an
            // asynchronous submission must not do. The pending handle owns the
            // completion event, which makes it the only place that can prove the
            // kernel finished, so it is what ends the lease.
            //
            // Everything that must still be enqueued under the lease goes INSIDE
            // the closure. A failure there releases at enqueue, with the
            // synchronize, because no pending handle will exist to await it.
            let launch_and_release_started = probe::mark();
            let (_, deferred_module_globals) = module_globals.launch_then_defer_release(
                stream_raw,
                "resident async dispatch launch",
                |module_globals| {
                    probe::measure(probe::Phase::LaunchLoop, || {
                        self.replay_fixpoint_launches(
                            module_globals,
                            func,
                            &mut kernel_args,
                            prepared,
                            stream_raw,
                        )
                    })?;
                    if let Some((_, end_event)) = guards.timing_events()? {
                        end_event.record(stream_raw)?;
                    }
                    Ok(())
                },
            )?;
            // On this path the span covers the trap reset and the timing-event
            // record. The release is paid by whoever awaits the pending handle.
            probe::charge_remainder(
                probe::Phase::Release,
                launch_and_release_started,
                probe::Phase::LaunchLoop,
            );
            // Output copies are enqueued on the same stream after the kernel.
            // The completion event recorded below fences uploads, compute, and
            // D2H transfer without blocking this submission call.
            Ok(deferred_module_globals)
        })();
        // Every fallible step that runs under the lease is inside the closure
        // above, so an error here means the lease was already ended: either it was
        // never taken, or `launch_then_defer_release` released it with the
        // synchronize a failed launch needs.
        let deferred_module_globals = match enqueue_result {
            Ok(lease) => lease,
            Err(error) => {
                return Err(guards.abandon(
                    error,
                    &self.telemetry,
                    stream_raw,
                    "cuStreamSynchronize (resident async error cleanup)",
                    "enqueue",
                ));
            }
        };

        let pending = (|| {
            let mut staged_readback_bytes = 0_u64;
            let mut staged_readback_ops = 0_u64;
            {
                let transfers = guards.recording()?.host_transfers;
                for &(src_base_ptr, readback) in &output_stage_readbacks {
                    let dst = transfers.push_output(readback.byte_len)?;
                    if readback.byte_len == 0 {
                        continue;
                    }
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
                    // SAFETY: the source is a validated resident or transient
                    // output and the pinned destination remains owned by the
                    // pending dispatch until its completion event retires.
                    unsafe {
                        crate::backend::copy::d2h_async_checked_with_label(
                            dst,
                            src_ptr,
                            readback.byte_len,
                            stream_raw,
                            "cuMemcpyDtoHAsync_v2 (resident staged output)",
                        )?;
                    }
                }
            }
            self.telemetry
                .record_device_to_host_readback(staged_readback_bytes);
            self.telemetry
                .record_device_readback_operations(staged_readback_ops);

            let outputs = reserved_vec(output_stage_readbacks.len(), "resident staged output")?;
            let event = self.launch_resources.acquire_event()?;
            if let Err(error) = event.record(stream_raw) {
                self.launch_resources.release_event(event);
                return Err(error);
            }
            let (stream, timing_events) = guards.take_stream_and_timing()?;
            let allocations = guards.take_allocations()?;
            let resident_use = guards.take_resident_use()?;
            let host_transfers = guards.take_host_transfers()?;
            let pending = match timing_events {
                Some((timing_start, timing_end)) => {
                    crate::stream::CudaPendingDispatch::new_with_timing(
                        Arc::clone(&self.ctx),
                        Arc::clone(&self.launch_resources),
                        event,
                        stream,
                        allocations,
                        Some(resident_use),
                        Some(host_transfers),
                        outputs,
                        timing_start,
                        timing_end,
                        Arc::clone(&self.telemetry),
                    )
                }
                None => crate::stream::CudaPendingDispatch::new(
                    Arc::clone(&self.ctx),
                    Arc::clone(&self.launch_resources),
                    event,
                    stream,
                    allocations,
                    Some(resident_use),
                    Some(host_transfers),
                    outputs,
                    Arc::clone(&self.telemetry),
                ),
            };
            Ok(pending)
        })();
        let pending = match pending {
            Ok(pending) => pending.holding_module_globals(deferred_module_globals),
            Err(error) => {
                let abandoned = guards.abandon(
                    error,
                    &self.telemetry,
                    stream_raw,
                    "cuStreamSynchronize (resident async output enqueue cleanup)",
                    "output enqueue",
                );
                // Dropped AFTER the abandon synchronized the stream, so the gate is
                // freed only once the grid cannot still be running. The trap record
                // is not read: this dispatch produced no answer, and the failure
                // worth reporting is the one that stopped the output enqueue.
                drop(deferred_module_globals);
                return Err(abandoned);
            }
        };
        if trace {
            tracing::debug!(
                "[cuda-trace] +{}ms resident asynchronous submission complete",
                start.elapsed().as_millis()
            );
        }
        Ok(CudaResidentDispatch {
            pending,
            output_handles,
            output_readbacks,
        })
    }
}
