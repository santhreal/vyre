use std::ffi::c_void;
use std::sync::Arc;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use vyre_driver::BindingRole;
use vyre_driver::{BackendError, DispatchConfig, ResidentHandle};
use vyre_foundation::ir::Program;

use crate::backend::allocations::{DispatchAllocations, HostTransferAllocations};
use crate::backend::dispatch::CudaBackend;
use crate::backend::enqueue_cleanup::EnqueueGuards;
use crate::backend::launch_params::launch_param_byte_len;
use crate::backend::module_cache::ModuleCacheKey;
use crate::backend::ordering::sort_unstable_by_key_if_needed;
use crate::backend::output_range::{cuda_output_readback_for_binding, CudaOutputReadback};
use crate::backend::plan::CudaDispatchPlan;
use crate::backend::resident::{CudaResidentBuffer, ResidentViewCache};
use crate::backend::resident_dispatch::dense_index_validation::validate_dense_resident_output_indices;
use crate::backend::resident_dispatch::descriptor_cursor::{
    next_resident_handle, resident_required_handles,
};
use crate::backend::resident_dispatch::host_uploads::enqueue_optional_resident_h2d_copy;
use crate::backend::resident_dispatch_support::{
    checked_resident_dispatch_capacity_mul, CudaResidentBatchDispatch,
};
use crate::backend::staging_reserve::{reserve_hash_set, reserve_smallvec};

impl CudaBackend {
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
        let (params_ptr, param_upload) = self.resolve_resident_params_ptr(
            &prepared.launch.param_words,
            param_bytes,
            static_params_ptr,
            "resident batch dispatch",
            &mut allocations,
            &mut host_transfers,
        )?;

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
        let mut seen_output_handles_small = SmallVec::<[ResidentHandle; 8]>::new();
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
                let handle =
                    next_resident_handle(handles, &mut next_handle, "resident batch dispatch")?;
                let resident = self.resident_store.view_cached(
                    handle,
                    &mut resident_view_cache,
                    "resident batch dispatch view cache",
                )?;
                resident.validate_binding(
                    &format!("resident batch dispatch item {batch_index}"),
                    &binding.name,
                    binding.static_byte_len,
                    handle.handle,
                )?;
                launch_ptrs.push(resident.ptr);
                if let Some(output_index) = binding.output_index {
                    let full_byte_len = match binding.static_byte_len {
                        Some(len) => len,
                        None => resident.byte_len,
                    };
                    let readback = cuda_output_readback_for_binding(
                        program.buffers(),
                        binding.buffer_index,
                        &binding.name,
                        full_byte_len,
                        "resident batch output readback",
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
            validate_dense_resident_output_indices(
                output_handles_by_index
                    .iter()
                    .map(|(output_index, _, _)| *output_index),
                prepared.output_binding_indices.len(),
                "resident batch output handles",
            )?;
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
                        if !seen_output_handles.insert(handle.handle) {
                            return Err(BackendError::InvalidProgram {
                                fix: format!(
                                    "Fix: CUDA resident batch dispatch cannot reuse output handle {} across submitted items; allocate one output resident buffer tuple per in-flight batch item so batched readback observes every result instead of the final overwrite.",
                                    handle.handle
                                ),
                            });
                        }
                    }
                } else {
                    if seen_output_handles_small.contains(&handle.handle) {
                        return Err(BackendError::InvalidProgram {
                            fix: format!(
                                "Fix: CUDA resident batch dispatch cannot reuse output handle {} across submitted items; allocate one output resident buffer tuple per in-flight batch item so batched readback observes every result instead of the final overwrite.",
                                handle.handle
                            ),
                        });
                    }
                    seen_output_handles_small.push(handle.handle);
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

        // Marked in-flight before the launch lease is taken: the lease blocks
        // while another launch holds it, and a handle must already be pinned
        // before this dispatch can wait behind one that reads it.
        let resident_use = self.resident_store.mark_inflight(&all_handles)?;
        let mut guards = EnqueueGuards::new(
            "resident batch",
            crate::stream::CudaLaunchResourceLease::acquire(
                Arc::clone(&self.launch_resources),
                false,
            )?,
            allocations,
            host_transfers,
            Some(resident_use),
        );
        let stream_raw = guards.stream_raw()?;
        let pending = (|| {
            enqueue_optional_resident_h2d_copy(param_upload, stream_raw)?;

            // One lease covers every batch element: they all enqueue on this one
            // stream, so stream order already separates their barrier counts.
            let grid_barrier = self.lease_grid_barrier(program, prepared, ptx_src, module_key)?;
            let mut kernel_args = SmallVec::<[*mut c_void; 8]>::new();
            // `launch_then_release` runs the launches and ends the lease in the
            // one safe order: the release synchronizes the stream before freeing
            // the gate, so a launch failure cannot leave a grid spinning while the
            // next sequence resets the counter underneath it.
            grid_barrier.launch_then_release(
                stream_raw,
                "resident batch grid-sync launch",
                |grid_barrier| {
                    for launch_ptrs in launch_ptrs_by_batch.iter_mut() {
                        let mut params_ref = params_ptr;
                        Self::kernel_args_into(launch_ptrs, &mut params_ref, &mut kernel_args)?;
                        self.replay_fixpoint_launches(
                            grid_barrier,
                            func,
                            &mut kernel_args,
                            prepared,
                            stream_raw,
                        )?;
                    }
                    Ok(())
                },
            )?;

            let event = self.launch_resources.acquire_event()?;
            if let Err(error) = event.record(stream_raw) {
                self.launch_resources.release_event(event);
                return Err(error);
            }
            let (stream, _) = guards.take_stream_and_timing()?;
            let allocations = guards.take_allocations()?;
            let resident_use = guards.take_resident_use()?;
            let host_transfers = guards.take_host_transfers()?;
            Ok(
                crate::stream::CudaPendingDispatch::new_resident_batch_pending(
                    Arc::clone(&self.ctx),
                    Arc::clone(&self.launch_resources),
                    event,
                    stream,
                    allocations,
                    resident_use,
                    host_transfers,
                    Arc::clone(&self.telemetry),
                ),
            )
        })();
        let pending = match pending {
            Ok(pending) => pending,
            Err(error) => {
                return Err(guards.abandon(
                    error,
                    &self.telemetry,
                    stream_raw,
                    "cuStreamSynchronize (resident batch error cleanup)",
                    "enqueue",
                ));
            }
        };
        Ok(CudaResidentBatchDispatch {
            pending,
            output_handles: output_handles_by_batch,
            output_readbacks: output_readbacks_by_batch,
        })
    }
}
