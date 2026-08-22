#![allow(unsafe_code)]
//! cudaGraph capture-and-replay path for repeat-shape Programs.
//!
//! Op id: `vyre-driver-cuda::cuda_graph`. Soundness: `Exact` over the
//! captured launch sequence. Cost-direction: read-only at the wire layer
//! (does not mutate Program); host-side dispatch overhead is amortized by
//! replacing repeated launch construction with a cached `CUgraphExec`.
//!
//! ## Why
//!
//! Latency-bound kernels can spend more time in host launch setup than in
//! device execution. cudaGraph captures the full launch sequence (memcpy +
//! kernel launch + readback) into a graph object once; subsequent dispatches
//! replay the cached executable graph with `cuGraphLaunch`.
//!
//! ## Constraints
//!
//! - **No allocation during capture.** `cuMemAlloc_v2` returns
//!   `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` while a stream is in capture
//!   mode. `record_cuda_graph` allocates ALL device buffers BEFORE
//!   `cuStreamBeginCapture_v2` and stores them in `CachedCudaGraph`.
//! - **Host pointers must persist.** The captured `cuMemcpyHtoDAsync_v2`
//!   records the host source pointer; the cached graph reuses the SAME
//!   pointer on every replay. `CachedCudaGraph` owns the input host buffers
//!   so callers can write new bytes into them without changing the address.
//! - **Shape-bound.** A cached graph captures one specific input/output
//!   byte layout. Calling `dispatch_via_cuda_graph` with mismatched input
//!   sizes returns `BackendError::InvalidProgram`  -  the caller must record
//!   a fresh graph for each shape.
//!
//! ## Lifecycle
//!
//! ```text
//! CachedCudaGraph::record  ──► CUgraph ──► CUgraphExec ──► live
//!                               │
//!                               ▼
//!                        owns input/output device pointers
//!                        owns input/output host buffers
//!                        owns dedicated CUstream
//!                        owns CUfunction (via module_cache)
//!                               │
//! CachedCudaGraph::drop ──► cuGraphExecDestroy ──► cuGraphDestroy
//!                       ──► cuStreamDestroy_v2
//!                       ──► cuMemFree_v2 for each device buffer

use std::sync::Arc;

use smallvec::SmallVec;
use vyre_driver::graph_capture::plan_graph_capture_bindings;
use vyre_driver::input_identity::exact_input_key;
use vyre_driver::BindingRole;
use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

use super::allocations::alloc_cuda_ptr;
use super::cuda_graph_lifecycle::*;
use super::dispatch::CudaBackend;
use super::output_range::cuda_output_readback_for_binding;
use super::pinned_allocations::HostTransferAllocations;
use super::staging_reserve::reserve_smallvec;
use crate::backend::copy::aligned_async_copy_len;
use crate::numeric::CUDA_NUMERIC;

pub use super::cuda_graph_lifecycle::CachedCudaGraph;

fn cuda_graph_usize_to_u64(value: usize, label: &'static str) -> Result<u64, BackendError> {
    CUDA_NUMERIC.usize_to_u64(value, label)
}

super::define_required_input!(
    cuda_graph_sample_input,
    "CUDA graph capture",
    "sample input",
    "Rebuild the binding plan or validate graph sample inputs before recording."
);

impl CudaBackend {
    /// Record one full Program dispatch into a CUDA graph for fast replay.
    ///
    /// Allocates all device + host buffers, captures the dispatch sequence
    /// (HtoD memcpy → kernel launch → DtoH memcpy), and instantiates the
    /// captured graph. The returned `CachedCudaGraph` is a handle the
    /// caller drives via `dispatch_via_cuda_graph`.
    ///
    /// `sample_inputs` is used only to determine the input byte-layout
    /// shape captured into the graph; the caller passes the actual
    /// per-dispatch bytes via `dispatch_via_cuda_graph`. The bytes in
    /// `sample_inputs` are also copied into the cached host buffers as the
    /// initial state.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when device allocation fails, the kernel
    /// cannot be compiled or loaded, or the CUDA driver rejects any of the
    /// graph capture / instantiate operations.
    pub fn record_cuda_graph(
        &self,
        program: &Program,
        sample_inputs: &[Vec<u8>],
        config: &DispatchConfig,
    ) -> Result<CachedCudaGraph, BackendError> {
        let mut sample_refs = SmallVec::<[&[u8]; 8]>::new();
        reserve_smallvec(
            &mut sample_refs,
            sample_inputs.len(),
            "cuda graph borrowed sample input references",
        )?;
        for input in sample_inputs {
            sample_refs.push(input.as_slice());
        }
        self.record_cuda_graph_borrowed(program, &sample_refs, config)
    }

    /// Record one full Program dispatch into a CUDA graph using borrowed
    /// sample inputs.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when device allocation fails, the kernel
    /// cannot be compiled or loaded, or the CUDA driver rejects graph capture.
    pub fn record_cuda_graph_borrowed(
        &self,
        program: &Program,
        sample_inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<CachedCudaGraph, BackendError> {
        if config.cooperative || vyre_driver::grid_sync::contains_grid_sync(program) {
            return Err(BackendError::UnsupportedFeature {
                name: "cuda_graph_cooperative_capture (regular CUDA graph capture records cuLaunchKernel, not cuLaunchCooperativeKernel, so a grid-sync kernel would replay non-cooperatively and without the per-launch _vyre_grid_barrier reset)"
                    .to_string(),
                backend: crate::CUDA_BACKEND_ID.to_string(),
            });
        }
        let _capture_serial = self.graph_capture_lock.lock().map_err(|_| {
            BackendError::DispatchFailed {
                code: None,
                message: "cuda graph capture lock poisoned. Fix: recreate CudaBackend after a panic during graph recording.".to_string(),
            }
        })?;
        self.warmup()?;

        // Compile + prepare. This lifts the program into PTX, computes the
        // binding plan, validates the program. All allocations / launches
        // below assume this succeeded.
        let prepared = self.prepare_host_dispatch(program, sample_inputs, config)?;
        let (ptx_src, ptx_source_key) = self.ptx_for_program_cached_with_key(program, config)?;
        let module_key = self.module_cache_key_for_ptx_source_key(ptx_source_key)?;
        let func = self.resolve_launch_function(&ptx_src, module_key, &prepared.launch, false)?;
        // A trap-declaring module cannot be captured. The trap record is read back
        // after a stream synchronize, and a synchronize is not permitted during
        // capture; a captured graph would replay the launches with no readback at
        // all, so a trapping replay would report success and hand back whatever the
        // lanes wrote before the guard fired. Refuse for the same reason
        // cooperative capture is refused: the recorded sequence would be missing a
        // step that the launch path treats as mandatory.
        if self
            .module_globals_with_key(&ptx_src, module_key)?
            .trap
            .is_some()
        {
            return Err(BackendError::UnsupportedFeature {
                name: "cuda_graph_trap_capture (the trap record is read back after a stream synchronize, which graph capture forbids, so a captured replay could not report a trap)"
                    .to_string(),
                backend: crate::CUDA_BACKEND_ID.to_string(),
            });
        }
        self.validate_transient_dispatch_memory_budget(
            &prepared,
            sample_inputs,
            "CUDA graph capture",
        )?;

        // Allocate all device buffers BEFORE capture. cuMemAlloc returns
        // CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED inside capture; allocating
        // up front is the only way to make capture work.
        let capture_binding_plan = plan_graph_capture_bindings(&prepared.bindings)?;
        let input_capacity = capture_binding_plan.input_device_capacity;
        let output_device_capacity = capture_binding_plan.output_device_capacity;
        let output_readback_capacity = capture_binding_plan.output_readback_capacity;
        let mut input_device_ptrs = SmallVec::<[DevicePtrGuard; 8]>::new();
        reserve_smallvec(
            &mut input_device_ptrs,
            input_capacity,
            "cuda graph input device pointer guards",
        )?;
        let mut input_indices = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut input_indices,
            input_capacity,
            "cuda graph logical input indices",
        )?;
        let mut output_device_ptrs = SmallVec::<[DevicePtrGuard; 8]>::new();
        reserve_smallvec(
            &mut output_device_ptrs,
            output_device_capacity,
            "cuda graph output device pointer guards",
        )?;
        let mut output_clears = SmallVec::<[OutputClear; 8]>::new();
        reserve_smallvec(
            &mut output_clears,
            output_device_capacity,
            "cuda graph output clear captures",
        )?;
        let mut output_indices = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut output_indices,
            output_readback_capacity,
            "cuda graph logical output indices",
        )?;
        let mut readback_device_ptrs = SmallVec::<[u64; 8]>::new();
        reserve_smallvec(
            &mut readback_device_ptrs,
            output_readback_capacity,
            "cuda graph readback device pointers",
        )?;
        let mut host_buffers = GraphHostBuffers::try_with_capacity(
            Arc::clone(&self.host_pool),
            input_capacity,
            output_readback_capacity,
        )?;
        let mut expected_input_lens = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut expected_input_lens,
            input_capacity,
            "cuda graph expected input byte lengths",
        )?;
        let mut input_transfer_lens = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut input_transfer_lens,
            input_capacity,
            "cuda graph input transfer byte lengths",
        )?;
        let mut output_lens = SmallVec::<[usize; 8]>::new();
        reserve_smallvec(
            &mut output_lens,
            output_readback_capacity,
            "cuda graph output byte lengths",
        )?;
        let mut replay_input_bytes = 0_u64;
        let mut replay_output_bytes = 0_u64;
        let mut replay_host_upload_operations = 0_u64;
        let mut replay_device_readback_operations = 0_u64;
        let resident_input_replay_safe = capture_binding_plan.resident_input_replay_safe;
        let cached_input_key = exact_input_key(sample_inputs)?;

        // Walk binding plan in order, allocating + classifying input vs output.
        for binding in &prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let byte_len = match binding.input_index {
                Some(input_index) => cuda_graph_sample_input(
                    sample_inputs,
                    input_index,
                    &binding.name,
                    "allocation sizing",
                )?
                .len(),
                None => binding
                    .static_byte_len
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA-graph output `{}` needs a static byte length to be \
                             cached. Set BufferDecl::with_count or output_byte_range before \
                             recording.",
                            binding.name
                        ),
                    })?,
            };
            let device_byte_len = if byte_len == 0 {
                1
            } else {
                aligned_async_copy_len(byte_len)?
            };
            let device_ptr = alloc_cuda_ptr(
                device_byte_len,
                "cuMemAlloc_v2 (cuda_graph input/output buffer)",
            )?;
            self.telemetry
                .record_transient_allocation_bytes(cuda_graph_usize_to_u64(
                    device_byte_len,
                    "cudaGraph input/output allocation bytes",
                )?);
            if let Some(input_index) = binding.input_index {
                let sample_input = cuda_graph_sample_input(
                    sample_inputs,
                    input_index,
                    &binding.name,
                    "input staging",
                )?;
                let input_len = sample_input.len();
                let input_transfer_len = if input_len == 0 {
                    0
                } else {
                    aligned_async_copy_len(input_len)?
                };
                expected_input_lens.push(input_len);
                input_transfer_lens.push(input_transfer_len);
                add_cuda_graph_replay_bytes(&mut replay_input_bytes, input_len, "input replay")?;
                if input_len != 0 {
                    add_cuda_graph_replay_operation(
                        &mut replay_host_upload_operations,
                        "host upload replay",
                    )?;
                }
                host_buffers.push_input_padded(sample_input, input_transfer_len)?;
                input_indices.push(input_index);
                input_device_ptrs.push(DevicePtrGuard::new(device_ptr));
            } else {
                output_clears.push(OutputClear {
                    dst: device_ptr,
                    byte_len: device_byte_len,
                });
                output_device_ptrs.push(DevicePtrGuard::new(device_ptr));
            }
            if let Some(output_index) = binding.output_index {
                let readback = cuda_output_readback_for_binding(
                    program.buffers(),
                    binding.buffer_index,
                    &binding.name,
                    byte_len,
                    "graph capture output readback",
                )?;
                host_buffers.push_output(readback.byte_len)?;
                output_indices.push(output_index);
                output_lens.push(readback.byte_len);
                add_cuda_graph_replay_bytes(
                    &mut replay_output_bytes,
                    readback.byte_len,
                    "output replay",
                )?;
                if readback.byte_len != 0 {
                    add_cuda_graph_replay_operation(
                        &mut replay_device_readback_operations,
                        "device readback replay",
                    )?;
                }
                let readback_ptr = vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
                    device_ptr,
                    readback.device_offset,
                    || {
                        BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA graph output readback device offset {} for `{}` does not fit CUdeviceptr arithmetic.",
                            readback.device_offset, binding.name
                        ),
                    }
                    },
                    || {
                        BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA graph readback pointer overflowed for output `{}` at device_ptr={device_ptr} offset={}. Re-record with a valid output range or split the output buffer.",
                            binding.name, readback.device_offset
                        ),
                    }
                    },
                )?;
                readback_device_ptrs.push(readback_ptr);
            }
        }

        // Allocate the param buffer separately (one per cached graph).
        let param_bytes = super::launch_params::launch_param_byte_len(
            &prepared.launch.param_words,
            "cudaGraph capture",
        )?;
        let param_copy_bytes = if param_bytes == 0 {
            0
        } else {
            aligned_async_copy_len(param_bytes)?
        };
        let params_device_ptr = if param_bytes != 0 {
            // SAFETY: param_bytes is u32-aligned and non-zero in this branch.
            let params_device_ptr =
                alloc_cuda_ptr(param_copy_bytes, "cuMemAlloc_v2 (cuda_graph param buffer)")?;
            self.telemetry
                .record_transient_allocation_bytes(cuda_graph_usize_to_u64(
                    param_copy_bytes,
                    "cudaGraph parameter allocation bytes",
                )?);
            params_device_ptr
        } else {
            0
        };
        let params_device_ptr = DevicePtrGuard::new(params_device_ptr);

        // Create dedicated stream for capture + replay.
        let stream = create_cuda_graph_stream()?;
        // SAFETY: FFI to libcuda.so. Pointer args were validated by the
        // matching alloc / store API; lifetimes are documented in the
        // surrounding function. cuda_check (or matching CUresult guard)
        // propagates non-success codes as BackendError.
        if param_bytes != 0 {
            let mut param_host_transfer =
                HostTransferAllocations::with_capacity(Arc::clone(&self.host_pool), 1, 0)?;
            let param_host_ptr = param_host_transfer
                .push_u32_words_padded(&prepared.launch.param_words, param_copy_bytes)?;
            // SAFETY: Safe FFI / low-level operation verified and audited for Release compliance.
            unsafe {
                // Upload the param words once; the kernel reads them on every replay.
                // The async copy targets the dedicated stream so recording cannot
                // create an implicit dependency on CUDA's legacy default stream.
                super::copy::h2d_async_checked_with_label(
                    params_device_ptr.ptr(),
                    param_host_ptr,
                    param_copy_bytes,
                    stream.ptr().as_ptr(),
                    "cuMemcpyHtoDAsync_v2 (cuda_graph param init)",
                )?;
                synchronize_cuda_graph_stream(
                    &stream,
                    "cuStreamSynchronize (cuda_graph param init)",
                )?;
            }
            self.telemetry.record_sync_point();
        }

        let _ = CU_STREAM_CAPTURE_MODE_THREAD_LOCAL; // suppress unused-const warning
                                                     // Begin capture. Every cuda call on `stream` from here until end
                                                     // capture is recorded into the graph.
                                                     //
                                                     // SAFETY: stream is freshly created. The capture mode is constructed
                                                     // directly via the typed enum variant (THREAD_LOCAL) rather than
                                                     // `std::mem::transmute::<u32, _>(1)`  -  the old transmute would have
                                                     // been UB if the local u32 constant ever drifted away from a valid
                                                     // variant value (the enum has gaps at 3..). The typed variant is
                                                     // compile-time-checked and just as efficient.
        let mut capture_guard = begin_cuda_graph_capture(&stream, "cuStreamBeginCapture_v2")?;

        // Record HtoD memcpys for each input.
        for ((host_buf, input_len), (input_transfer_len, device_ptr)) in host_buffers
            .input
            .iter()
            .zip(expected_input_lens.iter())
            .zip(input_transfer_lens.iter().zip(input_device_ptrs.iter()))
        {
            if *input_len == 0 {
                continue;
            }
            let copy_len = if *input_transfer_len == 0 {
                *input_len
            } else {
                *input_transfer_len
            };
            // SAFETY: host_buf.as_ptr() is stable for the lifetime of CachedCudaGraph
            // (the Vec is owned by CachedCudaGraph and never reallocated  -  capacity is
            // set at construction). device_ptr was allocated above. Both pointers
            // outlive the captured graph.
            unsafe {
                super::copy::h2d_async_checked_with_label(
                    device_ptr.ptr(),
                    host_buf.as_ptr(),
                    copy_len,
                    stream.ptr().as_ptr(),
                    "cuMemcpyHtoDAsync_v2 (capture input)",
                )?;
            }
        }

        record_cuda_graph_output_clears(
            &output_clears,
            &stream,
            "cuMemsetD8Async (capture output clear)",
        )?;

        // Record kernel launch. Build kernel_args mirroring the production
        // launch_module path: per-buffer u64 ptr-of-ptr, then param ptr.
        let launch_pointer_capacity = capture_binding_plan.kernel_pointer_capacity;
        let mut all_ptrs = SmallVec::<[u64; 16]>::new();
        reserve_smallvec(
            &mut all_ptrs,
            launch_pointer_capacity,
            "graph capture launch pointer",
        )?;
        let mut input_iter = input_device_ptrs.iter();
        let mut output_iter = output_device_ptrs.iter();
        for binding in &prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let ptr = if binding.input_index.is_some() {
                input_iter
                    .next()
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA graph capture binding plan expected an input pointer for `{}` but none was allocated.",
                            binding.name
                        ),
                    })?
                    .ptr()
            } else {
                output_iter
                    .next()
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA graph capture binding plan expected an output pointer for `{}` but none was allocated.",
                            binding.name
                        ),
                    })?
                    .ptr()
            };
            all_ptrs.push(ptr);
        }
        let kernel_arg_capacity = capture_binding_plan.kernel_argument_capacity;
        let mut kernel_args: SmallVec<[*mut std::ffi::c_void; 16]> = SmallVec::new();
        reserve_smallvec(
            &mut kernel_args,
            kernel_arg_capacity,
            "graph capture kernel argument",
        )?;
        for ptr in &mut all_ptrs {
            if *ptr == 0 {
                return Err(BackendError::InvalidProgram {
                    fix: "Fix: CUDA graph capture resolved a null kernel argument; graph launch arguments must preserve the lowered descriptor order."
                        .to_string(),
                });
            }
            kernel_args.push(ptr as *mut _ as *mut std::ffi::c_void);
        }
        let mut params_ref = params_device_ptr.ptr();
        kernel_args.push(&mut params_ref as *mut _ as *mut std::ffi::c_void);

        for _ in 0..prepared.fixpoint_iterations {
            super::launch::launch_cuda_function(
                func,
                kernel_args.as_mut_slice(),
                &prepared.launch,
                stream.ptr().as_ptr(),
                false,
                self.ptx_target_sm(),
                "cuLaunchKernel (capture)",
            )?;
        }

        record_cuda_graph_output_readbacks(
            &mut host_buffers.output,
            &output_lens,
            &readback_device_ptrs,
            &stream,
            "cuMemcpyDtoHAsync_v2 (capture output)",
        )?;

        // End capture and instantiate.
        let graph = capture_guard.finish(
            "cuStreamEndCapture",
            "cuStreamEndCapture returned a null graph after reporting success. Fix: update the CUDA driver or disable CUDA graph capture for this device.",
        )?;

        let graph_exec = instantiate_cuda_graph(
            &graph,
            "cuGraphInstantiateWithFlags",
            "cuGraphInstantiateWithFlags returned a null executable graph after reporting success. Fix: update the CUDA driver or disable CUDA graph capture for this device.",
        )?;

        // Capture a second steady-state graph for repeated identical inputs.
        // The full graph above remains the correctness path whenever input
        // bytes change; this graph removes only HtoD nodes after the device
        // input buffers are known-current.
        let mut resident_capture_guard = begin_cuda_graph_capture(
            &stream,
            "cuStreamBeginCapture_v2 (resident input cuda_graph)",
        )?;
        record_cuda_graph_output_clears(
            &output_clears,
            &stream,
            "cuMemsetD8Async (resident input capture output clear)",
        )?;
        for _ in 0..prepared.fixpoint_iterations {
            super::launch::launch_cuda_function(
                func,
                kernel_args.as_mut_slice(),
                &prepared.launch,
                stream.ptr().as_ptr(),
                false,
                self.ptx_target_sm(),
                "cuLaunchKernel (resident input capture)",
            )?;
        }
        record_cuda_graph_output_readbacks(
            &mut host_buffers.output,
            &output_lens,
            &readback_device_ptrs,
            &stream,
            "cuMemcpyDtoHAsync_v2 (resident input capture output)",
        )?;
        let resident_input_graph = resident_capture_guard.finish(
            "cuStreamEndCapture (resident input cuda_graph)",
            "cuStreamEndCapture returned a null resident-input graph after reporting success. Fix: update the CUDA driver or disable CUDA graph capture for this device.",
        )?;

        let resident_input_graph_exec = instantiate_cuda_graph(
            &resident_input_graph,
            "cuGraphInstantiateWithFlags (resident input cuda_graph)",
            "cuGraphInstantiateWithFlags returned a null resident-input executable graph after reporting success. Fix: update the CUDA driver or disable CUDA graph capture for this device.",
        )?;

        let upload_result = (|| {
            upload_cuda_graph_exec(&graph_exec, &stream, "cuGraphUpload")?;
            upload_cuda_graph_exec(
                &resident_input_graph_exec,
                &stream,
                "cuGraphUpload (resident input cuda_graph)",
            )
        })();
        if let Err(error) = upload_result {
            match synchronize_cuda_graph_stream(
                &stream,
                "cuStreamSynchronize (cuda_graph upload cleanup)",
            ) {
                Ok(()) => {
                    self.telemetry.record_sync_point();
                    return Err(error);
                }
                Err(sync_error) => {
                    tracing::error!(
                        "Fix: failed to synchronize CUDA graph upload stream after upload error: {sync_error}. In-flight CUDA graph resources will not be recycled."
                    );
                    std::mem::forget(stream);
                    std::mem::forget(graph_exec);
                    std::mem::forget(graph);
                    std::mem::forget(resident_input_graph_exec);
                    std::mem::forget(resident_input_graph);
                    std::mem::forget(params_device_ptr);
                    std::mem::forget(input_device_ptrs);
                    std::mem::forget(output_device_ptrs);
                    std::mem::forget(host_buffers);
                    return Err(error);
                }
            }
        }

        let (input_host_bufs, output_host_bufs) = host_buffers.into_raw();

        Ok(CachedCudaGraph {
            graph_exec,
            graph,
            resident_input_graph_exec,
            resident_input_graph,
            stream,
            input_host_bufs,
            input_indices,
            input_device_ptrs,
            output_device_ptrs,
            output_host_bufs,
            output_indices,
            output_lens,
            input_transfer_lens,
            replay_input_bytes,
            replay_output_bytes,
            replay_host_upload_operations,
            replay_device_readback_operations,
            expected_input_lens,
            cached_input_key,
            resident_input_replay_safe,
            device_inputs_initialized: false,
            host_outputs_initialized: false,
            params_device_ptr,
            backend: self.clone(),
        })
    }
}
