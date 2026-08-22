//! Replay helpers for captured CUDA graphs.

use std::ptr::NonNull;
use std::sync::Arc;

use smallvec::SmallVec;
use vyre_driver::BackendError;

use super::allocations::cuda_check;
use super::cuda_graph_lifecycle::{CachedCudaGraph, GraphExecGuard, StreamGuard};
use super::dispatch::CudaBackend;
use super::ordering::{classify_dense_permutation, DensePermutationDefect};
use super::staging_reserve::{reserve_smallvec, reserve_vec, reserved_vec, resize_vec_slots};
use vyre_driver::input_identity::{exact_input_key, ExactInputKey};

impl CachedCudaGraph {
    pub(crate) fn input_shape_matches(&self, inputs: &[&[u8]]) -> bool {
        inputs.len() == self.expected_input_lens.len()
            && self.input_indices.len() == self.expected_input_lens.len()
            && self
                .input_indices
                .iter()
                .zip(self.expected_input_lens.iter())
                .all(|(input_index, expected)| {
                    inputs
                        .get(*input_index)
                        .is_some_and(|input| input.len() == *expected)
                })
    }

    pub(crate) fn materialized_output_cache_matches(
        &self,
        inputs: &[&[u8]],
    ) -> Result<bool, BackendError> {
        let input_state = prepare_cuda_graph_replay_input_state(self, inputs)?;
        self.materialized_output_cache_matches_with_input_state(inputs, &input_state)
    }

    pub(crate) fn materialized_output_cache_matches_with_input_state(
        &self,
        inputs: &[&[u8]],
        input_state: &CudaGraphReplayInputState,
    ) -> Result<bool, BackendError> {
        if !(self.resident_input_replay_safe && self.host_outputs_initialized) {
            return Ok(false);
        }
        cached_input_bytes_match_with_key(self, inputs, &input_state.input_key)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaGraphReplayStats {
    input_bytes: u64,
    output_bytes: u64,
    host_upload_operations: u64,
    device_readback_operations: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaGraphReplayInputState {
    input_key: ExactInputKey,
}

#[derive(Clone, Copy, Debug)]
struct PreparedCudaGraphReplayLaunch {
    stats: CudaGraphReplayStats,
    resident_input_replay: bool,
}

fn launch_cuda_graph_exec(
    graph_exec: &GraphExecGuard,
    stream: &StreamGuard,
    label: &'static str,
) -> Result<(), BackendError> {
    let graph_exec = graph_exec.ptr();
    if graph_exec == NonNull::dangling() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA graph replay received a dangling CUgraphExec sentinel before {label}. Re-record the graph before replay."
            ),
        });
    }
    let stream = stream.ptr();
    if stream == NonNull::dangling() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA graph replay received a dangling CUstream sentinel before {label}. Re-record the graph before replay."
            ),
        });
    }
    // SAFETY: FFI to libcuda.so. `GraphExecGuard` and `StreamGuard` own
    // non-null CUDA handles and the dangling sentinels are rejected above.
    unsafe {
        cuda_check(
            cudarc::driver::sys::cuGraphLaunch(graph_exec.as_ptr(), stream.as_ptr()),
            label,
        )
    }
}

fn synchronize_cuda_graph_replay_stream(cached: &CachedCudaGraph) -> Result<(), BackendError> {
    // Single speculative poll: avoids the overhead of `cuStreamSynchronize`
    // on paths where the kernel has already completed by the time the host
    // reaches this point (e.g., very short kernels, warm caches).  If not
    // immediately ready, fall through directly to the blocking synchronize
    // rather than spinning: an unconditional multi-thousand-iteration spin
    // burns CPU on every replay regardless of kernel duration, adding host
    // overhead that outweighs any latency saved for long kernels.
    if crate::stream::query_raw_stream_ready(
        cached.stream.ptr().as_ptr(),
        "cuStreamQuery (cuda_graph)",
    )? {
        return Ok(());
    }
    crate::stream::synchronize_raw_stream(
        cached.stream.ptr().as_ptr(),
        "cuStreamSynchronize (cuda_graph)",
    )
}

fn cached_input_bytes_match(
    cached: &CachedCudaGraph,
    inputs: &[&[u8]],
) -> Result<bool, BackendError> {
    let input_key = exact_input_key(inputs)?;
    cached_input_bytes_match_with_key(cached, inputs, &input_key)
}

fn cached_input_bytes_match_with_key(
    cached: &CachedCudaGraph,
    inputs: &[&[u8]],
    input_key: &ExactInputKey,
) -> Result<bool, BackendError> {
    if cached.cached_input_key != *input_key {
        return Ok(false);
    }
    cached_input_bytes_match_after_key_match(cached, inputs)
}

fn cached_input_bytes_match_after_key_match(
    cached: &CachedCudaGraph,
    inputs: &[&[u8]],
) -> Result<bool, BackendError> {
    if cached.input_host_bufs.len() != inputs.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph has {} pinned input buffer(s) for {} caller input(s). Re-record the graph; zip-based replay would skip input uploads.",
                cached.input_host_bufs.len(),
                inputs.len()
            ),
        });
    }
    for (slot_index, (slot, input_index)) in cached
        .input_host_bufs
        .iter()
        .zip(cached.input_indices.iter())
        .enumerate()
    {
        let src = cached_graph_input(inputs, *input_index, slot_index, "cached input compare")?;
        if src.len() > slot.byte_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA graph cached input comparison saw {} byte(s) for a {} byte pinned allocation. Re-record the graph for this input shape.",
                    src.len(),
                    slot.byte_len
                ),
            });
        }
        if src.is_empty() {
            continue;
        }
        let cached_bytes = {
            // SAFETY: `slot` owns a pinned allocation of at least `slot.byte_len`
            // bytes, and the length check above proves `src.len() <= slot.byte_len`.
            unsafe { std::slice::from_raw_parts(slot.as_ptr().cast::<u8>(), src.len()) }
        };
        if cached_bytes != src {
            return Ok(false);
        }
    }
    Ok(true)
}

impl CudaBackend {
    pub(crate) fn try_cuda_graph_materialized_cache_into(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<bool, BackendError> {
        let input_state = self.prepare_cuda_graph_replay_input_state(cached, inputs)?;
        self.try_cuda_graph_materialized_cache_with_input_state_into(
            cached,
            inputs,
            &input_state,
            outputs,
        )
    }

    pub(crate) fn try_cuda_graph_materialized_cache_with_input_state_into(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        input_state: &CudaGraphReplayInputState,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<bool, BackendError> {
        if cached.materialized_output_cache_matches_with_input_state(inputs, input_state)? {
            collect_cuda_graph_outputs(cached, outputs)?;
            self.telemetry.record_cuda_graph_materialized_cache_hit();
            return Ok(true);
        }
        Ok(false)
    }

    pub(crate) fn enqueue_cuda_graph_replay(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
    ) -> Result<CudaGraphReplayStats, BackendError> {
        let input_state = self.prepare_cuda_graph_replay_input_state(cached, inputs)?;
        self.enqueue_cuda_graph_replay_with_input_state(cached, inputs, &input_state)
    }

    pub(crate) fn enqueue_cuda_graph_replay_with_input_state(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        input_state: &CudaGraphReplayInputState,
    ) -> Result<CudaGraphReplayStats, BackendError> {
        let prepared = prepare_cuda_graph_replay_launch(cached, inputs, input_state)?;
        launch_prepared_cuda_graph_replay(cached, &prepared, "cuGraphLaunch")?;
        self.telemetry.record_cuda_graph_launch();
        Ok(prepared.stats)
    }

    pub(crate) fn finish_cuda_graph_replay_into(
        &self,
        cached: &mut CachedCudaGraph,
        stats: CudaGraphReplayStats,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        synchronize_cuda_graph_replay_stream(cached)?;
        cached.device_inputs_initialized = true;
        self.telemetry.record_sync_point();
        self.record_cuda_graph_replay_stats(stats);
        collect_cuda_graph_outputs(cached, outputs)?;
        cached.host_outputs_initialized = true;
        Ok(())
    }

    pub(crate) fn record_cuda_graph_batched_replay_chunk(&self, lanes: u64) {
        self.telemetry.record_cuda_graph_batched_replay(lanes);
    }

    pub(crate) fn prepare_cuda_graph_replay_input_state(
        &self,
        cached: &CachedCudaGraph,
        inputs: &[&[u8]],
    ) -> Result<CudaGraphReplayInputState, BackendError> {
        prepare_cuda_graph_replay_input_state(cached, inputs)
    }

    pub(crate) fn prepare_cuda_graph_replay_input_state_with_key(
        &self,
        cached: &CachedCudaGraph,
        inputs: &[&[u8]],
        input_key: ExactInputKey,
    ) -> Result<CudaGraphReplayInputState, BackendError> {
        prepare_cuda_graph_replay_input_state_with_key(cached, inputs, input_key)
    }

    /// Replay a cached CUDA graph with new input bytes.
    pub fn dispatch_via_cuda_graph_into(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        let input_state = self.prepare_cuda_graph_replay_input_state(cached, inputs)?;
        self.dispatch_via_cuda_graph_with_input_state_into(cached, inputs, &input_state, outputs)
    }

    pub(crate) fn dispatch_via_cuda_graph_with_input_state_into(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        input_state: &CudaGraphReplayInputState,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<(), BackendError> {
        if self.try_cuda_graph_materialized_cache_with_input_state_into(
            cached,
            inputs,
            &input_state,
            outputs,
        )? {
            return Ok(());
        }
        let stats =
            self.enqueue_cuda_graph_replay_with_input_state(cached, inputs, &input_state)?;
        self.finish_cuda_graph_replay_into(cached, stats, outputs)
    }

    /// Replay a cached CUDA graph with CUDA event timing.
    ///
    /// Dispatches the graph on the GPU and measures device execution time via
    /// CUDA timing events. Returns `Some(device_ns)` of measured device time.
    pub(crate) fn dispatch_via_cuda_graph_timed_into(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<Option<u64>, BackendError> {
        let input_state = self.prepare_cuda_graph_replay_input_state(cached, inputs)?;
        self.dispatch_via_cuda_graph_timed_with_input_state_into(
            cached,
            inputs,
            &input_state,
            outputs,
        )
    }

    pub(crate) fn dispatch_via_cuda_graph_timed_with_input_state_into(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
        input_state: &CudaGraphReplayInputState,
        outputs: &mut Vec<Vec<u8>>,
    ) -> Result<Option<u64>, BackendError> {
        if self.try_cuda_graph_materialized_cache_with_input_state_into(
            cached,
            inputs,
            input_state,
            outputs,
        )? {
            // The outputs the graph would produce are already on the host for
            // exactly these inputs, so the launch is skipped and the device
            // time it would have measured is zero rather than absent: the
            // caller still gets a measurement, and it is the true one.
            return Ok(Some(0));
        }
        self.warmup()?;
        let prepared = prepare_cuda_graph_replay_launch(cached, inputs, &input_state)?;

        let mut timing_events =
            crate::stream::CudaTimingEventPairLease::acquire(Arc::clone(&self.launch_resources))?;
        {
            let (start, end) = timing_events.events()?;
            start.record(cached.stream.ptr().as_ptr())?;
            launch_prepared_cuda_graph_replay(cached, &prepared, "cuGraphLaunch")?;
            self.telemetry.record_cuda_graph_launch();
            end.record(cached.stream.ptr().as_ptr())?;
            end.synchronize()?;
        }
        timing_events.mark_synchronized();
        cached.device_inputs_initialized = true;
        self.telemetry.record_sync_point();
        let device_ns = {
            let (start, end) = timing_events.events()?;
            start.elapsed_time_ns(end)?
        };
        self.record_cuda_graph_replay_stats(prepared.stats);
        collect_cuda_graph_outputs(cached, outputs)?;
        cached.host_outputs_initialized = true;
        Ok(Some(device_ns))
    }

    /// Replay a cached CUDA graph with CUDA event timing and allocated outputs.
    pub fn dispatch_via_cuda_graph_timed(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        let started = std::time::Instant::now();
        let mut outputs = reserved_vec(
            cached.output_host_bufs.len(),
            "timed cuda graph replay output vector",
        )?;
        let device_ns = self.dispatch_via_cuda_graph_timed_into(cached, inputs, &mut outputs)?;
        let wall_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(started, "timed cuda graph replay wall latency")?;
        self.telemetry
            .record_timed_dispatch(wall_ns, device_ns, None, None);
        Ok(vyre_driver::TimedDispatchResult::device_timed(
            outputs, wall_ns, device_ns,
        ))
    }

    /// Convenience wrapper that allocates the output `Vec` internally.
    pub fn dispatch_via_cuda_graph(
        &self,
        cached: &mut CachedCudaGraph,
        inputs: &[&[u8]],
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = reserved_vec(
            cached.output_host_bufs.len(),
            "cuda graph replay output vector",
        )?;
        self.dispatch_via_cuda_graph_into(cached, inputs, &mut outputs)?;
        Ok(outputs)
    }
}

impl CudaGraphReplayStats {
    fn from_cached(cached: &CachedCudaGraph) -> Self {
        Self {
            input_bytes: cached.replay_input_bytes,
            output_bytes: cached.replay_output_bytes,
            host_upload_operations: cached.replay_host_upload_operations,
            device_readback_operations: cached.replay_device_readback_operations,
        }
    }
}

fn prepare_cuda_graph_replay(
    cached: &mut CachedCudaGraph,
    inputs: &[&[u8]],
    input_state: &CudaGraphReplayInputState,
) -> Result<(CudaGraphReplayStats, bool), BackendError> {
    let resident_input_replay = cached.resident_input_replay_safe
        && cached.device_inputs_initialized
        && cached_input_bytes_match_with_key(cached, inputs, &input_state.input_key)?;

    if !resident_input_replay {
        for (slot_index, ((slot, input_index), transfer_len)) in cached
            .input_host_bufs
            .iter_mut()
            .zip(cached.input_indices.iter())
            .zip(cached.input_transfer_lens.iter())
            .enumerate()
        {
            let src = cached_graph_input(inputs, *input_index, slot_index, "input replay staging")?;
            slot.copy_from_slice(src)?;
            if *transfer_len > src.len() {
                slot.zero_range(src.len(), transfer_len - src.len())?;
            }
        }
        cached.cached_input_key = input_state.input_key;
        cached.device_inputs_initialized = false;
        cached.host_outputs_initialized = false;
    }
    let mut stats = CudaGraphReplayStats::from_cached(cached);
    if resident_input_replay {
        stats.input_bytes = 0;
        stats.host_upload_operations = 0;
    }
    Ok((stats, resident_input_replay))
}

fn prepare_cuda_graph_replay_launch(
    cached: &mut CachedCudaGraph,
    inputs: &[&[u8]],
    input_state: &CudaGraphReplayInputState,
) -> Result<PreparedCudaGraphReplayLaunch, BackendError> {
    let (stats, resident_input_replay) = prepare_cuda_graph_replay(cached, inputs, input_state)?;
    Ok(PreparedCudaGraphReplayLaunch {
        stats,
        resident_input_replay,
    })
}

fn launch_prepared_cuda_graph_replay(
    cached: &mut CachedCudaGraph,
    prepared: &PreparedCudaGraphReplayLaunch,
    label: &'static str,
) -> Result<(), BackendError> {
    let graph_exec = if prepared.resident_input_replay {
        &cached.resident_input_graph_exec
    } else {
        &cached.graph_exec
    };
    launch_cuda_graph_exec(graph_exec, &cached.stream, label)
}

fn prepare_cuda_graph_replay_input_state(
    cached: &CachedCudaGraph,
    inputs: &[&[u8]],
) -> Result<CudaGraphReplayInputState, BackendError> {
    prepare_cuda_graph_replay_input_state_with_key(cached, inputs, exact_input_key(inputs)?)
}

fn prepare_cuda_graph_replay_input_state_with_key(
    cached: &CachedCudaGraph,
    inputs: &[&[u8]],
    input_key: ExactInputKey,
) -> Result<CudaGraphReplayInputState, BackendError> {
    validate_cached_graph_inputs(cached, inputs)?;
    Ok(CudaGraphReplayInputState { input_key })
}

fn cached_graph_input<'a>(
    inputs: &[&'a [u8]],
    input_index: usize,
    slot_index: usize,
    context: &'static str,
) -> Result<&'a [u8], BackendError> {
    inputs
        .get(input_index)
        .copied()
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph {context} slot {slot_index} maps to logical input {input_index}, but replay received only {} input(s). Re-record the graph from a valid BindingPlan.",
                inputs.len()
            ),
        })
}

fn validate_cached_graph_slot_index_map(
    indices: &[usize],
    expected_len: usize,
    slot_kind: &'static str,
    action: &'static str,
) -> Result<(), BackendError> {
    let mut sorted_indices = SmallVec::<[usize; 8]>::new();
    reserve_smallvec(
        &mut sorted_indices,
        indices.len(),
        "cuda graph slot index validation",
    )?;
    sorted_indices.extend(indices.iter().copied());
    crate::backend::ordering::sort_unstable_if_needed(sorted_indices.as_mut_slice());
    // Delegate the dense-permutation invariant to the single backend-neutral
    // owner (shared with the resident-dispatch index validators); format the
    // graph-replay-specific remediation from the classified defect.
    match classify_dense_permutation(&sorted_indices, expected_len) {
        Ok(()) => Ok(()),
        Err(DensePermutationDefect::Duplicate { index, slot }) => {
            Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph has a duplicate logical {slot_kind} index {index} at sorted slot {slot}; duplicate {slot_kind} indexes alias one logical slot onto two descriptors. Re-record the graph from Program::buffers logical {slot_kind} order before {action}.",
                ),
            })
        }
        Err(DensePermutationDefect::Sparse { index, slot }) => {
            Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph logical {slot_kind} index {index} at sorted slot {slot} is not dense over 0..{expected_len}. Re-record the graph from Program::buffers logical {slot_kind} order before {action}.",
                ),
            })
        }
        Err(DensePermutationDefect::LengthMismatch { resolved, expected }) => {
            Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph resolved {resolved} logical {slot_kind} index(es); expected {expected} {slot_kind} slot(s). Re-record the graph; descriptor-ordered graph {slot_kind}s must map back to Program::buffers {slot_kind} slots.",
                ),
            })
        }
    }
}

fn validate_cached_graph_input_index_map(
    input_indices: &[usize],
    expected_len: usize,
) -> Result<(), BackendError> {
    validate_cached_graph_slot_index_map(input_indices, expected_len, "input", "replay")
}

fn validate_cached_graph_output_index_map(
    output_indices: &[usize],
    expected_len: usize,
) -> Result<(), BackendError> {
    validate_cached_graph_slot_index_map(output_indices, expected_len, "output", "collection")
}

fn validate_cached_graph_inputs(
    cached: &CachedCudaGraph,
    inputs: &[&[u8]],
) -> Result<(), BackendError> {
    if cached.input_host_bufs.len() != cached.expected_input_lens.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph has {} pinned input buffer(s) but {} expected input length(s). Re-record the graph before replay.",
                cached.input_host_bufs.len(),
                cached.expected_input_lens.len()
            ),
        });
    }
    if cached.input_transfer_lens.len() != cached.expected_input_lens.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph has {} input transfer length(s) but {} expected input length(s). Re-record the graph; zip-based replay would skip or truncate input uploads.",
                cached.input_transfer_lens.len(),
                cached.expected_input_lens.len()
            ),
        });
    }
    validate_cached_graph_input_index_map(&cached.input_indices, cached.expected_input_lens.len())?;
    if inputs.len() != cached.expected_input_lens.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph expects {} inputs but received {}.",
                cached.expected_input_lens.len(),
                inputs.len()
            ),
        });
    }
    for (idx, ((input_index, expected_len), transfer_len)) in cached
        .input_indices
        .iter()
        .zip(cached.expected_input_lens.iter())
        .zip(cached.input_transfer_lens.iter())
        .enumerate()
    {
        let input = cached_graph_input(inputs, *input_index, idx, "shape validation")?;
        let received_len = input.len();
        if received_len != *expected_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph input {idx} expects {expected_len} bytes but \
                     received {}  -  re-record the graph for this input shape.",
                    received_len
                ),
            });
        }
        if *transfer_len < *expected_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph input {idx} expects {expected_len} bytes but its captured transfer length is {transfer_len}. Re-record the graph before replay; truncated graph memcpy would leave stale device input bytes.",
                ),
            });
        }
    }
    Ok(())
}

fn collect_cuda_graph_outputs(
    cached: &CachedCudaGraph,
    outputs: &mut Vec<Vec<u8>>,
) -> Result<(), BackendError> {
    if cached.output_host_bufs.len() != cached.output_lens.len()
        || cached.output_indices.len() != cached.output_lens.len()
    {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph has {} pinned output buffer(s), {} logical output index(es), and {} output length(s). Re-record the graph before collecting outputs.",
                cached.output_host_bufs.len(),
                cached.output_indices.len(),
                cached.output_lens.len()
            ),
        });
    }
    validate_cached_graph_output_index_map(&cached.output_indices, cached.output_lens.len())?;
    resize_vec_slots(
        outputs,
        cached.output_lens.len(),
        "cuda graph replay output vector",
    )?;
    reserve_cuda_graph_output_slots(&cached.output_indices, &cached.output_lens, outputs)?;
    let output_count = outputs.len();
    for (slot_index, (buf, (output_index, byte_len))) in cached
        .output_host_bufs
        .iter()
        .zip(cached.output_indices.iter().zip(cached.output_lens.iter()))
        .enumerate()
    {
        let output = outputs
            .get_mut(*output_index)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph output slot {slot_index} maps to logical output {output_index}, but collection has only {} output slot(s). Re-record the graph from a valid BindingPlan.",
                    output_count
                ),
            })?;
        buf.copy_prefix_into(*byte_len, output)?;
    }
    Ok(())
}

fn reserve_cuda_graph_output_slots(
    output_indices: &[usize],
    output_lens: &[usize],
    outputs: &mut [Vec<u8>],
) -> Result<(), BackendError> {
    if output_indices.len() != output_lens.len() || output_lens.len() != outputs.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: cached cuda graph output preflight expected {} logical output index(es), {} output length(s), and {} caller output slot(s). Re-record the graph before collecting outputs.",
                output_indices.len(),
                output_lens.len(),
                outputs.len()
            ),
        });
    }
    let output_count = outputs.len();
    for (slot_index, (output_index, byte_len)) in
        output_indices.iter().zip(output_lens.iter()).enumerate()
    {
        let output = outputs
            .get_mut(*output_index)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: cached cuda graph output preflight slot {slot_index} maps to logical output {output_index}, but collection has only {} output slot(s). Re-record the graph from a valid BindingPlan.",
                    output_count
                ),
            })?;
        reserve_vec(output, *byte_len, "cuda graph replay output bytes")?;
    }
    Ok(())
}

impl CudaBackend {
    fn record_cuda_graph_replay_stats(&self, stats: CudaGraphReplayStats) {
        self.telemetry
            .record_host_to_device_bytes(stats.input_bytes);
        self.telemetry
            .record_device_to_host_readback(stats.output_bytes);
        self.telemetry
            .record_host_upload_operations(stats.host_upload_operations);
        self.telemetry
            .record_device_readback_operations(stats.device_readback_operations);
    }
}

// Inline: covers `cached_graph_input`, `validate_cached_graph_input_index_map`,
// `validate_cached_graph_output_index_map`, which no integration test can name.
#[cfg(test)]
mod source_contract_tests {
    use super::{
        cached_graph_input, validate_cached_graph_input_index_map,
        validate_cached_graph_output_index_map,
    };

    #[test]
    fn cached_graph_replay_input_index_map_accepts_reordered_descriptor_inputs() {
        validate_cached_graph_input_index_map(&[2, 0, 1], 3).expect(
            "Fix: descriptor-ordered CUDA graph inputs may map to reordered logical slots.",
        );

        let first = [0xA1, 0xA2];
        let second = [0xB1];
        let third = [0xC1, 0xC2, 0xC3];
        let inputs: &[&[u8]] = &[first.as_slice(), second.as_slice(), third.as_slice()];

        assert_eq!(
            cached_graph_input(inputs, 2, 0, "test replay")
                .expect("Fix: graph replay should resolve logical input 2 for descriptor slot 0."),
            third.as_slice()
        );
        assert_eq!(
            cached_graph_input(inputs, 0, 1, "test replay")
                .expect("Fix: graph replay should resolve logical input 0 for descriptor slot 1."),
            first.as_slice()
        );
    }

    #[test]
    fn cached_graph_replay_input_index_map_rejects_stale_or_non_dense_maps() {
        let duplicate = validate_cached_graph_input_index_map(&[0, 0, 2], 3).unwrap_err();
        assert!(
            duplicate.to_string().contains("duplicate"),
            "Fix: duplicate CUDA graph logical input indexes must fail before replay can alias an input slot: {duplicate}"
        );
        let sparse = validate_cached_graph_input_index_map(&[0, 2, 3], 3).unwrap_err();
        assert!(
            sparse.to_string().contains("dense"),
            "Fix: sparse CUDA graph logical input indexes must fail before replay can skip an input slot: {sparse}"
        );
        let truncated = validate_cached_graph_input_index_map(&[0, 1], 3).unwrap_err();
        assert!(
            truncated.to_string().contains("expected 3"),
            "Fix: truncated CUDA graph logical input maps must fail before zip-based replay staging: {truncated}"
        );

        let only = [0xAA];
        let inputs: &[&[u8]] = &[only.as_slice()];
        let stale = cached_graph_input(inputs, 1, 0, "test replay").unwrap_err();
        assert!(
            stale.to_string().contains("logical input 1"),
            "Fix: stale CUDA graph logical input indexes must become BackendError, not a panic or wrong-slot replay: {stale}"
        );
    }

    #[test]
    fn cached_graph_replay_output_index_map_accepts_reordered_descriptor_outputs() {
        validate_cached_graph_output_index_map(&[1, 0, 2], 3).expect(
            "Fix: descriptor-ordered CUDA graph outputs may map to reordered logical slots.",
        );
        let duplicate = validate_cached_graph_output_index_map(&[0, 0, 2], 3).unwrap_err();
        assert!(
            duplicate.to_string().contains("duplicate"),
            "Fix: duplicate CUDA graph logical output indexes must fail before collection can alias an output slot: {duplicate}"
        );
        let sparse = validate_cached_graph_output_index_map(&[0, 2, 3], 3).unwrap_err();
        assert!(
            sparse.to_string().contains("dense"),
            "Fix: sparse CUDA graph logical output indexes must fail before collection can skip an output slot: {sparse}"
        );
        let truncated = validate_cached_graph_output_index_map(&[0, 1], 3).unwrap_err();
        assert!(
            truncated.to_string().contains("expected 3"),
            "Fix: truncated CUDA graph logical output maps must fail before positional collection can drop a slot: {truncated}"
        );
    }
}
