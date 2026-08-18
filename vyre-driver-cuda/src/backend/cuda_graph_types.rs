use std::ptr::NonNull;
use std::sync::Arc;

use cudarc::driver::sys::{CUgraphExec_st, CUgraph_st, CUstream_st};
use smallvec::SmallVec;
use vyre_driver::input_identity::ExactInputKey;
use vyre_driver::transfer_accounting::TransferAccountingPolicy;
use vyre_driver::BackendError;

use super::allocations::{cuda_check, free_cuda_ptr_with_label};
use super::dispatch::CudaBackend;
use super::pinned_allocations::{PinnedHostAllocation, PinnedHostAllocationPool};
use super::staging_reserve::reserve_smallvec;

pub(crate) const CUDA_GRAPH_REPLAY_ACCOUNTING: TransferAccountingPolicy =
    TransferAccountingPolicy::new("CUDA graph", "record a smaller graph shape");

pub(crate) fn log_cuda_drop_result(op: &str, result: cudarc::driver::sys::CUresult) {
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        tracing::error!(
            "Fix: {op} failed while releasing CUDA graph resources with {result:?}; ensure graph work has completed before resource drop."
        );
    }
}

pub(crate) const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: u32 = 1;

#[derive(Debug)]
pub(crate) struct DevicePtrGuard {
    ptr: u64,
}

impl DevicePtrGuard {
    pub(crate) fn new(ptr: u64) -> Self {
        Self { ptr }
    }

    pub(crate) fn ptr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for DevicePtrGuard {
    fn drop(&mut self) {
        free_cuda_ptr_with_label(self.ptr, "CUDA graph device buffer");
    }
}

#[derive(Debug)]
pub(crate) struct StreamGuard {
    stream: NonNull<CUstream_st>,
}

impl StreamGuard {
    pub(crate) fn new(stream: NonNull<CUstream_st>) -> Self {
        Self { stream }
    }

    pub(crate) fn ptr(&self) -> NonNull<CUstream_st> {
        self.stream
    }
}

impl Drop for StreamGuard {
    fn drop(&mut self) {
        if self.stream != NonNull::dangling() {
            crate::stream::destroy_raw_stream(
                self.stream.as_ptr(),
                "cuStreamDestroy_v2 (cuda_graph dedicated stream)",
            );
        }
    }
}

pub(crate) fn create_cuda_graph_stream() -> Result<StreamGuard, BackendError> {
    let _nonblocking_flag_contract = cudarc::driver::sys::CUstream_flags::CU_STREAM_NON_BLOCKING;
    crate::stream::create_non_blocking_raw_stream("cuStreamCreate (cuda_graph dedicated stream)")
        .map(StreamGuard::new)
}

pub(crate) fn synchronize_cuda_graph_stream(
    stream: &StreamGuard,
    label: &'static str,
) -> Result<(), BackendError> {
    crate::stream::synchronize_raw_stream(stream.ptr().as_ptr(), label)
}

#[derive(Clone, Copy)]
pub(crate) struct OutputClear {
    pub(crate) dst: u64,
    pub(crate) byte_len: usize,
}

pub(crate) fn record_cuda_graph_output_clears(
    clears: &[OutputClear],
    stream: &StreamGuard,
    label: &'static str,
) -> Result<(), BackendError> {
    for clear in clears {
        if clear.byte_len == 0 {
            continue;
        }
        // SAFETY: The output device allocation was created before capture
        // and retained by CachedCudaGraph for the captured graph lifetime.
        // Capturing this memset preserves the host-dispatch contract that
        // output-only buffers start as zero before sparse stores run.
        unsafe {
            super::copy::memset_d8_async_checked(
                clear.dst,
                0,
                clear.byte_len,
                stream.ptr().as_ptr(),
            )
            .map_err(|error| BackendError::DispatchFailed {
                code: None,
                message: format!("{label} failed: {error}"),
            })?;
        }
    }
    Ok(())
}

pub(crate) fn record_cuda_graph_output_readbacks(
    host_buffers: &mut [PinnedHostAllocation],
    output_lens: &[usize],
    readback_device_ptrs: &[u64],
    stream: &StreamGuard,
    label: &'static str,
) -> Result<(), BackendError> {
    if host_buffers.len() != output_lens.len() || output_lens.len() != readback_device_ptrs.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA graph output readback capture has {} host buffer(s), {} output length(s), and {} device pointer(s). Rebuild graph capture staging from one BindingPlan.",
                host_buffers.len(),
                output_lens.len(),
                readback_device_ptrs.len()
            ),
        });
    }
    for ((host_buf, output_len), device_ptr) in host_buffers
        .iter_mut()
        .zip(output_lens.iter())
        .zip(readback_device_ptrs.iter())
    {
        if *output_len == 0 {
            continue;
        }
        // SAFETY: The host buffer is pinned and retained by CachedCudaGraph
        // for the captured graph lifetime; device_ptr was validated from the
        // output allocation plus checked readback offset before capture.
        unsafe {
            super::copy::d2h_async_checked_with_label(
                host_buf.as_mut_ptr(),
                *device_ptr,
                *output_len,
                stream.ptr().as_ptr(),
                label,
            )?;
        }
    }
    Ok(())
}

pub(crate) struct CaptureGuard {
    stream: NonNull<CUstream_st>,
    active: bool,
}

impl CaptureGuard {
    pub(crate) fn armed(stream: NonNull<CUstream_st>) -> Self {
        Self {
            stream,
            active: true,
        }
    }

    pub(crate) fn finish(
        &mut self,
        label: &'static str,
        null_message: &'static str,
    ) -> Result<GraphGuard, BackendError> {
        let graph = end_cuda_graph_capture(self.stream, label, null_message);
        self.disarm();
        graph
    }

    pub(crate) fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if self.active {
            match end_cuda_graph_capture(
                self.stream,
                "cuStreamEndCapture (capture guard drop)",
                "cuStreamEndCapture returned a null graph while dropping an active capture guard. Fix: ensure graph capture is finished explicitly before resource cleanup.",
            ) {
                Ok(graph) => drop(graph),
                Err(error) => tracing::error!(
                    "Fix: failed to end CUDA graph capture during guard drop: {error}"
                ),
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct GraphGuard {
    graph: NonNull<CUgraph_st>,
}

impl GraphGuard {
    pub(crate) fn new(graph: NonNull<CUgraph_st>) -> Self {
        Self { graph }
    }

    pub(crate) fn ptr(&self) -> NonNull<CUgraph_st> {
        self.graph
    }
}

impl Drop for GraphGuard {
    fn drop(&mut self) {
        if self.graph != NonNull::dangling() {
            destroy_cuda_graph_or_log(self.graph.as_ptr(), "CUDA graph guard drop");
        }
    }
}

#[derive(Debug)]
pub(crate) struct GraphExecGuard {
    graph_exec: NonNull<CUgraphExec_st>,
}

impl GraphExecGuard {
    pub(crate) fn new(graph_exec: NonNull<CUgraphExec_st>) -> Self {
        Self { graph_exec }
    }

    pub(crate) fn ptr(&self) -> NonNull<CUgraphExec_st> {
        self.graph_exec
    }
}

impl Drop for GraphExecGuard {
    fn drop(&mut self) {
        if self.graph_exec != NonNull::dangling() {
            destroy_cuda_graph_exec_or_log(self.graph_exec.as_ptr(), "CUDA graph exec guard drop");
        }
    }
}

pub(crate) fn begin_cuda_graph_capture(
    stream: &StreamGuard,
    label: &'static str,
) -> Result<CaptureGuard, BackendError> {
    // SAFETY: stream is a backend-owned non-blocking stream; CUDA validates
    // the opaque handle and returns a CUresult.
    unsafe {
        cuda_check(
            cudarc::driver::sys::cuStreamBeginCapture_v2(
                stream.ptr().as_ptr(),
                cudarc::driver::sys::CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            ),
            label,
        )?;
    }
    Ok(CaptureGuard::armed(stream.ptr()))
}

pub(crate) fn end_cuda_graph_capture(
    stream: NonNull<CUstream_st>,
    label: &'static str,
    null_message: &'static str,
) -> Result<GraphGuard, BackendError> {
    let mut graph_ptr: cudarc::driver::sys::CUgraph = std::ptr::null_mut();
    let status = {
        // SAFETY: stream is in capture mode for normal callers; guard-drop callers
        // are best-effort cleanup paths and CUDA returns a status if capture ended.
        unsafe { cudarc::driver::sys::cuStreamEndCapture(stream.as_ptr(), &mut graph_ptr) }
    };
    if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS && !graph_ptr.is_null() {
        destroy_cuda_graph_or_log(graph_ptr, label);
    }
    cuda_check(status, label)?;
    let graph = NonNull::new(graph_ptr).ok_or_else(|| BackendError::DispatchFailed {
        code: None,
        message: null_message.to_string(),
    })?;
    Ok(GraphGuard::new(graph))
}

pub(crate) fn instantiate_cuda_graph(
    graph: &GraphGuard,
    label: &'static str,
    null_message: &'static str,
) -> Result<GraphExecGuard, BackendError> {
    let mut graph_exec_ptr: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
    // SAFETY: graph is a valid captured graph handle; flags = 0 selects CUDA's
    // default executable graph instantiation policy.
    unsafe {
        cuda_check(
            cudarc::driver::sys::cuGraphInstantiateWithFlags(
                &mut graph_exec_ptr,
                graph.ptr().as_ptr(),
                0,
            ),
            label,
        )?;
    }
    let graph_exec = NonNull::new(graph_exec_ptr).ok_or_else(|| BackendError::DispatchFailed {
        code: None,
        message: null_message.to_string(),
    })?;
    Ok(GraphExecGuard::new(graph_exec))
}

pub(crate) fn upload_cuda_graph_exec(
    graph_exec: &GraphExecGuard,
    stream: &StreamGuard,
    label: &'static str,
) -> Result<(), BackendError> {
    // SAFETY: both handles are owned by CachedCudaGraph and remain live for
    // the upload call.
    unsafe {
        cuda_check(
            cudarc::driver::sys::cuGraphUpload(graph_exec.ptr().as_ptr(), stream.ptr().as_ptr()),
            label,
        )
    }
}

pub(crate) fn destroy_cuda_graph_or_log(graph: cudarc::driver::sys::CUgraph, label: &str) {
    if graph.is_null() {
        return;
    }
    // SAFETY: graph is an owned CUDA graph handle; destroy is used from Drop
    // and cleanup paths so failures are logged.
    unsafe {
        log_cuda_drop_result(label, cudarc::driver::sys::cuGraphDestroy(graph));
    }
}

pub(crate) fn destroy_cuda_graph_exec_or_log(
    graph_exec: cudarc::driver::sys::CUgraphExec,
    label: &str,
) {
    if graph_exec.is_null() {
        return;
    }
    // SAFETY: graph_exec is an owned CUDA executable graph handle; destroy is
    // used from Drop paths so failures are logged.
    unsafe {
        log_cuda_drop_result(label, cudarc::driver::sys::cuGraphExecDestroy(graph_exec));
    }
}

pub(crate) fn add_cuda_graph_replay_bytes(
    total: &mut u64,
    bytes: usize,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_GRAPH_REPLAY_ACCOUNTING.add_bytes(total, bytes, label)
}

pub(crate) fn add_cuda_graph_replay_operation(
    total: &mut u64,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_GRAPH_REPLAY_ACCOUNTING.add_u64_counter(total, 1, label, "operation accounting")
}

pub(crate) struct GraphHostBuffers {
    pool: Arc<PinnedHostAllocationPool>,
    pub(crate) input: SmallVec<[PinnedHostAllocation; 8]>,
    pub(crate) output: SmallVec<[PinnedHostAllocation; 8]>,
}

impl GraphHostBuffers {
    pub(crate) fn try_with_capacity(
        pool: Arc<PinnedHostAllocationPool>,
        input_capacity: usize,
        output_capacity: usize,
    ) -> Result<Self, BackendError> {
        let mut buffers = Self {
            pool,
            input: SmallVec::new(),
            output: SmallVec::new(),
        };
        reserve_smallvec(
            &mut buffers.input,
            input_capacity,
            "cuda graph input host buffers",
        )?;
        reserve_smallvec(
            &mut buffers.output,
            output_capacity,
            "cuda graph output host buffers",
        )?;
        Ok(buffers)
    }

    pub(crate) fn push_input(&mut self, bytes: &[u8]) -> Result<(), BackendError> {
        if bytes.is_empty() {
            self.input.push(PinnedHostAllocation::default());
            return Ok(());
        }
        let mut allocation = self.pool.acquire(bytes.len())?;
        allocation.copy_from_slice(bytes)?;
        self.input.push(allocation);
        Ok(())
    }

    pub(crate) fn push_input_padded(
        &mut self,
        bytes: &[u8],
        transfer_byte_len: usize,
    ) -> Result<(), BackendError> {
        if bytes.is_empty() {
            self.input.push(PinnedHostAllocation::default());
            return Ok(());
        }
        if transfer_byte_len < bytes.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA graph transfer length {} is smaller than logical input length {}.",
                    transfer_byte_len,
                    bytes.len()
                ),
            });
        }
        let mut allocation = self.pool.acquire(transfer_byte_len)?;
        allocation.copy_from_slice(bytes)?;
        if transfer_byte_len > bytes.len() {
            allocation.zero_range(bytes.len(), transfer_byte_len - bytes.len())?;
        }
        self.input.push(allocation);
        Ok(())
    }

    pub(crate) fn push_output(&mut self, byte_len: usize) -> Result<(), BackendError> {
        if byte_len == 0 {
            self.output.push(PinnedHostAllocation::default());
            return Ok(());
        }
        self.output.push(self.pool.acquire(byte_len)?);
        Ok(())
    }

    pub(crate) fn into_raw(
        mut self,
    ) -> (
        SmallVec<[PinnedHostAllocation; 8]>,
        SmallVec<[PinnedHostAllocation; 8]>,
    ) {
        let input = std::mem::take(&mut self.input);
        let output = std::mem::take(&mut self.output);
        (input, output)
    }
}

impl Drop for GraphHostBuffers {
    fn drop(&mut self) {
        for allocation in self.input.drain(..).chain(self.output.drain(..)) {
            self.pool.release(allocation);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GraphHostBuffers;
    use crate::backend::pinned_allocations::PinnedHostAllocationPool;
    use std::sync::Arc;

    #[test]
    fn cuda_graph_zero_byte_host_buffers_do_not_acquire_pinned_memory() {
        let pool = Arc::new(PinnedHostAllocationPool::new(0));
        let mut buffers = GraphHostBuffers::try_with_capacity(Arc::clone(&pool), 1, 1)
            .expect("Fix: graph host buffers should reserve tiny test capacities");

        buffers
            .push_input(&[])
            .expect("Fix: zero-byte graph input must not call CUDA host allocation APIs");
        buffers
            .push_output(0)
            .expect("Fix: zero-byte graph output must not call CUDA host allocation APIs");

        assert!(buffers.input[0].as_ptr().is_null());
        assert!(buffers.output[0].as_ptr().is_null());
        assert_eq!(pool.cached_bytes(), 0);
    }

    #[test]
    fn cuda_graph_padded_input_upload_zero_fills_tail() {
        // Pinned host memory requires an initialized, thread-bound CUDA
        // context; acquire one first (held for the whole test so it outlives
        // the pinned buffers).
        let _device = crate::device::CudaDeviceHandle::acquire_ordinal(0)
            .expect("Fix: acquire a CUDA device/context before pinned host allocation");
        let pool = Arc::new(PinnedHostAllocationPool::new(0));
        let mut buffers = GraphHostBuffers::try_with_capacity(Arc::clone(&pool), 1, 1)
            .expect("Fix: padded input staging should use fallible pinned buffer acquisition");

        buffers.push_input_padded(&[1_u8, 2, 3], 16).expect(
            "Fix: padded input staging should allocate enough capacity for async DMA copies",
        );

        let mut out = Vec::new();
        buffers.input[0]
            .copy_prefix_into(16, &mut out)
            .expect("Fix: copy back staged input staging bytes to verify alignment padding");

        assert_eq!(out, &[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }
}

/// A pre-recorded CUDA graph wrapping one full Program-dispatch sequence
/// (input HtoD memcpy + kernel launch + output DtoH memcpy). Hold on to this
/// across many `dispatch_via_cuda_graph` calls to amortize launch overhead.
///
/// `CachedCudaGraph` owns:
///   - The captured `CUgraph` and instantiated `CUgraphExec`.
///   - A dedicated `CUstream` used for capture + replay.
///   - Device pointers for every input + output buffer.
///   - Host buffers for every input (so callers write new bytes into the
///     same address the captured memcpy reads from) and every output (so
///     readback target stays stable across replays).
///
/// Drop ordering contract:
/// `backend` is declared as the final field of `CachedCudaGraph`. Rust drops struct
/// fields in declaration order after running `CachedCudaGraph::drop`. Placing `backend`
/// (which holds `Arc<CudaContext>`) last guarantees that the CUDA driver context remains
/// live while `cuGraphExecDestroy`, `cuGraphDestroy`, `cuStreamDestroy_v2`, and
/// `cuMemFree_v2` run on all child handles.
#[derive(Debug)]
pub struct CachedCudaGraph {
    /// Instantiated graph executable (owned). Destroyed in `drop` BEFORE
    /// `graph`.
    pub(crate) graph_exec: GraphExecGuard,
    /// Captured graph (owned). Destroyed in `drop`.
    pub(crate) graph: GraphGuard,
    /// Steady-state graph executable that reuses resident device inputs when
    /// the caller replays the same bytes and no input buffer is also an
    /// output buffer.
    pub(crate) resident_input_graph_exec: GraphExecGuard,
    /// Captured steady-state graph without host-to-device input copy nodes.
    pub(crate) resident_input_graph: GraphGuard,
    /// Dedicated stream used for capture + replay (owned). Destroyed in
    /// `drop` AFTER graph + graph_exec.
    pub(crate) stream: StreamGuard,
    /// Per-input host buffers. Callers write new input bytes here before
    /// each replay; the captured memcpy reads from these addresses.
    pub(crate) input_host_bufs: SmallVec<[PinnedHostAllocation; 8]>,
    /// Logical caller input index for each descriptor-ordered input host
    /// buffer. CUDA graph capture records memcpy nodes in descriptor order,
    /// while public replay inputs follow Program::buffers logical order.
    pub(crate) input_indices: SmallVec<[usize; 8]>,
    /// Per-input device pointers (allocated via `cuMemAlloc_v2`). Freed in
    /// `drop`.
    pub(crate) input_device_ptrs: SmallVec<[DevicePtrGuard; 8]>,
    /// Per-output device pointers (allocated via `cuMemAlloc_v2`). Freed
    /// in `drop`.
    pub(crate) output_device_ptrs: SmallVec<[DevicePtrGuard; 8]>,
    /// Per-output pinned host buffers. The captured DtoH memcpy writes into
    /// these stable addresses on every replay.
    pub(crate) output_host_bufs: SmallVec<[PinnedHostAllocation; 8]>,
    /// Logical output index for each descriptor-ordered output host buffer.
    /// CUDA graph readback nodes stay in descriptor order, while public
    /// result vectors follow Program::buffers logical output order.
    pub(crate) output_indices: SmallVec<[usize; 8]>,
    /// Exact byte lengths for each output. Pinned allocations are bucketed and
    /// can be larger than the logical output buffer.
    pub(crate) output_lens: SmallVec<[usize; 8]>,
    /// Total input bytes copied by every replay of this fixed-shape graph.
    pub(crate) replay_input_bytes: u64,
    /// Total output bytes read back by every replay of this fixed-shape graph.
    pub(crate) replay_output_bytes: u64,
    /// Non-empty host-to-device copy operations captured in each replay.
    pub(crate) replay_host_upload_operations: u64,
    /// Non-empty device-to-host copy operations captured in each replay.
    pub(crate) replay_device_readback_operations: u64,
    /// Expected input byte lengths. `dispatch_via_cuda_graph` validates
    /// the caller's input sizes match these  -  a mismatch means the graph
    /// is wrong-shape for the input and must be re-recorded.
    pub(crate) expected_input_lens: SmallVec<[usize; 8]>,
    /// Host-side transfer lengths used for async input uploads during capture
    /// and replay updates.
    pub(crate) input_transfer_lens: SmallVec<[usize; 8]>,
    /// Exact tuple-boundary-preserving key for bytes currently stored in
    /// `input_host_bufs`.
    pub(crate) cached_input_key: ExactInputKey,
    /// Whether the no-upload steady-state graph is semantically safe. It is
    /// disabled for input-output bindings because the kernel mutates the
    /// resident input buffer.
    pub(crate) resident_input_replay_safe: bool,
    /// Whether resident device inputs are known to match the cached host
    /// input bytes.
    pub(crate) device_inputs_initialized: bool,
    /// Whether pinned host output buffers contain a completed replay result
    /// for the cached host input bytes.
    pub(crate) host_outputs_initialized: bool,
    /// Param-buffer device pointer (single allocation; freed in `drop`).
    /// The kernel reads launch parameters (workgroup-related constants)
    /// from this buffer.
    pub(crate) params_device_ptr: DevicePtrGuard,
    /// Backend reference, for the pinned-host pool the `drop` body returns the
    /// cached host buffers to, and ensuring the CUDA context outlives all raw handles.
    pub(crate) backend: CudaBackend,
}

// SAFETY: `CachedCudaGraph` holds raw CUDA resource pointers (graph,
// graph_exec, stream, device pointers). All access goes through cudarc FFI
// calls that are documented thread-safe per the CUDA Driver API
// (`cuGraphLaunch`, `cuStreamSynchronize`, etc.). The pinned host buffers
// are mutated only through `&mut self`.
unsafe impl Send for CachedCudaGraph {}

impl Drop for CachedCudaGraph {
    fn drop(&mut self) {
        let _owned_cuda_resource_counts = (
            self.graph.ptr().as_ptr(),
            self.resident_input_graph.ptr().as_ptr(),
            self.input_device_ptrs.len(),
            self.output_device_ptrs.len(),
            self.params_device_ptr.ptr(),
        );
        if let Err(error) = self.backend.warmup() {
            tracing::error!(
                "Fix: CUDA backend warmup failed before graph resource drop: {error}. Cleanup will continue, but the CUDA context may be unhealthy."
            );
        }
        for allocation in self
            .input_host_bufs
            .drain(..)
            .chain(self.output_host_bufs.drain(..))
        {
            self.backend.host_pool.release(allocation);
        }
    }
}
