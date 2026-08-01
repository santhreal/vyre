//! Shared command recording, dispatch, and readback for vyre IR pipelines.

use crate::buffer::BindGroupCache;
use crate::buffer::{BufferPool, GpuBufferHandle};
use crate::numeric::WGPU_NUMERIC;
use crate::pipeline::binding::consumes_host_input;
use crate::pipeline::element_size_bytes;
use crate::pipeline::{BufferBindingInfo, OutputBindingLayout};
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;
use vyre_driver::BackendError;
use vyre_emit_naga::program::{TrapTag, TRAP_SIDECAR_WORDS};

mod bind_groups;
pub(super) mod binding_lookup;
mod clears;
mod readback;
mod staging;
mod submit;
pub(crate) mod timestamp;
mod trap;
mod upload;

use readback::SubmittedMap;
pub(crate) use readback::WgpuPendingReadback;
pub(crate) use submit::{submit_recorded_batch, submit_recorded_dispatch};
use timestamp::TimestampRecorder;
use upload::{pool_backend_error, write_padded_input};

type GpuBuffers = SmallVec<[(u32, GpuBufferHandle, u64); 8]>;

/// Static labels used for wgpu resource creation.
#[derive(Clone, Copy)]
pub(crate) struct DispatchLabels {
    /// Bind group label.
    pub bind_group: &'static str,
    /// Command encoder label.
    pub encoder: &'static str,
    /// Compute pass label.
    pub compute: &'static str,
}

/// Command recording inputs shared by direct and compiled dispatch.
pub(crate) struct RecordAndReadback<'a> {
    /// Device and queue owned by the backend or compiled pipeline.
    pub device_queue: &'a Arc<(wgpu::Device, wgpu::Queue)>,
    /// Dispatch-local buffer arena.
    pub pool: &'a BufferPool,
    /// Size-classed direct readback rings for output staging.
    pub readback_rings: Option<&'a Arc<crate::runtime::readback_ring::ReadbackRingSet>>,
    /// Compiled compute pipeline to execute.
    pub pipeline: &'a wgpu::ComputePipeline,
    /// Bind-group layouts for `pipeline`.
    pub bind_group_layouts: &'a [Arc<wgpu::BindGroupLayout>],
    /// Cache for bind groups keyed by layout identity and pooled-buffer IDs.
    pub bind_group_cache: Option<&'a BindGroupCache>,
    /// Buffer binding metadata derived from the Program at compile time.
    pub buffer_bindings: &'a [BufferBindingInfo],
    /// Caller-provided bytes for each non-shared, non-output buffer in declaration order.
    pub inputs: &'a [&'a [u8]],
    /// Per-output copy and trimming layouts.
    ///
    /// Owned rather than borrowed because a runtime-sized writable binding is
    /// re-laid-out per dispatch from the caller's bytes; see
    /// [`resolve_runtime_sized_outputs`].
    pub output_bindings: Arc<[OutputBindingLayout]>,
    /// Trap tag table for backend-owned trap sidecar decoding.
    pub trap_tags: &'a [TrapTag],
    /// Workgroup count for direct dispatch.
    pub workgroup_count: [u32; 3],
    /// Optional indirect dispatch source.
    pub indirect: Option<&'a crate::pipeline::IndirectDispatch>,
    /// wgpu labels for trace readability.
    pub labels: DispatchLabels,
    /// Number of back-to-back compute dispatches to record before readback.
    pub iterations: u32,
    /// Enable opt-in GPU timestamp query profiling for this dispatch.
    pub timestamp_profile: bool,
    /// Workgroup shape the grid may be re-inferred from, when it was inferred.
    ///
    /// `Some(shape)` means `workgroup_count` was derived from the compile-time
    /// output word count, so it must be recomputed once a runtime-sized output
    /// resolves to its real length. `None` means the caller pinned the launch
    /// through `DispatchConfig::grid_override` and it is dispatched as asked.
    pub inferred_grid_shape: Option<[u32; 3]>,
}

impl<'a> RecordAndReadback<'a> {
    pub(crate) fn for_dispatch(
        pipeline: &'a crate::pipeline::WgpuPipeline,
        dispatch_arena: &'a crate::DispatchArena,
        inputs: &'a [&'a [u8]],
        workgroup_count: [u32; 3],
        config: &vyre_driver::DispatchConfig,
        timestamp_profile: bool,
        labels: DispatchLabels,
    ) -> Self {
        Self {
            device_queue: &pipeline.device_queue,
            pool: dispatch_arena.pool(),
            readback_rings: Some(dispatch_arena.readback_rings()),
            pipeline: &pipeline.pipeline,
            bind_group_layouts: &pipeline.bind_group_layouts,
            bind_group_cache: Some(pipeline.bind_group_cache.as_ref()),
            buffer_bindings: &pipeline.buffer_bindings,
            inputs,
            output_bindings: Arc::clone(&pipeline.output_bindings),
            trap_tags: &pipeline.trap_tags,
            workgroup_count,
            indirect: pipeline.indirect.as_ref(),
            labels,
            iterations: config.fixpoint_iterations.unwrap_or(1).max(1),
            timestamp_profile,
            inferred_grid_shape: config
                .grid_override
                .is_none()
                .then_some(pipeline.workgroup_shape),
        }
    }
}

pub(crate) struct RecordedDispatch {
    device_queue: Arc<(wgpu::Device, wgpu::Queue)>,
    command_buffer: Option<wgpu::CommandBuffer>,
    _gpu_buffers: GpuBuffers,
    _bind_groups: SmallVec<[Arc<wgpu::BindGroup>; 4]>,
    readback_buffers: smallvec::SmallVec<[SubmittedMap; 4]>,
    output_count: usize,
    output_bindings: Arc<[OutputBindingLayout]>,
    trap_tags: Arc<[TrapTag]>,
    timestamp_recorder: Option<TimestampRecorder>,
}

/// Record compute work, submit it, and return a pending readback handle.
///
/// # Errors
///
/// Returns a backend error when buffer sizing, bind-group construction, GPU
/// submission, or readback map request fails.
pub(crate) fn record_and_submit_async(
    request: RecordAndReadback<'_>,
) -> Result<WgpuPendingReadback, BackendError> {
    let recorded = record_dispatch_unsubmitted(request)?;
    submit_recorded_dispatch(recorded)
}

pub(crate) fn record_dispatch_unsubmitted(
    request: RecordAndReadback<'_>,
) -> Result<RecordedDispatch, BackendError> {
    super::dispatch_scratch::with_dispatch_scratch(|scratch| {
        record_dispatch_unsubmitted_impl(request, scratch)
    })
}

fn record_dispatch_unsubmitted_impl(
    mut request: RecordAndReadback<'_>,
    scratch: &mut super::dispatch_scratch::DispatchScratch,
) -> Result<RecordedDispatch, BackendError> {
    let (device, queue) = &**request.device_queue;
    let pool = request.pool;
    let host_upload_started = Instant::now();

    // Map buffer binding → index into `request.inputs`. Plain outputs are
    // allocated empty, but read-write state buffers with
    // `preserve_input_contents` are both inputs and outputs: their host bytes
    // must be uploaded before dispatch and read back after dispatch.
    let input_slot_count = request.inputs.len();
    let non_shared_binding_count = request
        .buffer_bindings
        .iter()
        .filter(|info| info.kind != vyre_foundation::ir::MemoryKind::Shared)
        .count();
    let full_input_order_count = request
        .buffer_bindings
        .iter()
        .filter(|info| info.kind != vyre_foundation::ir::MemoryKind::Shared && !info.internal_trap)
        .count();
    let full_input_order = input_slot_count == full_input_order_count;
    let input_idx_by_binding = &mut scratch.input_idx_by_binding;
    let mut next_input = 0usize;
    for info in request.buffer_bindings.iter() {
        if info.kind == vyre_foundation::ir::MemoryKind::Shared || info.internal_trap {
            continue;
        }
        if !consumes_host_input(info) {
            if full_input_order {
                next_input = next_input.checked_add(1).ok_or_else(|| {
                    BackendError::new(
                        "record-and-readback input binding index overflowed usize. Fix: split the dispatch input list before recording.",
                    )
                })?;
            }
            continue;
        }
        input_idx_by_binding.push(info.binding, next_input)?;
        next_input = next_input.checked_add(1).ok_or_else(|| {
            BackendError::new(
                "record-and-readback consumed input index overflowed usize. Fix: split the dispatch input list before recording.",
            )
        })?;
    }

    // Runtime-sized writable bindings are sized from the bytes the caller
    // supplied on THIS dispatch. The pipeline's cached layouts cannot carry that
    // number: they are derived once per program, at compile time, where the input
    // lengths do not exist yet, so a countless declaration was laid out as zero
    // elements. Resolving it here is what makes such a buffer read back the same
    // bytes the CPU reference and CUDA return, instead of an empty buffer under
    // an `Ok`, and it is what stops the upload from overrunning a 4-byte
    // allocation and aborting the host process inside `Queue::write_buffer`.
    if request
        .buffer_bindings
        .iter()
        .any(|info| info.is_output && info.count == 0 && !info.internal_trap)
    {
        request.output_bindings = resolve_runtime_sized_outputs(
            &request.output_bindings,
            request.buffer_bindings,
            request.inputs,
            input_idx_by_binding,
        )?;
        // The launch grid was inferred from that same compile-time count, so it
        // is wrong for the same reason the layout was: a countless output
        // reports zero words, which rounds up to exactly one workgroup no
        // matter how many elements the caller supplied. Left alone, the
        // dispatch covers one workgroup of a buffer that now reads back at its
        // full resolved length, so every element past that first workgroup
        // comes home as a zero under an `Ok`, which is the same silent
        // wrong-answer the zero-length readback was. Re-infer from the resolved
        // word count, and only when this dispatch inferred its grid: a caller
        // who pinned `grid_override` asked for that exact launch and keeps it.
        if let Some(workgroup_shape) = request.inferred_grid_shape {
            let resolved_words = request
                .output_bindings
                .iter()
                .max_by_key(|output| output.word_count)
                .map(|largest| {
                    u32::try_from(largest.word_count).map_err(|error| {
                        BackendError::new(format!(
                            "resolved output word count {} for `{}` does not fit u32: {error}. Fix: declare the buffer with a smaller .with_count(n), or shard the dispatch.",
                            largest.word_count, largest.name
                        ))
                    })
                })
                .transpose()?;
            if let Some(words) = resolved_words {
                request.workgroup_count =
                    vyre_driver::program_walks::infer_dispatch_grid_for_count(
                        words,
                        workgroup_shape,
                    )?;
            }
        }
    }

    // Create a GPU buffer for every binding that needs one.
    // Tuple = (binding, pooled buffer, logical-byte-size). The third
    // field is the size we want the descriptor binding range to use,
    // not the size_class allocation length the pool returns. See
    // ROADMAP Q3  -  using `as_entire_binding()` made `arrayLength`
    // report the rounded-up element count instead of the logical one.
    let mut gpu_buffers = GpuBuffers::with_capacity(non_shared_binding_count);
    let gpu_idx_by_binding = &mut scratch.gpu_idx_by_binding;
    let output_idx_by_binding = &mut scratch.output_idx_by_binding;
    for (idx, output) in request.output_bindings.iter().enumerate() {
        output_idx_by_binding.push(output.binding, idx)?;
    }
    // P0 #9: scratch.clear_requests is a thread-local Vec that retains
    // capacity across dispatches. Programs with > 8 buffers no longer pay a
    // fresh heap allocation on the spill path.
    let clear_requests = &mut scratch.clear_requests;

    for info in request.buffer_bindings.iter() {
        if info.kind == vyre_foundation::ir::MemoryKind::Shared {
            continue;
        }
        let input_idx = input_idx_by_binding
            .get(info.binding)
            .unwrap_or(input_slot_count);
        let data = request.inputs.get(input_idx).copied();

        let (buf, logical_size): (GpuBufferHandle, u64) = if info.internal_trap {
            let size = u64::from(TRAP_SIDECAR_WORDS) * 4;
            let b = pool
                .acquire(
                    size,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                )
                .map_err(pool_backend_error)?;
            clear_requests.push((info.binding, 0, size));
            (b, size)
        } else if info.is_output {
            let output_idx = output_idx_by_binding.get(info.binding).ok_or_else(|| {
                BackendError::new(format!(
                    "missing output layout metadata for binding {}. Fix: keep writable BufferDecl metadata synchronized during dispatch setup.",
                    info.binding
                ))
            })?;
            let output = request.output_bindings.get(output_idx).ok_or_else(|| {
                BackendError::new(format!(
                    "output layout index {output_idx} for binding {} is out of bounds. Fix: keep output binding lookup synchronized with output metadata.",
                    info.binding
                ))
            })?;
            let output_bytes = output.word_count.checked_mul(4).ok_or_else(|| {
                BackendError::new(format!(
                    "output buffer `{}` size overflows usize. Fix: reduce its element count.",
                    output.name
                ))
            })?;
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::INDIRECT;
            let output_bytes_u64 =
                WGPU_NUMERIC.usize_to_u64(output_bytes, "output allocation bytes")?;
            let b = pool
                .acquire(output_bytes_u64, usage)
                .map_err(pool_backend_error)?;
            if info.preserve_input_contents {
                if let Some(bytes) = data {
                    if let Some((offset, len)) =
                        write_padded_input(queue, b.buffer(), bytes, output_bytes)?
                    {
                        clear_requests.push((info.binding, offset, len));
                    }
                } else {
                    clear_requests.push((info.binding, 0, output_bytes_u64));
                }
            }
            (b, output_bytes_u64)
        } else {
            let element_size = element_size_bytes(&info.element)?;
            let declared_size = if info.count > 0 {
                (usize::try_from(info.count).map_err(|_| {
                    BackendError::new(format!(
                        "buffer `{}` element count cannot fit host usize. Fix: reduce buffer count or shard the binding.",
                        info.name
                    ))
                })?)
                    .checked_mul(element_size)
                    .ok_or_else(|| {
                        BackendError::new(format!(
                            "buffer `{}` declared size overflows usize. Fix: reduce buffer count.",
                            info.name
                        ))
                    })?
            } else {
                0
            };
            let (size, contents): (usize, Option<&[u8]>) = match (declared_size, data) {
                (d, Some(bytes)) if d > 0 => (d.max(bytes.len()), Some(bytes)),
                (0, Some(bytes)) => (bytes.len(), Some(bytes)),
                (_, None) => {
                    return Err(BackendError::new(format!(
                        "input binding {} (`{}`) has no host input bytes. Fix: pass input slices matching non-output BufferDecl order; only internal traps and pure outputs may be backend-allocated empty.",
                        info.binding, info.name
                    )));
                }
                _ => {
                    return Err(BackendError::new(
                        "unexpected (declared_size, data) combination. Fix: ensure buffer has either a declared count or input data.",
                    ));
                }
            };

            // wgpu requires buffer sizes to be a multiple of 4 for some usages.
            let size = padded_wgpu_usize(size, "record-and-readback input allocation bytes")?;

            let usage = match info.kind {
                vyre_foundation::ir::MemoryKind::Readonly
                | vyre_foundation::ir::MemoryKind::Global => {
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::INDIRECT
                }
                vyre_foundation::ir::MemoryKind::Uniform
                | vyre_foundation::ir::MemoryKind::Push => {
                    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                }
                vyre_foundation::ir::MemoryKind::Shared => {
                    return Err(BackendError::new(format!(
                        "buffer `{}` reached wgpu allocation with MemoryKind::Shared after filtering. Fix: this is an internal invariant violation; report as a bug.",
                        info.name
                    )));
                }
                vyre_foundation::ir::MemoryKind::Local => {
                    return Err(BackendError::new(format!(
                        "buffer `{}` reached wgpu allocation with MemoryKind::Local. Fix: lower Local regions into shader function variables before dispatch.",
                        info.name
                    )));
                }
                _ => {
                    return Err(BackendError::new(format!(
                        "buffer `{}` uses an unknown future MemoryKind in wgpu allocation. Fix: update vyre-wgpu before dispatching this Program.",
                        info.name
                    )));
                }
            };

            let size_u64 = WGPU_NUMERIC.usize_to_u64(size, "input allocation bytes")?;
            let b = pool.acquire(size_u64, usage).map_err(pool_backend_error)?;
            if let Some(c) = contents {
                if let Some((offset, len)) = write_padded_input(queue, b.buffer(), c, size)? {
                    clear_requests.push((info.binding, offset, len));
                }
            } else {
                clear_requests.push((info.binding, 0, size_u64));
            }
            (b, size_u64)
        };

        let idx = gpu_buffers.len();
        gpu_buffers.push((info.binding, buf, logical_size));
        gpu_idx_by_binding.push(info.binding, idx)?;
    }
    let host_upload_us = u64::try_from(host_upload_started.elapsed().as_micros()).map_err(|source| {
        BackendError::new(format!(
            "host upload elapsed time cannot fit u64 microseconds: {source}. Fix: split or timeout the dispatch before telemetry overflows."
        ))
    })?;

    let bind_groups = bind_groups::build_bind_groups(
        device,
        &request,
        &gpu_buffers,
        gpu_idx_by_binding,
        &mut scratch.bind_group_buffer_ids,
        &mut scratch.bind_group_bound_indices,
    )?;

    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(request.labels.encoder),
    });
    let timestamp_recorder = TimestampRecorder::new(
        device,
        queue,
        pool,
        request.timestamp_profile,
        host_upload_us,
    )?;

    clears::record_buffer_clears(
        &mut encoder,
        &request,
        &gpu_buffers,
        gpu_idx_by_binding,
        clear_requests,
    )?;

    {
        let timestamp_writes =
            timestamp_recorder
                .as_ref()
                .map(|recorder| wgpu::ComputePassTimestampWrites {
                    query_set: &recorder.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(request.labels.compute),
            timestamp_writes,
        });
        pass.set_pipeline(request.pipeline);
        for (i, bg) in bind_groups.iter().enumerate() {
            let bind_group_index = u32::try_from(i).map_err(|_| {
                BackendError::new("bind group index exceeds u32::MAX. Fix: reduce bind group fanout before WGPU dispatch.")
            })?;
            pass.set_bind_group(bind_group_index, bg.as_ref(), &[]);
        }

        let indirect_dispatch_buffer = if let Some(indirect) = request.indirect {
            let indirect_binding = request
                    .buffer_bindings
                    .iter()
                    .find(|b| b.name.as_ref() == indirect.count_buffer.as_str())
                    .map(|b| b.binding)
                    .ok_or_else(|| {
                        BackendError::new(format!(
                            "indirect dispatch count buffer `{}` not found in program bindings. Fix: declare the buffer in the Program.",
                            indirect.count_buffer
                        ))
                    })?;
            Some(
                gpu_idx_by_binding
                    .get(indirect_binding)
                    .and_then(|idx| gpu_buffers.get(idx))
                    .map(|(_, buf, _)| buf)
                    .ok_or_else(|| {
                        BackendError::new(format!(
                            "indirect dispatch count buffer `{}` was not allocated. Fix: ensure the buffer has a declared count or input data.",
                            indirect.count_buffer
                        ))
                    })?
                    .buffer(),
            )
        } else {
            None
        };

        for _ in 0..request.iterations.max(1) {
            if let (Some(indirect), Some(indirect_buffer)) =
                (request.indirect, indirect_dispatch_buffer)
            {
                pass.dispatch_workgroups_indirect(indirect_buffer, indirect.count_offset);
            } else {
                pass.dispatch_workgroups(
                    request.workgroup_count[0],
                    request.workgroup_count[1],
                    request.workgroup_count[2],
                );
            }
        }
    }

    if let Some(recorder) = &timestamp_recorder {
        encoder.write_timestamp(&recorder.query_set, 2);
    }

    let output_count = request.output_bindings.len();
    let readback_buffers = staging::record_readback_copies(
        device,
        pool,
        &mut encoder,
        &request,
        &gpu_buffers,
        gpu_idx_by_binding,
    )?;

    if let Some(recorder) = &timestamp_recorder {
        encoder.write_timestamp(&recorder.query_set, 3);
        recorder.resolve(&mut encoder)?;
    }

    let command_buffer = encoder.finish();
    if let Some(error) = crate::runtime::device::pop_error_scope_now(device).map_err(|message| {
        BackendError::DispatchFailed {
            code: None,
            message: format!(
                "wgpu command-recording validation did not complete without a host wait: {message}"
            ),
        }
    })? {
        return Err(BackendError::DispatchFailed {
            code: None,
            message: format!(
                "wgpu rejected command recording: {error}. Fix: verify bind groups, adapter limits, dispatch dimensions, and copy ranges before submitting."
            ),
        });
    }

    Ok(RecordedDispatch {
        device_queue: Arc::clone(request.device_queue),
        command_buffer: Some(command_buffer),
        _gpu_buffers: gpu_buffers,
        _bind_groups: bind_groups,
        readback_buffers,
        output_count,
        output_bindings: Arc::clone(&request.output_bindings),
        trap_tags: Arc::from(request.trap_tags),
        timestamp_recorder,
    })
}

/// Re-lay-out every runtime-sized writable binding from the bytes supplied now.
///
/// A writable buffer declared without `.with_count(n)` has `count == 0`, which
/// the IR defines as runtime-sized: its element count is whatever the caller
/// passed for it. `vyre_driver::dynamic_element_count_from_bytes` is the one rule
/// for deriving that count, and the CPU reference and CUDA already size such a
/// buffer through it, so applying it here is what puts WGPU on the same answer.
///
/// Bindings with a static count are copied through untouched, so a counted
/// declaration keeps the layout the pipeline compiled for it.
///
/// # Errors
///
/// Returns when a runtime-sized writable binding has no host input to take its
/// size from. That is the request the backend genuinely cannot honor, and it is
/// refused here by name rather than answered with a zero-length buffer.
fn resolve_runtime_sized_outputs(
    output_bindings: &Arc<[OutputBindingLayout]>,
    buffer_bindings: &[BufferBindingInfo],
    inputs: &[&[u8]],
    input_idx_by_binding: &binding_lookup::BindingLookup,
) -> Result<Arc<[OutputBindingLayout]>, BackendError> {
    let mut resolved = Vec::new();
    vyre_driver::allocation::try_reserve_vec_to_capacity(&mut resolved, output_bindings.len())
        .map_err(|error| {
            BackendError::new(format!(
                "runtime-sized output resolution could not reserve {} layout slot(s): {error}. Fix: reduce the program's writable binding fanout.",
                output_bindings.len()
            ))
        })?;
    for layout in output_bindings.iter() {
        let Some(info) = buffer_bindings
            .iter()
            .find(|info| info.binding == layout.binding)
        else {
            resolved.push(layout.clone());
            continue;
        };
        if info.count != 0 || info.internal_trap {
            resolved.push(layout.clone());
            continue;
        }
        let bytes = input_idx_by_binding
            .get(info.binding)
            .and_then(|index| inputs.get(index).copied())
            .ok_or_else(|| {
                BackendError::new(format!(
                    "writable buffer `{}` (binding {}) is runtime-sized, declared without .with_count(n), but this dispatch supplied no input bytes for it, so its readback size is unknown. Fix: declare it with .with_count(n), or pass one input slice for it in BufferDecl order.",
                    info.name, info.binding
                ))
            })?;
        let count = vyre_driver::dynamic_element_count_from_bytes(&info.element, bytes.len())
            .ok_or_else(|| {
                BackendError::new(format!(
                    "writable buffer `{}` (binding {}) is runtime-sized and was supplied {} byte(s), which do not resolve to a whole element count for `{}`. Fix: declare it with .with_count(n), or supply a whole number of elements.",
                    info.name,
                    info.binding,
                    bytes.len(),
                    info.element
                ))
            })?;
        resolved.push(vyre_driver::output_binding_layout_parts(
            info.binding,
            &info.name,
            &info.element,
            count,
            None,
        )?);
    }
    Ok(Arc::from(resolved))
}

fn padded_wgpu_usize(size: usize, label: &'static str) -> Result<usize, BackendError> {
    crate::numeric::WGPU_NUMERIC.align_up_usize(size, 4, 4, label)
}

/// Record compute work, submit it, and return trimmed readback bytes.
///
/// # Errors
///
/// Returns a backend error when buffer sizing, bind-group construction, GPU
/// submission, or readback mapping fails.
pub(crate) fn record_and_readback(
    request: RecordAndReadback<'_>,
) -> Result<vyre_driver::OutputBuffers, BackendError> {
    record_and_submit_async(request)?.await_result()
}
