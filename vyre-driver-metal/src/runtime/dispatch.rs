use std::collections::BTreeMap;
use std::time::Instant;

use metal::{
    Buffer, CommandBuffer, ComputePipelineState, Device, MTLCommandBufferStatus, MTLSize,
    NSUInteger,
};
use vyre_driver::{
    dispatch_element_count_for_program, enforce_actual_output_budget,
    infer_dispatch_grid_for_count, output_binding_layouts, sealed, BackendError, BindingPlan,
    DispatchConfig, OutputBindingLayout, PendingDispatch, TimedDispatchResult,
};
use vyre_foundation::ir::Program;

use super::buffer_plan::{allocation_len_for_output, output_layout_map, PlannedBuffer};
use super::metrics::{elapsed_ns, record_output_readback_metrics, MetalMetricCounters};
use super::resident::{checked_ns_uint, new_host_input_buffer};
use crate::METAL_BACKEND_ID;

pub(super) struct MetalDispatchResult {
    pub(super) outputs: Vec<Vec<u8>>,
    pub(super) enqueue_ns: u64,
    pub(super) wait_ns: u64,
}

pub(super) struct MetalPendingCommand {
    command_buffer: Option<CommandBuffer>,
    buffers: Option<Vec<PlannedBuffer>>,
    output_by_binding: BTreeMap<u32, OutputBindingLayout>,
    config: DispatchConfig,
    enqueue_ns: u64,
    _sizes_buffer: Option<(u8, Buffer)>,
    _pipeline: ComputePipelineState,
}

impl MetalPendingCommand {
    pub(super) fn is_ready(&self) -> bool {
        self.command_buffer.as_ref().is_none_or(|command_buffer| {
            matches!(
                command_buffer.status(),
                MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
            )
        })
    }

    fn wait_and_validate(&mut self) -> Result<u64, BackendError> {
        let Some(command_buffer) = self.command_buffer.as_ref() else {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: Metal pending command retirement was attempted more than once."
                    .to_string(),
            });
        };
        let wait_start = Instant::now();
        command_buffer.wait_until_completed();
        let wait_ns = elapsed_ns(wait_start, "Metal command buffer wait")?;
        let status = command_buffer.status();
        self.command_buffer.take();
        if status != MTLCommandBufferStatus::Completed {
            return Err(BackendError::DispatchFailed {
                code: Some(status as i32),
                message: format!(
                    "Metal command buffer finished with status {status:?} after {wait_ns} ns"
                ),
            });
        }
        Ok(wait_ns)
    }

    pub(super) fn retire(mut self) -> Result<MetalDispatchResult, BackendError> {
        let wait_ns = self.wait_and_validate()?;
        let buffers = self.buffers.as_deref().unwrap_or_default();
        let outputs = collect_outputs(buffers, &self.output_by_binding)?;
        enforce_actual_output_budget(&self.config, &outputs)?;
        self.buffers.take();
        Ok(MetalDispatchResult {
            outputs,
            enqueue_ns: self.enqueue_ns,
            wait_ns,
        })
    }
}

impl Drop for MetalPendingCommand {
    fn drop(&mut self) {
        if let Some(command_buffer) = self.command_buffer.take() {
            command_buffer.wait_until_completed();
        }
    }
}

pub(super) struct MetalPendingDispatch {
    pub(super) command: MetalPendingCommand,
    pub(super) metrics: MetalMetricCounters,
    pub(super) started: Instant,
}

// SAFETY: Metal command buffers, pipelines, and buffers are Objective-C Metal
// objects designed for cross-thread completion and status observation. The
// pending handle never shares an encoder and owns all resources until retirement.
unsafe impl Send for MetalPendingDispatch {}
// SAFETY: `is_ready` only reads command-buffer status. Retirement consumes the
// handle, so no mutable buffer access can race another pending-handle operation.
unsafe impl Sync for MetalPendingDispatch {}

impl sealed::Sealed for MetalPendingDispatch {}

impl MetalPendingDispatch {
    pub(super) fn retire_timed(self) -> Result<TimedDispatchResult, BackendError> {
        let Self {
            command,
            metrics,
            started,
        } = self;
        let result = command.retire()?;
        record_output_readback_metrics(&metrics, &result.outputs);
        Ok(TimedDispatchResult {
            outputs: result.outputs,
            wall_ns: elapsed_ns(started, "Metal resident timed dispatch")?,
            device_ns: None,
            enqueue_ns: Some(result.enqueue_ns),
            wait_ns: Some(result.wait_ns),
        })
    }
}

impl PendingDispatch for MetalPendingDispatch {
    fn is_ready(&self) -> bool {
        self.command.is_ready()
    }

    fn await_result(self: Box<Self>) -> Result<Vec<Vec<u8>>, BackendError> {
        Ok((*self).retire_timed()?.outputs)
    }

    fn await_timed_result(self: Box<Self>) -> Result<TimedDispatchResult, BackendError> {
        (*self).retire_timed()
    }
}

pub(super) fn dispatch_planned_buffers_with_queue(
    device: &Device,
    queue: &metal::CommandQueue,
    program: &Program,
    binding_plan: &BindingPlan,
    config: &DispatchConfig,
    artifact: &vyre_emit_metal::MetalArtifact,
    pipeline: &metal::ComputePipelineState,
    buffers: Vec<PlannedBuffer>,
) -> Result<MetalDispatchResult, BackendError> {
    let output_by_binding = output_layout_map(output_binding_layouts(program)?)?;
    submit_planned_buffers_with_queue(
        device,
        queue,
        program,
        binding_plan,
        config,
        artifact,
        pipeline,
        buffers,
        output_by_binding,
    )?
    .retire()
}

pub(super) fn submit_planned_buffers_with_queue(
    device: &Device,
    queue: &metal::CommandQueue,
    program: &Program,
    binding_plan: &BindingPlan,
    config: &DispatchConfig,
    artifact: &vyre_emit_metal::MetalArtifact,
    pipeline: &metal::ComputePipelineState,
    buffers: Vec<PlannedBuffer>,
    output_by_binding: BTreeMap<u32, OutputBindingLayout>,
) -> Result<MetalPendingCommand, BackendError> {
    let sizes_buffer = artifact
        .sizes_buffer_index
        .map(|slot| new_buffer_sizes_buffer(device, slot, &artifact.bindings, &buffers))
        .transpose()?;
    let mut threadgroup_memory_lengths = Vec::new();
    threadgroup_memory_lengths
        .try_reserve(artifact.threadgroup_memories.len())
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal threadgroup memory length list could not reserve {} entries: {error}. Split the workgroup allocations.",
                artifact.threadgroup_memories.len()
            ),
        })?;
    for memory in &artifact.threadgroup_memories {
        threadgroup_memory_lengths.push((
            memory.threadgroup_index,
            checked_ns_uint(
                usize::try_from(memory.aligned_byte_length).map_err(|error| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: Metal threadgroup memory `{}` length {} cannot fit usize: {error}. Split the workgroup allocation.",
                            memory.name, memory.aligned_byte_length
                        ),
                    }
                })?,
                "Metal threadgroup memory length",
            )?,
        ));
    }
    let workgroup_size = config.workgroup_override.unwrap_or(artifact.workgroup_size);
    let threads_per_group = metal_threadgroup_size(workgroup_size)?;
    let workgroups = match config.grid_override {
        Some(grid) => grid,
        None => infer_dispatch_grid_for_count(
            dispatch_element_count_for_program(program, &binding_plan.bindings),
            workgroup_size,
        )?,
    };
    let threadgroups = metal_grid_size(workgroups)?;

    let enqueue_start = Instant::now();
    let command_buffer = queue.new_command_buffer().to_owned();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for planned in &buffers {
        encoder.set_buffer(planned.metal_slot.into(), Some(&planned.buffer), 0);
    }
    if let Some((slot, buffer)) = sizes_buffer.as_ref() {
        encoder.set_buffer((*slot).into(), Some(buffer), 0);
    }
    for (index, length) in threadgroup_memory_lengths {
        encoder.set_threadgroup_memory_length(index.into(), length);
    }

    encoder.dispatch_thread_groups(threadgroups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    let enqueue_ns = match elapsed_ns(enqueue_start, "Metal command buffer enqueue") {
        Ok(enqueue_ns) => enqueue_ns,
        Err(error) => {
            command_buffer.wait_until_completed();
            return Err(error);
        }
    };
    Ok(MetalPendingCommand {
        command_buffer: Some(command_buffer),
        buffers: Some(buffers),
        output_by_binding,
        config: config.clone(),
        enqueue_ns,
        _sizes_buffer: sizes_buffer,
        _pipeline: pipeline.to_owned(),
    })
}

fn collect_outputs(
    buffers: &[PlannedBuffer],
    output_by_binding: &BTreeMap<u32, OutputBindingLayout>,
) -> Result<Vec<Vec<u8>>, BackendError> {
    let mut outputs = Vec::new();
    outputs
        .try_reserve(output_by_binding.len())
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal output collection could not reserve {} output slot(s): {error}. Split the Program outputs.",
                output_by_binding.len()
            ),
        })?;
    for (binding, layout) in output_by_binding {
        let buffer = buffers
            .iter()
            .find(|planned| planned.binding == *binding)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal output binding {binding} (`{}`) has no allocated buffer.",
                    layout.name
                ),
            })?;
        let allocation_len = allocation_len_for_output(layout)?;
        let copy_start = layout.layout.copy_offset;
        let copy_end = copy_start
            .checked_add(layout.layout.copy_size)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal output binding {binding} copy range overflows usize. Narrow output_byte_range."
                ),
            })?;
        if copy_end > allocation_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal output binding {binding} copy range {copy_start}..{copy_end} exceeds allocation length {allocation_len}."
                ),
            });
        }
        // SAFETY: the source pointer belongs to a live Metal shared buffer,
        // `allocation_len` is the host allocation length used for the buffer,
        // and range checks above prove the slice window is in bounds.
        let source = unsafe {
            std::slice::from_raw_parts(buffer.buffer.contents().cast::<u8>(), allocation_len)
        };
        let copied = &source[copy_start..copy_end];
        let trim_start = layout.layout.trim_start;
        let trim_end = trim_start
            .checked_add(layout.layout.read_size)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal output binding {binding} trim range overflows usize. Narrow output_byte_range."
                ),
            })?;
        if trim_end > copied.len() {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal output binding {binding} trim range {trim_start}..{trim_end} exceeds copied length {}.",
                    copied.len()
                ),
            });
        }
        outputs.push(copied[trim_start..trim_end].to_vec());
    }
    Ok(outputs)
}

fn new_buffer_sizes_buffer(
    device: &Device,
    slot: u8,
    bindings: &[vyre_emit_metal::MetalBindingMetadata],
    buffers: &[PlannedBuffer],
) -> Result<(u8, Buffer), BackendError> {
    let sidecar_words = bindings
        .iter()
        .map(|binding| usize::from(binding.metal_buffer_index) + 1)
        .max()
        .unwrap_or(1);
    let mut sizes = vec![0u32; sidecar_words];
    sizes.try_reserve(0).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal buffer-size sidecar could not reserve {sidecar_words} length word(s): {error}. Split the Program bindings.",
            ),
        }
    })?;
    for binding in bindings {
        let planned = buffers
            .iter()
            .find(|planned| planned.binding == binding.slot)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal buffer-size sidecar could not find planned binding {} (`{}`). Rebuild BindingPlan before dispatch.",
                    binding.slot, binding.name
                ),
            })?;
        let byte_len =
            u32::try_from(planned.buffer.length()).map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal binding {} (`{}`) length {} cannot fit the u32 _buffer_sizes ABI: {error}. Split the dispatch buffer.",
                    binding.slot,
                    binding.name,
                    planned.buffer.length()
                ),
            })?;
        let Some(size_slot) = sizes.get_mut(usize::from(binding.metal_buffer_index)) else {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal buffer-size sidecar index {} for binding {} (`{}`) exceeds the packed sidecar word count {sidecar_words}. Rebuild the Metal artifact binding map.",
                    binding.metal_buffer_index, binding.slot, binding.name
                ),
            });
        };
        *size_slot = byte_len;
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve(sizes.len() * std::mem::size_of::<u32>())
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal buffer-size sidecar byte packing could not reserve {} byte(s): {error}. Split the Program bindings.",
                sizes.len() * std::mem::size_of::<u32>()
            ),
        })?;
    for size in sizes {
        bytes.extend_from_slice(&size.to_le_bytes());
    }
    Ok((slot, new_host_input_buffer(device, &bytes)?))
}

pub(super) fn validate_metal_dispatch_config(
    config: &DispatchConfig,
    cooperative_feature: &'static str,
    repeated_feature: &'static str,
    zero_iteration_context: &'static str,
) -> Result<(), BackendError> {
    if config.cooperative {
        return Err(BackendError::UnsupportedFeature {
            name: cooperative_feature.to_string(),
            backend: METAL_BACKEND_ID.to_string(),
        });
    }
    if matches!(config.fixpoint_iterations, Some(0)) {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {zero_iteration_context} received fixpoint_iterations=0; use None or a positive iteration count."
            ),
        });
    }
    if let Some(iterations) = config.fixpoint_iterations {
        if iterations != 1 {
            return Err(BackendError::UnsupportedFeature {
                name: format!("{repeated_feature} with {iterations} iterations"),
                backend: METAL_BACKEND_ID.to_string(),
            });
        }
    }
    Ok(())
}

fn metal_threadgroup_size(workgroup_size: [u32; 3]) -> Result<MTLSize, BackendError> {
    Ok(MTLSize::new(
        checked_nonzero_dimension(workgroup_size[0], "workgroup x")?,
        checked_nonzero_dimension(workgroup_size[1], "workgroup y")?,
        checked_nonzero_dimension(workgroup_size[2], "workgroup z")?,
    ))
}

fn metal_grid_size(workgroups: [u32; 3]) -> Result<MTLSize, BackendError> {
    Ok(MTLSize::new(
        checked_nonzero_dimension(workgroups[0], "workgroups x")?,
        checked_nonzero_dimension(workgroups[1], "workgroups y")?,
        checked_nonzero_dimension(workgroups[2], "workgroups z")?,
    ))
}

fn checked_nonzero_dimension(value: u32, field: &'static str) -> Result<NSUInteger, BackendError> {
    if value == 0 {
        return Err(BackendError::InvalidProgram {
            fix: format!("Fix: Metal dispatch {field} dimension must be nonzero."),
        });
    }
    Ok(value.into())
}
