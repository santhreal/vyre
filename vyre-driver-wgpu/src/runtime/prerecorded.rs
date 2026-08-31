//! Pre-recorded persistent dispatch command buffers.

use std::sync::{Arc, Mutex};

use smallvec::SmallVec;
use vyre_driver::BackendError;

use crate::buffer::GpuBufferHandle;
use crate::pipeline::binding::{accepts_params_handle, clear_outputs_for_bound, validate_handle};
use crate::pipeline::persistent::DispatchNaming;
use crate::pipeline::{BufferBindingInfo, WgpuPipeline};

fn padded_prerecorded(size: u64, label: &'static str) -> Result<u64, BackendError> {
    crate::allocation::padded_wgpu_u64(size, label, "split the pre-recorded dispatch buffer")
}

const PRERECORDED_DISPATCH: DispatchNaming = DispatchNaming {
    path: "pre-recorded",
    action: "recording",
    bind_group_label: "vyre pre-recorded persistent bind group",
    key_size_label: "pre-recorded bind-group cache key byte length",
    bind_size_label: "pre-recorded bind-group binding size",
    pad: padded_prerecorded,
    compute_label: "vyre pre-recorded persistent compute",
    index_overflow: "pre-recorded bind group index exceeds u32::MAX. Fix: reduce bind group fanout before recording.",
};

/// GPU work recorded ahead of submission for encoder-free dispatch handoff.
///
/// `wgpu::CommandBuffer` is single-submit. This type prevents the raw wgpu
/// panic by consuming the stored command buffer on the first replay and
/// returning a structured error on repeated replay attempts.
pub struct PrerecordedDispatch {
    /// Pre-recorded command buffer.
    pub cb: Mutex<Option<wgpu::CommandBuffer>>,
    /// Bind groups captured by the command buffer.
    pub bind_groups: Vec<Arc<wgpu::BindGroup>>,
    /// Buffer handles kept alive for the lifetime of the recorded commands.
    pub handles: Vec<GpuBufferHandle>,
    /// Output handles recorded for terminal readback by tests and callers.
    pub output_handles: Vec<GpuBufferHandle>,
    /// Device used to record this dispatch.
    pub device: wgpu::Device,
    /// Queue paired with `device`.
    pub queue: wgpu::Queue,
}

impl PrerecordedDispatch {
    /// Submit the pre-recorded command buffer to `queue`.
    ///
    /// # Errors
    ///
    /// Returns a backend error when this command buffer was already submitted.
    pub fn replay(&self, queue: &wgpu::Queue) -> Result<wgpu::SubmissionIndex, BackendError> {
        let command_buffer = self
            .cb
            .lock()
            .map_err(|source| {
                BackendError::new(format!(
                    "pre-recorded dispatch mutex poisoned: {source}. Fix: drop this dispatch and record a fresh command buffer."
                ))
            })?
            .take()
            .ok_or_else(|| {
                BackendError::new(
                    "pre-recorded wgpu command buffer was already submitted. Fix: record a new PrerecordedDispatch for each replay slot; wgpu command buffers are single-submit.",
                )
            })?;
        Ok(queue.submit(std::iter::once(command_buffer)))
    }

    /// Read one recorded output buffer into a byte vector.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the output index is invalid or mapping
    /// fails.
    pub fn read_output(&self, index: usize) -> Result<Vec<u8>, BackendError> {
        let output = self.output_handles.get(index).ok_or_else(|| {
            BackendError::new(format!(
                "pre-recorded output index {index} is out of bounds for {} outputs. Fix: request an output produced by this dispatch.",
                self.output_handles.len()
            ))
        })?;
        let byte_capacity = usize::try_from(output.byte_len()).map_err(|error| {
            BackendError::new(format!(
                "pre-recorded output byte length {} does not fit usize on this host: {error}. Fix: shard the GPU output before readback.",
                output.byte_len()
            ))
        })?;
        let mut bytes = Vec::new();
        vyre_driver::allocation::try_reserve_vec_to_capacity(&mut bytes, byte_capacity).map_err(
            |source| {
                BackendError::new(format!(
                    "pre-recorded output readback could not reserve {byte_capacity} byte(s): {source}. Fix: shard the GPU output before readback."
                ))
            },
        )?;
        output.readback(&self.device, &self.queue, &mut bytes)?;
        Ok(bytes)
    }

    /// Read one recorded output buffer into caller-owned storage.
    ///
    /// Clears `out`, then reuses its allocation.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the output index is invalid or mapping
    /// fails.
    pub fn read_output_into(&self, index: usize, out: &mut Vec<u8>) -> Result<(), BackendError> {
        let output = self.output_handles.get(index).ok_or_else(|| {
            BackendError::new(format!(
                "pre-recorded output index {index} is out of bounds for {} outputs. Fix: request an output produced by this dispatch.",
                self.output_handles.len()
            ))
        })?;
        output.readback(&self.device, &self.queue, out)
    }
}

impl WgpuPipeline {
    /// Record a persistent dispatch once so later submission bypasses encoder
    /// construction, output clears, bind-group lookup, and compute-pass setup.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the handles do not match the compiled
    /// program's binding contract or command recording fails.
    pub fn prerecord_persistent_dispatch(
        &self,
        inputs: &[GpuBufferHandle],
        outputs: &[GpuBufferHandle],
        params: Option<&GpuBufferHandle>,
        workgroups: [u32; 3],
    ) -> Result<PrerecordedDispatch, BackendError> {
        let (device, queue) = &*self.device_queue;
        let bound = bind_handles(&self.buffer_bindings, inputs, outputs, params)?;
        let bind_groups = self.cached_bind_groups_named(device, &bound, &PRERECORDED_DISPATCH)?;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vyre pre-recorded persistent dispatch"),
        });
        clear_outputs_for_bound("pre-recorded", &mut encoder, &bound, |binding| {
            self.output_binding(binding).cloned()
        })?;
        self.record_compute_pass(
            &mut encoder,
            &bind_groups,
            &bound,
            workgroups,
            None,
            &PRERECORDED_DISPATCH,
        )?;

        let mut handles = Vec::new();
        vyre_driver::allocation::try_reserve_vec_to_capacity(&mut handles, bound.len()).map_err(
            |source| {
                BackendError::new(format!(
                    "pre-recorded dispatch handle retention could not reserve {} handle slot(s): {source}. Fix: split bound resources before recording.",
                    bound.len()
                ))
            },
        )?;
        handles.extend(bound.iter().map(|(_, handle)| (*handle).clone()));
        let mut output_handles = Vec::new();
        vyre_driver::allocation::try_reserve_vec_to_capacity(&mut output_handles, outputs.len())
            .map_err(|source| {
                BackendError::new(format!(
                    "pre-recorded output handle retention could not reserve {} output slot(s): {source}. Fix: split output resources before recording.",
                    outputs.len()
                ))
            })?;
        output_handles.extend(outputs.iter().cloned());
        Ok(PrerecordedDispatch {
            cb: Mutex::new(Some(encoder.finish())),
            bind_groups,
            handles,
            output_handles,
            device: device.clone(),
            queue: queue.clone(),
        })
    }

    /// Upload borrowed host inputs, allocate output handles, and pre-record
    /// one persistent dispatch using this pipeline's device.
    ///
    /// # Errors
    ///
    /// Returns a backend error when upload, output allocation, or command
    /// recording fails.
    pub fn prerecord_borrowed_dispatch(
        &self,
        inputs: &[&[u8]],
        workgroups: [u32; 3],
    ) -> Result<PrerecordedDispatch, BackendError> {
        let (input_handles, output_handles) = self.legacy_handles_from_inputs(inputs)?;
        self.prerecord_persistent_dispatch(&input_handles, &output_handles, None, workgroups)
    }
}

fn bind_handles<'a>(
    bindings: &'a [BufferBindingInfo],
    inputs: &'a [GpuBufferHandle],
    outputs: &'a [GpuBufferHandle],
    params: Option<&'a GpuBufferHandle>,
) -> Result<SmallVec<[(&'a BufferBindingInfo, &'a GpuBufferHandle); 8]>, BackendError> {
    let mut input_index = 0usize;
    let mut output_index = 0usize;
    let mut params_used = false;
    let mut bound = SmallVec::new();
    vyre_foundation::allocation::try_reserve_smallvec_to_capacity(&mut bound, bindings.len())
        .map_err(|source| {
            BackendError::new(format!(
                "pre-recorded binding resolution could not reserve {} binding slot(s): {source}. Fix: split bound resources before recording.",
                bindings.len()
            ))
        })?;
    for info in bindings {
        if info.kind == vyre_foundation::ir::MemoryKind::Shared {
            continue;
        }
        let handle = if info.is_output {
            let handle = outputs.get(output_index).ok_or_else(|| {
                BackendError::new(format!(
                    "pre-recorded dispatch missing output handle for binding {} (`{}`). Fix: pass one output handle per output BufferDecl.",
                    info.binding, info.name
                ))
            })?;
            output_index += 1;
            handle
        } else if accepts_params_handle(info) && params.is_some() && !params_used {
            params_used = true;
            if let Some(handle) = params {
                handle
            } else {
                return Err(BackendError::new(
                    "pre-recorded dispatch parameter handle disappeared after validation. Fix: retry recording with a stable params handle.",
                ));
            }
        } else {
            let handle = inputs.get(input_index).ok_or_else(|| {
                BackendError::new(format!(
                    "pre-recorded dispatch missing input handle for binding {} (`{}`). Fix: pass non-output handles in BufferDecl order.",
                    info.binding, info.name
                ))
            })?;
            input_index += 1;
            handle
        };
        validate_handle("pre-recorded", info, handle)?;
        bound.push((info, handle));
    }
    if input_index != inputs.len() {
        return Err(BackendError::new(format!(
            "pre-recorded dispatch received {} input handles but consumed {input_index}. Fix: pass handles matching non-output BufferDecl order.",
            inputs.len()
        )));
    }
    if output_index != outputs.len() {
        return Err(BackendError::new(format!(
            "pre-recorded dispatch received {} output handles but consumed {output_index}. Fix: pass handles matching output BufferDecl order.",
            outputs.len()
        )));
    }
    Ok(bound)
}
