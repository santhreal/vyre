//! `CompiledPipeline` implementation for WGPU pipeline dispatch.
//!
//! The parent `pipeline` module owns compilation and metadata assembly. This
//! module owns the trait entrypoints that turn caller inputs into persistent
//! GPU handles, execute the compiled compute pipeline, and read back outputs.

use std::sync::Arc;
use std::time::{Duration, Instant};

use smallvec::SmallVec;
use vyre_driver::program_walks::enforce_actual_output_budget;
use vyre_driver::{
    resolve_fixpoint_iterations_usize, BackendError, CompiledPipeline, DispatchConfig,
    OutputBuffers, TimedDispatchResult,
};

use crate::engine::record_and_readback::timestamp::{
    collect_timestamp_profile, PendingTimestampProfile, TimestampRecorder,
};
use crate::numeric::WGPU_NUMERIC;
use crate::pipeline::output_slots::resize_vec_with;
use crate::pipeline::WgpuPipeline;
use crate::staging_reserve::{reserve_pipeline_vec, reserve_smallvec, reserve_vec};

pub(crate) struct WgpuPendingPersistentDispatch {
    pipeline: Arc<WgpuPipeline>,
    resolved: crate::pipeline::persistent_resources::ResolvedPersistentResources,
    output_readbacks: SmallVec<[crate::buffer::PendingGpuBufferReadback; 8]>,
    trap_flag_readback: Option<crate::buffer::PendingGpuBufferReadback>,
    timestamp_profile: Option<PendingTimestampProfile>,
    started: Instant,
    deadline: Option<Instant>,
    timestamp_deadline: Instant,
    enqueue_ns: u64,
    config: DispatchConfig,
}

impl vyre_driver::backend::private::Sealed for WgpuPendingPersistentDispatch {}

impl WgpuPendingPersistentDispatch {
    fn is_ready_inner(&self) -> bool {
        let (device, _) = &*self.pipeline.device_queue;
        let outputs_ready = self
            .output_readbacks
            .iter()
            .all(|pending| pending.is_ready(device));
        let trap_ready = self
            .trap_flag_readback
            .as_ref()
            .map(|pending| pending.is_ready(device))
            .unwrap_or(true);
        let timestamp_ready = self
            .timestamp_profile
            .as_ref()
            .map(PendingTimestampProfile::is_ready)
            .unwrap_or(true);
        outputs_ready && trap_ready && timestamp_ready
    }

    fn retire(self) -> Result<TimedDispatchResult, BackendError> {
        let wait_started = Instant::now();
        let (device, queue) = &*self.pipeline.device_queue;
        if let Some(trap_flag) = self.trap_flag_readback {
            let mut flag = Vec::new();
            trap_flag.await_into(device, self.deadline, &mut flag)?;
            if flag.len() != 4 {
                return Err(BackendError::new(format!(
                    "internal wgpu trap flag readback returned {} bytes but 4 bytes are required. Fix: allocate the trap sidecar as four or more bytes.",
                    flag.len()
                )));
            }
            if u32::from_le_bytes([flag[0], flag[1], flag[2], flag[3]]) != 0 {
                self.pipeline.raise_if_trapped(
                    &self.resolved.inputs,
                    device,
                    queue,
                    self.deadline,
                )?;
            }
        }

        let mut outputs = Vec::new();
        resize_vec_with(
            &mut outputs,
            self.output_readbacks.len(),
            Vec::new,
            "asynchronous persistent output slots",
        )?;
        for ((pending, output), bytes) in self
            .output_readbacks
            .into_iter()
            .zip(self.pipeline.output_bindings.iter())
            .zip(outputs.iter_mut())
        {
            pending.await_into(device, self.deadline, bytes)?;
            if bytes.len() != output.layout.read_size {
                return Err(BackendError::new(format!(
                    "asynchronous persistent readback for `{}` returned {} bytes but {} bytes are required. Fix: keep async output range validation synchronized with OutputLayout.",
                    output.name,
                    bytes.len(),
                    output.layout.read_size
                )));
            }
        }
        enforce_actual_output_budget(&self.config, outputs.as_slice())?;
        let device_ns = collect_timestamp_profile(self.timestamp_profile, self.timestamp_deadline)?
            .map(|profile| profile.dispatch_ns);
        let wait_ns = WGPU_NUMERIC.elapsed_nanos_u64(wait_started, "persistent asynchronous wait")?;
        Ok(TimedDispatchResult {
            outputs,
            wall_ns: WGPU_NUMERIC.elapsed_nanos_u64(self.started, "persistent asynchronous dispatch")?,
            device_ns,
            enqueue_ns: Some(self.enqueue_ns),
            wait_ns: Some(wait_ns),
        })
    }
}

impl vyre_driver::PendingDispatch for WgpuPendingPersistentDispatch {
    fn is_ready(&self) -> bool {
        self.is_ready_inner()
    }

    fn await_result(self: Box<Self>) -> Result<OutputBuffers, BackendError> {
        Ok((*self).retire()?.outputs)
    }

    fn await_timed_result(self: Box<Self>) -> Result<TimedDispatchResult, BackendError> {
        (*self).retire()
    }
}

impl WgpuPipeline {
    pub(crate) fn dispatch_persistent_handles_async(
        self: &Arc<Self>,
        resources: &[vyre_driver::Resource],
        config: &DispatchConfig,
        started: Instant,
    ) -> Result<WgpuPendingPersistentDispatch, BackendError> {
        self.enforce_static_output_budget(config)?;
        let enqueue_started = Instant::now();
        let (device, queue) = &*self.device_queue;
        let deadline = config
            .timeout
            .and_then(|timeout| started.checked_add(timeout));
        let timestamp_deadline =
            deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(30));
        let resolved = self.resolve_persistent_resources(resources, queue)?;
        if resolved.outputs.len() != self.output_bindings.len() {
            return Err(BackendError::new(format!(
                "WGPU asynchronous persistent dispatch resolved {} output handles for {} output layouts. Fix: keep resident resource resolution synchronized with compiled output bindings.",
                resolved.outputs.len(),
                self.output_bindings.len()
            )));
        }
        let item = crate::pipeline::persistent::BorrowedDispatchItem {
            inputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.inputs),
            outputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.outputs),
            params: None,
            workgroups: self.workgroups_for_dispatch(config)?,
        };
        let timestamp_recorder =
            TimestampRecorder::new(device, queue, &self.persistent_pool, true, 0)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vyre asynchronous persistent dispatch"),
        });
        let timestamp_writes =
            timestamp_recorder
                .as_ref()
                .map(|recorder| wgpu::ComputePassTimestampWrites {
                    query_set: &recorder.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        self.record_borrowed_persistent_item_with_timestamps(
            device,
            &mut encoder,
            &item,
            timestamp_writes,
        )?;
        drop(item);
        if let Some(recorder) = &timestamp_recorder {
            encoder.write_timestamp(&recorder.query_set, 2);
            encoder.write_timestamp(&recorder.query_set, 3);
            recorder.resolve(&mut encoder)?;
        }
        queue.submit(std::iter::once(encoder.finish()));

        let pending_after_submit = (|| {
            let timestamp_profile = timestamp_recorder
                .map(TimestampRecorder::map_async)
                .transpose()?;
            let mut output_readbacks = SmallVec::new();
            reserve_smallvec(
                &mut output_readbacks,
                resolved.outputs.len(),
                "asynchronous persistent dispatch",
                "output readback",
                "split the dispatch output set before submission",
            )?;
            for (handle, output) in resolved.outputs.iter().zip(self.output_bindings.iter()) {
                output_readbacks.push(crate::pipeline::output_readback::start_trimmed_output(
                    handle,
                    output,
                    device,
                    &self.staging_pool,
                    queue,
                    "asynchronous persistent output",
                )?);
            }

            let trap_flag_readback = self
                .buffer_bindings
                .iter()
                .filter(|info| {
                    info.kind != vyre_foundation::ir::MemoryKind::Shared && !info.is_output
                })
                .enumerate()
                .find(|(_, info)| info.internal_trap)
                .map(|(index, _)| {
                    resolved
                        .inputs
                        .get(index)
                        .ok_or_else(|| {
                            BackendError::new(
                                "internal wgpu trap buffer was not allocated. Fix: keep trap buffer binding metadata synchronized with resident input resolution.",
                            )
                        })?
                        .readback_range_async(
                            device,
                            Some(&self.staging_pool),
                            queue,
                            0,
                            4,
                        )
                })
                .transpose()?;
            Ok::<_, BackendError>((output_readbacks, trap_flag_readback, timestamp_profile))
        })();
        let (output_readbacks, trap_flag_readback, timestamp_profile) = match pending_after_submit {
            Ok(pending) => pending,
            Err(error) => {
                let fence = queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
                crate::runtime::device::poll_device_wait_for(device, fence)?;
                return Err(error);
            }
        };

        let enqueue_ns =
            match WGPU_NUMERIC.elapsed_nanos_u64(enqueue_started, "persistent asynchronous enqueue") {
                Ok(elapsed) => elapsed,
                Err(error) => {
                    let fence = queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
                    crate::runtime::device::poll_device_wait_for(device, fence)?;
                    return Err(error);
                }
            };
        Ok(WgpuPendingPersistentDispatch {
            pipeline: Arc::clone(self),
            resolved,
            output_readbacks,
            trap_flag_readback,
            timestamp_profile,
            started,
            deadline,
            timestamp_deadline,
            enqueue_ns,
            config: config.clone(),
        })
    }
}

impl CompiledPipeline for WgpuPipeline {
    fn dispatch_persistent_handles(
        &self,
        inputs: &[vyre_driver::Resource],
        config: &DispatchConfig,
    ) -> Result<OutputBuffers, BackendError> {
        let mut outputs = Vec::new();
        reserve_vec(
            &mut outputs,
            self.output_bindings.len(),
            "WGPU pipeline",
            "persistent dispatch output buffers",
            "split the dispatch batch before submission",
        )?;
        self.dispatch_persistent_handles_into(inputs, config, &mut outputs)?;
        enforce_actual_output_budget(config, outputs.as_slice())?;
        Ok(outputs)
    }

    fn dispatch_persistent_handles_into(
        &self,
        inputs: &[vyre_driver::Resource],
        config: &DispatchConfig,
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        self.enforce_static_output_budget(config)?;
        let (device, queue) = &*self.device_queue;
        let workgroup_count = self.workgroups_for_dispatch(config)?;
        let deadline = config
            .timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));
        let resolved = self.resolve_persistent_resources(inputs, queue)?;
        let item = crate::pipeline::persistent::BorrowedDispatchItem {
            inputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.inputs),
            outputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.outputs),
            params: None,
            workgroups: workgroup_count,
        };
        self.dispatch_borrowed_persistent_batched(&[item])?;
        self.raise_if_trapped(&resolved.inputs, device, queue, deadline)?;
        self.readback_persistent_outputs(&resolved.outputs, deadline, outputs)?;
        enforce_actual_output_budget(config, outputs.as_slice())
    }

    fn dispatch_persistent_handles_timed(
        &self,
        inputs: &[vyre_driver::Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        self.enforce_static_output_budget(config)?;
        let started = Instant::now();
        let enqueue_started = Instant::now();
        let (device, queue) = &*self.device_queue;
        let deadline = config
            .timeout
            .and_then(|timeout| started.checked_add(timeout));
        let timestamp_deadline =
            deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(30));
        let resolved = self.resolve_persistent_resources(inputs, queue)?;
        let item = crate::pipeline::persistent::BorrowedDispatchItem {
            inputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.inputs),
            outputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.outputs),
            params: None,
            workgroups: self.workgroups_for_dispatch(config)?,
        };

        let timestamp_recorder =
            TimestampRecorder::new(device, queue, &self.persistent_pool, true, 0)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vyre timed persistent dispatch"),
        });
        let timestamp_writes =
            timestamp_recorder
                .as_ref()
                .map(|recorder| wgpu::ComputePassTimestampWrites {
                    query_set: &recorder.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
        self.record_borrowed_persistent_item_with_timestamps(
            device,
            &mut encoder,
            &item,
            timestamp_writes,
        )?;
        if let Some(recorder) = &timestamp_recorder {
            encoder.write_timestamp(&recorder.query_set, 2);
            encoder.write_timestamp(&recorder.query_set, 3);
            recorder.resolve(&mut encoder)?;
        }
        queue.submit(std::iter::once(encoder.finish()));
        let timestamp_profile = timestamp_recorder
            .map(TimestampRecorder::map_async)
            .transpose()?;
        let enqueue_ns = WGPU_NUMERIC.elapsed_nanos_u64(enqueue_started, "persistent enqueue")?;

        let wait_started = Instant::now();
        self.raise_if_trapped(&resolved.inputs, device, queue, deadline)?;
        let mut outputs = Vec::new();
        self.readback_persistent_outputs(&resolved.outputs, deadline, &mut outputs)?;
        enforce_actual_output_budget(config, outputs.as_slice())?;
        let device_ns = collect_timestamp_profile(timestamp_profile, timestamp_deadline)?
            .map(|profile| profile.dispatch_ns);
        let wait_ns = WGPU_NUMERIC.elapsed_nanos_u64(wait_started, "persistent wait")?;

        Ok(TimedDispatchResult {
            outputs,
            wall_ns: WGPU_NUMERIC.elapsed_nanos_u64(started, "persistent timed dispatch")?,
            device_ns,
            enqueue_ns: Some(enqueue_ns),
            wait_ns: Some(wait_ns),
        })
    }

    fn dispatch_persistent_resource_outputs(
        &self,
        inputs: &[vyre_driver::Resource],
        config: &DispatchConfig,
    ) -> Result<Vec<vyre_driver::Resource>, BackendError> {
        self.enforce_static_output_budget(config)?;
        let (device, queue) = &*self.device_queue;
        let resolved = self.resolve_persistent_resources_for_resource_outputs(inputs, queue)?;
        let item = crate::pipeline::persistent::BorrowedDispatchItem {
            inputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.inputs),
            outputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.outputs),
            params: None,
            workgroups: self.workgroups_for_dispatch(config)?,
        };
        self.dispatch_borrowed_persistent_batched(&[item])?;
        let deadline = config
            .timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));
        self.raise_if_trapped(&resolved.inputs, device, queue, deadline)?;
        Ok(resolved.output_resources.into_iter().collect())
    }

    fn dispatch_persistent_handles_batched(
        &self,
        batches: &[&[vyre_driver::Resource]],
        config: &DispatchConfig,
    ) -> Result<Vec<OutputBuffers>, BackendError> {
        let mut outputs = Vec::new();
        reserve_vec(
            &mut outputs,
            batches.len(),
            "WGPU pipeline",
            "persistent batched dispatch output sets",
            "split the dispatch batch before submission",
        )?;
        self.dispatch_persistent_handles_batched_into(batches, config, &mut outputs)?;
        Ok(outputs)
    }

    fn dispatch_persistent_handles_batched_into(
        &self,
        batches: &[&[vyre_driver::Resource]],
        config: &DispatchConfig,
        batch_outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        if batches.is_empty() {
            batch_outputs.clear();
            return Ok(());
        }
        self.enforce_static_output_budget(config)?;
        let (device, queue) = &*self.device_queue;
        let workgroup_count = self.workgroups_for_dispatch(config)?;
        let deadline = config
            .timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));

        let mut resolved = SmallVec::<[_; 8]>::new();
        reserve_smallvec(
            &mut resolved,
            batches.len(),
            "persistent batched dispatch",
            "resolved resource set",
            "split the persistent dispatch batch before submission",
        )?;
        for batch in batches {
            resolved.push(self.resolve_persistent_resources(batch, queue)?);
        }

        let mut items =
            SmallVec::<[crate::pipeline::persistent::BorrowedDispatchItem<'_>; 8]>::new();
        reserve_smallvec(
            &mut items,
            resolved.len(),
            "persistent batched dispatch",
            "command item",
            "split the persistent dispatch batch before submission",
        )?;
        for item in resolved.iter() {
            items.push(crate::pipeline::persistent::BorrowedDispatchItem {
                inputs: crate::pipeline::persistent::borrowed_handle_refs(&item.inputs),
                outputs: crate::pipeline::persistent::borrowed_handle_refs(&item.outputs),
                params: None,
                workgroups: workgroup_count,
            });
        }

        self.dispatch_borrowed_persistent_batched(&items)?;

        resize_vec_with(
            batch_outputs,
            resolved.len(),
            Vec::new,
            "persistent batched dispatch output slots",
        )?;
        for (item, outputs) in resolved.iter().zip(batch_outputs.iter_mut()) {
            self.raise_if_trapped(&item.inputs, device, queue, deadline)?;
            self.readback_persistent_outputs(&item.outputs, deadline, outputs)?;
            enforce_actual_output_budget(config, outputs.as_slice())?;
        }

        Ok(())
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn dispatch(
        &self,
        inputs: &[Vec<u8>],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let borrowed = vyre_driver::borrowed_input_slices(inputs, "wgpu compiled borrowed input")?;
        self.dispatch_borrowed(&borrowed, config)
    }

    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        let mut outputs = Vec::new();
        reserve_pipeline_vec(
            &mut outputs,
            self.output_bindings.len(),
            "borrowed dispatch output buffers",
        )?;
        self.dispatch_borrowed_into(inputs, config, &mut outputs)?;
        Ok(outputs)
    }

    fn dispatch_borrowed_timed(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        self.enforce_static_output_budget(config)?;
        let started = Instant::now();
        let enqueue_started = Instant::now();
        let iterations = resolve_fixpoint_iterations_usize(config, "WGPU")?;
        let iterations = u32::try_from(iterations).map_err(|source| {
            BackendError::new(format!(
                "WGPU compiled borrowed timed dispatch iteration count cannot fit u32: {source}. Fix: split fixpoint replay before command recording."
            ))
        })?;
        let workgroup_count = self.workgroups_for_dispatch(config)?;
        let pending = crate::engine::record_and_readback::record_and_submit_async(
            crate::engine::record_and_readback::RecordAndReadback {
                device_queue: &self.device_queue,
                pool: &self.persistent_pool,
                readback_rings: None,
                pipeline: &self.pipeline,
                bind_group_layouts: &self.bind_group_layouts,
                bind_group_cache: Some(self.bind_group_cache.as_ref()),
                buffer_bindings: &self.buffer_bindings,
                inputs,
                output_bindings: Arc::clone(&self.output_bindings),
                trap_tags: &self.trap_tags,
                workgroup_count,
                indirect: self.indirect.as_ref(),
                labels: crate::engine::record_and_readback::DispatchLabels {
                    bind_group: "vyre compiled timed bind group",
                    encoder: "vyre compiled timed dispatch",
                    compute: "vyre compiled timed compute",
                },
                iterations,
                timestamp_profile: true,
                inferred_grid_shape: config
                    .grid_override
                    .is_none()
                    .then_some(self.workgroup_shape),
            },
        )?;
        let enqueue_ns = WGPU_NUMERIC.elapsed_nanos_u64(enqueue_started, "compiled timed enqueue")?;

        let wait_started = Instant::now();
        let deadline = config
            .timeout
            .and_then(|timeout| started.checked_add(timeout));
        let (outputs, device_ns) = match deadline {
            Some(deadline) => pending.await_timed_result_until(deadline)?,
            None => pending.await_timed_result()?,
        };
        enforce_actual_output_budget(config, outputs.as_slice())?;
        let wait_ns = WGPU_NUMERIC.elapsed_nanos_u64(wait_started, "compiled timed wait")?;

        Ok(TimedDispatchResult {
            outputs,
            wall_ns: WGPU_NUMERIC.elapsed_nanos_u64(started, "compiled timed dispatch")?,
            device_ns,
            enqueue_ns: Some(enqueue_ns),
            wait_ns: Some(wait_ns),
        })
    }

    fn dispatch_borrowed_batched(
        &self,
        batches: &[&[&[u8]]],
        config: &DispatchConfig,
    ) -> Result<Vec<OutputBuffers>, BackendError> {
        let mut outputs = Vec::new();
        reserve_pipeline_vec(
            &mut outputs,
            batches.len(),
            "borrowed batched dispatch output sets",
        )?;
        self.dispatch_borrowed_batched_into(batches, config, &mut outputs)?;
        Ok(outputs)
    }

    fn dispatch_borrowed_batched_into(
        &self,
        batches: &[&[&[u8]]],
        config: &DispatchConfig,
        batch_outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        if batches.is_empty() {
            batch_outputs.clear();
            return Ok(());
        }
        self.enforce_static_output_budget(config)?;
        let deadline = config
            .timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));
        let workgroup_count = self.workgroups_for_dispatch(config)?;

        let mut resolved = SmallVec::<[_; 8]>::new();
        reserve_smallvec(
            &mut resolved,
            batches.len(),
            "borrowed batched dispatch",
            "resolved handle set",
            "split the borrowed dispatch batch before submission",
        )?;
        for inputs in batches {
            resolved.push(self.legacy_handles_from_inputs(inputs)?);
        }

        let mut items =
            SmallVec::<[crate::pipeline::persistent::BorrowedDispatchItem<'_>; 8]>::new();
        reserve_smallvec(
            &mut items,
            resolved.len(),
            "borrowed batched dispatch",
            "command item",
            "split the borrowed dispatch batch before submission",
        )?;
        for (inputs, outputs) in resolved.iter() {
            items.push(crate::pipeline::persistent::BorrowedDispatchItem {
                inputs: crate::pipeline::persistent::borrowed_handle_refs(inputs),
                outputs: crate::pipeline::persistent::borrowed_handle_refs(outputs),
                params: None,
                workgroups: workgroup_count,
            });
        }

        let max_iters = resolve_fixpoint_iterations_usize(config, "WGPU")?;
        for _ in 0..max_iters {
            self.dispatch_borrowed_persistent_batched(&items)?;
        }

        let (device, queue) = &*self.device_queue;
        resize_vec_with(
            batch_outputs,
            resolved.len(),
            Vec::new,
            "borrowed batched dispatch output slots",
        )?;
        for ((inputs, outputs), item_outputs) in resolved.iter().zip(batch_outputs.iter_mut()) {
            self.raise_if_trapped(inputs, device, queue, deadline)?;
            self.readback_persistent_outputs(outputs, deadline, item_outputs)?;
            enforce_actual_output_budget(config, item_outputs.as_slice())?;
        }
        Ok(())
    }

    fn dispatch_borrowed_into(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        self.enforce_static_output_budget(config)?;
        let deadline = config
            .timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));
        let workgroup_count = self.workgroups_for_dispatch(config)?;

        let (input_handles, mut output_handles) = self.legacy_handles_from_inputs(inputs)?;
        let max_iters = resolve_fixpoint_iterations_usize(config, "WGPU")?;
        for _iter in 0..max_iters {
            self.dispatch_persistent(&input_handles, &mut output_handles, None, workgroup_count)?;
        }
        if max_iters > 1 {
            tracing::trace!(
                target: "vyre.dispatch.fixpoint",
                iters = max_iters,
                substrate_path = "persistent_pipeline_fixpoint_loop",
                "persistent pipeline fixpoint loop ran",
            );
        }
        let (device, queue) = &*self.device_queue;
        self.raise_if_trapped(&input_handles, device, queue, deadline)?;
        resize_vec_with(
            outputs,
            output_handles.len(),
            Vec::new,
            "borrowed dispatch output slots",
        )?;
        for ((handle, output), bytes) in output_handles
            .iter()
            .zip(self.output_bindings.iter())
            .zip(outputs.iter_mut())
        {
            crate::pipeline::output_readback::read_trimmed_output(
                handle,
                output,
                device,
                &self.staging_pool,
                queue,
                "persistent pipeline output",
                deadline,
                bytes,
            )?;
        }
        enforce_actual_output_budget(config, outputs.as_slice())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vyre_driver::{resolve_fixpoint_iterations_usize, DispatchConfig};

    #[test]
    fn generated_fixpoint_iteration_count_uses_driver_policy() {
        let default_config = DispatchConfig::default();
        assert_eq!(
            resolve_fixpoint_iterations_usize(&default_config, "WGPU")
                .expect("Fix: default fixpoint count fits"),
            1
        );

        let mut zero_config = DispatchConfig::default();
        zero_config.fixpoint_iterations = Some(0);
        assert!(
            resolve_fixpoint_iterations_usize(&zero_config, "WGPU").is_err(),
            "Fix: WGPU must use the driver-owned policy and reject explicit zero fixpoint iterations."
        );

        for iterations in 1..4096u32 {
            let mut config = DispatchConfig::default();
            config.fixpoint_iterations = Some(iterations);
            assert_eq!(
                resolve_fixpoint_iterations_usize(&config, "WGPU")
                    .expect("Fix: generated fixpoint count should fit usize"),
                iterations as usize
            );
        }
    }
}
