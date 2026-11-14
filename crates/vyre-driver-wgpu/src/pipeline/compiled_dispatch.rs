//! `CompiledPipeline` implementation for WGPU pipeline dispatch.
//!
//! The parent `pipeline` module owns compilation and metadata assembly. This
//! module owns the trait entrypoints that turn caller inputs into persistent
//! GPU handles, execute the compiled compute pipeline, and read back outputs.

use std::sync::Arc;
use std::time::{Duration, Instant};

use smallvec::SmallVec;
use vyre_runtime::enforce_actual_output_budget;
use vyre_runtime::{
    resolve_fixpoint_iterations_usize, BackendError, CompiledPipeline, DispatchConfig,
    OutputBuffers, TimedDispatchResult,
};

use crate::engine::record_and_readback::timestamp::{
    collect_timestamp_profile, device_records_timestamps, PendingTimestampProfile,
    TimestampRecorder,
};
use crate::numeric::WGPU_NUMERIC;
use crate::pipeline::output_slots::resize_vec_with;
use crate::pipeline::WgpuPipeline;
use crate::staging_reserve::reserve_smallvec;

/// One persistent dispatch whose compute pass, output copies, trap-flag copy
/// and timestamp resolve are submitted and still in flight.
///
/// Every resident entry point of the compiled pipeline runs through this
/// record: the timed dispatch retires it at once, the asynchronous dispatch
/// returns it to the caller, and the batched dispatch submits one per item
/// before retiring any. Each item's output copies are queued directly behind
/// its own compute pass, so items that bind the same output buffer still read
/// back their own results.
struct EnqueuedPersistentDispatch {
    resolved: crate::pipeline::persistent_resources::ResolvedPersistentResources,
    output_readbacks: SmallVec<[crate::buffer::PendingGpuBufferReadback; 8]>,
    trap_flag_readback: Option<crate::buffer::PendingGpuBufferReadback>,
    timestamp_profile: Option<PendingTimestampProfile>,
    deadline: Option<Instant>,
    timestamp_deadline: Instant,
}

impl EnqueuedPersistentDispatch {
    fn is_ready(&self, device: &wgpu::Device) -> bool {
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
}

pub(crate) struct WgpuPendingPersistentDispatch {
    pipeline: Arc<WgpuPipeline>,
    enqueued: EnqueuedPersistentDispatch,
    started: Instant,
    enqueue_ns: u64,
    config: DispatchConfig,
}

impl vyre_runtime::sealed::Sealed for WgpuPendingPersistentDispatch {}

impl WgpuPendingPersistentDispatch {
    fn retire(self) -> Result<TimedDispatchResult, BackendError> {
        let wait_started = Instant::now();
        let mut outputs = Vec::new();
        let device_ns =
            self.pipeline
                .retire_persistent_into(self.enqueued, &self.config, &mut outputs)?;
        let wait_ns =
            WGPU_NUMERIC.elapsed_nanos_u64(wait_started, "persistent asynchronous wait")?;
        Ok(TimedDispatchResult::split_timed(
            outputs,
            WGPU_NUMERIC.elapsed_nanos_u64(self.started, "persistent asynchronous dispatch")?,
            device_ns,
            self.enqueue_ns,
            wait_ns,
        ))
    }
}

impl vyre_runtime::PendingDispatch for WgpuPendingPersistentDispatch {
    fn is_ready(&self) -> bool {
        self.enqueued.is_ready(&self.pipeline.device_queue.0)
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
        resources: &[vyre_runtime::Resource],
        config: &DispatchConfig,
        started: Instant,
    ) -> Result<WgpuPendingPersistentDispatch, BackendError> {
        let enqueue_started = Instant::now();
        let enqueued = self.enqueue_persistent(resources, config, started)?;
        let enqueue_ns =
            WGPU_NUMERIC.elapsed_nanos_u64(enqueue_started, "persistent asynchronous enqueue")?;
        Ok(WgpuPendingPersistentDispatch {
            pipeline: Arc::clone(self),
            enqueued,
            started,
            enqueue_ns,
            config: config.clone(),
        })
    }

    /// Record one resident dispatch, submit it, and start every readback it
    /// needs without waiting for any of them.
    ///
    /// The timeout in `config` counts from `started`.
    fn enqueue_persistent(
        &self,
        resources: &[vyre_runtime::Resource],
        config: &DispatchConfig,
        started: Instant,
    ) -> Result<EnqueuedPersistentDispatch, BackendError> {
        self.enforce_static_output_budget(config)?;
        let (device, queue) = &*self.device_queue;
        let deadline = config
            .timeout
            .and_then(|timeout| started.checked_add(timeout));
        let timestamp_deadline =
            deadline.unwrap_or_else(|| Instant::now() + Duration::from_secs(30));
        let resolved = self.resolve_persistent_resources(resources, queue)?;
        if resolved.outputs.len() != self.output_bindings.len() {
            return Err(BackendError::new(format!(
                "WGPU persistent dispatch resolved {} output handles for {} output layouts. Fix: keep resident resource resolution synchronized with compiled output bindings.",
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
        // Every `CompiledPipeline` timed entry reports timing the backend
        // already owns: the trait default is host wall time and `device_ns`
        // is an `Option`, so an adapter that records no timestamp answers
        // `None`. Requesting the queries unconditionally turned that into a
        // refusal, and both a retained execution through `launch_resident`
        // and a plain `MaterializedInstance::dispatch` failed on every
        // adapter advertising no timestamp pair.
        let timestamp_recorder = TimestampRecorder::new(
            device,
            queue,
            &self.persistent_pool,
            device_records_timestamps(device),
            0,
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vyre persistent dispatch"),
        });
        let timestamp_writes = timestamp_recorder
            .as_ref()
            .map(TimestampRecorder::pass_writes);
        self.record_borrowed_persistent_item_with_timestamps(
            device,
            &mut encoder,
            &item,
            timestamp_writes,
        )?;
        drop(item);
        if let Some(recorder) = &timestamp_recorder {
            recorder.finish_and_resolve(&mut encoder)?;
        }
        queue.submit(std::iter::once(encoder.finish()));

        let in_flight = (|| {
            let timestamp_profile = timestamp_recorder
                .map(TimestampRecorder::map_async)
                .transpose()?;
            let mut output_readbacks = SmallVec::new();
            reserve_smallvec(
                &mut output_readbacks,
                resolved.outputs.len(),
                "persistent dispatch",
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
                    "persistent output",
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
        let (output_readbacks, trap_flag_readback, timestamp_profile) = match in_flight {
            Ok(in_flight) => in_flight,
            Err(error) => {
                self.drain_queue()?;
                return Err(error);
            }
        };
        Ok(EnqueuedPersistentDispatch {
            resolved,
            output_readbacks,
            trap_flag_readback,
            timestamp_profile,
            deadline,
            timestamp_deadline,
        })
    }

    /// Wait for one enqueued dispatch, raise its trap if it set one, read its
    /// outputs into `outputs`, and return its device duration.
    ///
    /// `outputs` keeps the slots and byte capacity it already holds.
    fn retire_persistent_into(
        &self,
        enqueued: EnqueuedPersistentDispatch,
        config: &DispatchConfig,
        outputs: &mut OutputBuffers,
    ) -> Result<Option<u64>, BackendError> {
        let (device, queue) = &*self.device_queue;
        if let Some(trap_flag) = enqueued.trap_flag_readback {
            let mut flag = Vec::new();
            trap_flag.await_into(device, enqueued.deadline, &mut flag)?;
            if flag.len() != 4 {
                return Err(BackendError::new(format!(
                    "internal wgpu trap flag readback returned {} bytes but 4 bytes are required. Fix: allocate the trap sidecar as four or more bytes.",
                    flag.len()
                )));
            }
            if u32::from_le_bytes([flag[0], flag[1], flag[2], flag[3]]) != 0 {
                self.raise_if_trapped(&enqueued.resolved.inputs, device, queue, enqueued.deadline)?;
            }
        }

        resize_vec_with(
            outputs,
            enqueued.output_readbacks.len(),
            Vec::new,
            "persistent output slots",
        )?;
        for ((pending, output), bytes) in enqueued
            .output_readbacks
            .into_iter()
            .zip(self.output_bindings.iter())
            .zip(outputs.iter_mut())
        {
            pending.await_into(device, enqueued.deadline, bytes)?;
            if bytes.len() != output.layout.read_size {
                return Err(BackendError::new(format!(
                    "persistent readback for `{}` returned {} bytes but {} bytes are required. Fix: keep output range validation synchronized with OutputLayout.",
                    output.name,
                    bytes.len(),
                    output.layout.read_size
                )));
            }
        }
        enforce_actual_output_budget(config, outputs.as_slice())?;
        Ok(
            collect_timestamp_profile(enqueued.timestamp_profile, enqueued.timestamp_deadline)?
                .map(|profile| profile.dispatch_ns),
        )
    }

    /// Block until every submission on this pipeline's queue has completed.
    ///
    /// A failure after a submission leaves work in flight that references
    /// buffers the caller is about to drop.
    fn drain_queue(&self) -> Result<(), BackendError> {
        let (device, queue) = &*self.device_queue;
        let fence = queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
        crate::runtime::device::poll_device_wait_for(device, fence).map(|_| ())
    }
}

impl WgpuPipeline {
    /// Dispatch against resident resources and leave every output on the
    /// device.
    ///
    /// Device-buffer dispatch binds caller-owned output buffers that the
    /// caller downloads afterwards, so no output is read back here. Only the
    /// trap sidecar is checked before returning.
    pub(crate) fn dispatch_persistent_handles_without_readback(
        &self,
        inputs: &[vyre_runtime::Resource],
        config: &DispatchConfig,
    ) -> Result<(), BackendError> {
        self.enforce_static_output_budget(config)?;
        let (device, queue) = &*self.device_queue;
        let deadline = config
            .timeout
            .and_then(|timeout| Instant::now().checked_add(timeout));
        let resolved = self.resolve_persistent_resources(inputs, queue)?;
        let item = crate::pipeline::persistent::BorrowedDispatchItem {
            inputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.inputs),
            outputs: crate::pipeline::persistent::borrowed_handle_refs(&resolved.outputs),
            params: None,
            workgroups: self.workgroups_for_dispatch(config)?,
        };
        self.submit_borrowed_persistent(&item)?;
        self.raise_if_trapped(&resolved.inputs, device, queue, deadline)
    }
}

impl CompiledPipeline for WgpuPipeline {
    fn id(&self) -> &str {
        &self.id
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
                timestamp_profile: device_records_timestamps(&self.device_queue.0),
                inferred_launch: config.launch_grid().is_none().then(|| {
                    crate::engine::record_and_readback::InferredLaunch {
                        workgroup_shape: self.workgroup_shape,
                        max_per_axis: self.axis_ceiling(config),
                        grid_linearized: self.grid_linearized,
                    }
                }),
            },
        )?;
        let enqueue_ns =
            WGPU_NUMERIC.elapsed_nanos_u64(enqueue_started, "compiled timed enqueue")?;

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

        Ok(TimedDispatchResult::split_timed(
            outputs,
            WGPU_NUMERIC.elapsed_nanos_u64(started, "compiled timed dispatch")?,
            device_ns,
            enqueue_ns,
            wait_ns,
        ))
    }

    fn dispatch_persistent_handles_timed(
        &self,
        inputs: &[vyre_runtime::Resource],
        config: &DispatchConfig,
    ) -> Result<TimedDispatchResult, BackendError> {
        let started = Instant::now();
        let enqueued = self.enqueue_persistent(inputs, config, started)?;
        let enqueue_ns = WGPU_NUMERIC.elapsed_nanos_u64(started, "persistent enqueue")?;

        let wait_started = Instant::now();
        let mut outputs = Vec::new();
        let device_ns = self.retire_persistent_into(enqueued, config, &mut outputs)?;
        let wait_ns = WGPU_NUMERIC.elapsed_nanos_u64(wait_started, "persistent wait")?;

        Ok(TimedDispatchResult::split_timed(
            outputs,
            WGPU_NUMERIC.elapsed_nanos_u64(started, "persistent timed dispatch")?,
            device_ns,
            enqueue_ns,
            wait_ns,
        ))
    }

    /// Submit every item, each with its own output copies and timestamp
    /// queries, before waiting for any of them.
    ///
    /// The device stays busy across the batch instead of idling for one
    /// host round trip per item. Each item's timeout counts from its own
    /// submission.
    fn dispatch_persistent_handles_batched_timed(
        &self,
        batches: &[&[vyre_runtime::Resource]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
        device_ns_by_item: &mut Vec<Option<u64>>,
    ) -> Result<(), BackendError> {
        resize_vec_with(
            outputs,
            batches.len(),
            Vec::new,
            "persistent batch output sets",
        )?;
        device_ns_by_item.clear();
        device_ns_by_item
            .try_reserve_exact(batches.len())
            .map_err(|error| {
                BackendError::new(format!(
                    "failed to reserve {} persistent batch timing slot(s): {error}. Fix: submit a smaller resident batch.",
                    batches.len()
                ))
            })?;
        let mut enqueued = SmallVec::<[EnqueuedPersistentDispatch; 8]>::new();
        reserve_smallvec(
            &mut enqueued,
            batches.len(),
            "persistent batch dispatch",
            "enqueued item",
            "submit a smaller resident batch",
        )?;
        for batch in batches {
            match self.enqueue_persistent(batch, config, Instant::now()) {
                Ok(item) => enqueued.push(item),
                Err(error) => {
                    if !enqueued.is_empty() {
                        self.drain_queue()?;
                    }
                    return Err(error);
                }
            }
        }
        for (item, slot) in enqueued.into_iter().zip(outputs.iter_mut()) {
            device_ns_by_item.push(self.retire_persistent_into(item, config, slot)?);
        }
        Ok(())
    }
}

// Inline: `vyre_driver_wgpu::pipeline::compiled_dispatch` is `pub(crate)`, so no integration test
// can reach what this suite exercises.
#[cfg(test)]
mod tests {
    use vyre_runtime::{resolve_fixpoint_iterations_usize, DispatchConfig};

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
