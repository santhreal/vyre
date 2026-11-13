//! `CompiledPipeline` implementation for precompiled CUDA pipelines.
//!
//! The parent `pipeline` module owns construction and static launch state. This
//! module owns the dispatch entry points: CUDA graph replay selection, dynamic
//! GPU dispatch when runtime policy changes, and the grid-sync split.

use smallvec::SmallVec;
use vyre_driver::{
    dispatch_configs_share_launch_shape, BackendError, CompiledPipeline, DispatchConfig,
    OutputBuffers, Resource,
};

use crate::backend::staging_reserve::{reserve_smallvec, reserved_vec, resize_vec_slots};
use crate::numeric::CUDA_NUMERIC;
use crate::pipeline::{CudaCompiledPipeline, CudaPipelineExecutionStrategy};
use vyre_driver::input_identity::exact_input_key;

impl CudaCompiledPipeline {
    /// Whether this launch must take the host-orchestrated grid-sync split.
    ///
    /// A compiled pipeline is built around one native launch shape: a CUDA
    /// graph to replay, or a direct cooperative launch. Neither can express a
    /// grid whose blocks do not all fit on the device at once, so a program
    /// with a grid barrier and an over-residency grid has no pipeline route at
    /// all and must go back to the backend's split. The pipeline used to ask
    /// nothing and launch anyway, which is where a large multi-block scan met
    /// `CooperativeResidencyExceeded` through the artifact runtime while the
    /// same program dispatched correctly straight off the backend.
    fn needs_grid_sync_host_split(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        if !vyre_driver::grid_sync::contains_grid_sync(&self.program) {
            return Ok(false);
        }
        self.backend
            .grid_sync_program_needs_host_split(&self.program, inputs, config)
    }

    /// Whether this resident launch must take the grid-sync split.
    ///
    /// The pipeline's resident entry points bind device memory the caller
    /// already owns, and the launch grid comes from those buffers' byte
    /// lengths, so the borrowed predicate above cannot answer for them. They
    /// asked nothing at all: a compiled pipeline holds one native launch
    /// shape, and an over-residency grid-sync program submitted through a
    /// resident entry point launched cooperatively and failed with
    /// `CooperativeResidencyExceeded`, while the same program through a
    /// borrowed entry point returned an answer. The artifact runtime submits
    /// resident resources, so that was the whole shipped route for a
    /// multi-block grid-sync reduction.
    fn needs_resident_grid_sync_host_split(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<bool, BackendError> {
        if !vyre_driver::grid_sync::contains_grid_sync(&self.program) {
            return Ok(false);
        }
        self.backend
            .grid_sync_resident_program_needs_host_split(&self.program, inputs, config)
    }

    /// Dispatch `inputs` through the grid-sync split, one segment at a time
    /// against the same resident resources.
    fn dispatch_resident_grid_sync_split_timed(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        self.backend
            .dispatch_resident_with_grid_sync_split_timed(&self.program, inputs, config)
    }

    /// Dispatch every batch item through the grid-sync split.
    ///
    /// Items run back to back. The concurrent batch route submits every item
    /// before awaiting any, which a segmented item cannot join: its later
    /// segments read the device state its earlier segments leave in the bound
    /// resources, so a second item submitted into the same resources
    /// mid-sequence would overwrite them.
    fn dispatch_resident_batches_grid_sync_split_into(
        &self,
        batches: &[&[Resource]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
        device_ns_by_item: &mut Vec<Option<u64>>,
    ) -> Result<(), BackendError> {
        resize_vec_slots(outputs, batches.len(), "resident split batched output")?;
        *device_ns_by_item =
            reserved_vec(batches.len(), "resident split batched item device time")?;
        for (inputs, item_outputs) in batches.iter().zip(outputs.iter_mut()) {
            let split = self.dispatch_resident_grid_sync_split_timed(inputs, config)?;
            device_ns_by_item.push(split.device_ns);
            vyre_driver::replace_output_buffers_preserving_slots(split.outputs, item_outputs);
        }
        Ok(())
    }
}

impl CompiledPipeline for CudaCompiledPipeline {
    fn id(&self) -> &str {
        &self.id
    }

    fn dispatch_borrowed_timed(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_DISPATCH_RANGE);
        if self.needs_grid_sync_host_split(inputs, config)? {
            return self
                .backend
                .dispatch_borrowed_timed(&self.program, inputs, config);
        }
        if !dispatch_configs_share_launch_shape(&self.compiled_config, config)
            || self.execution_strategy() == CudaPipelineExecutionStrategy::DirectDispatch
        {
            return self.backend.dispatch_borrowed_timed_with_ptx_key(
                &self.program,
                inputs,
                config,
                &self.ptx_src,
                self.module_key,
            );
        }
        let started = std::time::Instant::now();
        let mut outputs = reserved_vec(self.prepared.output_binding_indices.len(), "timed output")?;
        let input_key = exact_input_key(inputs)?;
        let (mut cached, input_state) = match self
            .take_cached_graph_with_replay_state(inputs, &input_key)?
        {
            Some(selection) => (selection.graph, selection.input_state),
            None => {
                let cached =
                    self.backend
                        .record_cuda_graph_borrowed(&self.program, inputs, config)?;
                let input_state = self
                    .backend
                    .prepare_cuda_graph_replay_input_state_with_key(&cached, inputs, input_key)?;
                (cached, input_state)
            }
        };
        let replay_result = self
            .backend
            .dispatch_via_cuda_graph_timed_with_input_state_into(
                &mut cached,
                inputs,
                &input_state,
                &mut outputs,
            );
        if replay_result.is_ok() {
            self.return_cached_graph(cached)?;
        } else {
            std::mem::forget(cached);
        }
        let device_ns = replay_result?;
        let wall_ns = CUDA_NUMERIC.elapsed_nanos_u64(started, "cuda graph replay wall latency")?;
        self.backend
            .telemetry
            .record_timed_dispatch(wall_ns, device_ns, None, None);
        Ok(vyre_driver::TimedDispatchResult::device_timed(
            outputs, wall_ns, device_ns,
        ))
    }

    fn dispatch_persistent_handles_timed(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_DISPATCH_RANGE);
        if self.needs_resident_grid_sync_host_split(inputs, config)? {
            return self.dispatch_resident_grid_sync_split_timed(inputs, config);
        }
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            let started = std::time::Instant::now();
            let bindings = self.backend.resident_bindings_from_resources(inputs)?;
            let mut outputs = reserved_vec(
                self.prepared.output_binding_indices.len(),
                "persistent fallback output",
            )?;
            self.backend.dispatch_resident_outputs_with_ptx_key_into(
                &self.program,
                &bindings,
                config,
                &self.ptx_src,
                self.module_key,
                &mut outputs,
            )?;
            let wall_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(started, "compiled persistent fallback wall latency")?;
            self.backend
                .telemetry
                .record_timed_dispatch(wall_ns, None, None, None);
            return Ok(vyre_driver::TimedDispatchResult::host_timed(
                outputs, wall_ns,
            ));
        }
        if !dispatch_configs_share_launch_shape(&self.compiled_config, config) {
            let started = std::time::Instant::now();
            let enqueue_started = std::time::Instant::now();
            let bindings = self.backend.resident_bindings_from_resources(inputs)?;
            let prepared =
                self.backend
                    .prepare_resident_dispatch(&self.program, &bindings, config)?;
            let dispatch = self.backend.dispatch_resident_async_concrete_with_ptx_key(
                &self.program,
                &bindings,
                config,
                &self.ptx_src,
                self.module_key,
                true,
                None,
                true,
                &prepared,
            )?;
            let enqueue_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(enqueue_started, "compiled persistent enqueue latency")?;
            let wait_started = std::time::Instant::now();
            let (outputs, device_ns) = dispatch.pending.await_timed_result()?;
            let wait_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(wait_started, "compiled persistent wait latency")?;
            let wall_ns = crate::numeric::CUDA_NUMERIC
                .elapsed_nanos_u64(started, "compiled persistent wall latency")?;
            self.backend.telemetry.record_timed_dispatch(
                wall_ns,
                device_ns,
                Some(enqueue_ns),
                Some(wait_ns),
            );
            return Ok(vyre_driver::TimedDispatchResult::split_timed(
                outputs, wall_ns, device_ns, enqueue_ns, wait_ns,
            ));
        }

        let started = std::time::Instant::now();
        let enqueue_started = std::time::Instant::now();
        let bindings = self.backend.resident_bindings_from_resources(inputs)?;
        let _kernel_window = crate::backend::dispatch_phase_probe::arm_dispatch(
            &self.program,
            self.ptx_src.len(),
            bindings.len(),
            self.prepared.fixpoint_iterations as usize,
            self.prepared.launch.grid,
            &self.backend.launch_resources,
        );
        let dispatch = self.backend.dispatch_resident_async_concrete_with_ptx_key(
            &self.program,
            &bindings,
            config,
            &self.ptx_src,
            self.module_key,
            true,
            (self.static_params.ptr != 0).then_some(self.static_params.ptr),
            true,
            &self.prepared,
        )?;
        let enqueue_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(enqueue_started, "compiled persistent enqueue latency")?;
        let wait_started = std::time::Instant::now();
        let (outputs, device_ns) = dispatch.pending.await_timed_result()?;
        let wait_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(wait_started, "compiled persistent wait latency")?;
        let wall_ns = crate::numeric::CUDA_NUMERIC
            .elapsed_nanos_u64(started, "compiled persistent wall latency")?;
        self.backend.telemetry.record_timed_dispatch(
            wall_ns,
            device_ns,
            Some(enqueue_ns),
            Some(wait_ns),
        );
        crate::backend::dispatch_phase_probe::emit_dispatch(
            &self.backend,
            wall_ns,
            enqueue_ns,
            wait_ns,
            device_ns,
        );
        Ok(vyre_driver::TimedDispatchResult::split_timed(
            outputs, wall_ns, device_ns, enqueue_ns, wait_ns,
        ))
    }

    fn dispatch_persistent_handles_batched_timed(
        &self,
        batches: &[&[Resource]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
        device_ns_by_item: &mut Vec<Option<u64>>,
    ) -> Result<(), BackendError> {
        let _profiler_range = crate::profiler::cuda_profiler_range(
            crate::profiler::CUDA_PIPELINE_BATCH_DISPATCH_RANGE,
        );
        if batches.is_empty() {
            outputs.clear();
            device_ns_by_item.clear();
            return Ok(());
        }
        if self.needs_resident_grid_sync_host_split(batches[0], config)? {
            return self.dispatch_resident_batches_grid_sync_split_into(
                batches,
                config,
                outputs,
                device_ns_by_item,
            );
        }
        let mut resident_batches =
            SmallVec::<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>::new();
        reserve_smallvec(&mut resident_batches, batches.len(), "resident batch")?;
        for batch in batches {
            resident_batches.push(self.backend.resident_handles_from_resources(batch)?);
        }

        self.dispatch_resident_batches_timed_into(
            &resident_batches,
            config,
            outputs,
            device_ns_by_item,
        )
    }
}
