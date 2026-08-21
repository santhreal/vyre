//! `CompiledPipeline` implementation for precompiled CUDA pipelines.
//!
//! The parent `pipeline` module owns construction and static launch state. This
//! module owns dispatch entrypoints, CUDA graph replay selection, dynamic GPU
//! dispatch when runtime policy changes, and persistent-resource output routing.

use smallvec::SmallVec;
use vyre_driver::{
    borrowed_input_batch_shapes_match, dispatch_configs_share_launch_shape, BackendError,
    BindingRole, CompiledPipeline, DispatchConfig, OutputBuffers, Resource,
};

use crate::backend::resident::CudaDispatchBinding;
use crate::backend::resident_dispatch::next_dispatch_binding;
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
}

impl CompiledPipeline for CudaCompiledPipeline {
    fn id(&self) -> &str {
        &self.id
    }

    fn dispatch_borrowed(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, BackendError> {
        if self.needs_grid_sync_host_split(inputs, config)? {
            return self
                .backend
                .dispatch_borrowed(&self.program, inputs, config);
        }
        if !dispatch_configs_share_launch_shape(&self.compiled_config, config) {
            return self
                .backend
                .dispatch_borrowed_async_with_ptx_key(
                    &self.program,
                    inputs,
                    config,
                    &self.ptx_src,
                    self.module_key,
                )?
                .await_result();
        }
        let mut outputs = reserved_vec(self.prepared.output_binding_indices.len(), "output")?;
        self.dispatch_borrowed_into(inputs, config, &mut outputs)?;
        Ok(outputs)
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
            self.remember_materialized_output_cache_with_key(inputs, input_key, &outputs)?;
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

    fn dispatch_borrowed_into(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_DISPATCH_RANGE);
        if self.needs_grid_sync_host_split(inputs, config)? {
            let split = self
                .backend
                .dispatch_borrowed(&self.program, inputs, config)?;
            vyre_driver::replace_output_buffers_preserving_slots(split, outputs);
            return Ok(());
        }
        if !dispatch_configs_share_launch_shape(&self.compiled_config, config)
            || self.execution_strategy() == CudaPipelineExecutionStrategy::DirectDispatch
        {
            self.backend
                .dispatch_borrowed_async_with_ptx_key(
                    &self.program,
                    inputs,
                    config,
                    &self.ptx_src,
                    self.module_key,
                )?
                .await_result_into(outputs)?;
            return Ok(());
        }
        let input_key = exact_input_key(inputs)?;
        if self.materialized_output_cache_hit_with_key_into(inputs, &input_key, outputs)? {
            return Ok(());
        }
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
        let replay_result = self.backend.dispatch_via_cuda_graph_with_input_state_into(
            &mut cached,
            inputs,
            &input_state,
            outputs,
        );
        if replay_result.is_ok() {
            self.return_cached_graph(cached)?;
            self.remember_materialized_output_cache_with_key(inputs, input_key, outputs)?;
        } else {
            std::mem::forget(cached);
        }
        replay_result
    }

    fn dispatch_borrowed_batched_into(
        &self,
        batches: &[&[&[u8]]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        let _profiler_range = crate::profiler::cuda_profiler_range(
            crate::profiler::CUDA_PIPELINE_BATCH_DISPATCH_RANGE,
        );
        if batches.is_empty() {
            outputs.clear();
            return Ok(());
        }
        if self.needs_grid_sync_host_split(batches[0], config)? {
            resize_vec_slots(outputs, batches.len(), "split batched output")?;
            for (inputs, item_outputs) in batches.iter().zip(outputs.iter_mut()) {
                let split = self
                    .backend
                    .dispatch_borrowed(&self.program, inputs, config)?;
                vyre_driver::replace_output_buffers_preserving_slots(split, item_outputs);
            }
            return Ok(());
        }
        if self.execution_strategy() == CudaPipelineExecutionStrategy::GraphReplay
            && dispatch_configs_share_launch_shape(&self.compiled_config, config)
            && borrowed_input_batch_shapes_match(batches)
        {
            return self.dispatch_borrowed_batched_via_cuda_graph_lanes(batches, config, outputs);
        }
        let mut pending = SmallVec::<[_; 8]>::new();
        reserve_smallvec(&mut pending, batches.len(), "pending dispatch")?;
        if dispatch_configs_share_launch_shape(&self.compiled_config, config) {
            for inputs in batches {
                pending.push(self.backend.dispatch_prepared_borrowed_async_with_ptx_key(
                    &self.program,
                    inputs,
                    &self.ptx_src,
                    self.module_key,
                    &self.prepared,
                )?);
            }
        } else {
            for inputs in batches {
                pending.push(self.backend.dispatch_borrowed_async_with_ptx_key(
                    &self.program,
                    inputs,
                    config,
                    &self.ptx_src,
                    self.module_key,
                )?);
            }
        }

        resize_vec_slots(outputs, pending.len(), "batched output")?;
        for (dispatch, item_outputs) in pending.into_iter().zip(outputs.iter_mut()) {
            dispatch.await_result_into(item_outputs)?;
        }
        Ok(())
    }

    fn dispatch_persistent_handles(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<OutputBuffers, BackendError> {
        let mut outputs = reserved_vec(
            self.prepared.output_binding_indices.len(),
            "persistent output",
        )?;
        self.dispatch_persistent_handles_into(inputs, config, &mut outputs)?;
        Ok(outputs)
    }

    fn dispatch_persistent_handles_timed(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<vyre_driver::TimedDispatchResult, BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_DISPATCH_RANGE);
        if crate::instrumentation::cuda_resident_borrowed_fallback_enabled() {
            let started = std::time::Instant::now();
            let outputs = self.dispatch_persistent_handles(inputs, config)?;
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
        Ok(vyre_driver::TimedDispatchResult::split_timed(
            outputs, wall_ns, device_ns, enqueue_ns, wait_ns,
        ))
    }

    fn dispatch_persistent_handles_into(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
        outputs: &mut OutputBuffers,
    ) -> Result<(), BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_DISPATCH_RANGE);
        let bindings = self.backend.resident_bindings_from_resources(inputs)?;
        if dispatch_configs_share_launch_shape(&self.compiled_config, config)
            && !crate::instrumentation::cuda_resident_borrowed_fallback_enabled()
        {
            let dispatch = self.backend.dispatch_resident_async_concrete_with_ptx_key(
                &self.program,
                &bindings,
                config,
                &self.ptx_src,
                self.module_key,
                false,
                (self.static_params.ptr != 0).then_some(self.static_params.ptr),
                true,
                &self.prepared,
            )?;
            let (dispatch_outputs, _) = dispatch.pending.await_timed_result()?;
            vyre_driver::replace_output_buffers_preserving_slots(dispatch_outputs, outputs);
            return Ok(());
        }
        self.backend.dispatch_resident_outputs_with_ptx_key_into(
            &self.program,
            &bindings,
            config,
            &self.ptx_src,
            self.module_key,
            outputs,
        )
    }

    fn dispatch_persistent_handles_batched_into(
        &self,
        batches: &[&[Resource]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        let _profiler_range = crate::profiler::cuda_profiler_range(
            crate::profiler::CUDA_PIPELINE_BATCH_DISPATCH_RANGE,
        );
        if batches.is_empty() {
            outputs.clear();
            return Ok(());
        }
        let mut resident_batches =
            SmallVec::<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>::new();
        reserve_smallvec(&mut resident_batches, batches.len(), "resident batch")?;
        for batch in batches {
            resident_batches.push(self.backend.resident_handles_from_resources(batch)?);
        }

        self.dispatch_resident_batches_into(&resident_batches, config, outputs)
    }

    fn dispatch_persistent_handle_rows_into(
        &self,
        rows: &[[Resource; 4]],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
    ) -> Result<(), BackendError> {
        let _profiler_range = crate::profiler::cuda_profiler_range(
            crate::profiler::CUDA_PIPELINE_BATCH_DISPATCH_RANGE,
        );
        if rows.is_empty() {
            outputs.clear();
            return Ok(());
        }
        let mut resident_batches =
            SmallVec::<[SmallVec<[crate::backend::CudaResidentBuffer; 8]>; 8]>::new();
        reserve_smallvec(&mut resident_batches, rows.len(), "resident row batch")?;
        for row in rows {
            resident_batches.push(
                self.backend
                    .resident_handles_from_resources(row.as_slice())?,
            );
        }

        self.dispatch_resident_batches_into(&resident_batches, config, outputs)
    }

    fn dispatch_persistent_resource_outputs(
        &self,
        inputs: &[Resource],
        config: &DispatchConfig,
    ) -> Result<Vec<Resource>, BackendError> {
        let _profiler_range =
            crate::profiler::cuda_profiler_range(crate::profiler::CUDA_PIPELINE_DISPATCH_RANGE);
        let bindings = self.backend.resident_bindings_from_resources(inputs)?;
        let borrowed_fallback = crate::instrumentation::cuda_resident_borrowed_fallback_enabled();
        let same_shape = dispatch_configs_share_launch_shape(&self.compiled_config, config);
        let prepared_storage;
        let (prepared, static_params_ptr) = if same_shape {
            (
                &self.prepared,
                (self.static_params.ptr != 0).then_some(self.static_params.ptr),
            )
        } else {
            prepared_storage =
                self.backend
                    .prepare_resident_dispatch(&self.program, &bindings, config)?;
            (&prepared_storage, None)
        };
        let mut output_handles = SmallVec::<[crate::backend::CudaResidentBuffer; 8]>::new();
        reserve_smallvec(
            &mut output_handles,
            prepared.output_binding_indices.len(),
            "compiled resident resource output handle",
        )?;
        let mut next_binding = 0usize;
        for binding in &prepared.bindings.bindings {
            if binding.role == BindingRole::Shared {
                continue;
            }
            let source = next_dispatch_binding(
                &bindings,
                &mut next_binding,
                "compiled resident resource output routing",
            )?;
            if binding.output_index.is_some() {
                // This entry point hands back a Resource::Resident per output,
                // naming device memory that outlives the call. A borrowed
                // output is staged from the transient pool and recycled when
                // the dispatch ends, so there is no lasting id to return.
                let CudaDispatchBinding::Resident(handle) = source else {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA compiled resident resource output `{}` was given a borrowed resource, but this entry point returns Resource::Resident outputs that outlive the dispatch. Bind a resident buffer for that output, or use a dispatch entry point that returns output bytes.",
                            binding.name
                        ),
                    });
                };
                output_handles.push(handle);
            }
        }
        if borrowed_fallback {
            self.backend
                .dispatch_resident_via_borrowed(&self.program, &bindings, config)?;
        } else {
            self.backend
                .dispatch_resident_async_concrete_with_ptx_key(
                    &self.program,
                    &bindings,
                    config,
                    &self.ptx_src,
                    self.module_key,
                    false,
                    static_params_ptr,
                    false,
                    prepared,
                )?
                .pending
                .await_timed_result()?;
        }
        let mut resources = reserved_vec(output_handles.len(), "resource output")?;
        for handle in output_handles {
            resources.push(Resource::Resident(handle.handle));
        }
        Ok(resources)
    }
}
