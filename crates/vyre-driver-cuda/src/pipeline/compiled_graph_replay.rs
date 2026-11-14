use smallvec::SmallVec;
use vyre_runtime::input_identity::ExactInputKey;
use vyre_runtime::{BackendError, DispatchConfig, OutputBuffers};

use crate::backend::cuda_graph_replay::CudaGraphReplayInputState;
use crate::backend::resident_dispatch::output_disjoint_prefix;
use crate::backend::staging_reserve::reserve_vec;
use crate::backend::CachedCudaGraph;
use crate::pipeline::{CudaCompiledPipeline, MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE};

#[derive(Debug)]
pub(crate) struct CachedGraphReplaySelection {
    pub(crate) graph: CachedCudaGraph,
    pub(crate) input_state: CudaGraphReplayInputState,
}

impl CudaCompiledPipeline {
    /// Run every resident item and report each item's outputs and device
    /// duration, as a submission per item would.
    ///
    /// Items are submitted in runs whose output handles are pairwise distinct,
    /// each run retired before the next is submitted. A batch whose items share
    /// no output is one run. Two items that write one handle land in separate
    /// runs, so each reads back its own bytes rather than the later item's.
    pub(crate) fn dispatch_resident_batches_timed_into(
        &self,
        resident_batches: &[SmallVec<[crate::backend::CudaResidentBuffer; 8]>],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
        device_ns_by_item: &mut Vec<Option<u64>>,
    ) -> Result<(), BackendError> {
        let mut run_len = output_disjoint_prefix(&self.prepared, resident_batches)?;
        if run_len == resident_batches.len() {
            return self.dispatch_output_disjoint_batch_into(
                resident_batches,
                config,
                outputs,
                device_ns_by_item,
            );
        }
        outputs.clear();
        device_ns_by_item.clear();
        reserve_vec(
            outputs,
            resident_batches.len(),
            "aliased resident batch outputs",
        )?;
        reserve_vec(
            device_ns_by_item,
            resident_batches.len(),
            "aliased resident batch timing",
        )?;
        let mut run_outputs = Vec::new();
        let mut run_device_ns = Vec::new();
        let mut start = 0;
        loop {
            let end = start + run_len;
            self.dispatch_output_disjoint_batch_into(
                &resident_batches[start..end],
                config,
                &mut run_outputs,
                &mut run_device_ns,
            )?;
            outputs.append(&mut run_outputs);
            device_ns_by_item.append(&mut run_device_ns);
            if end == resident_batches.len() {
                return Ok(());
            }
            start = end;
            run_len = output_disjoint_prefix(&self.prepared, &resident_batches[start..])?;
        }
    }

    /// Submit every item of an output-disjoint batch on ONE stream and report
    /// each item's device duration.
    ///
    /// The items are enqueued back to back before any of them is awaited, so
    /// item N's start event retires when item N-1's kernel ends while item N's
    /// kernel is already submitted. Its event pair therefore spans the kernel,
    /// not the host latency of submitting it, and the device stays busy across
    /// the whole batch. Timing each item through a separate submit-and-wait
    /// charges every item the launch front-end latency of an idle stream, which
    /// on a 9 us kernel is close to a fifth of the reported figure.
    fn dispatch_output_disjoint_batch_into(
        &self,
        resident_batches: &[SmallVec<[crate::backend::CudaResidentBuffer; 8]>],
        config: &DispatchConfig,
        outputs: &mut Vec<OutputBuffers>,
        device_ns_by_item: &mut Vec<Option<u64>>,
    ) -> Result<(), BackendError> {
        if resident_batches.is_empty() {
            outputs.clear();
            device_ns_by_item.clear();
            return Ok(());
        }
        let resident_dispatch = self
            .backend
            .dispatch_resident_batch_async_concrete_with_ptx_key(
                &self.program,
                resident_batches,
                config,
                &self.ptx_src,
                self.module_key,
                (self.static_params.ptr != 0).then_some(self.static_params.ptr),
                &self.prepared,
            );
        let resident_dispatch = resident_dispatch?;
        let output_handles = resident_dispatch.output_handles;
        let output_readbacks = resident_dispatch.output_readbacks;
        let item_timing = resident_dispatch.item_timing;
        resident_dispatch.pending.await_timed_result()?;
        self.backend
            .collect_batch_item_device_ns(item_timing, device_ns_by_item)?;
        self.backend.download_resident_readback_batches_many_into(
            &output_handles,
            &output_readbacks,
            outputs,
        )
    }

    pub(crate) fn take_cached_graph_with_replay_state(
        &self,
        inputs: &[&[u8]],
        input_key: &ExactInputKey,
    ) -> Result<Option<CachedGraphReplaySelection>, BackendError> {
        let mut graphs = self.graph_cache.lock().map_err(|_| {
            BackendError::DispatchFailed {
                code: None,
                message: "CUDA compiled-pipeline graph cache lock poisoned. Fix: rebuild the compiled pipeline after a panic during graph replay.".to_string(),
            }
        })?;
        let mut first_shape_match = None;
        for (index, cached) in graphs.iter().enumerate() {
            if !cached.input_shape_matches(inputs) {
                continue;
            }
            let input_state = self
                .backend
                .prepare_cuda_graph_replay_input_state_with_key(cached, inputs, *input_key)?;
            if cached.materialized_output_cache_matches_with_input_state(inputs, &input_state)? {
                return Ok(Some(CachedGraphReplaySelection {
                    graph: graphs.swap_remove(index),
                    input_state,
                }));
            }
            if first_shape_match.is_none() {
                first_shape_match = Some((index, input_state));
            }
        }
        Ok(
            first_shape_match.map(|(index, input_state)| CachedGraphReplaySelection {
                graph: graphs.swap_remove(index),
                input_state,
            }),
        )
    }

    pub(crate) fn return_cached_graph(&self, cached: CachedCudaGraph) -> Result<(), BackendError> {
        let mut graphs = self.graph_cache.lock().map_err(|_| {
            BackendError::DispatchFailed {
                code: None,
                message: "CUDA compiled-pipeline graph cache lock poisoned while returning a graph. Fix: rebuild the compiled pipeline after a panic during graph replay.".to_string(),
            }
        })?;
        if graphs.len() < MAX_GRAPH_CACHE_ENTRIES_PER_PIPELINE {
            graphs.push(cached);
        }
        Ok(())
    }
}
