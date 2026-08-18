use super::{BidirectionalGpuScratch, CachedBidirectionalProgram};
use crate::graph::csr_bidirectional::plan_csr_bidirectional_step;
use crate::graph::csr_closure_inputs::CsrClosureInputs;
use crate::graph::dispatch::csr_bidirectional::dispatch::{
    bidirectional_step_dispatch_prepared_inputs_into, refresh_bidirectional_step_inputs,
};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Dispatcher-backed bidirectional closure.
///
/// # Errors
///
/// Propagates dispatch failures from each bidirectional step.
pub fn bidirectional_closure_via(
    dispatcher: &dyn ProgramDispatcher,
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
) -> Result<Vec<u32>, DispatchError> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    bidirectional_closure_via_into(dispatcher, inputs, seed, &mut current, &mut next)?;
    Ok(current)
}

/// Dispatcher-backed bidirectional closure using caller-owned buffers.
///
/// # Errors
///
/// Propagates dispatch failures from each bidirectional step.
pub fn bidirectional_closure_via_into(
    dispatcher: &dyn ProgramDispatcher,
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = BidirectionalGpuScratch::default();
    bidirectional_closure_via_with_scratch_into(
        dispatcher,
        inputs,
        seed,
        &mut scratch,
        current,
        next,
    )
}

/// Dispatcher-backed bidirectional closure with caller-owned dispatch scratch.
///
/// # Errors
///
/// Propagates dispatch failures from each bidirectional step.
pub fn bidirectional_closure_via_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    scratch: &mut BidirectionalGpuScratch,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let graph = inputs.graph;
    let plan = plan_csr_bidirectional_step(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
        seed,
        inputs.allow_mask,
    )
    .map_err(DispatchError::BadInputs)?;

    let BidirectionalGpuScratch {
        inputs: dispatch_inputs,
        static_input_key,
        program_cache,
    } = scratch;
    let program_key = plan.program_key();
    let static_key = plan
        .static_input_key(graph.edge_offsets, graph.edge_targets, graph.edge_kind_mask)
        .map_err(DispatchError::BadInputs)?;
    crate::graph::csr_bidirectional::run_csr_bidirectional_closure_plan_with_step(
        &plan,
        seed,
        inputs.max_iters,
        current,
        next,
        DispatchError::BadInputs,
        |curr, nxt| {
            let cached =
                program_cache.get_or_insert_with(program_key, || CachedBidirectionalProgram {
                    program: plan.program(),
                });
            refresh_bidirectional_step_inputs(
                dispatch_inputs,
                static_input_key,
                static_key,
                &plan,
                graph.edge_offsets,
                graph.edge_targets,
                graph.edge_kind_mask,
                curr,
            )?;
            bidirectional_step_dispatch_prepared_inputs_into(
                dispatcher,
                &plan,
                &cached.program,
                dispatch_inputs,
                nxt,
            )
        },
    )
}
