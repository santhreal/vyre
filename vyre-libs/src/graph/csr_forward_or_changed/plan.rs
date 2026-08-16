use super::dispatch_plan::CsrForwardOrChangedDispatchPlan;
use super::launch_plan::CsrForwardOrChangedLaunchPlan;
use super::layout::{
    csr_forward_or_changed_parallel_grid, CsrForwardOrChangedProgramKey,
    CSR_FORWARD_OR_CHANGED_HISTORY_FAST_PATH_MAX_ITERS,
};
use super::validate::validate_csr_inputs;
use crate::graph::csr_closure_inputs::CsrClosureInputs;

/// Validate CSR inputs and select a primitive-owned launch plan without
/// allocating the generated program.
///
/// # Errors
///
/// Returns an actionable diagnostic when CSR inputs are malformed.
pub fn plan_csr_forward_or_changed_launch(
    inputs: CsrClosureInputs<'_>,
) -> Result<CsrForwardOrChangedLaunchPlan, String> {
    let graph = inputs.graph;
    let layout = validate_csr_inputs(
        graph.node_count,
        graph.edge_offsets,
        graph.edge_targets,
        graph.edge_kind_mask,
    )?;
    let max_iters = inputs.max_iters;
    let uses_changed_history =
        max_iters > 0 && max_iters <= CSR_FORWARD_OR_CHANGED_HISTORY_FAST_PATH_MAX_ITERS;
    let changed_slots = if uses_changed_history { max_iters } else { 1 };
    Ok(CsrForwardOrChangedLaunchPlan::new(
        CsrForwardOrChangedProgramKey::new(
            layout,
            inputs.allow_mask,
            changed_slots,
            uses_changed_history,
        ),
        csr_forward_or_changed_parallel_grid(layout.node_count),
    ))
}

/// Validate CSR inputs and select the primitive-owned expansion launch plan.
///
/// # Errors
///
/// Returns an actionable diagnostic when CSR inputs are malformed or the
/// changed-history fast path cannot be represented by the primitive builders.
pub(crate) fn plan_csr_forward_or_changed_dispatch(
    inputs: CsrClosureInputs<'_>,
) -> Result<CsrForwardOrChangedDispatchPlan, String> {
    let launch = plan_csr_forward_or_changed_launch(inputs)?;
    let program = launch.program()?;

    Ok(CsrForwardOrChangedDispatchPlan::new(launch, program))
}
