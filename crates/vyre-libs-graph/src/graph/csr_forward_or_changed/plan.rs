//! Input validation and program selection, with no IR built yet.
//!
//! The changed-history fast path is chosen here, from the iteration ceiling in
//! [`super::layout`], so the selection rule lives beside the validation that
//! makes it legal rather than inside a program builder.

use super::launch_plan::CsrForwardOrChangedLaunchPlan;
use super::layout::{
    CsrForwardOrChangedProgramKey, CSR_FORWARD_OR_CHANGED_HISTORY_FAST_PATH_MAX_ITERS,
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
    ))
}
