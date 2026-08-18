//! Rule-graph change-impact as a Pearl do-calculus query (#36 substrate).
//!
//! Frames vyre's cache-invalidation as a `do(rule_X)` query on the
//! dependency graph. When rule `X` changes, `do(X)` on the graph
//! predicts which downstream Programs invalidate.
//!
//! This replaces ad-hoc cache invalidation with formal causal analysis.

#[cfg(test)]
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::dispatch_buffers::{
    ceil_div_u32, checked_square_cells, decode_u32_output_exact, ensure_input_slots,
    write_u32_slice_le_bytes, write_zero_bytes,
};
use crate::graph::do_calculus::{
    do_impact_mask_from_closure, do_intervention_delete_incoming, do_rule2_reverse_incoming,
    do_rule3_subgraph,
};
use crate::prelude::reachability_closure_via_into;
use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Reusable matrix buffers for do-calculus impact queries.
#[derive(Debug, Default)]
pub struct DoCalculusImpactScratch {
    surgically_modified_adj: Vec<u32>,
    closure: Vec<u32>,
    scratch: Vec<u32>,
    impact_mask: Vec<u32>,
    reduced_adjacency: Vec<u32>,
    kept_indices: Vec<u32>,
    dispatch_inputs: Vec<Vec<u8>>,
}

impl DoCalculusImpactScratch {
    /// Last computed impact mask.
    #[must_use]
    pub fn impact_mask(&self) -> &[u32] {
        &self.impact_mask
    }

    /// Last computed reduced adjacency.
    #[must_use]
    pub fn reduced_adjacency(&self) -> &[u32] {
        &self.reduced_adjacency
    }

    /// Original indices retained in the last reduced adjacency.
    #[must_use]
    pub fn kept_indices(&self) -> &[u32] {
        &self.kept_indices
    }
}

fn dispatch_impact_mask_from_closure_into(
    dispatcher: &dyn ProgramDispatcher,
    mask: &[u32],
    closure: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    if n == 0 {
        if !mask.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: dispatch_impact_mask_from_closure requires mask.len() == 0 for n=0, got len={}.",
                mask.len()
            )));
        }
        if !closure.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: dispatch_impact_mask_from_closure requires closure.len() == 0 for n=0, got len={}.",
                closure.len()
            )));
        }
        out.clear();
        return Ok(());
    }

    let cells = checked_square_cells(n, "dispatch_impact_mask_from_closure")?;
    if closure.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: dispatch_impact_mask_from_closure requires closure.len() == n*n, got len={}, n={n}, n*n={cells}.",
            closure.len()
        )));
    }
    if mask.len() != n as usize {
        return Err(DispatchError::BadInputs(format!(
            "Fix: dispatch_impact_mask_from_closure requires mask.len() == n, got len={}, n={n}.",
            mask.len()
        )));
    }
    let program = do_impact_mask_from_closure("mask", "closure", "out", n);
    let mask_bytes = (n as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: dispatch_impact_mask_from_closure mask byte size overflows usize for n={n}."
            ))
        })?;
    ensure_input_slots(inputs, 3);
    write_u32_slice_le_bytes(&mut inputs[0], mask);
    write_u32_slice_le_bytes(&mut inputs[1], closure);
    write_zero_bytes(&mut inputs[2], mask_bytes);
    let outputs =
        dispatcher.dispatch(&program, &inputs[..3], Some([ceil_div_u32(n, 256), 1, 1]))?;
    let [impact_out] = match outputs.as_slice() {
        [impact_out] => [impact_out],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: dispatch_impact_mask_from_closure expected exactly one output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(
        impact_out,
        n as usize,
        "dispatch_impact_mask_from_closure",
        out,
    )
}

/// GPU-backed impact prediction using primitive-native graph surgery and
/// reachability closure dispatch.
///
/// This keeps the graph rewrite and transitive closure off the CPU. The final
/// host projection only materializes the already-read-back `n`-word impact mask
/// needed by cache invalidation callers.
#[must_use = "GPU impact prediction returns a mask or dispatch error that must be handled"]
pub fn predict_impact_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut scratch = DoCalculusImpactScratch::default();
    predict_impact_via_into(dispatcher, adj, intervention_mask, n, &mut scratch)?;
    Ok(scratch.impact_mask)
}

/// GPU-backed impact prediction into caller-owned scratch.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn predict_impact_via_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) -> Result<(), DispatchError> {
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);
    if n == 0 {
        if !adj.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: predict_impact_via requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !intervention_mask.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: predict_impact_via requires intervention_mask.len() == 0 for n=0, got len={}.",
                intervention_mask.len()
            )));
        }
        scratch.impact_mask.clear();
        scratch.surgically_modified_adj.clear();
        scratch.closure.clear();
        return Ok(());
    }
    intervention_delete_incoming_via_into_with_inputs(
        dispatcher,
        adj,
        intervention_mask,
        n,
        &mut scratch.dispatch_inputs,
        &mut scratch.surgically_modified_adj,
    )?;
    reachability_closure_via_into(
        dispatcher,
        &scratch.surgically_modified_adj,
        n,
        n,
        &mut scratch.closure,
        &mut scratch.scratch,
    )?;
    dispatch_impact_mask_from_closure_into(
        dispatcher,
        intervention_mask,
        &scratch.closure,
        n,
        &mut scratch.dispatch_inputs,
        &mut scratch.impact_mask,
    )?;
    Ok(())
}

/// Primitive-native dispatcher path for Pearl Rule 1 graph surgery:
/// remove incoming edges to every intervened node.
///
/// This is the GPU-backed first stage of `predict_impact`. Full impact
/// prediction also needs reachability closure; callers that already keep the
/// closure on-device can compose this output with the closure primitive.
///
/// # Errors
///
/// Returns [`DispatchError`] when shapes are invalid, lane counts overflow,
/// or the backend returns malformed output.
pub fn intervention_delete_incoming_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    intervention_delete_incoming_via_into_with_inputs(
        dispatcher,
        adj,
        intervention_mask,
        n,
        &mut inputs,
        &mut out,
    )?;
    Ok(out)
}

/// Dispatcher-backed intervention graph surgery into caller-owned storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn intervention_delete_incoming_via_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut inputs = Vec::new();
    intervention_delete_incoming_via_into_with_inputs(
        dispatcher,
        adj,
        intervention_mask,
        n,
        &mut inputs,
        out,
    )
}

fn intervention_delete_incoming_via_into_with_inputs(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    dispatch_do_calculus_surgery_into(
        dispatcher,
        adj,
        intervention_mask,
        n,
        inputs,
        out,
        "intervention_delete_incoming_via",
        "intervention_mask",
        do_intervention_delete_incoming,
    )
}

/// Primitive-native dispatcher path for Pearl Rule 2 graph surgery:
/// reverse incoming edges to every observed/treatment node.
///
/// This is the GPU-backed first stage of `predict_impact_observation_form`.
/// Full observation-form impact also needs reachability closure; callers that
/// keep closure on-device can compose this output directly with the closure
/// primitive.
///
/// # Errors
///
/// Returns [`DispatchError`] when shapes are invalid, lane counts overflow, or
/// the backend returns malformed output.
pub fn rule2_reverse_incoming_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    treatment_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    rule2_reverse_incoming_via_into_with_inputs(
        dispatcher,
        adj,
        treatment_mask,
        n,
        &mut inputs,
        &mut out,
    )?;
    Ok(out)
}

/// Dispatcher-backed Rule 2 graph surgery into caller-owned storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn rule2_reverse_incoming_via_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    treatment_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut inputs = Vec::new();
    rule2_reverse_incoming_via_into_with_inputs(
        dispatcher,
        adj,
        treatment_mask,
        n,
        &mut inputs,
        out,
    )
}

fn rule2_reverse_incoming_via_into_with_inputs(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    treatment_mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    dispatch_do_calculus_surgery_into(
        dispatcher,
        adj,
        treatment_mask,
        n,
        inputs,
        out,
        "rule2_reverse_incoming_via",
        "treatment_mask",
        do_rule2_reverse_incoming,
    )
}

fn dispatch_do_calculus_surgery_into<F>(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
    op_name: &'static str,
    mask_buffer: &'static str,
    build_program: F,
) -> Result<(), DispatchError>
where
    F: FnOnce(&str, &str, &str, u32) -> Program,
{
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);

    if n == 0 {
        if !adj.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: {op_name} requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !mask.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: {op_name} requires {mask_buffer}.len() == 0 for n=0, got len={}.",
                mask.len()
            )));
        }
        out.clear();
        return Ok(());
    }

    let cells = checked_square_cells(n, op_name)?;
    let cells_u32 = u32::try_from(cells).map_err(|_| {
        DispatchError::BadInputs(format!(
            "Fix: {op_name} n*n exceeds the primitive u32 lane limit for n={n}."
        ))
    })?;
    if adj.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: {op_name} requires adj.len() == n*n, got len={}, n={n}, n*n={cells}.",
            adj.len()
        )));
    }
    if mask.len() != n as usize {
        return Err(DispatchError::BadInputs(format!(
            "Fix: {op_name} requires {mask_buffer}.len() == n, got len={}, n={n}.",
            mask.len()
        )));
    }

    let program = build_program("adj", mask_buffer, "out", n);
    // Real-backend dispatch-input contract (vyre-driver `role_for_buffer`): one input per
    // INPUT-CONSUMING buffer in buffer order: `adj` RO (0), `mask` RO (1), `out` plain-ReadWrite (2,
    // InputOutput). `out` is a plain-RW output, so the backend requires a zero-filled input slot for
    // its initial contents (the per-lane kernel overwrites every cell). Passing only the two RO
    // buffers would fail the backend's strict `validate_input_lengths` count.
    let out_bytes = cells
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: {op_name} out byte size overflows usize for {cells} cells."
            ))
        })?;
    ensure_input_slots(inputs, 3);
    write_u32_slice_le_bytes(&mut inputs[0], adj);
    write_u32_slice_le_bytes(&mut inputs[1], mask);
    write_zero_bytes(&mut inputs[2], out_bytes);
    let outputs = dispatcher.dispatch(
        &program,
        &inputs[..3],
        Some([ceil_div_u32(cells_u32, 256), 1, 1]),
    )?;
    let [out_buf] = match outputs.as_slice() {
        [out_buf] => [out_buf],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: {op_name} expected exactly one output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(out_buf, cells, op_name, out)
}

/// Primitive-native dispatcher path for Pearl Rule 3 graph surgery:
/// **subgraph extraction**. Restricts `adj` to the nodes whose `keep_mask` bit
/// is set, returning the dense `k × k` `reduced` block (row-major, stride `k`)
/// and the `kept` original-index map, where `k = popcount(keep_mask)`.
///
/// This is the GPU/IR counterpart of the Rule-3 subgraph-extraction oracle and
/// the missing third member of the do-calculus surgery family (the two per-cell
/// maps: [`intervention_delete_incoming_via`] / [`rule2_reverse_incoming_via`]
///: have long had a `_via` form; Rule 3, a compaction + gather with
/// data-dependent output size, did not until now). The underlying kernel
/// serializes the compaction on a single lane so the kept order is deterministic
/// (ascending original index), byte-identical to the host oracle.
///
/// # Errors
///
/// Returns [`DispatchError`] when shapes are invalid, `n * n` overflows the lane
/// limit, or the backend returns anything other than the three output buffers.
pub fn rule3_subgraph_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    keep_mask: &[u32],
    n: u32,
) -> Result<(Vec<u32>, Vec<u32>), DispatchError> {
    let mut reduced = Vec::new();
    let mut kept = Vec::new();
    let mut inputs = Vec::new();
    rule3_subgraph_via_into_with_inputs(
        dispatcher,
        adj,
        keep_mask,
        n,
        &mut inputs,
        &mut reduced,
        &mut kept,
    )?;
    Ok((reduced, kept))
}

/// Dispatcher-backed Rule 3 subgraph extraction into caller-owned storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn rule3_subgraph_via_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    keep_mask: &[u32],
    n: u32,
    reduced: &mut Vec<u32>,
    kept: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut inputs = Vec::new();
    rule3_subgraph_via_into_with_inputs(dispatcher, adj, keep_mask, n, &mut inputs, reduced, kept)
}

fn rule3_subgraph_via_into_with_inputs(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    keep_mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    reduced: &mut Vec<u32>,
    kept: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);

    if n == 0 {
        if !adj.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: rule3_subgraph_via requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !keep_mask.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: rule3_subgraph_via requires keep_mask.len() == 0 for n=0, got len={}.",
                keep_mask.len()
            )));
        }
        reduced.clear();
        kept.clear();
        return Ok(());
    }

    let cells = checked_square_cells(n, "rule3_subgraph_via")?;
    if adj.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: rule3_subgraph_via requires adj.len() == n*n, got len={}, n={n}, n*n={cells}.",
            adj.len()
        )));
    }
    if keep_mask.len() != n as usize {
        return Err(DispatchError::BadInputs(format!(
            "Fix: rule3_subgraph_via requires keep_mask.len() == n, got len={}, n={n}.",
            keep_mask.len()
        )));
    }
    let k = keep_mask.iter().filter(|&&v| v != 0).count();
    let k_cells = k.checked_mul(k).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: rule3_subgraph_via reduced k*k overflows usize for k={k}."
        ))
    })?;

    let program = do_rule3_subgraph("adj", "keep_mask", "reduced", "kept", "kept_len", n);
    // Real-backend dispatch-input contract (vyre-driver `role_for_buffer`): one input per
    // input-consuming buffer in buffer order: `adj` RO (0), `keep_mask` RO (1), then the three
    // plain-ReadWrite outputs `reduced` (2, n*n), `kept` (3, n), and `kept_len` (4, 1). Each
    // plain-RW output needs a zero-filled input slot for its initial contents.
    let reduced_bytes = cells
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: rule3_subgraph_via reduced byte size overflows usize for {cells} cells."
            ))
        })?;
    let kept_bytes = (n as usize) * std::mem::size_of::<u32>();
    ensure_input_slots(inputs, 5);
    write_u32_slice_le_bytes(&mut inputs[0], adj);
    write_u32_slice_le_bytes(&mut inputs[1], keep_mask);
    write_zero_bytes(&mut inputs[2], reduced_bytes);
    write_zero_bytes(&mut inputs[3], kept_bytes);
    write_zero_bytes(&mut inputs[4], std::mem::size_of::<u32>());
    // The kernel is lane-0-serial, so a single workgroup covers it.
    let outputs = dispatcher.dispatch(&program, &inputs[..5], Some([1, 1, 1]))?;
    let [reduced_out, kept_out, _kept_len_out] = match outputs.as_slice() {
        [r, k_out, kl] => [r, k_out, kl],
        _ => {
            return Err(DispatchError::BackendError(format!(
            "Fix: rule3_subgraph_via expected 3 output buffers (reduced, kept, kept_len), got {}.",
            outputs.len()
        )))
        }
    };

    let mut reduced_full = Vec::new();
    decode_u32_output_exact(
        reduced_out,
        cells,
        "rule3_subgraph_via reduced",
        &mut reduced_full,
    )?;
    let mut kept_full = Vec::new();
    decode_u32_output_exact(
        kept_out,
        n as usize,
        "rule3_subgraph_via kept",
        &mut kept_full,
    )?;

    reduced.clear();
    reduced.extend_from_slice(&reduced_full[..k_cells]);
    kept.clear();
    kept.extend_from_slice(&kept_full[..k]);
    Ok(())
}

/// GPU-backed observation-form impact prediction.
///
/// Uses the Rule 2 graph-surgery primitive plus GPU reachability closure. The
/// remaining host work only projects the returned closure into the `n`-word
/// mask required by cache invalidation and diagnostics.
#[must_use = "GPU observation-form impact prediction returns a mask or dispatch error that must be handled"]
pub fn predict_impact_observation_form_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    observation_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut scratch = DoCalculusImpactScratch::default();
    predict_impact_observation_form_via_into(dispatcher, adj, observation_mask, n, &mut scratch)?;
    Ok(scratch.impact_mask)
}

/// GPU-backed observation-form impact prediction into caller-owned scratch.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn predict_impact_observation_form_via_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    observation_mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) -> Result<(), DispatchError> {
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);
    if n == 0 {
        if !adj.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: predict_impact_observation_form_via requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !observation_mask.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: predict_impact_observation_form_via requires observation_mask.len() == 0 for n=0, got len={}.",
                observation_mask.len()
            )));
        }
        scratch.impact_mask.clear();
        scratch.surgically_modified_adj.clear();
        scratch.closure.clear();
        return Ok(());
    }
    rule2_reverse_incoming_via_into_with_inputs(
        dispatcher,
        adj,
        observation_mask,
        n,
        &mut scratch.dispatch_inputs,
        &mut scratch.surgically_modified_adj,
    )?;
    reachability_closure_via_into(
        dispatcher,
        &scratch.surgically_modified_adj,
        n,
        n,
        &mut scratch.closure,
        &mut scratch.scratch,
    )?;
    dispatch_impact_mask_from_closure_into(
        dispatcher,
        observation_mask,
        &scratch.closure,
        n,
        &mut scratch.dispatch_inputs,
        &mut scratch.impact_mask,
    )?;
    Ok(())
}

/// Canonical op id for projecting impacted rules and provenance closure to lineage cells.
pub(crate) const PROJECT_LINEAGE_IMPACT_OP_ID: &str =
    "vyre-libs::reasoning::do_project_impacted_lineage_entries";

/// Emit a Program that projects an `impact_mask` and `closure` through `lineage_cells`
/// into an `m`-element 0/1 invalidation mask on device.
#[must_use]
pub(crate) fn do_project_impacted_lineage_entries(
    impact_mask: &str,
    closure: &str,
    lineage_cells: &str,
    out: &str,
    n: u32,
    m: u32,
) -> Program {
    match try_do_project_impacted_lineage_entries(impact_mask, closure, lineage_cells, out, n, m) {
        Ok(program) => program,
        Err(error) => trap_program(
            PROJECT_LINEAGE_IMPACT_OP_ID,
            Some((out, DataType::U32)),
            error,
        ),
    }
}

/// Emit an impacted-lineage projection Program with checked input shapes.
///
/// # Errors
///
/// Returns an error message if `m == 0` or nonzero `n * n` overflows `u32`.
pub(crate) fn try_do_project_impacted_lineage_entries(
    impact_mask: &str,
    closure: &str,
    lineage_cells: &str,
    out: &str,
    n: u32,
    m: u32,
) -> Result<Program, String> {
    if m == 0 {
        return Err(format!(
            "Fix: {PROJECT_LINEAGE_IMPACT_OP_ID} requires m > 0."
        ));
    }
    let impact_count = n.max(1);
    let closure_count = if n == 0 {
        1
    } else {
        crate::plumbing::operand::shape::square_matrix_cells(PROJECT_LINEAGE_IMPACT_OP_ID, n)?
    };
    let j = Expr::InvocationId { axis: 0 };
    let body = vec![Node::if_then(
        Expr::lt(j.clone(), Expr::u32(m)),
        vec![
            Node::let_bind("cell", Expr::load(lineage_cells, j.clone())),
            Node::let_bind("is_impacted", Expr::u32(0)),
            Node::if_then(
                Expr::lt(Expr::var("cell"), Expr::u32(n)),
                vec![
                    Node::assign(
                        "is_impacted",
                        Expr::select(
                            Expr::ne(Expr::load(impact_mask, Expr::var("cell")), Expr::u32(0)),
                            Expr::u32(1),
                            Expr::u32(0),
                        ),
                    ),
                    Node::loop_for(
                        "k",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![
                            Node::let_bind(
                                "k_impacted",
                                Expr::ne(Expr::load(impact_mask, Expr::var("k")), Expr::u32(0)),
                            ),
                            Node::let_bind(
                                "reach",
                                Expr::ne(
                                    Expr::load(
                                        closure,
                                        Expr::add(
                                            Expr::mul(Expr::var("cell"), Expr::u32(n)),
                                            Expr::var("k"),
                                        ),
                                    ),
                                    Expr::u32(0),
                                ),
                            ),
                            Node::if_then(
                                Expr::and(Expr::var("k_impacted"), Expr::var("reach")),
                                vec![Node::assign("is_impacted", Expr::u32(1))],
                            ),
                        ],
                    ),
                ],
            ),
            Node::store(out, j, Expr::var("is_impacted")),
        ],
    )];

    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(impact_mask, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(impact_count),
            BufferDecl::storage(closure, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(closure_count),
            BufferDecl::storage(lineage_cells, 2, BufferAccess::ReadOnly, DataType::U32)
                .with_count(m),
            BufferDecl::output(out, 3, DataType::U32).with_count(m),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(PROJECT_LINEAGE_IMPACT_OP_ID, body)],
    ))
}

/// Reusable scratch for impacted lineage projection dispatch.
#[derive(Debug, Default)]
pub struct ImpactedLineageProjectionScratch {
    dispatch_inputs: Vec<Vec<u8>>,
}

/// Dispatch-backed projection of impacted rules and provenance closure through `lineage_cells`
/// into an `m`-element 0/1 mask in caller-owned storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when shapes are invalid, lane counts overflow, or the backend fails.
pub fn project_impacted_lineage_entries_via_into(
    dispatcher: &dyn ProgramDispatcher,
    impact_mask: &[u32],
    closure: &[u32],
    n: u32,
    lineage_cells: &[u32],
    scratch: &mut ImpactedLineageProjectionScratch,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    if lineage_cells.is_empty() {
        out.clear();
        return Ok(());
    }

    if n == 0 {
        if !impact_mask.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: project_impacted_lineage_entries requires impact_mask.len() == 0 for n=0, got len={}.",
                impact_mask.len()
            )));
        }
        if !closure.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: project_impacted_lineage_entries requires closure.len() == 0 for n=0, got len={}.",
                closure.len()
            )));
        }
    } else {
        let cells = checked_square_cells(n, "project_impacted_lineage_entries")?;
        if closure.len() != cells {
            return Err(DispatchError::BadInputs(format!(
                "Fix: project_impacted_lineage_entries requires closure.len() == n*n, got len={}, n={n}, n*n={cells}.",
                closure.len()
            )));
        }
        if impact_mask.len() != n as usize {
            return Err(DispatchError::BadInputs(format!(
                "Fix: project_impacted_lineage_entries requires impact_mask.len() == n, got len={}, n={n}.",
                impact_mask.len()
            )));
        }
    }
    let m = u32::try_from(lineage_cells.len()).map_err(|_| {
        DispatchError::BadInputs(format!(
            "Fix: project_impacted_lineage_entries lineage_cells.len() {} overflows u32.",
            lineage_cells.len()
        ))
    })?;
    let program =
        do_project_impacted_lineage_entries("impact_mask", "closure", "lineage_cells", "out", n, m);
    ensure_input_slots(&mut scratch.dispatch_inputs, 3);
    if n == 0 {
        write_zero_bytes(&mut scratch.dispatch_inputs[0], std::mem::size_of::<u32>());
        write_zero_bytes(&mut scratch.dispatch_inputs[1], std::mem::size_of::<u32>());
    } else {
        write_u32_slice_le_bytes(&mut scratch.dispatch_inputs[0], impact_mask);
        write_u32_slice_le_bytes(&mut scratch.dispatch_inputs[1], closure);
    }
    write_u32_slice_le_bytes(&mut scratch.dispatch_inputs[2], lineage_cells);
    let outputs = dispatcher.dispatch(
        &program,
        &scratch.dispatch_inputs[..3],
        Some([ceil_div_u32(m, 256), 1, 1]),
    )?;
    let [impact_out] = match outputs.as_slice() {
        [impact_out] => [impact_out],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: project_impacted_lineage_entries expected exactly one output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(
        impact_out,
        lineage_cells.len(),
        "project_impacted_lineage_entries",
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::Program;
    use vyre_reference::composition_witness::{
        do_rule3_subgraph_witness, do_rule3_subgraph_witness_into,
        predict_impact_observation_form_witness, predict_impact_observation_form_witness_into,
        predict_impact_witness, predict_impact_witness_into,
    };

    fn predict_impact(adj: &[u32], intervention_mask: &[u32], n: u32) -> Vec<u32> {
        predict_impact_witness(adj, intervention_mask, n)
    }

    fn predict_impact_with_scratch(
        adj: &[u32],
        intervention_mask: &[u32],
        n: u32,
        scratch: &mut DoCalculusImpactScratch,
    ) {
        predict_impact_witness_into(
            adj,
            intervention_mask,
            n,
            &mut scratch.surgically_modified_adj,
            &mut scratch.closure,
            &mut scratch.impact_mask,
        );
        scratch.scratch.clear();
        scratch.scratch.resize((n * n) as usize, 0);
    }

    fn impact_subgraph(adj: &[u32], mask: &[u32], n: u32) -> (Vec<u32>, Vec<u32>) {
        let impact = predict_impact_witness(adj, mask, n);
        do_rule3_subgraph_witness(adj, &impact, n)
    }

    fn reference_impact_subgraph_with_scratch(
        adj: &[u32],
        mask: &[u32],
        n: u32,
        scratch: &mut DoCalculusImpactScratch,
    ) {
        predict_impact_with_scratch(adj, mask, n, scratch);
        do_rule3_subgraph_witness_into(
            adj,
            &scratch.impact_mask,
            n,
            &mut scratch.reduced_adjacency,
            &mut scratch.kept_indices,
        );
    }

    fn predict_impact_observation_form(adj: &[u32], observation_mask: &[u32], n: u32) -> Vec<u32> {
        predict_impact_observation_form_witness(adj, observation_mask, n)
    }

    fn predict_impact_observation_form_with_scratch(
        adj: &[u32],
        observation_mask: &[u32],
        n: u32,
        scratch: &mut DoCalculusImpactScratch,
    ) {
        predict_impact_observation_form_witness_into(
            adj,
            observation_mask,
            n,
            &mut scratch.surgically_modified_adj,
            &mut scratch.closure,
            &mut scratch.impact_mask,
        );
        scratch.scratch.clear();
        scratch.scratch.resize((n * n) as usize, 0);
    }

    #[test]
    fn zero_node_validation_precedes_scratch_mutation() {
        struct NoDispatch;
        impl ProgramDispatcher for NoDispatch {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                panic!("invalid zero-node inputs must fail before dispatch");
            }
        }

        let assert_untouched = |scratch: &DoCalculusImpactScratch, case: &str| {
            assert_eq!(scratch.impact_mask, [11], "{case}: impact mask changed");
            assert_eq!(
                scratch.surgically_modified_adj,
                [12],
                "{case}: surgery scratch changed"
            );
            assert_eq!(scratch.closure, [13], "{case}: closure scratch changed");
        };

        let mut scratch = seeded_impact_scratch();
        let result = predict_impact_via_into(&NoDispatch, &[1], &[], 0, &mut scratch);
        assert!(matches!(result, Err(DispatchError::BadInputs(_))));
        assert_untouched(&scratch, "impact adjacency");

        let mut scratch = seeded_impact_scratch();
        let result = predict_impact_via_into(&NoDispatch, &[], &[1], 0, &mut scratch);
        assert!(matches!(result, Err(DispatchError::BadInputs(_))));
        assert_untouched(&scratch, "impact mask");

        let mut scratch = seeded_impact_scratch();
        let result =
            predict_impact_observation_form_via_into(&NoDispatch, &[1], &[], 0, &mut scratch);
        assert!(matches!(result, Err(DispatchError::BadInputs(_))));
        assert_untouched(&scratch, "observation adjacency");

        let mut scratch = seeded_impact_scratch();
        let result =
            predict_impact_observation_form_via_into(&NoDispatch, &[], &[1], 0, &mut scratch);
        assert!(matches!(result, Err(DispatchError::BadInputs(_))));
        assert_untouched(&scratch, "observation mask");
    }

    fn seeded_impact_scratch() -> DoCalculusImpactScratch {
        DoCalculusImpactScratch {
            impact_mask: vec![11],
            surgically_modified_adj: vec![12],
            closure: vec![13],
            ..DoCalculusImpactScratch::default()
        }
    }

    #[test]
    fn chain_impact() {
        // 0 -> 1 -> 2
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        // Change node 0
        let mask = vec![1, 0, 0];
        let impact = predict_impact(&adj, &mask, 3);
        // All impacted
        assert_eq!(impact, vec![1, 1, 1]);
    }

    #[test]
    fn impact_scratch_reuses_matrix_buffers() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mask = vec![1, 0, 0];
        let mut scratch = DoCalculusImpactScratch::default();
        predict_impact_with_scratch(&adj, &mask, 3, &mut scratch);
        let modified_capacity = scratch.surgically_modified_adj.capacity();
        let closure_capacity = scratch.closure.capacity();
        let temp_capacity = scratch.scratch.capacity();
        let mask_capacity = scratch.impact_mask.capacity();
        assert_eq!(scratch.impact_mask(), &[1, 1, 1]);

        predict_impact_with_scratch(&adj, &[0, 1, 0], 3, &mut scratch);
        assert_eq!(
            scratch.surgically_modified_adj.capacity(),
            modified_capacity
        );
        assert_eq!(scratch.closure.capacity(), closure_capacity);
        assert_eq!(scratch.scratch.capacity(), temp_capacity);
        assert_eq!(scratch.impact_mask.capacity(), mask_capacity);
        assert_eq!(scratch.impact_mask(), &[0, 1, 1]);
    }

    #[test]
    fn middle_chain_impact() {
        // 0 -> 1 -> 2
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        // Change node 1
        let mask = vec![0, 1, 0];
        let impact = predict_impact(&adj, &mask, 3);
        // 1 and 2 impacted, 0 not impacted
        assert_eq!(impact, vec![0, 1, 1]);
    }

    #[test]
    fn branched_impact() {
        // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        let adj = vec![0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0];
        // Change node 2
        let mask = vec![0, 0, 1, 0];
        let impact = predict_impact(&adj, &mask, 4);
        // 2 and 3 impacted
        assert_eq!(impact, vec![0, 0, 1, 1]);
    }

    #[test]
    fn disjoint_impact() {
        // 0 -> 1, 2 -> 3
        let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        // Change node 0
        let mask = vec![1, 0, 0, 0];
        let impact = predict_impact(&adj, &mask, 4);
        // 0 and 1 impacted
        assert_eq!(impact, vec![1, 1, 0, 0]);
    }

    #[test]
    fn cycle_impact() {
        // 0 -> 1, 1 -> 0, 1 -> 2
        let adj = vec![0, 1, 0, 1, 0, 1, 0, 0, 0];
        // Change node 0.
        // do(0) removes 1 -> 0.
        // 0 -> 1 -> 2 remains.
        let mask = vec![1, 0, 0];
        let impact = predict_impact(&adj, &mask, 3);
        // All impacted
        assert_eq!(impact, vec![1, 1, 1]);
    }

    #[test]
    fn empty_graph() {
        let impact = predict_impact(&[], &[], 0);
        assert!(impact.is_empty());
    }

    // ---- impact_subgraph (Rule 3 consumer) ----

    #[test]
    fn impact_subgraph_chain_extracts_downstream() {
        // 0 -> 1 -> 2. Intervene 0: impact = {0,1,2}, subgraph = full.
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mask = vec![1, 0, 0];
        let (reduced, kept) = impact_subgraph(&adj, &mask, 3);
        assert_eq!(kept, vec![0, 1, 2]);
        assert_eq!(reduced, adj);
    }

    #[test]
    fn impact_subgraph_branch_compresses_unimpacted_rows() {
        // 0 -> 1, 2 -> 3 (disjoint). Intervene 0: impact = {0,1};
        // reduced is 2×2, kept = [0, 1].
        let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mask = vec![1, 0, 0, 0];
        let (reduced, kept) = impact_subgraph(&adj, &mask, 4);
        assert_eq!(kept, vec![0, 1]);
        // Edge 0->1 preserved, 2x2 layout.
        assert_eq!(reduced, vec![0, 1, 0, 0]);
    }

    #[test]
    fn impact_subgraph_scratch_reuses_reduction_buffers() {
        let adj = vec![0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0];
        let mut scratch = DoCalculusImpactScratch::default();
        reference_impact_subgraph_with_scratch(&adj, &[1, 0, 0, 0], 4, &mut scratch);
        let reduced_capacity = scratch.reduced_adjacency.capacity();
        let kept_capacity = scratch.kept_indices.capacity();
        assert_eq!(scratch.kept_indices(), &[0, 1]);
        assert_eq!(scratch.reduced_adjacency(), &[0, 1, 0, 0]);

        reference_impact_subgraph_with_scratch(&adj, &[0, 0, 1, 0], 4, &mut scratch);
        assert_eq!(scratch.reduced_adjacency.capacity(), reduced_capacity);
        assert_eq!(scratch.kept_indices.capacity(), kept_capacity);
        assert_eq!(scratch.kept_indices(), &[2, 3]);
        assert_eq!(scratch.reduced_adjacency(), &[0, 1, 0, 0]);
    }

    #[test]
    fn impact_subgraph_empty_intervention_empty_subgraph() {
        let adj = vec![0, 1, 0, 0];
        let mask = vec![0, 0];
        let (reduced, kept) = impact_subgraph(&adj, &mask, 2);
        assert!(reduced.is_empty());
        assert!(kept.is_empty());
    }

    #[test]
    fn impact_subgraph_empty_graph() {
        let (r, k) = impact_subgraph(&[], &[], 0);
        assert!(r.is_empty());
        assert!(k.is_empty());
    }

    /// Closure-bar test: the reduced adjacency must have **exactly**
    /// `kept.len()²` cells AND every cell must equal the original
    /// adjacency restricted to the corresponding kept-index pair. If
    /// the consumer ever drifts (off-by-one indexing into the kept
    /// vector, mis-sized output buffer, etc.) this test fires.
    #[test]
    fn impact_subgraph_size_invariant_holds_under_partial_impact() {
        // 0 -> 1 -> 2, plus disjoint 3 -> 4. Intervene 1.
        // Impact = {1, 2}; subgraph keeps those two with edge 1->2.
        let adj = vec![
            0, 1, 0, 0, 0, // 0 -> 1
            0, 0, 1, 0, 0, // 1 -> 2
            0, 0, 0, 0, 0, // 2
            0, 0, 0, 0, 1, // 3 -> 4
            0, 0, 0, 0, 0, // 4
        ];
        let mask = vec![0, 1, 0, 0, 0];
        let (reduced, kept) = impact_subgraph(&adj, &mask, 5);
        // Exact size invariant.
        assert_eq!(reduced.len(), kept.len() * kept.len());
        assert_eq!(kept, vec![1, 2]);
        // Edge 1->2 preserved at (0,1) in the reduced 2×2.
        assert_eq!(reduced, vec![0, 1, 0, 0]);
    }

    /// Adversarial: intervention on a leaf must not pull in upstream
    /// nodes. `do(leaf)` only impacts leaf itself; if the consumer
    /// accidentally also kept ancestors, the kept vec would grow.
    #[test]
    fn impact_subgraph_adversarial_leaf_intervention_keeps_only_leaf() {
        // 0 -> 1 -> 2. Intervene 2 (leaf).
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mask = vec![0, 0, 1];
        let (reduced, kept) = impact_subgraph(&adj, &mask, 3);
        assert_eq!(kept, vec![2]);
        // 1×1, value = adj[2,2] = 0.
        assert_eq!(reduced, vec![0]);
    }

    /// Adversarial: every edge between kept nodes must survive in
    /// the reduced adjacency, and no edge to a dropped node may
    /// appear. A common bug is to copy the edge weight from the
    /// wrong (i, j) cell of the original  -  a permutation error.
    #[test]
    fn impact_subgraph_adversarial_dense_must_drop_unkept_edges() {
        // K3 over {0,1,2} plus isolated 3.
        let adj = vec![
            0, 1, 1, 0, // 0 -> 1, 0 -> 2
            1, 0, 1, 0, // 1 -> 0, 1 -> 2
            1, 1, 0, 0, // 2 -> 0, 2 -> 1
            0, 0, 0, 0, // 3 isolated
        ];
        // Intervene 0: rule-1 impact closure walks 0 -> 1 -> 2.
        let mask = vec![1, 0, 0, 0];
        let (reduced, kept) = impact_subgraph(&adj, &mask, 4);
        assert_eq!(kept, vec![0, 1, 2]);
        // Reduced is the original 3×3 corner. Every original edge
        // among {0,1,2} preserved; no row/col for 3.
        assert_eq!(
            reduced,
            vec![
                0, 1, 1, // 0 -> 1, 0 -> 2
                1, 0, 1, // 1 -> 0, 1 -> 2
                1, 1, 0, // 2 -> 0, 2 -> 1
            ]
        );
    }

    // ---- predict_impact_observation_form (Rule 2 consumer) ----

    /// On a DAG, observation-form impact equals intervention-form
    /// impact at the observed node itself (no feedback edges to
    /// reverse).
    #[test]
    fn observation_form_dag_observed_self_only() {
        // 0 -> 1 -> 2 (no incoming edges into observed node 0).
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mask = vec![1, 0, 0];
        let observed = predict_impact_observation_form(&adj, &mask, 3);
        let intervened = predict_impact(&adj, &mask, 3);
        // On this DAG, observing 0 = intervening on 0.
        assert_eq!(observed, intervened);
    }

    #[test]
    fn observation_form_scratch_reuses_buffers() {
        let adj = vec![0, 1, 0, 0, 0, 1, 0, 0, 0];
        let mut scratch = DoCalculusImpactScratch::default();
        predict_impact_observation_form_with_scratch(&adj, &[1, 0, 0], 3, &mut scratch);
        let reversed_capacity = scratch.surgically_modified_adj.capacity();
        let closure_capacity = scratch.closure.capacity();
        assert_eq!(scratch.impact_mask(), &[1, 1, 1]);

        predict_impact_observation_form_with_scratch(&adj, &[0, 1, 0], 3, &mut scratch);
        assert_eq!(
            scratch.surgically_modified_adj.capacity(),
            reversed_capacity
        );
        assert_eq!(scratch.closure.capacity(), closure_capacity);
        assert_eq!(scratch.impact_mask(), &[1, 1, 1]);
    }

    /// Closure-bar: observation-form must include the observed node
    /// itself as impact.
    #[test]
    fn observation_form_marks_observed_node() {
        let adj = vec![0, 1, 0, 0];
        let mask = vec![0, 1];
        let impact = predict_impact_observation_form(&adj, &mask, 2);
        assert_eq!(impact[1], 1, "observed node must be in impact set");
    }

    /// Adversarial: feedback loop into observed node. Rule-2 reverses
    /// the loop edge, so observation-form sees the loop's source as
    /// reachable along the reversed edge.
    #[test]
    fn observation_form_walks_reversed_feedback_edge() {
        // 0 -> 1, 1 -> 0 (mutual feedback), 1 -> 2.
        // Observe 0. Rule-2 reverses 1 -> 0 to 0 -> 1 (already exists,
        // OR-merged); it does NOT reverse 0 -> 1 (target is 0 only).
        // Reachable from 0 in modified graph: 0, 1, 2.
        let adj = vec![0, 1, 0, 1, 0, 1, 0, 0, 0];
        let mask = vec![1, 0, 0];
        let impact = predict_impact_observation_form(&adj, &mask, 3);
        assert_eq!(impact, vec![1, 1, 1]);
    }

    /// Adversarial: empty observation yields empty impact.
    #[test]
    fn observation_form_empty_mask_yields_empty() {
        let adj = vec![0, 1, 0, 0];
        let mask = vec![0, 0];
        let impact = predict_impact_observation_form(&adj, &mask, 2);
        assert_eq!(impact, vec![0, 0]);
    }

    /// Adversarial: empty graph returns empty result.
    #[test]
    fn observation_form_empty_graph() {
        assert!(predict_impact_observation_form(&[], &[], 0).is_empty());
    }

    fn assert_mock_dispatch_contract(inputs: &[Vec<u8>], grid_override: Option<[u32; 3]>, expected_len: usize) {
        assert_eq!(grid_override, Some([1, 1, 1]));
        assert_eq!(inputs.len(), expected_len);
    }

    struct InterventionDispatcher;

    impl ProgramDispatcher for InterventionDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_mock_dispatch_contract(inputs, grid_override, 3);
            let adj = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let mask = crate::dispatch_buffers::read_u32s(&inputs[1]);
            let n = mask.len();
            let mut out = adj;
            for j in 0..n {
                if mask[j] != 0 {
                    for i in 0..n {
                        out[i * n + j] = 0;
                    }
                }
            }
            Ok(vec![u32_slice_to_le_bytes(&out)])
        }
    }

    #[test]
    fn intervention_delete_incoming_via_dispatches_rule1() {
        let adj = vec![1, 2, 3, 4];
        let out =
            intervention_delete_incoming_via(&InterventionDispatcher, &adj, &[1, 0], 2).unwrap();
        assert_eq!(out, vec![0, 2, 0, 4]);
    }

    #[test]
    fn intervention_delete_incoming_via_rejects_bad_shape() {
        let err = intervention_delete_incoming_via(&InterventionDispatcher, &[1, 2, 3], &[1, 0], 2)
            .unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    struct Rule2Dispatcher;

    impl ProgramDispatcher for Rule2Dispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_mock_dispatch_contract(inputs, grid_override, 3);
            let adj = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let mask = crate::dispatch_buffers::read_u32s(&inputs[1]);
            let n = mask.len();
            let mut out = vec![0u32; n * n];
            for row in 0..n {
                for col in 0..n {
                    let idx = row * n + col;
                    if row == col {
                        out[idx] = adj[idx];
                        continue;
                    }
                    if mask[col] == 0 {
                        out[idx] |= adj[idx];
                    }
                    if mask[row] != 0 {
                        out[idx] |= adj[col * n + row];
                    }
                }
            }
            Ok(vec![u32_slice_to_le_bytes(&out)])
        }
    }

    #[test]
    fn rule2_reverse_incoming_via_dispatches_rule2() {
        let adj = vec![
            0, 1, 0, //
            0, 0, 1, //
            0, 0, 0,
        ];
        let out = rule2_reverse_incoming_via(&Rule2Dispatcher, &adj, &[0, 1, 0], 3).unwrap();
        assert_eq!(
            out,
            vec![
                0, 0, 0, //
                1, 0, 1, //
                0, 0, 0,
            ]
        );
    }

    #[test]
    fn rule2_reverse_incoming_via_preserves_bidirectional_fully_treated_edges() {
        let adj = vec![0, 1, 1, 0];
        let out = rule2_reverse_incoming_via(&Rule2Dispatcher, &adj, &[1, 1], 2).unwrap();
        assert_eq!(out, adj);
    }

    #[test]
    fn rule2_reverse_incoming_via_rejects_bad_shape() {
        let err = rule2_reverse_incoming_via(&Rule2Dispatcher, &[1, 2, 3], &[1, 0], 2).unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    #[test]
    fn intervention_delete_incoming_via_handles_zero_nodes() {
        let out = intervention_delete_incoming_via(&InterventionDispatcher, &[], &[], 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn intervention_delete_incoming_via_rejects_non_empty_when_n_zero() {
        let err =
            intervention_delete_incoming_via(&InterventionDispatcher, &[1], &[], 0).unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    #[test]
    fn intervention_delete_incoming_via_rejects_extra_outputs() {
        struct ExtraOutDispatcher;
        impl ProgramDispatcher for ExtraOutDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 2, 0, 4]),
                    u32_slice_to_le_bytes(&[0, 0]),
                ])
            }
        }
        let err = intervention_delete_incoming_via(&ExtraOutDispatcher, &[1, 2, 3, 4], &[1, 0], 2)
            .unwrap_err();
        assert!(matches!(err, DispatchError::BackendError(_)));
    }

    #[test]
    fn rule2_reverse_incoming_via_handles_zero_nodes() {
        let out = rule2_reverse_incoming_via(&Rule2Dispatcher, &[], &[], 0).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn rule2_reverse_incoming_via_rejects_extra_outputs() {
        struct ExtraOutDispatcher;
        impl ProgramDispatcher for ExtraOutDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 1, 0]),
                    u32_slice_to_le_bytes(&[0]),
                ])
            }
        }
        let err =
            rule2_reverse_incoming_via(&ExtraOutDispatcher, &[0, 1, 1, 0], &[1, 1], 2).unwrap_err();
        assert!(matches!(err, DispatchError::BackendError(_)));
    }

    #[test]
    fn rule3_subgraph_via_handles_zero_nodes() {
        struct DummyDispatcher;
        impl ProgramDispatcher for DummyDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                panic!("dispatch should not be invoked for n=0");
            }
        }
        let (reduced, kept) = rule3_subgraph_via(&DummyDispatcher, &[], &[], 0).unwrap();
        assert!(reduced.is_empty());
        assert!(kept.is_empty());
    }

    #[test]
    fn rule3_subgraph_via_derives_shape_from_inputs_not_gpu_scalar() {
        struct CorruptedRedundantScalarDispatcher;
        impl ProgramDispatcher for CorruptedRedundantScalarDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 0, 0]),
                    u32_slice_to_le_bytes(&[0, 1]),
                    u32_slice_to_le_bytes(&[999]),
                ])
            }
        }
        let (reduced, kept) = rule3_subgraph_via(
            &CorruptedRedundantScalarDispatcher,
            &[0, 1, 0, 0],
            &[1, 1],
            2,
        )
        .unwrap();
        assert_eq!(reduced, [0, 1, 0, 0]);
        assert_eq!(kept, [0, 1]);
    }

    #[test]
    fn rule3_subgraph_via_rejects_missing_outputs() {
        struct MissingOutDispatcher;
        impl ProgramDispatcher for MissingOutDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 0, 0]),
                    u32_slice_to_le_bytes(&[0, 1]),
                ])
            }
        }
        let err = rule3_subgraph_via(&MissingOutDispatcher, &[0, 1, 0, 0], &[1, 1], 2).unwrap_err();
        assert!(matches!(err, DispatchError::BackendError(_)));
    }

    #[test]
    fn rule3_subgraph_via_rejects_extra_outputs() {
        struct ExtraOutDispatcher;
        impl ProgramDispatcher for ExtraOutDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[0, 1, 0, 0]),
                    u32_slice_to_le_bytes(&[0, 1]),
                    u32_slice_to_le_bytes(&[0]),
                    u32_slice_to_le_bytes(&[0]),
                ])
            }
        }
        let err = rule3_subgraph_via(&ExtraOutDispatcher, &[0, 1, 0, 0], &[1, 1], 2).unwrap_err();
        assert!(matches!(err, DispatchError::BackendError(_)));
    }
    #[test]
    fn project_impacted_lineage_entries_handles_empty_lineage() {
        struct PanicDispatcher;
        impl ProgramDispatcher for PanicDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                _inputs: &[Vec<u8>],
                _grid: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                panic!("empty lineage cells must not dispatch");
            }
        }
        let mut out = vec![99; 4];
        let mut scratch = ImpactedLineageProjectionScratch::default();
        project_impacted_lineage_entries_via_into(
            &PanicDispatcher,
            &[1, 0],
            &[0; 4],
            2,
            &[],
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn project_impacted_lineage_entries_parity_via_reference() {
        use vyre_driver_reference::ReferenceEvalDispatcher;
        let dispatcher = ReferenceEvalDispatcher;
        let impact_mask = vec![1, 0, 0];
        let mut closure = vec![0u32; 9];
        closure[2 * 3 + 0] = 1; // 2 -> 0
        let lineage_cells = vec![0, 1, 2, 99];
        let mut out = Vec::new();
        let mut scratch = ImpactedLineageProjectionScratch::default();
        project_impacted_lineage_entries_via_into(
            &dispatcher,
            &impact_mask,
            &closure,
            3,
            &lineage_cells,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![1, 0, 1, 0]);
    }

    #[test]
    fn project_impacted_lineage_entries_zero_n_nonempty_lineage_dispatches_zeros() {
        use vyre_driver_reference::ReferenceEvalDispatcher;
        let dispatcher = ReferenceEvalDispatcher;
        let lineage_cells = vec![0, 1, 99];
        let mut out = Vec::new();
        let mut scratch = ImpactedLineageProjectionScratch::default();
        project_impacted_lineage_entries_via_into(
            &dispatcher,
            &[],
            &[],
            0,
            &lineage_cells,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![0, 0, 0]);
    }
}
