//! Rule-graph change-impact as a Pearl do-calculus query.
//!
//! Frames vyre's cache-invalidation as a `do(rule_X)` query on the
//! dependency graph. When rule `X` changes, `do(X)` on the graph
//! predicts which downstream Programs invalidate.
//!
//! This replaces ad-hoc cache invalidation with formal causal analysis.

use crate::dispatch_buffers::{
    checked_square_cells, decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    write_zero_bytes,
};
use crate::graph::do_calculus::{
    impact_mask_from_closure, intervention_delete_incoming, rule2_reverse_incoming, rule3_subgraph,
};
use crate::prelude::reachability_closure_via_into;
use vyre_foundation::composition::{trap_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

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
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    mask: &[u32],
    closure: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    if n == 0 {
        if !mask.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: dispatch_impact_mask_from_closure requires mask.len() == 0 for n=0, got len={}.",
            mask.len()
        )));
        }
        if !closure.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: dispatch_impact_mask_from_closure requires closure.len() == 0 for n=0, got len={}.",
            closure.len()
        )));
        }
        out.clear();
        return Ok(());
    }

    let cells = checked_square_cells(n, "dispatch_impact_mask_from_closure")?;
    if closure.len() != cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: dispatch_impact_mask_from_closure requires closure.len() == n*n, got len={}, n={n}, n*n={cells}.",
        closure.len()
    )));
    }
    if mask.len() != n as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: dispatch_impact_mask_from_closure requires mask.len() == n, got len={}, n={n}.",
            mask.len()
        )));
    }
    let program = impact_mask_from_closure("mask", "closure", "out", n);
    let mask_bytes = (n as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: dispatch_impact_mask_from_closure mask byte size overflows usize for n={n}."
            ))
        })?;
    ensure_input_slots(inputs, 3);
    write_u32_slice_le_bytes(&mut inputs[0], mask);
    write_u32_slice_le_bytes(&mut inputs[1], closure);
    write_zero_bytes(&mut inputs[2], mask_bytes);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &inputs[..3],
        policy,
    )
    .map(|output| output.outputs)?;
    let [impact_out] = match outputs.as_slice() {
        [impact_out] => [impact_out],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
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
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut scratch = DoCalculusImpactScratch::default();
    predict_impact_via_into(dispatcher, policy, adj, intervention_mask, n, &mut scratch)?;
    Ok(scratch.impact_mask)
}

/// GPU-backed impact prediction into caller-owned scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation or semantic execution fails.
pub fn predict_impact_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);
    if n == 0 {
        if !adj.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: predict_impact_via requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !intervention_mask.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
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
        policy,
        adj,
        intervention_mask,
        n,
        &mut scratch.dispatch_inputs,
        &mut scratch.surgically_modified_adj,
    )?;
    reachability_closure_via_into(
        dispatcher,
        policy,
        &scratch.surgically_modified_adj,
        n,
        n,
        &mut scratch.closure,
        &mut scratch.scratch,
    )?;
    dispatch_impact_mask_from_closure_into(
        dispatcher,
        policy,
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
/// Returns [`SemanticExecutionError`] when shapes are invalid, lane counts overflow,
/// or the admitted artifact produces malformed output.
pub fn intervention_delete_incoming_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    intervention_delete_incoming_via_into_with_inputs(
        dispatcher,
        policy,
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
/// Returns [`SemanticExecutionError`] when validation or semantic execution fails.
pub fn intervention_delete_incoming_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut inputs = Vec::new();
    intervention_delete_incoming_via_into_with_inputs(
        dispatcher,
        policy,
        adj,
        intervention_mask,
        n,
        &mut inputs,
        out,
    )
}

fn intervention_delete_incoming_via_into_with_inputs(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    intervention_mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    dispatch_do_calculus_surgery_into(
        dispatcher,
        policy,
        adj,
        intervention_mask,
        n,
        inputs,
        out,
        "intervention_delete_incoming_via",
        "intervention_mask",
        intervention_delete_incoming,
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
/// Returns [`SemanticExecutionError`] when shapes are invalid, lane counts overflow,
/// or the admitted artifact produces malformed output.
pub fn rule2_reverse_incoming_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    treatment_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    let mut inputs = Vec::new();
    rule2_reverse_incoming_via_into_with_inputs(
        dispatcher,
        policy,
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
/// Returns [`SemanticExecutionError`] when validation or semantic execution fails.
pub fn rule2_reverse_incoming_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    treatment_mask: &[u32],
    n: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut inputs = Vec::new();
    rule2_reverse_incoming_via_into_with_inputs(
        dispatcher,
        policy,
        adj,
        treatment_mask,
        n,
        &mut inputs,
        out,
    )
}

fn rule2_reverse_incoming_via_into_with_inputs(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    treatment_mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    dispatch_do_calculus_surgery_into(
        dispatcher,
        policy,
        adj,
        treatment_mask,
        n,
        inputs,
        out,
        "rule2_reverse_incoming_via",
        "treatment_mask",
        rule2_reverse_incoming,
    )
}

fn dispatch_do_calculus_surgery_into<F>(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    out: &mut Vec<u32>,
    op_name: &'static str,
    mask_buffer: &'static str,
    build_program: F,
) -> Result<(), SemanticExecutionError>
where
    F: FnOnce(&str, &str, &str, u32) -> Program,
{
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);

    if n == 0 {
        if !adj.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: {op_name} requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !mask.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: {op_name} requires {mask_buffer}.len() == 0 for n=0, got len={}.",
                mask.len()
            )));
        }
        out.clear();
        return Ok(());
    }

    let cells = checked_square_cells(n, op_name)?;
    let cells_u32 = u32::try_from(cells).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {op_name} n*n exceeds the primitive u32 lane limit for n={n}."
        ))
    })?;
    if adj.len() != cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {op_name} requires adj.len() == n*n, got len={}, n={n}, n*n={cells}.",
            adj.len()
        )));
    }
    if mask.len() != n as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
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
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: {op_name} out byte size overflows usize for {cells} cells."
            ))
        })?;
    ensure_input_slots(inputs, 3);
    write_u32_slice_le_bytes(&mut inputs[0], adj);
    write_u32_slice_le_bytes(&mut inputs[1], mask);
    write_zero_bytes(&mut inputs[2], out_bytes);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &inputs[..3],
        policy,
    )
    .map(|output| output.outputs)?;
    let [out_buf] = match outputs.as_slice() {
        [out_buf] => [out_buf],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
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
/// Returns [`SemanticExecutionError`] when shapes are invalid, `n * n` overflows the lane
/// limit, or the admitted artifact does not produce the three required output buffers.
pub fn rule3_subgraph_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    keep_mask: &[u32],
    n: u32,
) -> Result<(Vec<u32>, Vec<u32>), SemanticExecutionError> {
    let mut reduced = Vec::new();
    let mut kept = Vec::new();
    let mut inputs = Vec::new();
    rule3_subgraph_via_into_with_inputs(
        dispatcher,
        policy,
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
/// Returns [`SemanticExecutionError`] when validation or semantic execution fails.
pub fn rule3_subgraph_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    keep_mask: &[u32],
    n: u32,
    reduced: &mut Vec<u32>,
    kept: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut inputs = Vec::new();
    rule3_subgraph_via_into_with_inputs(
        dispatcher,
        policy,
        adj,
        keep_mask,
        n,
        &mut inputs,
        reduced,
        kept,
    )
}

fn rule3_subgraph_via_into_with_inputs(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    keep_mask: &[u32],
    n: u32,
    inputs: &mut Vec<Vec<u8>>,
    reduced: &mut Vec<u32>,
    kept: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);

    if n == 0 {
        if !adj.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: rule3_subgraph_via requires adj.len() == 0 for n=0, got len={}.",
                adj.len()
            )));
        }
        if !keep_mask.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
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
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: rule3_subgraph_via requires adj.len() == n*n, got len={}, n={n}, n*n={cells}.",
            adj.len()
        )));
    }
    if keep_mask.len() != n as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: rule3_subgraph_via requires keep_mask.len() == n, got len={}, n={n}.",
            keep_mask.len()
        )));
    }
    let k = keep_mask.iter().filter(|&&v| v != 0).count();
    let k_cells = k.checked_mul(k).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: rule3_subgraph_via reduced k*k overflows usize for k={k}."
        ))
    })?;

    let program = rule3_subgraph("adj", "keep_mask", "reduced", "kept", "kept_len", n);
    // Real-backend dispatch-input contract (vyre-driver `role_for_buffer`): one input per
    // input-consuming buffer in buffer order: `adj` RO (0), `keep_mask` RO (1), then the three
    // plain-ReadWrite outputs `reduced` (2, n*n), `kept` (3, n), and `kept_len` (4, 1). Each
    // plain-RW output needs a zero-filled input slot for its initial contents.
    let reduced_bytes = cells
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
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
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &inputs[..5],
        policy,
    )
    .map(|output| output.outputs)?;
    let [reduced_out, kept_out, _kept_len_out] = match outputs.as_slice() {
        [r, k_out, kl] => [r, k_out, kl],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
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
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    observation_mask: &[u32],
    n: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut scratch = DoCalculusImpactScratch::default();
    predict_impact_observation_form_via_into(
        dispatcher,
        policy,
        adj,
        observation_mask,
        n,
        &mut scratch,
    )?;
    Ok(scratch.impact_mask)
}

/// GPU-backed observation-form impact prediction into caller-owned scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation or semantic execution fails.
pub fn predict_impact_observation_form_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    adj: &[u32],
    observation_mask: &[u32],
    n: u32,
    scratch: &mut DoCalculusImpactScratch,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, do_calculus_change_impact_calls};
    bump(&do_calculus_change_impact_calls);
    if n == 0 {
        if !adj.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: predict_impact_observation_form_via requires adj.len() == 0 for n=0, got len={}.",
            adj.len()
        )));
        }
        if !observation_mask.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
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
        policy,
        adj,
        observation_mask,
        n,
        &mut scratch.dispatch_inputs,
        &mut scratch.surgically_modified_adj,
    )?;
    reachability_closure_via_into(
        dispatcher,
        policy,
        &scratch.surgically_modified_adj,
        n,
        n,
        &mut scratch.closure,
        &mut scratch.scratch,
    )?;
    dispatch_impact_mask_from_closure_into(
        dispatcher,
        policy,
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
    let j = Expr::LogicalIndex { axis: 0 };
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
/// Returns [`SemanticExecutionError`] when shapes are invalid or semantic execution fails.
pub fn project_impacted_lineage_entries_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    impact_mask: &[u32],
    closure: &[u32],
    n: u32,
    lineage_cells: &[u32],
    scratch: &mut ImpactedLineageProjectionScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    if lineage_cells.is_empty() {
        out.clear();
        return Ok(());
    }

    if n == 0 {
        if !impact_mask.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: project_impacted_lineage_entries requires impact_mask.len() == 0 for n=0, got len={}.",
            impact_mask.len()
        )));
        }
        if !closure.is_empty() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: project_impacted_lineage_entries requires closure.len() == 0 for n=0, got len={}.",
            closure.len()
        )));
        }
    } else {
        let cells = checked_square_cells(n, "project_impacted_lineage_entries")?;
        if closure.len() != cells {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: project_impacted_lineage_entries requires closure.len() == n*n, got len={}, n={n}, n*n={cells}.",
            closure.len()
        )));
        }
        if impact_mask.len() != n as usize {
            return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: project_impacted_lineage_entries requires impact_mask.len() == n, got len={}, n={n}.",
            impact_mask.len()
        )));
        }
    }
    let m = u32::try_from(lineage_cells.len()).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
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
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.dispatch_inputs[..3],
        policy,
    )
    .map(|output| output.outputs)?;
    let [impact_out] = match outputs.as_slice() {
        [impact_out] => [impact_out],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
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
#[path = "do_calculus_change_impact_tests.rs"]
mod tests;
