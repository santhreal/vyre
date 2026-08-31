//! Host-side dispatch of the CSR topological sort.
//!
//! The returned order is validated against the graph before it is handed back,
//! so a cycle surfaces as a rejected order rather than as a plausible one.

use super::{CachedToposortProgram, ToposortGpuScratch};
use crate::graph::dispatch::dispatch_bridge::{
    dispatch_single_u32_output_from_prepared_into, refresh_keyed_dispatch_inputs, DispatchInput,
};
use crate::graph::toposort::{
    plan_toposort_csr_dispatch, validate_toposort_csr_order, ToposortCsrDispatchPlan,
    ToposortCsrError, ToposortCsrStaticInputKey, TOPOSORT_INDEGREE_SCRATCH_BUFFER,
    TOPOSORT_ORDER_OUT_BUFFER, TOPOSORT_QUEUE_SCRATCH_BUFFER,
};
use vyre_megakernel::{SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor};

/// Topologically sort a dependency graph through the dispatcher using the
/// primitive-native CSR representation.
///
/// `offsets` has `node_count + 1` entries and `targets` stores outgoing edges
/// from each prerequisite node to its dependent nodes. This is the adjacency
/// shape consumed by the primitive topological-sort dispatch plan.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when CSR shape validation fails, the backend
/// rejects the primitive, or the returned order is not a full permutation of
/// `0..node_count` (cycle or malformed backend output).
pub fn topo_order_csr_via(
    dispatcher: &impl SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut scratch = ToposortGpuScratch::default();
    let mut order = Vec::new();
    topo_order_csr_via_with_scratch_into(
        dispatcher,
        policy,
        node_count,
        offsets,
        targets,
        &mut scratch,
        &mut order,
    )?;
    Ok(order)
}

/// Topologically sort a dependency graph through the dispatcher using caller-owned scratch.
pub fn topo_order_csr_via_with_scratch(
    dispatcher: &impl SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    scratch: &mut ToposortGpuScratch,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut order = Vec::new();
    topo_order_csr_via_with_scratch_into(
        dispatcher, policy, node_count, offsets, targets, scratch, &mut order,
    )?;
    Ok(order)
}

/// Topologically sort a dependency graph into caller-owned output storage.
pub fn topo_order_csr_via_with_scratch_into(
    dispatcher: &impl SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    scratch: &mut ToposortGpuScratch,
    order: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, toposort_calls};
    bump(&toposort_calls);

    let plan =
        plan_toposort_csr_dispatch(node_count, offsets, targets).map_err(map_toposort_csr_error)?;
    if plan.layout.node_count == 0 {
        order.clear();
        return Ok(());
    }

    let ToposortGpuScratch {
        inputs,
        program_cache,
        static_input_key,
    } = scratch;
    let cached =
        program_cache.get_or_insert_with(plan.layout.node_count, || CachedToposortProgram {
            program: plan.program(),
        });
    refresh_toposort_inputs(inputs, static_input_key, &plan, offsets, targets)?;
    dispatch_single_u32_output_from_prepared_into(
        dispatcher,
        policy,
        cached.program.clone(),
        inputs,
        plan.node_words,
        TOPOSORT_ORDER_OUT_BUFFER,
        order,
    )?;
    validate_toposort_csr_order(node_count, offsets, targets, order).map_err(map_toposort_csr_error)
}

fn refresh_toposort_inputs(
    inputs: &mut Vec<Vec<u8>>,
    current_key: &mut Option<ToposortCsrStaticInputKey>,
    plan: &ToposortCsrDispatchPlan,
    offsets: &[u32],
    targets: &[u32],
) -> Result<(), SemanticExecutionError> {
    let next_key = plan
        .static_input_key(offsets, targets)
        .map_err(map_toposort_csr_error)?;
    refresh_keyed_dispatch_inputs(
        inputs,
        current_key,
        next_key,
        &[
            DispatchInput::U32Slice(offsets),
            DispatchInput::U32Slice(targets),
            DispatchInput::ZeroU32Words {
                words: plan.node_words,
                context: TOPOSORT_INDEGREE_SCRATCH_BUFFER,
            },
            DispatchInput::ZeroU32Words {
                words: plan.node_words,
                context: TOPOSORT_QUEUE_SCRATCH_BUFFER,
            },
        ],
        &[
            (
                2,
                DispatchInput::ZeroU32Words {
                    words: plan.node_words,
                    context: TOPOSORT_INDEGREE_SCRATCH_BUFFER,
                },
            ),
            (
                3,
                DispatchInput::ZeroU32Words {
                    words: plan.node_words,
                    context: TOPOSORT_QUEUE_SCRATCH_BUFFER,
                },
            ),
        ],
    )?;
    Ok(())
}

fn map_toposort_csr_error(error: ToposortCsrError) -> SemanticExecutionError {
    // `ToposortCsrError` is `#[non_exhaustive]`, which stops another crate from
    // matching it without a wildcard. Both ends live in `vyre-libs` now, so a
    // new variant fails this match at compile time instead of falling into a
    // catch-all that reported the variant name to the caller as a backend error.
    match error {
        ToposortCsrError::BadCsr { message } => SemanticExecutionError::InvalidRequest(message),
        ToposortCsrError::BadOrder { message } => SemanticExecutionError::Backend(message),
    }
}
