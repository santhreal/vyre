//! Semantic execution of the forward-or-changed fixpoint graph.

use std::collections::BTreeMap;

use crate::graph::csr_closure_inputs::CsrClosureInputs;
use crate::graph::csr_forward_or_changed::{
    copy_csr_forward_seed_frontier_into, plan_csr_forward_or_changed_launch,
    validate_csr_forward_or_changed_flag, CsrForwardOrChangedLaunchPlan,
    CsrForwardOrChangedProgramKey, CsrForwardOrChangedStaticInputKey,
};
use crate::graph::dispatch::dispatch_bridge::{
    refresh_keyed_dispatch_inputs, CachedProgram, DispatchInput, ProgramCache,
};
use crate::plumbing::host::scratch::reserve_vec as reserve_graph_vec;
use vyre_foundation::ir::{
    GraphInput, GraphOutput, GraphValueId, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};

/// Caller-owned GPU dispatch scratch for `csr_forward_or_changed` fixpoint loops.
#[derive(Debug, Default)]
pub struct ForwardChangedGpuScratch {
    pub(super) inputs: Vec<Vec<u8>>,
    changed_out: Vec<u32>,
    static_input_key: Option<CsrForwardOrChangedStaticInputKey>,
    program_cache: ProgramCache<CsrForwardOrChangedProgramKey, CachedForwardChangedProgram>,
}

type CachedForwardChangedProgram = CachedProgram;

impl ForwardChangedGpuScratch {
    #[cfg(test)]
    pub(super) fn program_builds(&self) -> usize {
        self.program_cache.builds()
    }

    #[cfg(test)]
    pub(super) fn with_input_capacities(
        input_capacities: &[usize],
        changed_capacity: usize,
    ) -> Self {
        let mut inputs = Vec::new();
        inputs.reserve_exact(input_capacities.len());
        for &capacity in input_capacities {
            let mut input = Vec::new();
            input.reserve_exact(capacity);
            inputs.push(input);
        }
        let mut changed_out = Vec::new();
        changed_out.reserve_exact(changed_capacity);
        Self {
            inputs,
            changed_out,
            static_input_key: None,
            program_cache: ProgramCache::default(),
        }
    }
}

/// Dispatcher-backed closure: build the `csr_forward_or_changed` Program once,
/// then iterate dispatch + read the `changed` flag to detect fixpoint.
/// Terminates when no new bits land in the frontier or after `max_iters`.
/// Returns the saturated frontier.
///
/// Uses the supplied `SemanticExecutor` so callers can swap device and
/// reference backends without touching this layer.
///
/// # Errors
///
/// Propagates any [`SemanticExecutionError`] surfaced by the dispatcher.
pub fn forward_closure_via_change_flag_gpu(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut frontier = Vec::new();
    forward_closure_via_change_flag_gpu_into(dispatcher, policy, inputs, seed, &mut frontier)?;
    Ok(frontier)
}

/// Dispatcher-backed closure into caller-owned storage.
///
/// # Errors
///
/// Propagates any [`SemanticExecutionError`] surfaced by the dispatcher.
pub fn forward_closure_via_change_flag_gpu_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    frontier: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = ForwardChangedGpuScratch::default();
    forward_closure_via_change_flag_gpu_with_scratch_into(
        dispatcher,
        policy,
        inputs,
        seed,
        &mut scratch,
        frontier,
    )
}

/// Dispatcher-backed closure using caller-owned dispatch scratch for the seven
/// input slots and changed flag.
///
/// # Errors
///
/// Propagates any [`SemanticExecutionError`] surfaced by the dispatcher.
pub fn forward_closure_via_change_flag_gpu_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    inputs: CsrClosureInputs<'_>,
    seed: &[u32],
    scratch: &mut ForwardChangedGpuScratch,
    frontier: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let source_graph = inputs.graph;
    let max_iters = inputs.max_iters;
    let plan = plan_csr_forward_or_changed_launch(inputs)
        .map_err(SemanticExecutionError::InvalidRequest)?;
    let changed_words = plan.changed_words();
    let frontier_words = plan.frontier_words();

    copy_csr_forward_seed_frontier_into(
        seed,
        frontier_words,
        frontier,
        reserve_graph_vec,
        SemanticExecutionError::InvalidRequest,
    )?;
    if max_iters == 0 {
        return Ok(());
    }

    let ForwardChangedGpuScratch {
        inputs: dispatch_inputs,
        changed_out,
        static_input_key,
        program_cache,
    } = scratch;
    let cached = program_cache.try_get_or_insert_with(plan.program_key(), || {
        let program = plan
            .program()
            .map_err(SemanticExecutionError::InvalidRequest)?;
        Ok::<CachedForwardChangedProgram, SemanticExecutionError>(CachedForwardChangedProgram {
            program,
        })
    })?;
    let next_static_input_key = plan
        .static_input_key(
            source_graph.edge_offsets,
            source_graph.edge_targets,
            source_graph.edge_kind_mask,
        )
        .map_err(SemanticExecutionError::InvalidRequest)?;
    refresh_forward_changed_inputs(
        dispatch_inputs,
        static_input_key,
        next_static_input_key,
        &plan,
        source_graph.edge_offsets,
        source_graph.edge_targets,
        source_graph.edge_kind_mask,
        frontier,
        changed_words,
    )?;

    let program = cached.program.clone();
    let host_buffers = program
        .buffers()
        .iter()
        .filter(|buffer| {
            buffer.access() != vyre_foundation::ir::BufferAccess::Workgroup
                && !buffer.is_backend_allocated_output()
        })
        .collect::<Vec<_>>();
    if host_buffers.len() != dispatch_inputs.len() {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "csr_forward_or_changed graph requires {} host input value(s), received {}. Fix: supply every canonical program input",
            host_buffers.len(),
            dispatch_inputs.len()
        )));
    }

    let slot_values = (0..max_iters)
        .map(|iter| plan.changed_slot_value(iter))
        .collect::<Vec<_>>();
    let slot_payloads = slot_values
        .iter()
        .map(|slot| slot.map_or_else(Vec::new, |value| value.to_le_bytes().to_vec()))
        .collect::<Vec<_>>();

    let mut graph = ProgramGraph::new();
    let mut request_inputs = BTreeMap::new();
    let mut shared_ids = BTreeMap::new();
    let mut frontier_id = None;
    let mut frontier_contract = None;
    let mut changed_contract = None;
    let mut slot_contract = None;

    for (buffer, bytes) in host_buffers.iter().zip(dispatch_inputs.iter()) {
        let lifetime = match buffer.name() {
            "frontier_out" | "changed" => ValueLifetime::Retained,
            _ => ValueLifetime::Invocation,
        };
        let contract = ValueContract {
            dtype: buffer.element(),
            shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
            access: buffer.access(),
            lifetime,
        };
        match buffer.name() {
            "frontier_out" => {
                let value = graph
                    .add_external_value("frontier_initial", contract.clone())
                    .map_err(|error| {
                        SemanticExecutionError::InvalidRequest(format!(
                            "forward fixpoint graph frontier is invalid: {error}"
                        ))
                    })?;
                request_inputs.insert(value, bytes.as_slice());
                frontier_id = Some(value);
                frontier_contract = Some(contract);
            }
            "changed" => changed_contract = Some(contract),
            "changed_slot" => slot_contract = Some(contract),
            name => {
                let value = graph
                    .add_external_value(format!("input_{name}"), contract.clone())
                    .map_err(|error| {
                        SemanticExecutionError::InvalidRequest(format!(
                            "forward fixpoint graph input `{name}` is invalid: {error}"
                        ))
                    })?;
                request_inputs.insert(value, bytes.as_slice());
                shared_ids.insert(name.to_string(), (value, contract));
            }
        }
    }

    let mut current_frontier = frontier_id.ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(
            "forward fixpoint program omitted read-write `frontier_out`".to_string(),
        )
    })?;
    let frontier_contract = frontier_contract.ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(
            "forward fixpoint program omitted the frontier contract".to_string(),
        )
    })?;
    let changed_contract = changed_contract.ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(
            "forward fixpoint program omitted read-write `changed`".to_string(),
        )
    })?;
    let changed_bytes = host_buffers
        .iter()
        .position(|buffer| buffer.name() == "changed")
        .map(|index| dispatch_inputs[index].as_slice())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(
                "forward fixpoint inputs omitted `changed` bytes".to_string(),
            )
        })?;

    let mut stage_frontiers = Vec::with_capacity(max_iters as usize);
    let mut stage_changed = Vec::with_capacity(max_iters as usize);
    for iter in 0..max_iters {
        let changed_input = graph
            .add_external_value(format!("changed_zero_{iter}"), changed_contract.clone())
            .map_err(|error| {
                SemanticExecutionError::InvalidRequest(format!(
                    "forward fixpoint changed input {iter} is invalid: {error}"
                ))
            })?;
        request_inputs.insert(changed_input, changed_bytes);

        let mut node_inputs = shared_ids
            .iter()
            .map(|(name, (value, contract))| GraphInput {
                buffer: name.clone(),
                value: *value,
                contract: contract.clone(),
            })
            .collect::<Vec<_>>();
        node_inputs.push(GraphInput {
            buffer: "frontier_out".to_string(),
            value: current_frontier,
            contract: frontier_contract.clone(),
        });
        node_inputs.push(GraphInput {
            buffer: "changed".to_string(),
            value: changed_input,
            contract: changed_contract.clone(),
        });
        if let Some(slot) = slot_values[iter as usize] {
            let contract = slot_contract.clone().ok_or_else(|| {
                SemanticExecutionError::InvalidRequest(
                    "forward fixpoint plan requires `changed_slot`, but the program omitted it"
                        .to_string(),
                )
            })?;
            let slot_id = graph
                .add_external_value(format!("changed_slot_{iter}"), contract.clone())
                .map_err(|error| {
                    SemanticExecutionError::InvalidRequest(format!(
                        "forward fixpoint changed slot {iter} is invalid: {error}"
                    ))
                })?;
            debug_assert_eq!(slot_payloads[iter as usize], slot.to_le_bytes());
            request_inputs.insert(slot_id, slot_payloads[iter as usize].as_slice());
            node_inputs.push(GraphInput {
                buffer: "changed_slot".to_string(),
                value: slot_id,
                contract,
            });
        }

        let (_, outputs) = graph
            .add_node(
                format!("forward_step_{iter}"),
                program.clone(),
                node_inputs,
                vec![
                    GraphOutput {
                        buffer: "frontier_out".to_string(),
                        name: format!("frontier_{iter}"),
                        contract: frontier_contract.clone(),
                        retained_successor_of: Some(current_frontier),
                    },
                    GraphOutput {
                        buffer: "changed".to_string(),
                        name: format!("changed_{iter}"),
                        contract: changed_contract.clone(),
                        retained_successor_of: Some(changed_input),
                    },
                ],
            )
            .map_err(|error| {
                SemanticExecutionError::InvalidRequest(format!(
                    "forward fixpoint stage {iter} is invalid: {error}"
                ))
            })?;
        current_frontier = outputs[0];
        stage_frontiers.push(outputs[0]);
        stage_changed.push(outputs[1]);
    }

    let logical = LogicalProgramGraph::validate(&graph, &policy.external_facts().symbolic_bindings)
        .map_err(|error| {
            SemanticExecutionError::InvalidRequest(format!(
                "forward fixpoint logical graph is invalid: {error}"
            ))
        })?;
    let request = SemanticExecutionRequest::new(
        &logical,
        request_inputs,
        policy.external_facts().clone(),
        policy.target_facts(),
        policy.objective(),
        policy.budget(),
        policy.max_artifact_bytes(),
    )?;
    crate::telemetry::bump(&crate::telemetry::graph_dispatch_calls);
    let outputs = dispatcher.execute(&request)?.outputs;
    decode_forward_fixpoint_outputs(
        outputs,
        &ForwardFixpointReadback {
            final_frontier: current_frontier,
            stage_frontiers: &stage_frontiers,
            stage_changed: &stage_changed,
            frontier_words,
            changed_words,
            plan: &plan,
        },
        frontier,
        changed_out,
    )
}

/// Which graph values one forward-fixpoint submission retains, and the widths
/// their bytes decode to.
struct ForwardFixpointReadback<'a> {
    final_frontier: GraphValueId,
    stage_frontiers: &'a [GraphValueId],
    stage_changed: &'a [GraphValueId],
    frontier_words: usize,
    changed_words: usize,
    plan: &'a CsrForwardOrChangedLaunchPlan,
}

/// Decode the graph values one forward-fixpoint submission returned.
///
/// The submission seam returns retained values keyed by identity, so decoding
/// is a separate step from submitting: this reads declared widths out of the
/// returned bytes and rejects a value the graph never retained. It derives
/// nothing the device did not compute.
fn decode_forward_fixpoint_outputs(
    mut outputs: BTreeMap<GraphValueId, Vec<u8>>,
    readback: &ForwardFixpointReadback<'_>,
    frontier: &mut Vec<u32>,
    changed_out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let expected_output_count = readback.stage_frontiers.len() + readback.stage_changed.len();
    if outputs.len() != expected_output_count {
        return Err(SemanticExecutionError::Backend(format!(
            "forward fixpoint executor returned {} graph values, expected {expected_output_count}. Fix: return every retained stage value exactly once",
            outputs.len()
        )));
    }
    let frontier_bytes = outputs.remove(&readback.final_frontier).ok_or_else(|| {
        SemanticExecutionError::Backend(format!(
            "forward fixpoint executor omitted final frontier graph value {}",
            readback.final_frontier.0
        ))
    })?;
    crate::dispatch_buffers::decode_u32_output_exact(
        &frontier_bytes,
        readback.frontier_words,
        "csr_forward_or_changed frontier_out",
        frontier,
    )
    .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;

    for (iter, value) in readback.stage_changed.iter().enumerate() {
        let bytes = outputs.remove(value).ok_or_else(|| {
            SemanticExecutionError::Backend(format!(
                "forward fixpoint executor omitted changed graph value {}",
                value.0
            ))
        })?;
        crate::dispatch_buffers::decode_u32_output_exact(
            &bytes,
            readback.changed_words,
            "csr_forward_or_changed changed",
            changed_out,
        )
        .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;
        let changed_index = readback
            .plan
            .changed_read_index(iter as u32)
            .map_err(SemanticExecutionError::InvalidRequest)?;
        validate_csr_forward_or_changed_flag(changed_out[changed_index])
            .map_err(SemanticExecutionError::Backend)?;
    }
    for value in readback.stage_frontiers {
        outputs.remove(value);
    }
    if !outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "forward fixpoint executor returned {} undeclared graph value(s)",
            outputs.len()
        )));
    }
    Ok(())
}

fn refresh_forward_changed_inputs(
    inputs: &mut Vec<Vec<u8>>,
    static_input_key: &mut Option<CsrForwardOrChangedStaticInputKey>,
    next_static_input_key: CsrForwardOrChangedStaticInputKey,
    plan: &CsrForwardOrChangedLaunchPlan,
    edge_offsets: &[u32],
    edge_targets: &[u32],
    edge_kind_mask: &[u32],
    frontier: &[u32],
    changed_words: usize,
) -> Result<(), SemanticExecutionError> {
    if plan.uses_changed_history() {
        return refresh_keyed_dispatch_inputs(
            inputs,
            static_input_key,
            next_static_input_key,
            &[
                DispatchInput::zero_u32_words(
                    plan.node_words(),
                    "csr_forward_or_changed source scratch",
                ),
                DispatchInput::u32_slice_or_zero_words(
                    edge_offsets,
                    plan.edge_offset_words(),
                    "csr_forward_or_changed edge_offsets",
                ),
                DispatchInput::u32_slice_or_zero_words(
                    edge_targets,
                    plan.edge_storage_words(),
                    "csr_forward_or_changed edge_targets",
                ),
                DispatchInput::u32_slice_or_zero_words(
                    edge_kind_mask,
                    plan.edge_storage_words(),
                    "csr_forward_or_changed edge_kind_mask",
                ),
                DispatchInput::zero_u32_words(
                    plan.node_words(),
                    "csr_forward_or_changed frontier seed scratch",
                ),
                DispatchInput::u32_slice(frontier),
                DispatchInput::zero_u32_words(
                    changed_words,
                    "csr_forward_or_changed changed history scratch",
                ),
                DispatchInput::u32_slice(&[0]),
            ],
            &[
                (5, DispatchInput::u32_slice(frontier)),
                (
                    6,
                    DispatchInput::zero_u32_words(
                        changed_words,
                        "csr_forward_or_changed changed history scratch",
                    ),
                ),
                (7, DispatchInput::u32_slice(&[0])),
            ],
        );
    }
    refresh_keyed_dispatch_inputs(
        inputs,
        static_input_key,
        next_static_input_key,
        &[
            DispatchInput::zero_u32_words(
                plan.node_words(),
                "csr_forward_or_changed source scratch",
            ),
            DispatchInput::u32_slice_or_zero_words(
                edge_offsets,
                plan.edge_offset_words(),
                "csr_forward_or_changed edge_offsets",
            ),
            DispatchInput::u32_slice_or_zero_words(
                edge_targets,
                plan.edge_storage_words(),
                "csr_forward_or_changed edge_targets",
            ),
            DispatchInput::u32_slice_or_zero_words(
                edge_kind_mask,
                plan.edge_storage_words(),
                "csr_forward_or_changed edge_kind_mask",
            ),
            DispatchInput::zero_u32_words(
                plan.node_words(),
                "csr_forward_or_changed frontier seed scratch",
            ),
            DispatchInput::u32_slice(frontier),
            DispatchInput::zero_u32_words(1, "csr_forward_or_changed changed scratch"),
        ],
        &[
            (5, DispatchInput::u32_slice(frontier)),
            (
                6,
                DispatchInput::zero_u32_words(1, "csr_forward_or_changed changed scratch"),
            ),
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod semantic_tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use vyre_foundation::ir::{ShapeDim, ValueLifetime};
    use vyre_megakernel::{Digest, SearchBudget, SemanticExecutionOutput};

    use super::*;

    struct InspectingGraphExecutor {
        calls: AtomicUsize,
        omit_final_frontier: bool,
    }

    impl SemanticExecutor for InspectingGraphExecutor {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let graph = request.logical().graph();
            assert_eq!(graph.nodes().len(), 3);
            assert!(graph.nodes().iter().all(|node| node.inputs.len() >= 7));
            assert!(graph
                .values()
                .iter()
                .filter(|value| value.producer.is_none())
                .all(|value| request.inputs().contains_key(&value.id)));

            let mut outputs = BTreeMap::new();
            for value in graph.values().iter().filter(|value| {
                value.producer.is_some() && value.contract.lifetime == ValueLifetime::Retained
            }) {
                if self.omit_final_frontier && value.name == "frontier_2" {
                    continue;
                }
                let [ShapeDim::Known(words)] = value.contract.shape.as_slice() else {
                    panic!("test graph values use one known dimension")
                };
                let word = if value.name.starts_with("frontier_") {
                    0b1111_u32
                } else {
                    0_u32
                };
                let mut bytes = Vec::with_capacity(*words as usize * 4);
                for _ in 0..*words {
                    bytes.extend_from_slice(&word.to_le_bytes());
                }
                outputs.insert(value.id, bytes);
            }
            Ok(SemanticExecutionOutput {
                artifact: Digest([1; 32]),
                payload: Digest([2; 32]),
                outputs,
            })
        }
    }

    fn policy() -> SemanticExecutionPolicy {
        vyre_test_support::semantic_requests::unknown_policy(
            Digest([0; 32]),
            SearchBudget::new(8, 64, 1, 0, 1_000),
            1_000_000,
        )
    }

    fn chain_inputs() -> CsrClosureInputs<'static> {
        CsrClosureInputs::new(4, &[0, 1, 2, 3, 3], &[1, 2, 3], &[1, 1, 1], u32::MAX, 3)
    }

    #[test]
    fn fixpoint_builds_one_retained_graph_and_executes_once() {
        let executor = InspectingGraphExecutor {
            calls: AtomicUsize::new(0),
            omit_final_frontier: false,
        };
        let frontier =
            forward_closure_via_change_flag_gpu(&executor, &policy(), chain_inputs(), &[1])
                .expect("semantic fixpoint graph should execute");
        assert_eq!(frontier, vec![0b1111]);
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn fixpoint_rejects_any_omitted_retained_stage_value() {
        let executor = InspectingGraphExecutor {
            calls: AtomicUsize::new(0),
            omit_final_frontier: true,
        };
        let error = forward_closure_via_change_flag_gpu(&executor, &policy(), chain_inputs(), &[1])
            .expect_err("missing canonical output must fail closed");
        assert!(error.to_string().contains("expected 6"));
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    }
}
