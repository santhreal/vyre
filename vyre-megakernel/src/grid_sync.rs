//! Whole-grid fences are cut out of the graph before schedule search.
//!
//! A `Node::Barrier { ordering: MemoryOrdering::GridSync }` synchronizes every
//! invocation in the dispatch. No shading language has an instruction for it, and
//! only a cooperative launch satisfies it device-side, so on every other route it
//! is a launch boundary and not a barrier.
//!
//! Rejecting the fusion of two fenced nodes is not enough. Program fusion writes
//! the fence INSIDE one node's body (`vyre_foundation::execution_plan::fusion`
//! inserts one between a divergent writer arm and the arm that reads what it
//! wrote), so a single-node graph carrying a fence has no fusion pair to reject
//! and used to reach the emitter, which refused it. The cut happens here instead:
//! the node becomes one node per segment, ordered by an explicit retained-state
//! succession, and the fence is gone before any candidate is costed.
//!
//! Ordering the segments through explicit retained-value successions rather than
//! a schedule convention is what makes the cut hold. The ordering carrier keeps
//! the launches separate, while sibling successions preserve every mutable value
//! crossing a fence. `legality::analyze_fusion_pair` prevents a later pass from
//! contracting the segments and reintroducing the fence.
//!
//! A fence inside a loop body has no correct cut and is refused. The loop body is
//! emitted once and branched back to, so a single boundary would synchronize the
//! first iteration and no later one.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, GraphInput, GraphOutput, GraphValueId, Program, ProgramGraph, ProgramGraphError,
    ProgramGraphNode, ValueLifetime,
};
use vyre_foundation::transform::grid_sync_split;

use crate::{failure, CompileError, CompilerFailureKind};

/// Whether `program` needs whole-grid synchronization.
///
/// One owner, `vyre_foundation::transform::grid_sync_split::contains_grid_sync`,
/// answers this for the planner cut, the dispatch-time split, and the device
/// admission gate. A second walk that disagreed by one `Node` variant would read
/// a nested fence as absent, and the program would take the ordinary path where
/// the fence lowers to a workgroup barrier and the kernel runs with no
/// cross-block synchronization at all.
#[must_use]
pub fn requires_grid_sync(program: &Program) -> bool {
    grid_sync_split::contains_grid_sync(program)
}

/// Rewrite `graph` so no node body carries a whole-grid fence.
///
/// A graph with no fence is returned untouched, so an artifact compiled from a
/// fence-free graph keeps its digest.
///
/// # Errors
///
/// Returns a [`CompileError`] when a fence is loop-nested, when a fenced node
/// publishes no mutable carrier that can order its segments, or when the
/// segmented graph does not satisfy the `ProgramGraph` contract.
pub(crate) fn split_graph(graph: ProgramGraph) -> Result<ProgramGraph, CompileError> {
    if !graph
        .nodes()
        .iter()
        .any(|node| requires_grid_sync(&node.program))
    {
        return Ok(graph);
    }
    for node in graph.nodes() {
        if let Some(loop_var) = grid_sync_split::loop_nested_grid_sync(&node.program) {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("request.graph.nodes[{}].program", node.id.0),
                format!(
                    "node `{}` holds a whole-grid fence inside loop `{}`. A loop body is emitted once and branched back to, so one launch boundary would synchronize the first iteration and no later one",
                    node.name,
                    loop_var.as_str()
                ),
                format!(
                    "hoist the fence out of loop `{}`, or unroll the loop so each iteration carries its own fence",
                    loop_var.as_str()
                ),
            ));
        }
    }

    let mut rebuilt = ProgramGraph::new();
    let mut value_map = BTreeMap::new();
    let mut externals = Vec::new();
    let mut external_ids = Vec::new();
    for value in graph.values() {
        if value.producer.is_none() {
            externals.push((value.name.clone(), value.contract.clone()));
            external_ids.push(value.id);
        }
    }
    let rebound = rebuilt
        .add_external_values(externals)
        .map_err(|error| graph_failure("graph.values", error))?;
    for (original, new) in external_ids.iter().zip(rebound) {
        value_map.insert(original.0, new);
    }

    for node in graph.nodes() {
        let segments = grid_sync_split::split_on_grid_sync(&node.program).map_err(|error| {
            failure(
                CompilerFailureKind::InvalidProgram,
                format!("request.graph.nodes[{}].program", node.id.0),
                error.to_string(),
                "reduce the number of whole-grid fences in the node body",
            )
        })?;
        add_segments(&mut rebuilt, &mut value_map, node, segments)?;
    }
    Ok(rebuilt)
}

/// Append one graph node per segment and record the values downstream nodes bind.
///
/// The final segment is taken off the list first, so the leading segments and the
/// one that carries the node's own output ports are two values the types keep
/// apart. An empty list cannot express a program, and the splitter never returns
/// one, so it is refused through the failure channel this function already has
/// rather than through a panic that says the same thing later.
fn add_segments(
    rebuilt: &mut ProgramGraph,
    value_map: &mut BTreeMap<u32, GraphValueId>,
    node: &ProgramGraphNode,
    segments: Vec<Program>,
) -> Result<(), CompileError> {
    let mut leading = segments;
    let Some(final_segment) = leading.pop() else {
        return Err(failure(
            CompilerFailureKind::InvalidProgram,
            format!("request.graph.nodes[{}].program", node.id.0),
            format!("splitting node `{}` on its whole-grid fences produced no segment, so the node has no program left to compile", node.name),
            "keep the fence splitter from returning an empty segment list",
        ));
    };
    let last = leading.len();
    if last == 0 {
        let inputs = remap_inputs(value_map, node)?;
        let outputs = remap_ports(value_map, node)?;
        let produced = insert(
            rebuilt,
            node.name.clone(),
            final_segment,
            0,
            inputs,
            outputs,
        )?;
        record_ports(value_map, node, &produced);
        return Ok(());
    }

    let ports = retained_ports(node);
    let mut carriers = Vec::with_capacity(ports.len());
    for port in ports {
        let base_name = remapped_name(rebuilt, value_map, port.value)?;
        let successor_port = node
            .output_ports
            .iter()
            .position(|output| output.buffer == port.buffer);
        if let Some(position) = successor_port {
            if node.output_ports[position].retained_successor_of != Some(port.value) {
                return Err(failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("request.graph.nodes[{}].output_ports", node.id.0),
                    format!(
                        "node `{}` binds buffer `{}` as both a retained input and an output port that does not declare it a retained successor, so the segments of a cut fence have no state succession to order them",
                        node.name, port.buffer
                    ),
                    "declare the output port a retained successor of the value bound to the same buffer",
                ));
            }
        }
        let current = mapped(value_map, port.value, &node.name)?;
        carriers.push((port.clone(), base_name, successor_port, Some(current)));
    }
    for (position, (port, value)) in node.output_ports.iter().zip(&node.outputs).enumerate() {
        if !matches!(
            port.contract.lifetime,
            ValueLifetime::Retained | ValueLifetime::Output
        ) || port.contract.access != BufferAccess::ReadWrite
            || carriers
                .iter()
                .any(|(input, _, _, _)| input.buffer == port.buffer)
        {
            continue;
        }
        let mut carrier_contract = port.contract.clone();
        carrier_contract.lifetime = ValueLifetime::Retained;
        carriers.push((
            GraphInput {
                buffer: port.buffer.clone(),
                value: *value,
                contract: carrier_contract,
            },
            port.name.clone(),
            Some(position),
            None,
        ));
    }

    if carriers.is_empty() {
        return Err(failure(
            CompilerFailureKind::InvalidProgram,
            format!("request.graph.nodes[{}].inputs", node.id.0),
            format!(
                "node `{}` holds a whole-grid fence but publishes no mutable value across the boundary, so its segments have no state succession to order them",
                node.name
            ),
            "bind every buffer the fence publishes as retained read-write graph state",
        ));
    }
    for (index, segment) in leading.into_iter().enumerate() {
        let mut inputs = remap_inputs(value_map, node)?;
        for (port, _, _, current) in &carriers {
            if let Some(current) = current {
                set_carrier_input(&mut inputs, port, *current);
            }
        }
        let outputs = carriers
            .iter()
            .map(|(port, base_name, _, current)| {
                Ok(GraphOutput {
                    buffer: port.buffer.clone(),
                    name: format!("{base_name}__gridsync{index}"),
                    contract: port.contract.clone(),
                    retained_successor_of: *current,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?;
        let produced = insert(
            rebuilt,
            format!("{}__gridsync{index}", node.name),
            segment,
            index,
            inputs,
            outputs,
        )?;
        for (carrier, produced) in carriers.iter_mut().zip(produced) {
            carrier.3 = Some(produced);
        }
    }

    let index = last;
    let mut inputs = remap_inputs(value_map, node)?;
    for (port, _, _, current) in &carriers {
        let current = current.ok_or_else(|| {
            failure(
                CompilerFailureKind::InvalidProgram,
                format!("request.graph.nodes[{}].program", node.id.0),
                format!(
                    "retained carrier `{}` has no predecessor after the first split segment",
                    port.buffer
                ),
                "publish every retained carrier from the first segment onward",
            )
        })?;
        set_carrier_input(&mut inputs, port, current);
    }
    let mut outputs = remap_ports(value_map, node)?;
    let mut newest = Vec::with_capacity(carriers.len());
    for (port, base_name, successor_port, current) in &carriers {
        let current = current.ok_or_else(|| {
            failure(
                CompilerFailureKind::InvalidProgram,
                format!("request.graph.nodes[{}].program", node.id.0),
                format!(
                    "retained carrier `{}` has no final predecessor",
                    port.buffer
                ),
                "publish every retained carrier from the first segment onward",
            )
        })?;
        let position = match successor_port {
            Some(position) => {
                let caller_output = node
                    .program
                    .buffers()
                    .iter()
                    .find(|buffer| buffer.name() == port.buffer)
                    .is_some_and(|buffer| buffer.is_output())
                    && outputs[*position].contract.lifetime == ValueLifetime::Output;
                if caller_output {
                    outputs[*position].retained_successor_of = Some(current);
                } else {
                    if outputs[*position].contract.lifetime == ValueLifetime::Output {
                        outputs[*position].contract = port.contract.clone();
                    }
                    outputs[*position].retained_successor_of = Some(current);
                }
                *position
            }
            None => {
                outputs.push(GraphOutput {
                    buffer: port.buffer.clone(),
                    name: format!("{base_name}__gridsync{index}"),
                    contract: port.contract.clone(),
                    retained_successor_of: Some(current),
                });
                outputs.len() - 1
            }
        };
        newest.push(position);
    }
    let produced = insert(
        rebuilt,
        format!("{}__gridsync{index}", node.name),
        final_segment,
        index,
        inputs,
        outputs,
    )?;
    record_ports(value_map, node, &produced);
    // Every retained carrier must point downstream at the LAST segment's value.
    // Updating only one ordering carrier loses writes made to sibling state
    // buffers between fences.
    for ((port, _, _, _), position) in carriers.iter().zip(newest) {
        value_map.insert(port.value.0, produced[position]);
    }
    Ok(())
}

/// Read-write inputs whose retained state crosses every split segment.
///
/// A fence publishes writes for later reads of the same storage. Every retained
/// read-write input therefore gets a successor chain. A first-write output can
/// instead publish the initial carrier from the first segment, so this set may
/// be empty until output carriers are added.
fn retained_ports(node: &ProgramGraphNode) -> Vec<&GraphInput> {
    node.inputs
        .iter()
        .filter(|input| {
            input.contract.lifetime == ValueLifetime::Retained
                && input.contract.access == BufferAccess::ReadWrite
        })
        .collect()
}

fn set_carrier_input(inputs: &mut Vec<GraphInput>, port: &GraphInput, value: GraphValueId) {
    for input in inputs.iter_mut() {
        if input.buffer == port.buffer {
            input.value = value;
            return;
        }
    }
    let mut input = port.clone();
    input.value = value;
    inputs.push(input);
}

fn insert(
    rebuilt: &mut ProgramGraph,
    name: String,
    program: Program,
    index: usize,
    inputs: Vec<GraphInput>,
    outputs: Vec<GraphOutput>,
) -> Result<Vec<GraphValueId>, CompileError> {
    let path = format!("graph.nodes[{name}].segment[{index}]");
    rebuilt
        .add_node(name, program, inputs, outputs)
        .map(|(_, produced)| produced)
        .map_err(|error| graph_failure(path, error))
}

fn remap_inputs(
    value_map: &BTreeMap<u32, GraphValueId>,
    node: &ProgramGraphNode,
) -> Result<Vec<GraphInput>, CompileError> {
    node.inputs
        .iter()
        .map(|input| {
            Ok(GraphInput {
                buffer: input.buffer.clone(),
                value: mapped(value_map, input.value, &node.name)?,
                contract: input.contract.clone(),
            })
        })
        .collect()
}

fn remap_ports(
    value_map: &BTreeMap<u32, GraphValueId>,
    node: &ProgramGraphNode,
) -> Result<Vec<GraphOutput>, CompileError> {
    node.output_ports
        .iter()
        .map(|port| {
            let prior = match port.retained_successor_of {
                Some(prior) => Some(mapped(value_map, prior, &node.name)?),
                None => None,
            };
            Ok(GraphOutput {
                buffer: port.buffer.clone(),
                name: port.name.clone(),
                contract: port.contract.clone(),
                retained_successor_of: prior,
            })
        })
        .collect()
}

fn record_ports(
    value_map: &mut BTreeMap<u32, GraphValueId>,
    node: &ProgramGraphNode,
    produced: &[GraphValueId],
) {
    for (original, new) in node.outputs.iter().zip(produced) {
        value_map.insert(original.0, *new);
    }
}

fn mapped(
    value_map: &BTreeMap<u32, GraphValueId>,
    value: GraphValueId,
    node: &str,
) -> Result<GraphValueId, CompileError> {
    value_map.get(&value.0).copied().ok_or_else(|| {
        failure(
            CompilerFailureKind::InvalidProgram,
            "graph.values",
            format!(
                "node `{node}` binds value {} before any node produces it",
                value.0
            ),
            "supply a ProgramGraph whose node order is a topological schedule",
        )
    })
}

fn remapped_name(
    rebuilt: &ProgramGraph,
    value_map: &BTreeMap<u32, GraphValueId>,
    value: GraphValueId,
) -> Result<String, CompileError> {
    let id = mapped(value_map, value, "segment chain")?;
    rebuilt
        .values()
        .get(id.0 as usize)
        .map(|value| value.name.clone())
        .ok_or_else(|| {
            failure(
                CompilerFailureKind::InvalidProgram,
                "graph.values",
                format!("segmented graph has no value {}", id.0),
                "supply a ProgramGraph whose node order is a topological schedule",
            )
        })
}

fn graph_failure(path: impl Into<String>, error: ProgramGraphError) -> CompileError {
    failure(
        CompilerFailureKind::InvalidProgram,
        path,
        format!("cutting a whole-grid fence produced an invalid graph: {error}"),
        "bind the buffers the fence publishes as retained read-write graph values",
    )
}
