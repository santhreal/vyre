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
//! Ordering the segments through a retained value rather than a schedule
//! convention is what makes the cut hold. `legality::analyze_fusion_pair` rejects
//! any edge whose value is not `ValueLifetime::Invocation`, so no later pass can
//! contract two segments back into one module and reintroduce the fence it was
//! split to remove.
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
/// Returns a [`CompileError`] when a fence is loop-nested, when a fenced node has
/// no retained read-write port to order its segments through, or when the
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
fn add_segments(
    rebuilt: &mut ProgramGraph,
    value_map: &mut BTreeMap<u32, GraphValueId>,
    node: &ProgramGraphNode,
    segments: Vec<Program>,
) -> Result<(), CompileError> {
    let last = segments.len() - 1;
    if last == 0 {
        let inputs = remap_inputs(value_map, node)?;
        let outputs = remap_ports(value_map, node)?;
        let only = segments
            .into_iter()
            .next()
            .expect("a segment list of length one yields its only segment");
        let produced = insert(rebuilt, node.name.clone(), only, 0, inputs, outputs)?;
        record_ports(value_map, node, &produced);
        return Ok(());
    }

    let chain = chain_port(node)?;
    let chain_name = remapped_name(rebuilt, value_map, chain.value)?;
    let successor_port = node
        .output_ports
        .iter()
        .position(|port| port.buffer == chain.buffer);
    if let Some(position) = successor_port {
        if node.output_ports[position].retained_successor_of != Some(chain.value) {
            return Err(failure(
                CompilerFailureKind::InvalidProgram,
                format!("request.graph.nodes[{}].output_ports", node.id.0),
                format!(
                    "node `{}` binds buffer `{}` as both a retained input and an output port that does not declare it a retained successor, so the segments of a cut fence have no state succession to order them",
                    node.name, chain.buffer
                ),
                "declare the output port a retained successor of the value bound to the same buffer",
            ));
        }
    }

    let mut current = mapped(value_map, chain.value, &node.name)?;
    let mut segments = segments.into_iter().enumerate();
    for (index, segment) in segments.by_ref().take(last) {
        let mut inputs = remap_inputs(value_map, node)?;
        set_chain_input(&mut inputs, &chain.buffer, current);
        let outputs = vec![GraphOutput {
            buffer: chain.buffer.clone(),
            name: format!("{chain_name}__gridsync{index}"),
            contract: chain.contract.clone(),
            retained_successor_of: Some(current),
        }];
        let produced = insert(
            rebuilt,
            format!("{}__gridsync{index}", node.name),
            segment,
            index,
            inputs,
            outputs,
        )?;
        current = produced[0];
    }

    let (index, segment) = segments
        .next()
        .expect("a segment list of length last + 1 yields a final segment");
    let mut inputs = remap_inputs(value_map, node)?;
    set_chain_input(&mut inputs, &chain.buffer, current);
    let mut outputs = remap_ports(value_map, node)?;
    let newest = match successor_port {
        Some(position) => {
            outputs[position].retained_successor_of = Some(current);
            position
        }
        None => {
            outputs.push(GraphOutput {
                buffer: chain.buffer.clone(),
                name: format!("{chain_name}__gridsync{index}"),
                contract: chain.contract.clone(),
                retained_successor_of: Some(current),
            });
            outputs.len() - 1
        }
    };
    let produced = insert(
        rebuilt,
        format!("{}__gridsync{index}", node.name),
        segment,
        index,
        inputs,
        outputs,
    )?;
    record_ports(value_map, node, &produced);
    // Downstream nodes that bound the pre-split retained value must observe the
    // state the LAST segment left, not the state the first one read. Without this
    // the consumer takes a dependency on the producer of the original value and
    // may be scheduled between two segments of the cut node.
    value_map.insert(chain.value.0, produced[newest]);
    Ok(())
}

/// The retained read-write port whose state succession orders the segments.
///
/// A fence publishes a write for a later read of the same storage, so a fenced
/// node necessarily binds that storage read-write; a write-only port cannot be
/// read back and a read-only port cannot be written. `from_program` maps a
/// read-write program buffer to a retained graph value, so the port exists
/// whenever the fence is real.
fn chain_port(node: &ProgramGraphNode) -> Result<&GraphInput, CompileError> {
    node.inputs
        .iter()
        .find(|input| {
            input.contract.lifetime == ValueLifetime::Retained
                && input.contract.access == BufferAccess::ReadWrite
        })
        .ok_or_else(|| {
            failure(
                CompilerFailureKind::InvalidProgram,
                format!("request.graph.nodes[{}].inputs", node.id.0),
                format!(
                    "node `{}` holds a whole-grid fence but binds no retained read-write value, so its segments have no state succession to order them",
                    node.name
                ),
                "bind the buffer the fence publishes as a retained read-write graph value",
            )
        })
}

fn set_chain_input(inputs: &mut [GraphInput], buffer: &str, value: GraphValueId) {
    for input in inputs {
        if input.buffer == buffer {
            input.value = value;
        }
    }
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
