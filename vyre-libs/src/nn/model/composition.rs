//! Transactional import of reusable typed ProgramGraph fragments.

use std::collections::HashMap;

use thiserror::Error;
use vyre_foundation::ir::{
    BufferAccess, GraphInput, GraphOutput, GraphValueId, ProgramGraph, ProgramGraphError,
};

/// Invalid reusable subgraph import.
#[derive(Debug, Error)]
pub(crate) enum SubgraphImportError {
    /// A caller supplied a binding for no external value.
    #[error("subgraph binding `{0}` does not name an external value")]
    UnknownBinding(String),
    /// A source value was not mapped before its consumer.
    #[error("subgraph value {0:?} was not mapped before import")]
    MissingValue(GraphValueId),
    /// Program liveouts and graph outputs disagree.
    #[error(
        "subgraph node `{node}` has {graph_outputs} graph outputs but {produced_buffers} produced live buffers"
    )]
    OutputArity {
        /// Stable source node name.
        node: String,
        /// Graph output count.
        graph_outputs: usize,
        /// Produced live-buffer count.
        produced_buffers: usize,
    },
    /// Target graph rejected a typed port or duplicate stable name.
    #[error(transparent)]
    Graph(#[from] ProgramGraphError),
}

/// Import one validated graph under `prefix`, rebinding selected external values.
///
/// The target is mutated only through the same transactional `ProgramGraph`
/// operations as direct construction. Returned keys are source value names.
pub(crate) fn import_subgraph(
    target: &mut ProgramGraph,
    source: &ProgramGraph,
    prefix: &str,
    bindings: &[(&str, GraphValueId)],
) -> Result<HashMap<String, GraphValueId>, SubgraphImportError> {
    source
        .analyze()
        .map_err(|error| SubgraphImportError::Graph(ProgramGraphError::Wire(error.to_string())))?;
    let external_names = source
        .values()
        .iter()
        .filter(|value| value.producer.is_none())
        .map(|value| value.name.as_str())
        .collect::<Vec<_>>();
    for (name, _) in bindings {
        if !external_names.contains(name) {
            return Err(SubgraphImportError::UnknownBinding((*name).to_string()));
        }
    }
    let binding_map = bindings.iter().copied().collect::<HashMap<_, _>>();
    let mut mapped = HashMap::<GraphValueId, GraphValueId>::new();
    let mut by_name = HashMap::<String, GraphValueId>::new();

    for value in source
        .values()
        .iter()
        .filter(|value| value.producer.is_none())
    {
        let target_id = if let Some(bound) = binding_map.get(value.name.as_str()) {
            *bound
        } else {
            target.add_external_value(format!("{prefix}.{}", value.name), value.contract.clone())?
        };
        mapped.insert(value.id, target_id);
        by_name.insert(value.name.clone(), target_id);
    }

    for node in source.nodes() {
        let inputs = node
            .inputs
            .iter()
            .map(|input| {
                Ok(GraphInput {
                    buffer: input.buffer.clone(),
                    value: *mapped
                        .get(&input.value)
                        .ok_or(SubgraphImportError::MissingValue(input.value))?,
                    contract: input.contract.clone(),
                })
            })
            .collect::<Result<Vec<_>, SubgraphImportError>>()?;
        let produced_buffers = node
            .program
            .buffers()
            .iter()
            .filter(|buffer| {
                (buffer.is_output || buffer.access == BufferAccess::ReadWrite)
                    && node
                        .inputs
                        .iter()
                        .all(|input| input.buffer.as_str() != buffer.name.as_ref())
            })
            .collect::<Vec<_>>();
        if produced_buffers.len() != node.outputs.len() {
            return Err(SubgraphImportError::OutputArity {
                node: node.name.clone(),
                graph_outputs: node.outputs.len(),
                produced_buffers: produced_buffers.len(),
            });
        }
        let outputs = node
            .outputs
            .iter()
            .zip(produced_buffers)
            .map(|(output_id, buffer)| {
                let value = source
                    .values()
                    .get(output_id.0 as usize)
                    .ok_or(SubgraphImportError::MissingValue(*output_id))?;
                Ok(GraphOutput {
                    buffer: buffer.name.to_string(),
                    name: format!("{prefix}.{}", value.name),
                    contract: value.contract.clone(),
                    retained_successor_of: value
                        .retained_successor_of
                        .map(|prior| {
                            mapped
                                .get(&prior)
                                .copied()
                                .ok_or(SubgraphImportError::MissingValue(prior))
                        })
                        .transpose()?,
                })
            })
            .collect::<Result<Vec<_>, SubgraphImportError>>()?;
        let (_, target_outputs) = target.add_node(
            format!("{prefix}.{}", node.name),
            node.program.clone(),
            inputs,
            outputs,
        )?;
        for (source_id, target_id) in node.outputs.iter().copied().zip(target_outputs) {
            mapped.insert(source_id, target_id);
            by_name.insert(
                source.values()[source_id.0 as usize].name.clone(),
                target_id,
            );
        }
    }
    Ok(by_name)
}
