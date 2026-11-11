use vyre_foundation::ir::{ProgramGraph, ValueLifetime};

use crate::{
    ensure_node_dag, failure, ArtifactNodeId, ArtifactValueId, CompileError, CompilerFailureKind,
    DependencyEdge, DependencyEndpoint, DependencyKind,
};

#[derive(Debug)]
pub(crate) struct NormalizedGraph {
    pub(crate) dependencies: Vec<DependencyEdge>,
}

pub(crate) fn normalize(graph: &ProgramGraph) -> Result<NormalizedGraph, CompileError> {
    let mut dependencies = Vec::new();
    for value in graph.values() {
        let value_id = ArtifactValueId(value.id.0);
        if let Some(producer) = value.producer {
            let from = ArtifactNodeId(producer.0);
            for consumer in &value.consumers {
                dependencies.push(DependencyEdge {
                    from: DependencyEndpoint::Node(from),
                    to: DependencyEndpoint::Node(ArtifactNodeId(consumer.0)),
                    kind: DependencyKind::Data,
                    value: Some(value_id),
                });
            }
            if matches!(
                value.contract.lifetime,
                ValueLifetime::Output | ValueLifetime::Retained
            ) {
                dependencies.push(DependencyEdge {
                    from: DependencyEndpoint::Node(from),
                    to: DependencyEndpoint::Value(value_id),
                    kind: DependencyKind::Materialization,
                    value: Some(value_id),
                });
            }
        }
        if let (Some(prior), Some(successor_node)) = (value.retained_successor_of, value.producer) {
            let prior = graph.values().get(prior.0 as usize).ok_or_else(|| {
                failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("graph.values[{}].retained_successor_of", value.id.0),
                    format!("retained predecessor {} does not exist", prior.0),
                    "repair the validated ProgramGraph retained transition",
                )
            })?;
            let from = prior
                .producer
                .map(|node| DependencyEndpoint::Node(ArtifactNodeId(node.0)))
                .unwrap_or_else(|| DependencyEndpoint::Value(ArtifactValueId(prior.id.0)));
            dependencies.push(DependencyEdge {
                from,
                to: DependencyEndpoint::Node(ArtifactNodeId(successor_node.0)),
                kind: DependencyKind::Retained,
                value: Some(value_id),
            });
        }
    }
    dependencies.sort();
    dependencies.dedup();
    ensure_node_dag(
        graph.nodes().len(),
        &dependencies,
        CompilerFailureKind::DependencyCycle,
    )?;
    Ok(NormalizedGraph { dependencies })
}
