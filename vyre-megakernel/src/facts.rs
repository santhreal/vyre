use vyre_foundation::ir::ProgramGraph;

use crate::{ArtifactNodeId, ArtifactValueId, DependencyEdge, DependencyEndpoint, DependencyKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DataflowEdge {
    pub(crate) from: ArtifactNodeId,
    pub(crate) to: ArtifactNodeId,
    pub(crate) value: ArtifactValueId,
}

#[derive(Debug)]
pub(crate) struct PlanningFacts {
    pub(crate) node_work: Vec<u64>,
    pub(crate) dataflow: Vec<DataflowEdge>,
}

pub(crate) fn derive(graph: &ProgramGraph, dependencies: &[DependencyEdge]) -> PlanningFacts {
    let node_work = graph
        .nodes()
        .iter()
        .map(|node| u64::try_from(node.program.stats().node_count).unwrap_or(u64::MAX))
        .collect();
    let dataflow = dependencies
        .iter()
        .filter_map(|edge| match (edge.from, edge.to, edge.kind, edge.value) {
            (
                DependencyEndpoint::Node(from),
                DependencyEndpoint::Node(to),
                DependencyKind::Data,
                Some(value),
            ) => Some(DataflowEdge { from, to, value }),
            _ => None,
        })
        .collect();
    PlanningFacts {
        node_work,
        dataflow,
    }
}
