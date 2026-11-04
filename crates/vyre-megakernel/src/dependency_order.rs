//! Dependency ordering over canonical records: acyclicity, stage assignment,
//! and the barriers those stages imply.

use std::collections::BTreeSet;

use crate::error::{failure, overflow, CompileError, CompilerFailureKind};
use crate::identity::{DependencyEdge, DependencyEndpoint, FusionGroupId};
use crate::schema::BarrierRecord;

pub(crate) fn ensure_node_dag(
    count: usize,
    dependencies: &[DependencyEdge],
    code: CompilerFailureKind,
) -> Result<(), CompileError> {
    let groups: Vec<_> = (0..count).map(|id| FusionGroupId(id as u32)).collect();
    ensure_group_dag(count, dependencies, &groups, code)
}

fn ensure_group_dag(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
    code: CompilerFailureKind,
) -> Result<(), CompileError> {
    group_stages_inner(count, dependencies, node_groups)
        .map(|_| ())
        .map_err(|_| {
            failure(
                code,
                "artifact.dependencies",
                "dependency graph contains a cycle",
                "remove the cyclic semantic dependency",
            )
        })
}

pub(crate) fn group_stages(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
) -> Result<Vec<u32>, CompileError> {
    group_stages_inner(count, dependencies, node_groups).map_err(|_| {
        failure(
            CompilerFailureKind::DependencyCycle,
            "artifact.dependencies",
            "selected-plan dependency graph contains a cycle",
            "fix compiler legality before plan selection",
        )
    })
}

fn group_stages_inner(
    count: usize,
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
) -> Result<Vec<u32>, ()> {
    let mut outgoing = vec![BTreeSet::<usize>::new(); count];
    let mut indegree = vec![0usize; count];
    for edge in dependencies {
        let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        else {
            continue;
        };
        let from = node_groups[from.0 as usize].0 as usize;
        let to = node_groups[to.0 as usize].0 as usize;
        if from != to && outgoing[from].insert(to) {
            indegree[to] += 1;
        }
    }
    let mut ready: BTreeSet<usize> = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, degree)| (*degree == 0).then_some(index))
        .collect();
    let mut stage = vec![0u32; count];
    let mut visited = 0usize;
    while let Some(next) = ready.pop_first() {
        visited += 1;
        for successor in outgoing[next].iter().copied() {
            stage[successor] = stage[successor].max(stage[next].checked_add(1).ok_or(())?);
            indegree[successor] -= 1;
            if indegree[successor] == 0 {
                ready.insert(successor);
            }
        }
    }
    (visited == count).then_some(stage).ok_or(())
}

pub(crate) fn build_barriers(
    dependencies: &[DependencyEdge],
    node_groups: &[FusionGroupId],
    stages: &[u32],
) -> Result<Vec<BarrierRecord>, CompileError> {
    let max_stage = stages.iter().copied().max().unwrap_or(0);
    let mut barriers = Vec::new();
    for after_stage in 1..=max_stage {
        let mut edge_ids = Vec::new();
        for (index, edge) in dependencies.iter().enumerate() {
            let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) =
                (edge.from, edge.to)
            else {
                continue;
            };
            let from_stage = stages[node_groups[from.0 as usize].0 as usize];
            let to_stage = stages[node_groups[to.0 as usize].0 as usize];
            if from_stage < after_stage && to_stage == after_stage {
                edge_ids.push(
                    u32::try_from(index).map_err(|_| {
                        overflow("artifact.dependencies", "edge identity exceeds u32")
                    })?,
                );
            }
        }
        barriers.push(BarrierRecord {
            before_stage: after_stage - 1,
            after_stage,
            dependencies: edge_ids,
        });
    }
    Ok(barriers)
}
