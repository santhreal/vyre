//! Sequential witnesses for frontier scheduling and dependency waves.

use std::collections::BTreeMap;

/// Work domain for a frontier scheduling witness node.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FrontierDomainWitness {
    /// Lexing, token classification, or preprocessing.
    Parser,
    /// Declaration, type, scope, or semantic facts.
    Semantic,
    /// Dataflow facts over graph layouts.
    Dataflow,
    /// Diagnostic aggregation and provenance.
    Diagnostic,
}

/// One frontier scheduling witness node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierNodeWitness {
    /// Stable node identifier.
    pub id: u32,
    /// Work domain.
    pub domain: FrontierDomainWitness,
    /// Estimated active items in this frontier.
    pub active_items: u32,
}

/// Directed dependency edge `before -> after`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierDependencyWitness {
    /// Prerequisite node identifier.
    pub before: u32,
    /// Dependent node identifier.
    pub after: u32,
}

/// One sequential dependency wave.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontierWaveWitness {
    /// Wave index.
    pub index: u32,
    /// Domains present in the wave.
    pub domains: Vec<FrontierDomainWitness>,
    /// Node identifiers in stable order.
    pub node_ids: Vec<u32>,
    /// Total active items in the wave.
    pub active_items: u64,
}

/// Sequential frontier scheduling witness result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontierTypedPlanWitness {
    /// Dependency waves.
    pub waves: Vec<FrontierWaveWitness>,
}

/// Frontier scheduling witness errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontierTypedPlanWitnessError {
    /// Duplicate node identifier.
    DuplicateNode {
        /// Duplicated node identifier.
        id: u32,
    },
    /// Dependency references an unknown node.
    UnknownDependencyNode {
        /// Unknown node identifier.
        id: u32,
    },
    /// Dependency graph contains a cycle.
    Cycle {
        /// Nodes left unscheduled when cycle detection stopped.
        unscheduled_nodes: usize,
    },
    /// The plan exceeds the stable frontier wave encoding.
    PlanTooLarge {
        /// Field that exceeded its representable range.
        field: &'static str,
    },
}

impl std::fmt::Display for FrontierTypedPlanWitnessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNode { id } => write!(
                f,
                "frontier scheduling witness has duplicate node id {id}. Fix: assign globally unique wave node ids."
            ),
            Self::UnknownDependencyNode { id } => write!(
                f,
                "frontier scheduling witness dependency references unknown node {id}. Fix: emit all nodes before planning dependencies."
            ),
            Self::Cycle { unscheduled_nodes } => write!(
                f,
                "frontier scheduling witness contains a dependency cycle with {unscheduled_nodes} unscheduled node(s). Fix: insert an explicit fixed-point frontier node."
            ),
            Self::PlanTooLarge { field } => write!(
                f,
                "frontier scheduling witness {field} exceeds the stable frontier encoding. Fix: shard the frontier graph before planning."
            ),
        }
    }
}

impl std::error::Error for FrontierTypedPlanWitnessError {}

/// Build stable sequential dependency waves for frontier nodes.
///
/// # Errors
///
/// Returns [`FrontierTypedPlanWitnessError`] for duplicate or unknown node
/// identifiers, dependency cycles, and counts that exceed the stable encoding.
pub fn plan_frontier_typed_ir_witness(
    nodes: &[FrontierNodeWitness],
    dependencies: &[FrontierDependencyWitness],
) -> Result<FrontierTypedPlanWitness, FrontierTypedPlanWitnessError> {
    let mut node_indices = BTreeMap::new();
    for (index, node) in nodes.iter().enumerate() {
        if node_indices.insert(node.id, index).is_some() {
            return Err(FrontierTypedPlanWitnessError::DuplicateNode { id: node.id });
        }
    }

    let mut successors: Vec<Vec<usize>> = (0..nodes.len()).map(|_| Vec::new()).collect();
    let mut indegree = vec![0_u32; nodes.len()];
    for dependency in dependencies {
        let before = node_indices.get(&dependency.before).copied().ok_or(
            FrontierTypedPlanWitnessError::UnknownDependencyNode {
                id: dependency.before,
            },
        )?;
        let after = node_indices.get(&dependency.after).copied().ok_or(
            FrontierTypedPlanWitnessError::UnknownDependencyNode {
                id: dependency.after,
            },
        )?;
        successors[before].push(after);
        indegree[after] =
            indegree[after]
                .checked_add(1)
                .ok_or(FrontierTypedPlanWitnessError::PlanTooLarge {
                    field: "dependency indegree",
                })?;
    }

    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect::<Vec<_>>();
    let mut scheduled = 0_usize;
    let mut waves = Vec::new();
    let mut next_ready = Vec::new();
    while !ready.is_empty() {
        ready.sort_unstable_by_key(|&index| (nodes[index].domain, nodes[index].id));
        let wave_index = u32::try_from(waves.len()).map_err(|_| {
            FrontierTypedPlanWitnessError::PlanTooLarge {
                field: "wave count",
            }
        })?;
        let mut domains = Vec::new();
        let mut node_ids = Vec::with_capacity(ready.len());
        let mut active_items = 0_u64;
        next_ready.clear();
        for &node_index in &ready {
            let node = nodes[node_index];
            if !domains.contains(&node.domain) {
                domains.push(node.domain);
            }
            node_ids.push(node.id);
            active_items = active_items
                .checked_add(u64::from(node.active_items))
                .ok_or(FrontierTypedPlanWitnessError::PlanTooLarge {
                    field: "active item count",
                })?;
            scheduled += 1;
            for &successor in &successors[node_index] {
                indegree[successor] -= 1;
                if indegree[successor] == 0 {
                    next_ready.push(successor);
                }
            }
        }
        waves.push(FrontierWaveWitness {
            index: wave_index,
            domains,
            node_ids,
            active_items,
        });
        std::mem::swap(&mut ready, &mut next_ready);
    }

    if scheduled != nodes.len() {
        return Err(FrontierTypedPlanWitnessError::Cycle {
            unscheduled_nodes: nodes.len() - scheduled,
        });
    }

    Ok(FrontierTypedPlanWitness { waves })
}
