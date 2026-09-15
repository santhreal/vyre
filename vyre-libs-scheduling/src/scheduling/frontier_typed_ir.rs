//! Frontier-typed IR dependency waves.

/// Work domain for a frontier-typed node.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum FrontierDomain {
    /// Lexing, token classification, or preprocessing.
    Parser,
    /// Declaration, type, scope, or semantic facts.
    Semantic,
    /// Dataflow facts over graph layouts.
    Dataflow,
    /// Diagnostic aggregation and provenance.
    Diagnostic,
}

/// One frontier-typed IR node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierNode {
    /// Stable node id.
    pub id: u32,
    /// Work domain.
    pub domain: FrontierDomain,
    /// Estimated active items in this frontier.
    pub active_items: u32,
}

/// Directed dependency edge `before -> after`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrontierDependency {
    /// Prerequisite node.
    pub before: u32,
    /// Dependent node.
    pub after: u32,
}

/// One dependency wave.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontierWave {
    /// Wave index.
    pub index: u32,
    /// Domains present in the wave.
    pub domains: Vec<FrontierDomain>,
    /// Node ids in stable order.
    pub node_ids: Vec<u32>,
    /// Total active items in the wave.
    pub active_items: u64,
}

/// Frontier-typed execution plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrontierTypedPlan {
    /// Dependency waves.
    pub waves: Vec<FrontierWave>,
}

/// Frontier planning errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrontierTypedPlanError {
    /// Duplicate node id.
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

impl std::fmt::Display for FrontierTypedPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateNode { id } => write!(
                f,
                "frontier-typed IR has duplicate node id {id}. Fix: assign globally unique wave node ids."
            ),
            Self::UnknownDependencyNode { id } => write!(
                f,
                "frontier-typed IR dependency references unknown node {id}. Fix: emit all nodes before planning dependencies."
            ),
            Self::Cycle { unscheduled_nodes } => write!(
                f,
                "frontier-typed IR contains a dependency cycle with {unscheduled_nodes} unscheduled node(s). Fix: insert an explicit fixed-point frontier node."
            ),
            Self::PlanTooLarge { field } => write!(
                f,
                "frontier-typed IR {field} exceeds the stable frontier encoding. Fix: shard the frontier graph before planning."
            ),
        }
    }
}

impl std::error::Error for FrontierTypedPlanError {}

/// Plan frontier-typed dependency waves.
#[cfg(test)]
pub fn plan_frontier_typed_ir(
    nodes: &[FrontierNode],
    dependencies: &[FrontierDependency],
) -> Result<FrontierTypedPlan, FrontierTypedPlanError> {
    use vyre_reference::composition_witness::{
        plan_frontier_typed_ir_witness, FrontierDependencyWitness, FrontierDomainWitness,
        FrontierNodeWitness, FrontierTypedPlanWitnessError,
    };

    let witness_nodes = nodes
        .iter()
        .map(|node| FrontierNodeWitness {
            id: node.id,
            domain: match node.domain {
                FrontierDomain::Parser => FrontierDomainWitness::Parser,
                FrontierDomain::Semantic => FrontierDomainWitness::Semantic,
                FrontierDomain::Dataflow => FrontierDomainWitness::Dataflow,
                FrontierDomain::Diagnostic => FrontierDomainWitness::Diagnostic,
            },
            active_items: node.active_items,
        })
        .collect::<Vec<_>>();
    let witness_dependencies = dependencies
        .iter()
        .map(|dependency| FrontierDependencyWitness {
            before: dependency.before,
            after: dependency.after,
        })
        .collect::<Vec<_>>();
    let witness_plan = plan_frontier_typed_ir_witness(&witness_nodes, &witness_dependencies)
        .map_err(|error| match error {
            FrontierTypedPlanWitnessError::DuplicateNode { id } => {
                FrontierTypedPlanError::DuplicateNode { id }
            }
            FrontierTypedPlanWitnessError::UnknownDependencyNode { id } => {
                FrontierTypedPlanError::UnknownDependencyNode { id }
            }
            FrontierTypedPlanWitnessError::Cycle { unscheduled_nodes } => {
                FrontierTypedPlanError::Cycle { unscheduled_nodes }
            }
            FrontierTypedPlanWitnessError::PlanTooLarge { field } => {
                FrontierTypedPlanError::PlanTooLarge { field }
            }
        })?;
    Ok(FrontierTypedPlan {
        waves: witness_plan
            .waves
            .into_iter()
            .map(|wave| FrontierWave {
                index: wave.index,
                domains: wave
                    .domains
                    .into_iter()
                    .map(|domain| match domain {
                        FrontierDomainWitness::Parser => FrontierDomain::Parser,
                        FrontierDomainWitness::Semantic => FrontierDomain::Semantic,
                        FrontierDomainWitness::Dataflow => FrontierDomain::Dataflow,
                        FrontierDomainWitness::Diagnostic => FrontierDomain::Diagnostic,
                    })
                    .collect(),
                node_ids: wave.node_ids,
                active_items: wave.active_items,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_typed_ir_groups_independent_work_into_waves() {
        let plan = plan_frontier_typed_ir(
            &[
                node(0, FrontierDomain::Parser, 10),
                node(1, FrontierDomain::Semantic, 20),
                node(2, FrontierDomain::Dataflow, 30),
                node(3, FrontierDomain::Diagnostic, 4),
            ],
            &[
                FrontierDependency {
                    before: 0,
                    after: 1,
                },
                FrontierDependency {
                    before: 1,
                    after: 2,
                },
                FrontierDependency {
                    before: 1,
                    after: 3,
                },
            ],
        )
        .expect("Fix: valid frontier-typed plan should build");

        assert_eq!(plan.waves.len(), 3);
        assert_eq!(plan.waves[0].node_ids, vec![0]);
        assert_eq!(plan.waves[1].node_ids, vec![1]);
        assert_eq!(plan.waves[2].node_ids, vec![2, 3]);
        assert_eq!(plan.waves[2].active_items, 34);
    }

    #[test]
    fn frontier_typed_ir_rejects_unknown_duplicate_and_cycle() {
        assert_eq!(
            plan_frontier_typed_ir(
                &[
                    node(1, FrontierDomain::Parser, 1),
                    node(1, FrontierDomain::Semantic, 1)
                ],
                &[],
            )
            .expect_err("duplicate node ids should fail"),
            FrontierTypedPlanError::DuplicateNode { id: 1 }
        );
        assert_eq!(
            plan_frontier_typed_ir(
                &[node(1, FrontierDomain::Parser, 1)],
                &[FrontierDependency {
                    before: 1,
                    after: 2,
                }],
            )
            .expect_err("unknown dependency should fail"),
            FrontierTypedPlanError::UnknownDependencyNode { id: 2 }
        );
        assert_eq!(
            plan_frontier_typed_ir(
                &[node(2, FrontierDomain::Parser, 1)],
                &[FrontierDependency {
                    before: 1,
                    after: 2,
                }],
            )
            .expect_err("unknown dependency before should fail"),
            FrontierTypedPlanError::UnknownDependencyNode { id: 1 }
        );
        assert_eq!(
            plan_frontier_typed_ir(
                &[
                    node(1, FrontierDomain::Parser, 1),
                    node(2, FrontierDomain::Semantic, 1)
                ],
                &[
                    FrontierDependency {
                        before: 1,
                        after: 2,
                    },
                    FrontierDependency {
                        before: 2,
                        after: 1,
                    },
                ],
            )
            .expect_err("cycle should fail"),
            FrontierTypedPlanError::Cycle {
                unscheduled_nodes: 2,
            }
        );
        assert_eq!(
            plan_frontier_typed_ir(
                &[node(1, FrontierDomain::Parser, 1)],
                &[FrontierDependency {
                    before: 1,
                    after: 1,
                }],
            )
            .expect_err("self cycle should fail"),
            FrontierTypedPlanError::Cycle {
                unscheduled_nodes: 1,
            }
        );
    }

    #[test]
    fn frontier_typed_plan_error_display() {
        let err = FrontierTypedPlanError::DuplicateNode { id: 7 };
        assert!(err.to_string().contains("duplicate node id 7"));

        let err = FrontierTypedPlanError::UnknownDependencyNode { id: 13 };
        assert!(err.to_string().contains("unknown node 13"));

        let err = FrontierTypedPlanError::Cycle {
            unscheduled_nodes: 4,
        };
        assert!(err.to_string().contains("cycle with 4 unscheduled node(s)"));

        let err = FrontierTypedPlanError::PlanTooLarge {
            field: "active item count",
        };
        assert!(err.to_string().contains("active item count exceeds"));
    }

    #[test]
    fn frontier_typed_ir_plans_wide_dag_with_stable_wave_order() {
        let mut nodes = Vec::new();
        let mut dependencies = Vec::new();
        for id in 0..512_u32 {
            nodes.push(node(
                id,
                if id % 2 == 0 {
                    FrontierDomain::Dataflow
                } else {
                    FrontierDomain::Parser
                },
                1,
            ));
            nodes.push(node(10_000 + id, FrontierDomain::Diagnostic, 2));
            dependencies.push(FrontierDependency {
                before: id,
                after: 10_000 + id,
            });
        }

        let plan =
            plan_frontier_typed_ir(&nodes, &dependencies).expect("Fix: wide DAG should plan");

        assert_eq!(plan.waves.len(), 2);
        assert_eq!(plan.waves[0].node_ids.len(), 512);
        assert_eq!(plan.waves[1].node_ids.len(), 512);
        assert_eq!(plan.waves[0].active_items, 512);
        assert_eq!(plan.waves[1].active_items, 1024);
        assert_eq!(plan.waves[0].node_ids[0], 1);
        assert_eq!(plan.waves[0].node_ids[1], 3);
    }

    fn node(id: u32, domain: FrontierDomain, active_items: u32) -> FrontierNode {
        FrontierNode {
            id,
            domain,
            active_items,
        }
    }
}
