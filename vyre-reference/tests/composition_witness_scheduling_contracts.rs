//! Contracts for reference-owned frontier dependency scheduling.
//!
//! These tests keep sequential topological planning in the reference crate so
//! production GPU crates consume only neutral plan data. They do not cover
//! backend byte sizing or launch-barrier construction.

use vyre_reference::composition_witness::{
    plan_frontier_typed_ir_witness, FrontierDependencyWitness, FrontierDomainWitness,
    FrontierNodeWitness, FrontierTypedPlanWitnessError,
};

fn node(id: u32, domain: FrontierDomainWitness, active_items: u32) -> FrontierNodeWitness {
    FrontierNodeWitness {
        id,
        domain,
        active_items,
    }
}

#[test]
fn independent_nodes_are_stably_grouped_by_domain_and_id() {
    let plan = plan_frontier_typed_ir_witness(
        &[
            node(3, FrontierDomainWitness::Diagnostic, 4),
            node(1, FrontierDomainWitness::Semantic, 20),
            node(0, FrontierDomainWitness::Parser, 10),
            node(2, FrontierDomainWitness::Dataflow, 30),
        ],
        &[
            FrontierDependencyWitness {
                before: 0,
                after: 1,
            },
            FrontierDependencyWitness {
                before: 1,
                after: 2,
            },
            FrontierDependencyWitness {
                before: 1,
                after: 3,
            },
        ],
    )
    .expect("valid frontier graph must schedule");

    assert_eq!(plan.waves.len(), 3);
    assert_eq!(plan.waves[0].node_ids, [0]);
    assert_eq!(plan.waves[1].node_ids, [1]);
    assert_eq!(plan.waves[2].node_ids, [2, 3]);
    assert_eq!(plan.waves[2].active_items, 34);
    assert_eq!(
        plan.waves[2].domains,
        [
            FrontierDomainWitness::Dataflow,
            FrontierDomainWitness::Diagnostic,
        ]
    );
}

#[test]
fn malformed_dependency_graphs_fail_closed() {
    assert_eq!(
        plan_frontier_typed_ir_witness(
            &[
                node(1, FrontierDomainWitness::Parser, 1),
                node(1, FrontierDomainWitness::Semantic, 1),
            ],
            &[],
        ),
        Err(FrontierTypedPlanWitnessError::DuplicateNode { id: 1 })
    );
    assert_eq!(
        plan_frontier_typed_ir_witness(
            &[node(1, FrontierDomainWitness::Parser, 1)],
            &[FrontierDependencyWitness {
                before: 1,
                after: 2,
            }],
        ),
        Err(FrontierTypedPlanWitnessError::UnknownDependencyNode { id: 2 })
    );
    assert_eq!(
        plan_frontier_typed_ir_witness(
            &[node(2, FrontierDomainWitness::Parser, 1)],
            &[FrontierDependencyWitness {
                before: 1,
                after: 2,
            }],
        ),
        Err(FrontierTypedPlanWitnessError::UnknownDependencyNode { id: 1 })
    );
    assert_eq!(
        plan_frontier_typed_ir_witness(
            &[
                node(1, FrontierDomainWitness::Parser, 1),
                node(2, FrontierDomainWitness::Semantic, 1),
            ],
            &[
                FrontierDependencyWitness {
                    before: 1,
                    after: 2,
                },
                FrontierDependencyWitness {
                    before: 2,
                    after: 1,
                },
            ],
        ),
        Err(FrontierTypedPlanWitnessError::Cycle {
            unscheduled_nodes: 2,
        })
    );
    assert_eq!(
        plan_frontier_typed_ir_witness(
            &[node(1, FrontierDomainWitness::Parser, 1)],
            &[FrontierDependencyWitness {
                before: 1,
                after: 1,
            }],
        ),
        Err(FrontierTypedPlanWitnessError::Cycle {
            unscheduled_nodes: 1,
        })
    );
}

#[test]
fn empty_graph_plans_zero_waves() {
    let plan = plan_frontier_typed_ir_witness(&[], &[]).expect("empty graph must schedule");
    assert!(plan.waves.is_empty());
}

#[test]
fn witness_error_display_and_error_trait() {
    let err = FrontierTypedPlanWitnessError::DuplicateNode { id: 42 };
    assert!(err.to_string().contains("duplicate node id 42"));

    let err = FrontierTypedPlanWitnessError::UnknownDependencyNode { id: 99 };
    assert!(err.to_string().contains("unknown node 99"));

    let err = FrontierTypedPlanWitnessError::Cycle {
        unscheduled_nodes: 3,
    };
    assert!(err.to_string().contains("cycle with 3 unscheduled node(s)"));

    let err = FrontierTypedPlanWitnessError::PlanTooLarge {
        field: "wave count",
    };
    assert!(err.to_string().contains("wave count exceeds"));
}
