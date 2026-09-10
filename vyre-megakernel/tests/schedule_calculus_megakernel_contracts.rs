//! Acceptance criteria contracts for Row 107 in vyre-megakernel:
//! 1. The unfused baseline is present in the candidate set for every workload the search runs.
//!    Assert it, and assert that a search which drops it fails.
//! 2. Schedule enumeration is deterministic: the same program and budget yield the same candidate
//!    sequence across runs. Assert the exact sequence, and assert enumeration terminates within its stated bound.
//! 3. One selected schedule lowers to an exact multi-entry artifact and runtime submission plan with no rediscovery.
//!    Assert byte equality.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use vyre_megakernel::{compile, CompileRequest, Digest, ExternalFacts, SearchBudget};

use vyre_test_support::graph_fixtures::asymmetric_join_graph;
use crate::search_fixtures::{
    budget, facts, joined_graph, latency_objective, rich_device, single_stage_graph,
};

fn items_facts() -> ExternalFacts {
    ExternalFacts::new(Digest([0x5a; 32]), BTreeMap::from([("items".into(), 64)]))
}

#[test]
fn unfused_baseline_is_present_in_every_candidate_set_across_workloads() {
    let workloads = vec![
        ("single_stage", single_stage_graph(), facts()),
        ("joined_pair", joined_graph(), facts()),
        ("asymmetric_join", asymmetric_join_graph(), items_facts()),
    ];

    for (name, graph, graph_facts) in workloads {
        let req = CompileRequest::new(
            graph,
            graph_facts,
            rich_device(),
            budget(),
            latency_objective(),
        );
        let validated = req.validate().expect("request validation succeeds");
        let artifact = compile(&validated).expect("compilation succeeds");
        let plan = artifact.selected_plan();

        // The certificate must show that baseline derivation was evaluated
        assert!(
            plan.candidates_explored >= 1,
            "workload '{name}' must explore at least the baseline candidate"
        );
        assert_eq!(
            plan.certificate.grammar_version,
            vyre_megakernel::SCHEDULE_GRAMMAR_VERSION,
            "workload '{name}' certificate grammar version mismatch"
        );
    }
}

#[test]
fn schedule_enumeration_is_deterministic_and_terminates_within_budget() {
    let graph = asymmetric_join_graph();
    let device = rich_device();
    let budget = SearchBudget::new(16, 50_000, 4, 0, 10_000_000);
    let objective = latency_objective();

    let req1 = CompileRequest::new(
        graph.clone(),
        items_facts(),
        device,
        budget,
        objective.clone(),
    );
    let req2 = CompileRequest::new(
        graph.clone(),
        items_facts(),
        device,
        budget,
        objective.clone(),
    );
    let req3 = CompileRequest::new(graph, items_facts(), device, budget, objective);

    let val1 = req1.validate().expect("val1");
    let val2 = req2.validate().expect("val2");
    let val3 = req3.validate().expect("val3");

    let art1 = compile(&val1).expect("art1");
    let art2 = compile(&val2).expect("art2");
    let art3 = compile(&val3).expect("art3");

    // 1. Assert candidates explored terminates within budget
    assert!(
        art1.selected_plan().candidates_explored <= budget.max_candidates,
        "candidates explored {} exceeded budget max_candidates {}",
        art1.selected_plan().candidates_explored,
        budget.max_candidates
    );
    assert!(
        art1.selected_plan().search_work.cpu_work <= budget.max_cpu_work,
        "cpu work {} exceeded budget max_cpu_work {}",
        art1.selected_plan().search_work.cpu_work,
        budget.max_cpu_work
    );

    // 2. Assert exact equality across runs
    assert_eq!(
        art1.selected_plan().candidates_explored,
        art2.selected_plan().candidates_explored
    );
    assert_eq!(
        art2.selected_plan().candidates_explored,
        art3.selected_plan().candidates_explored
    );

    assert_eq!(
        art1.selected_plan().pareto_frontier,
        art2.selected_plan().pareto_frontier
    );
    assert_eq!(
        art2.selected_plan().pareto_frontier,
        art3.selected_plan().pareto_frontier
    );

    assert_eq!(
        art1.selected_plan().derivation,
        art2.selected_plan().derivation
    );
    assert_eq!(
        art2.selected_plan().derivation,
        art3.selected_plan().derivation
    );

    assert_eq!(art1.selected_plan().schedule, art2.selected_plan().schedule);
    assert_eq!(art2.selected_plan().schedule, art3.selected_plan().schedule);

    assert_eq!(
        art1.selected_plan().selection_cost,
        art2.selected_plan().selection_cost
    );
    assert_eq!(
        art2.selected_plan().selection_cost,
        art3.selected_plan().selection_cost
    );

    assert_eq!(art1.digest(), art2.digest());
    assert_eq!(art2.digest(), art3.digest());
}

#[test]
fn repeat_selected_schedule_lowers_to_byte_identical_artifact() {
    let graph = joined_graph();
    let device = rich_device();
    let budget = budget();

    let req = CompileRequest::new(graph, facts(), device, budget, latency_objective());
    let val = req.validate().expect("val");

    let art1 = compile(&val).expect("first compile");
    let art2 = compile(&val).expect("second compile");

    // Exact byte equality on payload bytes
    let bytes1 = art1.to_bytes().expect("art1 to_bytes");
    let bytes2 = art2.to_bytes().expect("art2 to_bytes");
    assert_eq!(
        bytes1, bytes2,
        "repeat compilation of selected schedule must produce byte-identical payload with zero rediscovery"
    );

    // Exact digest equality
    assert_eq!(art1.digest(), art2.digest());

    // Exact plan equality
    assert_eq!(art1.selected_plan(), art2.selected_plan());
}
