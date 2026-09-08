//! Contract tests for Row 108: Proof-producing multi-level optimization framework.

use smallvec::SmallVec;
use vyre_foundation::optimizer::eqsat::{
    EChildren, EClassId, EGraph, ENodeLang, HardwarePropertyRule, Rule, TargetFact,
};
use vyre_foundation::optimizer::multi_level_eqsat::{
    MultiObjectiveCost, ParetoCandidate, ParetoFront, PassEngine,
};
use vyre_foundation::optimizer::rewrite_contract::RewriteWitness;
use vyre_foundation::schedule::{ScheduleOp, SchedulePlan, ScheduleResourceBounds, ScheduleTree};

#[allow(dead_code)]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum TestLang {
    Leaf(u32),
}

impl ENodeLang for TestLang {
    fn children(&self) -> EChildren {
        SmallVec::new()
    }
    fn with_children(&self, _children: &[EClassId]) -> Self {
        self.clone()
    }
}

struct DummyRule;

impl Rule<TestLang> for DummyRule {
    fn name(&self) -> &'static str {
        "dummy_rule"
    }
    fn witness(&self) -> RewriteWitness {
        RewriteWitness::Structural("dummy_structural_proof")
    }
    fn matches(&self, _egraph: &EGraph<TestLang>) -> Vec<(EClassId, EClassId)> {
        vec![(EClassId(0), EClassId(1))]
    }
}

#[test]
fn typed_hardware_property_rules_evaluate_deterministically() {
    let required = vec![
        TargetFact::SubgroupSize(32),
        TargetFact::TensorCoreAvailable,
    ];

    let facts_matching = vec![
        TargetFact::SubgroupSize(32),
        TargetFact::TensorCoreAvailable,
        TargetFact::AsyncCopySupported,
    ];

    let facts_lacking = vec![
        TargetFact::SubgroupSize(32),
        TargetFact::AsyncCopySupported,
    ];

    let inner1: Box<dyn Rule<TestLang>> = Box::new(DummyRule);
    let rule_match = HardwarePropertyRule::new(inner1, required.clone(), facts_matching);
    assert!(rule_match.is_satisfied(), "all required target facts present");

    let inner2: Box<dyn Rule<TestLang>> = Box::new(DummyRule);
    let rule_lack = HardwarePropertyRule::new(inner2, required, facts_lacking);
    assert!(!rule_lack.is_satisfied(), "missing TensorCoreAvailable rejects");
}

#[test]
fn pareto_frontier_multi_objective_extraction() {
    let mut front = ParetoFront::default();

    let plan1 = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile { axis: 0, tile_size: 16, inner_axis: 1 }),
        ScheduleResourceBounds { logical_points: 512, shared_bytes: 1024, private_bytes: 16, registers_per_invocation: 16, pipeline_slots: 1, queue_capacity: 0 },
    );

    let plan2 = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile { axis: 0, tile_size: 32, inner_axis: 1 }),
        ScheduleResourceBounds { logical_points: 512, shared_bytes: 2048, private_bytes: 16, registers_per_invocation: 24, pipeline_slots: 1, queue_capacity: 0 },
    );

    let plan3_dominated = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile { axis: 0, tile_size: 8, inner_axis: 1 }),
        ScheduleResourceBounds { logical_points: 512, shared_bytes: 4096, private_bytes: 32, registers_per_invocation: 48, pipeline_slots: 1, queue_capacity: 0 },
    );

    let cand1 = ParetoCandidate {
        plan: plan1,
        cost: MultiObjectiveCost {
            latency_cycles: 200,
            memory_traffic_bytes: 8192,
            register_pressure: 0.25,
            numerical_error_bound: 0.0,
            compile_time_us: 15,
        },
    };

    let cand2 = ParetoCandidate {
        plan: plan2,
        cost: MultiObjectiveCost {
            latency_cycles: 120,
            memory_traffic_bytes: 4096,
            register_pressure: 0.50,
            numerical_error_bound: 0.0,
            compile_time_us: 20,
        },
    };

    let cand3 = ParetoCandidate {
        plan: plan3_dominated,
        cost: MultiObjectiveCost {
            latency_cycles: 300,
            memory_traffic_bytes: 16384,
            register_pressure: 0.80,
            numerical_error_bound: 0.0,
            compile_time_us: 25,
        },
    };

    assert!(front.insert(cand1));
    assert!(front.insert(cand2));
    assert!(!front.insert(cand3), "dominated candidate is not inserted into Pareto front");
    assert_eq!(front.len(), 2);

    let best = front.select_weighted(1.0, 0.0).expect("non-empty frontier selects best");
    assert_eq!(best.cost.latency_cycles, 120);
}

#[test]
fn optimization_pass_engine_fixpoint_and_proof_generation() {
    let mut engine = PassEngine::default();

    let initial_plan = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile { axis: 0, tile_size: 16, inner_axis: 1 }),
        ScheduleResourceBounds { logical_points: 1024, shared_bytes: 1024, private_bytes: 16, registers_per_invocation: 16, pipeline_slots: 1, queue_capacity: 0 },
    );

    let mut step = 0;
    let (_final_plan, telemetry) = engine
        .run_to_fixpoint(initial_plan, |_current| {
            if step == 0 {
                step += 1;
                let next = SchedulePlan::new(
                    ScheduleTree::leaf(ScheduleOp::Tile { axis: 0, tile_size: 32, inner_axis: 1 }),
                    ScheduleResourceBounds { logical_points: 1024, shared_bytes: 2048, private_bytes: 16, registers_per_invocation: 24, pipeline_slots: 1, queue_capacity: 0 },
                );
                Some((next, "tile_expansion", "improves_l1_reuse"))
            } else {
                None // Fixpoint reached
            }
        })
        .expect("pass engine runs to fixpoint cleanly");

    assert_eq!(telemetry.rules_applied, 1);
    assert!(!telemetry.cycle_detected);

    let replay = engine
        .checker
        .export_replay_artifact(telemetry)
        .expect("replay artifact generated");

    assert_eq!(replay.proof_log.len(), 1);
    assert_eq!(replay.proof_log[0].rule_name, "tile_expansion");
    assert!(replay.proof_log[0].preserves_semantics);
}
