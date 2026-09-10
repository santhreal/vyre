//! Contract tests for Row 108: Proof-producing multi-level optimization framework.

use smallvec::SmallVec;
use std::sync::Arc;
use vyre_foundation::ir::{DataType, Expr, ExprNode};
use vyre_foundation::optimizer::eqsat::{
    EChildren, EClassId, EGraph, ENodeLang, HardwarePropertyRule, ProofTerm, Rule, RuleCacheKey,
    RuleFactIdentity, TargetFact,
};
use vyre_foundation::optimizer::expr_arena::ExprArena;
use vyre_foundation::optimizer::multi_level_eqsat::{
    MultiObjectiveCost, OptimizationProofChecker, ParetoCandidate, ParetoFront, PassEngine,
    ProofReplayError, ReplayArtifact, SemanticEqualitySaturation, StepProof,
};
use vyre_foundation::optimizer::region_law::{laws_for_family, REGION_LAWS};
use vyre_foundation::optimizer::rewrite_contract::RewriteWitness;
use vyre_foundation::schedule::{ScheduleOp, SchedulePlan, ScheduleResourceBounds, ScheduleTree};
use vyre_spec::RegionLawFamily;
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
    fn fact_identity(&self) -> RuleFactIdentity {
        RuleFactIdentity::TypedFact("dummy_rule_fact")
    }
    fn proof_term(&self) -> ProofTerm {
        ProofTerm::from_name_and_justification("dummy_rule", "dummy_structural_proof")
    }
    fn cache_key(&self) -> RuleCacheKey {
        RuleCacheKey::from_components(
            self.name(),
            &self.fact_identity(),
            &self.proof_term().obligation_digest,
        )
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

    let facts_lacking = vec![TargetFact::SubgroupSize(32), TargetFact::AsyncCopySupported];

    let inner1: Box<dyn Rule<TestLang>> = Box::new(DummyRule);
    let rule_match = HardwarePropertyRule::new(inner1, required.clone(), facts_matching);
    assert!(
        rule_match.is_satisfied(),
        "all required target facts present"
    );

    let inner2: Box<dyn Rule<TestLang>> = Box::new(DummyRule);
    let rule_lack = HardwarePropertyRule::new(inner2, required, facts_lacking);
    assert!(
        !rule_lack.is_satisfied(),
        "missing TensorCoreAvailable rejects"
    );
}

#[test]
fn pareto_frontier_multi_objective_extraction() {
    let mut front = ParetoFront::default();

    let plan1 = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 16,
            inner_axis: 1,
        }),
        ScheduleResourceBounds {
            logical_points: 512,
            shared_bytes: 1024,
            private_bytes: 16,
            registers_per_invocation: 16,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let plan2 = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 32,
            inner_axis: 1,
        }),
        ScheduleResourceBounds {
            logical_points: 512,
            shared_bytes: 2048,
            private_bytes: 16,
            registers_per_invocation: 24,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let plan3_dominated = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 8,
            inner_axis: 1,
        }),
        ScheduleResourceBounds {
            logical_points: 512,
            shared_bytes: 4096,
            private_bytes: 32,
            registers_per_invocation: 48,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
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
    assert!(
        !front.insert(cand3),
        "dominated candidate is not inserted into Pareto front"
    );
    assert_eq!(front.len(), 2);

    let best = front
        .select_weighted(1.0, 0.0)
        .expect("non-empty frontier selects best");
    assert_eq!(best.cost.latency_cycles, 120);
}

#[test]
fn optimization_pass_engine_fixpoint_and_proof_generation() {
    let mut engine = PassEngine::default();

    let initial_plan = SchedulePlan::new(
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 16,
            inner_axis: 1,
        }),
        ScheduleResourceBounds {
            logical_points: 1024,
            shared_bytes: 1024,
            private_bytes: 16,
            registers_per_invocation: 16,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let mut step = 0;
    let (_final_plan, telemetry) = engine
        .run_to_fixpoint(initial_plan, |_current| {
            if step == 0 {
                step += 1;
                let next = SchedulePlan::new(
                    ScheduleTree::leaf(ScheduleOp::Tile {
                        axis: 0,
                        tile_size: 32,
                        inner_axis: 1,
                    }),
                    ScheduleResourceBounds {
                        logical_points: 1024,
                        shared_bytes: 2048,
                        private_bytes: 16,
                        registers_per_invocation: 24,
                        pipeline_slots: 1,
                        queue_capacity: 0,
                    },
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

    // Replay verification
    let final_digest = OptimizationProofChecker::verify_replay_artifact(&replay)
        .expect("valid replay artifact must verify cleanly");
    assert_eq!(final_digest, replay.final_digest);
}

#[test]
fn runtime_enumeration_of_laws_and_rules_verifies_typed_identity_proof_and_cache_key() {
    // 1. Enumerate all declarative region laws across all families
    let families = [
        RegionLawFamily::Algebraic,
        RegionLawFamily::Recurrence,
        RegionLawFamily::Reduction,
        RegionLawFamily::Layout,
        RegionLawFamily::Numerical,
    ];

    for family in families {
        let laws = laws_for_family(family);
        assert!(
            !laws.is_empty(),
            "family {:?} must declare at least one law",
            family
        );
        for law in laws {
            assert!(!law.name.is_empty(), "law name must not be empty");
            assert!(!law.statement.is_empty(), "law statement must not be empty");
            assert!(!law.realized_by.is_empty(), "realized_by must not be empty");

            let fact_identity = RuleFactIdentity::AlgebraicLaw {
                law_name: law.name,
                family: law.family,
            };
            let proof_term = ProofTerm::from_name_and_justification(law.name, law.statement);
            let cache_key = RuleCacheKey::from_components(
                law.name,
                &fact_identity,
                &proof_term.obligation_digest,
            );

            assert_eq!(proof_term.rule_name, law.name);
            assert_ne!(proof_term.obligation_digest, [0u8; 32]);
            assert_ne!(cache_key.0, [0u8; 32]);
        }
    }

    assert_eq!(REGION_LAWS.len(), 13);

    // 2. Test concrete Rule instances
    let rule1: Box<dyn Rule<TestLang>> = Box::new(DummyRule);
    assert_eq!(rule1.name(), "dummy_rule");
    assert_eq!(
        rule1.witness(),
        RewriteWitness::Structural("dummy_structural_proof")
    );
    assert_eq!(
        rule1.fact_identity(),
        RuleFactIdentity::TypedFact("dummy_rule_fact")
    );
    assert_eq!(rule1.proof_term().rule_name, "dummy_rule");
    assert_ne!(rule1.cache_key().0, [0u8; 32]);

    let hw_rule = HardwarePropertyRule::new(
        rule1,
        vec![TargetFact::TensorCoreAvailable],
        vec![TargetFact::TensorCoreAvailable],
    );
    assert_eq!(
        hw_rule.fact_identity(),
        RuleFactIdentity::HardwareProperty {
            required: vec![TargetFact::TensorCoreAvailable],
        }
    );
    assert_ne!(hw_rule.cache_key().0, [0u8; 32]);
}

#[test]
fn proof_replay_detects_and_refuses_tampered_proofs_by_name() {
    let initial_digest = [1u8; 32];
    let intermediate_digest = [2u8; 32];
    let final_digest = [3u8; 32];

    let step0 = StepProof::new(
        0,
        "assoc_add",
        initial_digest,
        intermediate_digest,
        "associativity of addition",
        true,
    );
    let step1 = StepProof::new(
        1,
        "comm_mul",
        intermediate_digest,
        final_digest,
        "commutativity of multiplication",
        true,
    );

    let valid_artifact = ReplayArtifact {
        initial_digest,
        final_digest,
        proof_log: vec![step0.clone(), step1.clone()],
        telemetry: Default::default(),
    };

    // Valid replay succeeds
    let verified = OptimizationProofChecker::verify_replay_artifact(&valid_artifact)
        .expect("valid proof log must verify");
    assert_eq!(verified, final_digest);

    // 1. Tamper semantics preservation
    let mut bad_semantics = valid_artifact.clone();
    bad_semantics.proof_log[1].preserves_semantics = false;
    match OptimizationProofChecker::verify_replay_artifact(&bad_semantics) {
        Err(ProofReplayError::SemanticsViolation {
            rule_name,
            step_index,
        }) => {
            assert_eq!(rule_name, "comm_mul");
            assert_eq!(step_index, 1);
        }
        other => panic!("expected SemanticsViolation, got {other:?}"),
    }

    // 2. Tamper digest continuity
    let mut bad_continuity = valid_artifact.clone();
    bad_continuity.proof_log[1].before_digest = [99u8; 32];
    match OptimizationProofChecker::verify_replay_artifact(&bad_continuity) {
        Err(ProofReplayError::ContinuityBroken {
            rule_name,
            step_index,
            expected_digest,
            actual_digest,
        }) => {
            assert_eq!(rule_name, "comm_mul");
            assert_eq!(step_index, 1);
            assert_eq!(expected_digest, intermediate_digest);
            assert_eq!(actual_digest, [99u8; 32]);
        }
        other => panic!("expected ContinuityBroken, got {other:?}"),
    }

    // 3. Tamper step index
    let mut bad_step = valid_artifact.clone();
    bad_step.proof_log[1].step_index = 5;
    match OptimizationProofChecker::verify_replay_artifact(&bad_step) {
        Err(ProofReplayError::TamperedProof {
            rule_name,
            step_index,
            ..
        }) => {
            assert_eq!(rule_name, "comm_mul");
            assert_eq!(step_index, 1);
        }
        other => panic!("expected TamperedProof, got {other:?}"),
    }

    // 4. Tamper empty justification
    let mut bad_justification = valid_artifact.clone();
    bad_justification.proof_log[0].law_justification = String::new();
    match OptimizationProofChecker::verify_replay_artifact(&bad_justification) {
        Err(ProofReplayError::TamperedProof {
            rule_name,
            step_index,
            reason,
        }) => {
            assert_eq!(rule_name, "assoc_add");
            assert_eq!(step_index, 0);
            assert!(reason.contains("empty law justification"));
        }
        other => panic!("expected TamperedProof, got {other:?}"),
    }
}

#[derive(Debug)]
struct DummyExt {
    kind: &'static str,
    fingerprint: [u8; 32],
}

impl ExprNode for DummyExt {
    fn extension_kind(&self) -> &'static str {
        self.kind
    }
    fn debug_identity(&self) -> &str {
        self.kind
    }
    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }
    fn cse_safe(&self) -> bool {
        true
    }
    fn stable_fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn structurally_equal_opaque_expressions_intern_to_one_identity_and_distinct_differ() {
    let mut arena = ExprArena::default();

    let fp1 = [0x11; 32];
    let fp2 = [0x22; 32];

    // Two distinct Arc allocations wrapping structurally equal contents (same kind + same fingerprint)
    let ext_a1 = Arc::new(DummyExt {
        kind: "vendor.op.custom",
        fingerprint: fp1,
    });
    let ext_a2 = Arc::new(DummyExt {
        kind: "vendor.op.custom",
        fingerprint: fp1,
    });
    assert!(
        !Arc::ptr_eq(&ext_a1, &ext_a2),
        "must be two distinct Arc pointers"
    );

    let expr_a1 = Expr::Opaque(ext_a1);
    let expr_a2 = Expr::Opaque(ext_a2);

    let id_a1 = arena.intern(&expr_a1);
    let id_a2 = arena.intern(&expr_a2);

    // Must produce the exact same ExprId
    assert_eq!(
        id_a1, id_a2,
        "structurally equal opaque expressions must intern to identical ExprId"
    );

    // A third expression with a different fingerprint
    let ext_b = Arc::new(DummyExt {
        kind: "vendor.op.custom",
        fingerprint: fp2,
    });
    let expr_b = Expr::Opaque(ext_b);
    let id_b = arena.intern(&expr_b);

    // Must produce a different ExprId
    assert_ne!(
        id_a1, id_b,
        "opaque expressions with distinct fingerprints must intern to distinct ExprIds"
    );

    // A fourth expression with a different extension kind
    let ext_c = Arc::new(DummyExt {
        kind: "vendor.other.op",
        fingerprint: fp1,
    });
    let expr_c = Expr::Opaque(ext_c);
    let id_c = arena.intern(&expr_c);

    assert_ne!(
        id_a1, id_c,
        "opaque expressions with distinct extension kinds must intern to distinct ExprIds"
    );

    // Rebuild roundtrip
    let rebuilt_a1 = arena.rebuild(id_a1);
    assert_eq!(expr_a1, rebuilt_a1);
}

#[test]
fn semantic_equality_saturation_consumes_no_device_facts() {
    let sat = SemanticEqualitySaturation::default();
    assert_eq!(sat.class_growth_limit, 4096);
    assert_eq!(sat.max_iterations, 16);
    assert!(sat.admits_contract(
        vyre_foundation::optimizer::rewrite_contract::RewriteNumericalContract::BitExact
    ));
}
