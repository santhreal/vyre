//! Contract records, guarded laws, counterexample generator, and transform decision tests.

use vyre_spec::{
    AlgebraicLaw, AliasingContract, ContractValidationError, CounterexampleGenerator, DataType,
    DeterminismClass, GuardedLaw, LawDirection, LawGuard, LawValidationError, MemoryEffect,
    NumericBehavior, OpSignature, ProofMethod, RangeContract, ResourceBoundsContract,
    SemanticContractRecord, ShapeIndexContract, TransformDecision,
};

#[test]
fn guarded_law_with_executable_proof_validates() {
    let law = GuardedLaw::declared(AlgebraicLaw::Commutative);
    assert_eq!(law.validate(), Ok(()));
}

#[test]
fn law_with_no_executable_proof_evidence_is_rejected_by_name() {
    let law = GuardedLaw::declared(AlgebraicLaw::Associative).with_proof_method(ProofMethod::None);
    let result = law.validate();
    assert_eq!(
        result,
        Err(LawValidationError::NoExecutableProofEvidence {
            law: "associative".to_string()
        })
    );
    assert_eq!(
        result.unwrap_err().to_string(),
        "law `associative` rejected: no executable proof evidence provided"
    );
}

#[test]
fn counterexample_generator_failing_hypothesis_fails_law_not_test() {
    let generator = CounterexampleGenerator::deterministic("subtraction-commutativity-falsifier");
    let law =
        GuardedLaw::declared(AlgebraicLaw::Commutative).with_counterexample_generator(generator);

    // Test a non-commutative operation (e.g. integer subtraction: a - b != b - a)
    let subtraction_commutative = |inputs: &[u64]| -> bool {
        if inputs.len() < 2 {
            return true;
        }
        let (a, b) = (inputs[0], inputs[1]);
        a.wrapping_sub(b) == b.wrapping_sub(a)
    };

    let result = law.verify_with_predicate(2, subtraction_commutative);
    assert!(
        matches!(result, Err(LawValidationError::CounterexampleFound { ref law, .. }) if law == "commutative"),
        "Counterexample generator must fail the law rather than the test; got {result:?}"
    );
}

#[test]
fn counterexample_generator_passing_on_true_hypothesis() {
    let generator = CounterexampleGenerator::deterministic("addition-commutativity-verifier");
    let law =
        GuardedLaw::declared(AlgebraicLaw::Commutative).with_counterexample_generator(generator);

    // Test a truly commutative operation (integer addition: a + b == b + a)
    let addition_commutative = |inputs: &[u64]| -> bool {
        if inputs.len() < 2 {
            return true;
        }
        let (a, b) = (inputs[0], inputs[1]);
        a.wrapping_add(b) == b.wrapping_add(a)
    };

    assert_eq!(law.verify_with_predicate(2, addition_commutative), Ok(()));
}

#[test]
fn contract_record_with_opaque_decision_validates() {
    let record = SemanticContractRecord::opaque(
        "vyre-primitives::hardware::storage_barrier",
        "hardware memory fence with side-effects and synchronization",
    );
    assert_eq!(record.validate(), Ok(()));
    assert!(record.decision.is_opaque());
    assert_eq!(
        record.decision.opaque_reason(),
        Some("hardware memory fence with side-effects and synchronization")
    );
}

#[test]
fn contract_record_with_placeholder_opaque_decision_is_rejected() {
    for bad_reason in [
        "todo",
        "none",
        "TBD",
        "placeholder",
        "opaque",
        "no-op",
        "abc",
    ] {
        let record = SemanticContractRecord::opaque("test::op", bad_reason);
        let result = record.validate();
        assert!(
            matches!(result, Err(ContractValidationError::InvalidOpaqueReason(_))),
            "Reason `{bad_reason}` must be rejected as placeholder/invalid; got {result:?}"
        );
    }
}

#[test]
fn contract_record_with_empty_laws_decision_is_rejected() {
    let decision = TransformDecision::GuardedLaws(Vec::new());
    assert_eq!(decision.validate(), Err(ContractValidationError::EmptyLaws));
}

#[test]
fn contract_record_with_guarded_laws_validates() {
    let law = GuardedLaw::declared(AlgebraicLaw::Commutative)
        .with_direction(LawDirection::Bidirectional)
        .with_guard(LawGuard::ExactOnly)
        .with_proof_method(ProofMethod::ExhaustiveU8);
    let record = SemanticContractRecord::with_laws("vyre-libs::math::add", vec![law]);
    assert_eq!(record.validate(), Ok(()));
    assert_eq!(record.decision.laws().len(), 1);
}

#[test]
fn contract_record_serde_roundtrip() {
    let law = GuardedLaw::declared(AlgebraicLaw::Idempotent)
        .with_direction(LawDirection::LeftToRight)
        .with_guard(LawGuard::Unconditional)
        .with_proof_method(ProofMethod::ExhaustiveU16);
    let record = SemanticContractRecord {
        id: "vyre-libs::bitset::and".to_string(),
        signature: Some(OpSignature {
            inputs: vec![DataType::U32, DataType::U32],
            output: DataType::U32,
            input_params: None,
            output_params: None,
            contract: None,
        }),
        effects: MemoryEffect::Pure,
        aliasing: AliasingContract::Disjoint,
        shape_index: ShapeIndexContract::elementwise(),
        numerical: NumericBehavior::Exact,
        determinism: DeterminismClass::Deterministic,
        range_preconditions: RangeContract::unbounded(),
        resource_bounds: ResourceBoundsContract::Unbounded,
        decision: TransformDecision::GuardedLaws(vec![law]),
        rejected_labels: Vec::new(),
    };

    let serialized = serde_json::to_string(&record).expect("serialization succeeds");
    let deserialized: SemanticContractRecord =
        serde_json::from_str(&serialized).expect("deserialization succeeds");
    assert_eq!(record, deserialized);
}

#[test]
fn contract_record_with_no_transform_decision_validates() {
    let record = SemanticContractRecord::no_transform(
        "vyre-libs::monolith::op",
        "non-composable monolithic domain algorithm with no algebraic decomposition",
    );
    assert_eq!(record.validate(), Ok(()));
    assert!(record.decision.is_no_transform());
    assert!(record.decision.is_opaque());
    assert_eq!(record.state_name(), "no-transform");
    assert_eq!(
        record.decision.opaque_reason(),
        Some("non-composable monolithic domain algorithm with no algebraic decomposition")
    );
}

#[test]
fn contract_record_with_unrecorded_decision_fails_validation() {
    let record = SemanticContractRecord::unrecorded("vyre-libs::unrecorded::op");
    assert!(record.decision.is_not_recorded());
    assert!(!record.decision.has_decision());
    assert_eq!(record.state_name(), "not-recorded");
    let result = record.validate();
    assert_eq!(result, Err(ContractValidationError::NotRecorded));
    assert_eq!(
        result.unwrap_err().to_string(),
        "operation contract transform decision is not recorded"
    );
}

#[test]
fn four_states_are_mutually_exclusive_and_exhaustive() {
    let law = GuardedLaw::declared(AlgebraicLaw::Commutative);
    let s1 = TransformDecision::GuardedLaws(vec![law]);
    let s2 = TransformDecision::NoTransform {
        reason: "non-composable architecture".to_string(),
    };
    let s3 = TransformDecision::Opaque {
        reason: "opaque hardware primitive".to_string(),
    };
    let s4 = TransformDecision::NotRecorded;

    assert_eq!(s1.state_name(), "guarded-laws");
    assert_eq!(s2.state_name(), "no-transform");
    assert_eq!(s3.state_name(), "opaque");
    assert_eq!(s4.state_name(), "not-recorded");

    assert!(s1.is_guarded_laws());
    assert!(!s1.is_opaque());
    assert!(!s1.is_no_transform());
    assert!(!s1.is_not_recorded());

    assert!(!s2.is_guarded_laws());
    assert!(s2.is_opaque());
    assert!(s2.is_no_transform());
    assert!(!s2.is_not_recorded());

    assert!(!s3.is_guarded_laws());
    assert!(s3.is_opaque());
    assert!(!s3.is_no_transform());
    assert!(!s3.is_not_recorded());

    assert!(!s4.is_guarded_laws());
    assert!(!s4.is_opaque());
    assert!(!s4.is_no_transform());
    assert!(s4.is_not_recorded());
}

#[test]
fn law_with_zero_witness_count_is_rejected_as_no_executable_proof() {
    let law = GuardedLaw::declared(AlgebraicLaw::Associative).with_proof_method(
        ProofMethod::WitnessedU32 {
            seed: 0x1234,
            count: 0,
        },
    );
    let result = law.validate();
    assert_eq!(
        result,
        Err(LawValidationError::NoExecutableProofEvidence {
            law: "associative".to_string()
        })
    );
}

#[test]
fn law_with_inverted_guard_range_is_rejected() {
    let law = GuardedLaw::declared(AlgebraicLaw::Bounded { lo: 0, hi: 100 })
        .with_proof_method(ProofMethod::ExhaustiveU16)
        .with_guard(LawGuard::Range { lo: 100, hi: 10 });
    let result = law.validate();
    assert!(matches!(
        result,
        Err(LawValidationError::InvalidGuard { ref reason, .. }) if reason.contains("exceeds")
    ));
}

#[test]
fn law_with_empty_compiler_levels_is_rejected() {
    let mut law = GuardedLaw::declared(AlgebraicLaw::Idempotent);
    law.affected_compiler_levels.clear();
    let result = law.validate();
    assert!(matches!(
        result,
        Err(LawValidationError::InvalidGuard { ref reason, .. }) if reason.contains("affected compiler levels")
    ));
}
