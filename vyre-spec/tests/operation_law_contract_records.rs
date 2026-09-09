//! Contract records, guarded laws, counterexample generator, and transform decision tests.

use vyre_spec::{
    AlgebraicLaw, AliasingContract, ContractValidationError, CounterexampleGenerator, DataType,
    DeterminismClass, GuardedLaw, LawDirection, LawGuard, LawValidationError, MemoryEffect,
    NumericBehavior, OpSignature, ProofMethod, RangeContract, ResourceBoundsContract,
    SemanticContractRecord, ShapeIndexContract, TransformDecision,
};

#[test]
fn guarded_law_with_executable_proof_validates() {
    let law = GuardedLaw::unconditional(AlgebraicLaw::Commutative);
    assert_eq!(law.validate(), Ok(()));
}

#[test]
fn law_with_no_executable_proof_evidence_is_rejected_by_name() {
    let law =
        GuardedLaw::unconditional(AlgebraicLaw::Associative).with_proof_method(ProofMethod::None);
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
    let law = GuardedLaw::unconditional(AlgebraicLaw::Commutative)
        .with_counterexample_generator(generator);

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
    let law = GuardedLaw::unconditional(AlgebraicLaw::Commutative)
        .with_counterexample_generator(generator);

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
    let law = GuardedLaw::unconditional(AlgebraicLaw::Commutative)
        .with_direction(LawDirection::Bidirectional)
        .with_guard(LawGuard::ExactOnly)
        .with_proof_method(ProofMethod::ExhaustiveU8);
    let record = SemanticContractRecord::with_laws("vyre-libs::math::add", vec![law]);
    assert_eq!(record.validate(), Ok(()));
    assert_eq!(record.decision.laws().len(), 1);
}

#[test]
fn contract_record_serde_roundtrip() {
    let law = GuardedLaw::unconditional(AlgebraicLaw::Idempotent)
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
    };

    let serialized = serde_json::to_string(&record).expect("serialization succeeds");
    let deserialized: SemanticContractRecord =
        serde_json::from_str(&serialized).expect("deserialization succeeds");
    assert_eq!(record, deserialized);
}
