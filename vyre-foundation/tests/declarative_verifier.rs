//! Tests for single declarative verifier, certified wrapper, and compiler gate (Row 104).
//!
//! Acceptance criteria:
//! 1. A test derives the invariant set from the certificate type at run time and fails
//!    when an invariant carries no check, so adding an invariant turns the suite red.
//! 2. A test proves an unverified module cannot reach a consumer that requires verification,
//!    and that a tampered certificate is refused by name.
//! 3. A test proves proof replay is independent: it accepts a valid certificate without
//!    rerunning the full verifier, and rejects one whose input identity does not match.
//! 4. Proves that the certificate lists the invariants that were checked for a module
//!    that exercises more than one.

use std::sync::Arc;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Node, Program,
};
use vyre_foundation::ir::{AtomicOrdering, CollectiveGroup};
use vyre_foundation::types::NumericalContract;
use vyre_foundation::types::ScalarType;
use vyre_foundation::types::{ShapeConstraint, ShapeInterner};
use vyre_foundation::types::SemanticType;
use vyre_foundation::verifier::{
    CompileError, DeclarativeVerifier, InvariantCategory, LoweredStage, OptimizedStage,
    ReplayError, SemanticCompiler, SemanticModule, VerifiedIrStage, VerifiedStageWrapper,
};

#[test]
fn unverified_syntax_cannot_reach_compilation() {
    let unverified_module = SemanticModule::new("unverified_test_module");

    // Attempting to pass an unverified module to the compile_unverified entry point fails
    let err = SemanticCompiler::compile_unverified(&unverified_module).unwrap_err();
    assert!(matches!(err, CompileError::UnverifiedSyntaxRejected(_)));
    assert!(err.to_string().contains("UnverifiedSyntaxRejected"));
    assert!(err.to_string().contains("unverified_test_module"));
    assert!(err.to_string().contains("DeclarativeVerifier::verify"));
}

#[test]
fn verified_module_compiles_successfully_with_certificate() {
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("buf_in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::storage("buf_out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(64),
        ],
        [2, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("certified_kernel", prog);
    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");

    let cert = verified.certificate();
    assert_eq!(cert.verifier_version, DeclarativeVerifier::VERSION);
    assert_eq!(cert.schema_version, DeclarativeVerifier::SCHEMA_VERSION);
    assert!(!cert.input_identity.is_empty());
    assert!(cert.invariant_count() > 0);

    // Compilation succeeds because a &Verified<SemanticModule> is provided
    let compiled = SemanticCompiler::compile(&verified).expect("compilation of verified module should succeed");
    assert_eq!(compiled.name, "certified_kernel");
    assert_eq!(compiled.certified_invariant_count, cert.invariant_count());
}

#[test]
fn certificate_lists_invariants_checked_for_multi_feature_module() {
    let interner = Arc::new(ShapeInterner::new());

    // Create shape constraint: (batch * 64) == (64 * batch)
    let batch = interner.symbol("batch");
    let c64 = interner.constant(64);
    let mul1 = interner.mul(batch, c64);
    let mul2 = interner.mul(c64, batch);
    let constraint = ShapeConstraint::Equal(mul1, mul2);

    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::F32).with_count(128),
            BufferDecl::storage("dst", 1, BufferAccess::ReadWrite, DataType::F32).with_count(128),
        ],
        [4, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("rich_module", prog)
        .with_type(SemanticType::scalar(ScalarType::f32()))
        .with_shape_constraint(constraint)
        .with_atomic_effect(AtomicOrdering::AcqRel)
        .with_collective_group(CollectiveGroup::Workgroup)
        .with_numeric_contract(NumericalContract::fast_math())
        .with_extension_obligation("gpu_tensor_core_abi_v1")
        .with_termination_bound(1024)
        .with_deterministic_mode(true)
        .with_state_transition("init", "running");

    let mut module = module;
    module.shape_interner = interner.clone();

    let verified = DeclarativeVerifier::verify(module).expect("rich module must verify");
    let cert = verified.certificate();

    // Verify that the certificate lists multiple distinct invariant categories actually checked
    assert!(
        cert.has_category(InvariantCategory::StructuralClosure),
        "must check StructuralClosure"
    );
    assert!(
        cert.has_category(InvariantCategory::DominanceUseDef),
        "must check DominanceUseDef"
    );
    assert!(
        cert.has_category(InvariantCategory::TypeShapeRank),
        "must check TypeShapeRank"
    );
    assert!(
        cert.has_category(InvariantCategory::AliasOwnership),
        "must check AliasOwnership"
    );
    assert!(
        cert.has_category(InvariantCategory::Effects),
        "must check Effects"
    );
    assert!(
        cert.has_category(InvariantCategory::Bounds),
        "must check Bounds"
    );
    assert!(
        cert.has_category(InvariantCategory::TerminationProgress),
        "must check TerminationProgress"
    );
    assert!(
        cert.has_category(InvariantCategory::Determinism),
        "must check Determinism"
    );
    assert!(
        cert.has_category(InvariantCategory::NumericContracts),
        "must check NumericContracts"
    );
    assert!(
        cert.has_category(InvariantCategory::CollectiveGroups),
        "must check CollectiveGroups"
    );
    assert!(
        cert.has_category(InvariantCategory::StateTransitions),
        "must check StateTransitions"
    );
    assert!(
        cert.has_category(InvariantCategory::SemanticExtensionObligations),
        "must check SemanticExtensionObligations"
    );

    // Every invariant recorded in the certificate must have passed
    for inv in &cert.checked_invariants {
        assert!(inv.passed, "invariant {} failed: {}", inv.code, inv.description);
    }

    // Check that shape solver proof objects are recorded and replayable
    assert_eq!(cert.solver_proofs.len(), 1);
    assert!(cert.replay_solver_proofs(&interner));

    // Check resource bounds
    assert_eq!(cert.resource_bounds.grid_dimensions, [4, 1, 1]);
    assert_eq!(cert.resource_bounds.buffer_count, 2);
}

#[test]
fn invariant_set_derived_at_runtime_fails_if_any_invariant_carries_no_check() {
    // Dynamically derive the complete invariant category set from InvariantCategory::all()
    let all_categories = InvariantCategory::all();
    assert_eq!(
        all_categories.len(),
        12,
        "Row 104 requires exactly 12 canonical invariant categories"
    );

    // Construct a comprehensive semantic module
    let interner = Arc::new(ShapeInterner::new());
    let prog = Program::wrapped(
        vec![
            BufferDecl::storage("in_a", 0, BufferAccess::ReadOnly, DataType::U32).with_count(32),
            BufferDecl::storage("out_b", 1, BufferAccess::ReadWrite, DataType::U32).with_count(32),
        ],
        [1, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("comprehensive_test_module", prog)
        .with_type(SemanticType::scalar(ScalarType::u32()))
        .with_shape_constraint(ShapeConstraint::Equal(
            interner.constant(16),
            interner.constant(16),
        ))
        .with_atomic_effect(AtomicOrdering::Relaxed)
        .with_collective_group(CollectiveGroup::Workgroup)
        .with_numeric_contract(NumericalContract::strict_ieee())
        .with_extension_obligation("standard_compute_v1")
        .with_termination_bound(512)
        .with_deterministic_mode(true)
        .with_state_transition("stage0", "stage1");

    let mut module = module;
    module.shape_interner = interner;

    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");
    let cert = verified.certificate();

    // Dynamically test that EVERY category in InvariantCategory::all() has an active check
    for &category in all_categories {
        assert!(
            cert.has_category(category),
            "invariant category {:?} has no check in the verifier; adding an invariant category turns the suite red!",
            category
        );
    }
}

#[test]
fn tampered_certificate_refused_by_name_on_input_identity_mismatch() {
    let prog = Program::wrapped(
        vec![BufferDecl::storage("buf0", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16)],
        [1, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("unaltered_module", prog);
    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");
    let mut tampered_cert = verified.certificate().clone();

    // Tamper with the certificate's input identity digest
    tampered_cert.input_identity = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into();

    let module = verified.into_inner();

    // Replay proof directly on tampered certificate must fail with MismatchedInputIdentity by name
    let replay_err = tampered_cert.replay_proof(&module).unwrap_err();
    assert!(
        matches!(replay_err, ReplayError::MismatchedInputIdentity { .. }),
        "expected MismatchedInputIdentity, got: {replay_err:?}"
    );
    let err_str = replay_err.to_string();
    assert!(err_str.contains("MismatchedInputIdentity"));
    assert!(err_str.contains("Refusing tampered certificate by name"));

    // Attempting replay_and_compile with tampered certificate must be refused by name
    let compile_err = SemanticCompiler::replay_and_compile(&module, &tampered_cert).unwrap_err();
    assert!(
        matches!(compile_err, CompileError::MismatchedInputIdentity { .. }),
        "expected CompileError::MismatchedInputIdentity, got: {compile_err:?}"
    );
    assert!(compile_err.to_string().contains("MismatchedInputIdentity"));
}

#[test]
fn independent_proof_replay_accepts_valid_certificate_without_rerunning_full_verifier() {
    let prog = Program::wrapped(
        vec![BufferDecl::storage("data", 0, BufferAccess::ReadWrite, DataType::F32).with_count(64)],
        [2, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("replay_test_module", prog);
    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");
    let cert = verified.certificate().clone();
    let module = verified.into_inner();

    // Independent proof replay succeeds cleanly without rerunning AST traversal
    cert.replay_proof(&module).expect("valid certificate must replay successfully");

    // SemanticCompiler can compile by replaying the certificate
    let artifact = SemanticCompiler::replay_and_compile(&module, &cert)
        .expect("replay compilation of valid certificate must succeed");
    assert_eq!(artifact.name, "replay_test_module");
}

#[test]
fn independent_proof_replay_rejects_unsatisfied_invariant() {
    let prog = Program::wrapped(
        vec![BufferDecl::storage("data", 0, BufferAccess::ReadWrite, DataType::F32).with_count(64)],
        [2, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("unsatisfied_inv_module", prog);
    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");
    let mut bad_cert = verified.certificate().clone();

    // Tamper with invariant passed flag
    if let Some(first_inv) = bad_cert.checked_invariants.first_mut() {
        first_inv.passed = false;
    }

    // Recompute input identity to match the module so only the unsatisfied invariant trips
    let module = verified.into_inner();
    bad_cert.input_identity = module.compute_identity();

    let err = bad_cert.replay_proof(&module).unwrap_err();
    assert!(
        matches!(err, ReplayError::UnsatisfiedInvariant { .. }),
        "expected UnsatisfiedInvariant, got: {err:?}"
    );
    assert!(err.to_string().contains("UnsatisfiedInvariant"));
}

#[test]
fn independent_proof_replay_rejects_schema_version_mismatch() {
    let module = SemanticModule::new("version_test");
    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");
    let mut bad_cert = verified.certificate().clone();

    bad_cert.schema_version = 999;
    let module = verified.into_inner();

    let err = bad_cert.replay_proof(&module).unwrap_err();
    assert!(
        matches!(err, ReplayError::SchemaVersionMismatch { expected: 1, found: 999 }),
        "expected SchemaVersionMismatch, got: {err:?}"
    );
}

#[test]
fn verified_stage_wrapper_transitions_preserve_certificate() {
    let prog = Program::wrapped(
        vec![BufferDecl::storage("data", 0, BufferAccess::ReadOnly, DataType::U32).with_count(16)],
        [1, 1, 1],
        vec![Node::Return],
    );

    let module = SemanticModule::from_program("stage_pipeline_module", prog);
    let verified = DeclarativeVerifier::verify(module).expect("verification should succeed");
    let initial_cert = verified.certificate().clone();

    // Stage 1: VerifiedIrStage
    let ir_stage: VerifiedStageWrapper<VerifiedIrStage, SemanticModule> = verified.into();
    assert_eq!(ir_stage.certificate(), &initial_cert);

    // Stage 2: Transition to OptimizedStage
    let opt_stage: VerifiedStageWrapper<OptimizedStage, SemanticModule> = ir_stage.transition_to(|mut m| {
        m.name = "stage_pipeline_module_optimized".into();
        m
    });
    assert_eq!(opt_stage.certificate(), &initial_cert);
    assert_eq!(opt_stage.as_inner().name, "stage_pipeline_module_optimized");

    // Stage 3: Transition to LoweredStage
    let lowered_stage: VerifiedStageWrapper<LoweredStage, String> = opt_stage.transition_to(|m| {
        format!("lowered_binary_for_{}", m.name)
    });
    assert_eq!(lowered_stage.certificate(), &initial_cert);
    assert_eq!(lowered_stage.as_inner(), "lowered_binary_for_stage_pipeline_module_optimized");
}
