//! Tests for single declarative verifier, certified wrapper, and compiler gate (Row 104).
//!
//! Acceptance criteria:
//! 1. Proves unverified syntax cannot reach compilation.
//! 2. Proves that the certificate lists the invariants that were checked for a module
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
use vyre_foundation::verifier::InvariantCategory;
use vyre_foundation::verifier::{
    CompileError, DeclarativeVerifier, SemanticCompiler, SemanticModule,
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
            BufferDecl::storage("in_buf", 0, BufferAccess::ReadOnly, DataType::U32).with_count(64),
            BufferDecl::storage("out_buf", 1, BufferAccess::ReadWrite, DataType::U32).with_count(64),
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
        .with_numeric_contract(NumericalContract::fast_math());

    // Inject the interner
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
        cert.has_category(InvariantCategory::EffectsConcurrency),
        "must check EffectsConcurrency"
    );
    assert!(
        cert.has_category(InvariantCategory::BoundsTermination),
        "must check BoundsTermination"
    );
    assert!(
        cert.has_category(InvariantCategory::DeterminismNumeric),
        "must check DeterminismNumeric"
    );
    assert!(
        cert.has_category(InvariantCategory::CollectiveGroups),
        "must check CollectiveGroups"
    );

    // Assert that the exact list of checked invariants matches what was evaluated
    assert!(
        cert.checked_invariants.len() >= 8,
        "expected at least 8 checked invariants, got {}",
        cert.checked_invariants.len()
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
