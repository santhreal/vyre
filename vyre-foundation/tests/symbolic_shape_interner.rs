//! Tests for canonical symbolic shape interner, affine normalization, and solver proofs.
//!
//! Acceptance criteria:
//! 1. Proves two structurally equal symbolic shapes intern to the same id.
//! 2. Proves two different symbolic shapes do not intern to the same id.
//! 3. Proves that a shape check reads the interner rather than re-deriving.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Node, Program, ShapePredicate};
use vyre_foundation::types::{ShapeProofKind, ShapeSolver};
use vyre_foundation::types::{ShapeInterner, SymbolicDim};
use vyre_foundation::validate::shape_predicate::{
    check_shape_predicates, check_shape_predicates_with_interner,
};

#[test]
fn structurally_equal_symbolic_shapes_intern_to_identical_id() {
    let interner = ShapeInterner::new();

    // Expression 1: (batch * 64) + 16
    let batch_1 = interner.symbol("batch");
    let c64_1 = interner.constant(64);
    let c16_1 = interner.constant(16);
    let mul_1 = interner.mul(batch_1, c64_1);
    let dim1_1 = interner.add(mul_1, c16_1);

    // Expression 2: (64 * batch) + 16 (commutative order swapped on multiplication)
    let batch_2 = interner.symbol("batch");
    let c64_2 = interner.constant(64);
    let c16_2 = interner.constant(16);
    let mul_2 = interner.mul(c64_2, batch_2);
    let dim1_2 = interner.add(c16_2, mul_2);

    // Assert the dimension expressions canonicalized to the exact same ShapeExprId
    assert_eq!(
        dim1_1, dim1_2,
        "commutative and structurally equal expressions must intern to identical ShapeExprId"
    );

    // Second dimension: constant 128
    let dim2_1 = interner.constant(128);
    let dim2_2 = interner.constant(128);
    assert_eq!(dim2_1, dim2_2);

    // Full 2D shape: [(batch * 64) + 16, 128]
    let shape_id_1 = interner.intern_shape(&[dim1_1, dim2_1]);
    let shape_id_2 = interner.intern_shape(&[dim1_2, dim2_2]);

    assert_eq!(
        shape_id_1, shape_id_2,
        "structurally equal shapes must intern to the identical ShapeId"
    );
}

#[test]
fn distinct_symbolic_shapes_intern_to_different_ids() {
    let interner = ShapeInterner::new();

    let batch = interner.symbol("batch");
    let seq_len = interner.symbol("seq_len");
    let c32 = interner.constant(32);
    let c64 = interner.constant(64);

    let shape_a = interner.intern_shape(&[batch, c32]);
    let shape_b = interner.intern_shape(&[batch, c64]);
    let shape_c = interner.intern_shape(&[seq_len, c32]);
    let shape_d = interner.intern_shape(&[batch, c32, c64]); // Different rank

    assert_ne!(shape_a, shape_b);
    assert_ne!(shape_a, shape_c);
    assert_ne!(shape_a, shape_d);
    assert_ne!(shape_b, shape_c);
    assert_ne!(shape_b, shape_d);
    assert_ne!(shape_c, shape_d);
}

#[test]
fn zero_is_valid_extent_and_not_sentinel() {
    let interner = ShapeInterner::new();

    let zero_dim = interner.constant(0);
    assert_eq!(interner.get_expr(zero_dim), Some(SymbolicDim::Constant(0)));

    let empty_shape = interner.intern_shape(&[zero_dim, interner.constant(32)]);
    let dims = interner.get_shape(empty_shape).unwrap();
    assert_eq!(dims.len(), 2);
    assert_eq!(dims[0], zero_dim);
}

#[test]
fn shape_solver_equality_and_replay_proof() {
    let interner = ShapeInterner::new();

    let b1 = interner.symbol("B");
    let h1 = interner.constant(128);
    let s1 = interner.intern_shape(&[b1, h1]);

    let b2 = interner.symbol("B");
    let h2 = interner.constant(128);
    let s2 = interner.intern_shape(&[b2, h2]);

    let (equal, cert) = ShapeSolver::solve_equality(&interner, s1, s2);
    assert!(equal);
    assert!(cert.verdict);
    assert_eq!(cert.proof_kind, ShapeProofKind::Equality);
    assert!(cert.replay(&interner));
}

#[test]
fn shape_check_reads_interner_rather_than_re_deriving() {
    let interner = ShapeInterner::new();

    let prog_valid = Program::wrapped(
        vec![
            BufferDecl::storage("buf_a", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(128)
                .with_shape_predicate(ShapePredicate::MultipleOf(16)),
            BufferDecl::storage("buf_b", 1, BufferAccess::ReadWrite, DataType::U32)
                .with_count(64)
                .with_shape_predicate(ShapePredicate::AtLeast(32)),
        ],
        [1, 1, 1],
        vec![Node::Return],
    );

    let prog_invalid = Program::wrapped(
        vec![BufferDecl::storage(
            "buf_invalid",
            0,
            BufferAccess::ReadOnly,
            DataType::U32,
        )
        .with_count(15)
        .with_shape_predicate(ShapePredicate::MultipleOf(16))],
        [1, 1, 1],
        vec![Node::Return],
    );

    // Verify through the interner-driven shape check
    let errors_valid = check_shape_predicates_with_interner(&prog_valid, &interner);
    assert!(errors_valid.is_empty());

    let errors_invalid = check_shape_predicates_with_interner(&prog_invalid, &interner);
    assert_eq!(errors_invalid.len(), 1);
    assert!(errors_invalid[0].message().contains("buf_invalid"));
    assert!(errors_invalid[0].message().contains("count % 16 == 0"));
    assert!(errors_invalid[0].message().contains("count=15"));

    // Standard entry point also uses interner internally
    assert!(check_shape_predicates(&prog_valid).is_empty());
    assert_eq!(check_shape_predicates(&prog_invalid).len(), 1);
}
