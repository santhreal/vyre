//! The foundation decisions the oracle still shares, pinned.
//!
//! The interpreter evaluates the program as submitted: no optimizer pass, no
//! schedule selection, no lowering, and no rewrite of a node on the way in. Two
//! decisions on that path are still foundation's, and neither is a program
//! rewrite. Both are pinned here so removing one is a test failure rather than
//! a silent change in what the oracle admits.
//!
//! Admission is shared on purpose. A program the compiler rejects has no
//! expected output for a backend to be wrong against, so the oracle refuses
//! exactly what `validate` refuses. The consequence is the part worth stating:
//! a validator that admits an illegal program admits it on both sides.
//!
//! The top-level `Region` contract is shape, not semantics.
//! `Program::wrapped` puts the entry sequence in a `Region`, the cleanup pass
//! that flattens small regions can leave a statement-shaped entry list behind,
//! and `Program::reconcile_runnable_top_level` puts the wrapper back. The
//! interpreter applies it to what it is handed rather than carrying a second
//! opinion about the shape.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::reference_eval;
use vyre_reference::value::Value;

/// Duplicate binding slots are a validation rejection and nothing else.
///
/// The interpreter keys every buffer by name, so it would run this program and
/// answer with a value: two declarations sharing binding 1 collide only in a
/// descriptor set, which the oracle never builds. The refusal is therefore
/// entirely foundation's V108, and it disappears the moment the oracle stops
/// consulting the shared validator.
#[test]
fn the_oracle_refuses_exactly_what_the_shared_validator_refuses() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("left", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
            BufferDecl::storage("right", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "left",
            Expr::u32(0),
            Expr::load("right", Expr::u32(0)),
        )],
    );

    let error = reference_eval(
        &program,
        &[
            Value::from(vec![0u8; 4]),
            Value::from(7u32.to_le_bytes().to_vec()),
        ],
    )
    .expect_err("a program the shared validator rejects has no reference result");

    let source = error
        .validation_source()
        .expect("the refusal must carry the validation issue that caused it");
    assert_eq!(
        source.code().to_string(),
        "V108",
        "the oracle must refuse the duplicate binding slot the validator names, got: {error}"
    );
}

/// A statement-shaped entry list is re-wrapped, not refused.
///
/// `reconcile_runnable_top_level` re-applies the same `Region` wrapper
/// `Program::wrapped` builds and moves nothing else, so the program evaluates
/// to the value its statements compute.
#[test]
fn a_statement_entry_list_is_re_wrapped_and_evaluated() {
    let program = Program::from_raw_parts(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![
            Node::let_bind("value", Expr::add(Expr::u32(40), Expr::u32(2))),
            Node::store("out", Expr::u32(0), Expr::var("value")),
        ],
    );

    let outputs = reference_eval(&program, &[])
        .expect("a statement-shaped entry list must be re-wrapped and evaluated");

    assert_eq!(outputs[0].to_bytes(), 42u32.to_le_bytes().to_vec());
}

/// The re-wrap is not a blanket repair.
///
/// An entry list whose first node is a `Store` keeps the refusal, so the
/// wrapper is applied where the cleanup pass could have removed one and
/// nowhere else.
#[test]
fn a_store_first_entry_list_is_still_refused() {
    let program = Program::from_raw_parts(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );

    let error = reference_eval(&program, &[])
        .expect_err("a store-first entry list must still name the region contract");

    assert!(
        error.to_string().contains("top-level Region-wrapped Program"),
        "the refusal must name the region contract, got: {error}"
    );
}
