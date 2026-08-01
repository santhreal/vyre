//! The interpreter owns both halves of its buffer ABI.
//!
//! `reference_eval` consumes one `Value` per [`is_reference_input`] decl and
//! returns one buffer per [`is_reference_output`] decl, both in
//! `Program::buffers` order. Callers that build an input vector or index a
//! named output have to read those predicates rather than restate them: a
//! restatement that disagrees on one decl shifts every later position by one,
//! and the failure surfaces as a missing value for whichever buffer the offset
//! ran past, which points nowhere near the copy that drifted.
//!
//! The restatement that actually shipped was `!decl.is_output()`, in the
//! pairwise composition harness. See BACKLOG.md R72.

use vyre::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;
use vyre_reference::{is_reference_input, is_reference_output, reference_eval};

fn read_only(name: &str, binding: u32) -> BufferDecl {
    BufferDecl::read(name, binding, DataType::U32).with_count(4)
}

fn read_write(name: &str, binding: u32) -> BufferDecl {
    BufferDecl::read_write(name, binding, DataType::U32).with_count(4)
}

/// A read-only input is supplied by the caller and is not returned.
#[test]
fn a_read_only_buffer_is_an_input_and_not_an_output() {
    let decl = read_only("in", 0);
    assert!(is_reference_input(&decl));
    assert!(!is_reference_output(&decl));
}

/// A read-write buffer is both: the caller seeds it and reads the result back.
/// This is the case a `!is_output()` restatement gets right by accident and the
/// reason the two predicates are not complements of each other.
#[test]
fn a_read_write_buffer_is_both_an_input_and_an_output() {
    let decl = read_write("acc", 1);
    assert!(is_reference_input(&decl));
    assert!(is_reference_output(&decl));
}

/// A workgroup buffer is allocated per dispatch, so it is neither.
#[test]
fn a_workgroup_buffer_is_neither_supplied_nor_returned() {
    let decl = BufferDecl::workgroup("scratch", 4, DataType::U32);
    assert!(!is_reference_input(&decl));
    assert!(!is_reference_output(&decl));
    assert_eq!(decl.access(), BufferAccess::Workgroup);
}

/// A backend-allocated output is zero-filled by the interpreter, so the caller
/// supplies nothing for it.
#[test]
fn a_backend_allocated_output_is_returned_but_not_supplied() {
    let decl = BufferDecl::output("out", 2, DataType::U32).with_count(4);
    assert!(decl.is_backend_allocated_output());
    assert!(!is_reference_input(&decl));
    assert!(is_reference_output(&decl));
}

/// A write-only buffer is allocated by the backend, so the caller supplies
/// nothing for it. This is one of the two shapes `!decl.is_output()` gets
/// wrong: `is_output` is false here, so that form would demand a value the
/// interpreter never reads, shifting every later input by one.
#[test]
fn a_write_only_buffer_is_not_supplied_by_the_caller() {
    let decl = BufferDecl::storage("sink", 0, BufferAccess::WriteOnly, DataType::U32).with_count(4);
    assert!(
        !decl.is_output(),
        "the flag is not set, which is exactly why the narrower predicate got this wrong"
    );
    assert!(decl.is_backend_allocated_output());
    assert!(!is_reference_input(&decl));
    assert!(is_reference_output(&decl));
}

/// A read-write buffer marked live-out is the other shape. It leaves the
/// dispatch, so the backend owns its storage and the caller supplies nothing.
#[test]
fn a_live_out_read_write_buffer_is_not_supplied_by_the_caller() {
    let decl = read_write("acc", 1).with_pipeline_live_out(true);
    assert!(!decl.is_output(), "again the flag is not what decides it");
    assert!(decl.is_backend_allocated_output());
    assert!(!is_reference_input(&decl));
    assert!(is_reference_output(&decl));
}

/// The predicates describe what `reference_eval` actually does, not merely what
/// the doc says. Counting decls by predicate must predict the length of the
/// input vector it accepts and of the output vector it returns.
#[test]
fn the_predicates_predict_the_shapes_reference_eval_uses() {
    let buffers = vec![
        read_only("in", 0),
        read_write("acc", 1),
        BufferDecl::output("out", 2, DataType::U32).with_count(4),
        BufferDecl::workgroup("scratch", 3, DataType::U32),
    ];
    let body = vec![
        Node::store("acc", Expr::gid_x(), Expr::load("in", Expr::gid_x())),
        Node::store("out", Expr::gid_x(), Expr::load("acc", Expr::gid_x())),
    ];
    let program = Program::wrapped(buffers, [4, 1, 1], body);

    let expected_inputs = program
        .buffers()
        .iter()
        .filter(|decl| is_reference_input(decl))
        .count();
    let expected_outputs = program
        .buffers()
        .iter()
        .filter(|decl| is_reference_output(decl))
        .count();
    assert_eq!(
        expected_inputs, 2,
        "in and acc are supplied; out and scratch are not"
    );
    assert_eq!(
        expected_outputs, 2,
        "acc and out come back; in and scratch do not"
    );

    let pack = |words: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(words));
    let outputs = reference_eval(&program, &[pack(&[1, 2, 3, 4]), pack(&[0, 0, 0, 0])])
        .expect("one Value per is_reference_input decl is exactly what the interpreter wants");
    assert_eq!(
        outputs.len(),
        expected_outputs,
        "the returned vector has one entry per is_reference_output decl"
    );
    assert_eq!(
        outputs[0].to_bytes(),
        vyre_primitives::wire::pack_u32_slice(&[1, 2, 3, 4]),
        "acc is the first output in Program::buffers order"
    );
    assert_eq!(
        outputs[1].to_bytes(),
        vyre_primitives::wire::pack_u32_slice(&[1, 2, 3, 4]),
        "out is the second"
    );
}

/// Supplying a value for a buffer the predicate excludes is rejected rather
/// than silently absorbed, which is what makes the predicate worth reading.
#[test]
fn supplying_one_value_too_many_is_rejected() {
    let buffers = vec![read_only("in", 0), read_write("acc", 1)];
    let body = vec![Node::store(
        "acc",
        Expr::gid_x(),
        Expr::load("in", Expr::gid_x()),
    )];
    let program = Program::wrapped(buffers, [4, 1, 1], body);
    let pack = |words: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(words));
    let error = reference_eval(
        &program,
        &[
            pack(&[1, 2, 3, 4]),
            pack(&[0, 0, 0, 0]),
            pack(&[9, 9, 9, 9]),
        ],
    )
    .expect_err("three values for two input decls is a contract violation");
    assert!(
        error.to_string().contains("unused input"),
        "the diagnostic must say the extra value went unused: {error}"
    );
}
