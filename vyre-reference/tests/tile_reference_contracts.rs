//! Contract tests for reference interpreter execution of Tile operations.
//!
//! The value cases are `vyre_test_support::tile_programs::tile_cases`, the same
//! corpus the lowering contract lowers and simulates, so the oracle and the
//! lowering are compared on one set of programs with one set of expected
//! values. A case that only the oracle can answer, a rejection, stays here.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Ident, Layout, Node, Program, Residency, Tile,
};
use vyre_reference::reference_eval;
use vyre_reference::value::Value;
use vyre_test_support::tile_programs::tile_cases;

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

#[test]
fn reference_eval_computes_every_tile_case() {
    for case in tile_cases() {
        let name = case.name;
        let inputs: Vec<Value> = case
            .inputs
            .iter()
            .map(|buffer| Value::from(encode_f32(buffer)))
            .collect();

        let outputs = reference_eval(&case.program, &inputs)
            .unwrap_or_else(|e| panic!("case {name} must evaluate on the oracle: {e}"));

        let actual = decode_f32(&outputs[0].to_bytes());
        assert_eq!(
            actual, case.expected,
            "case {name} oracle output did not match expected"
        );
    }
}

#[test]
fn reference_eval_rejects_elementwise_operand_that_does_not_divide_the_output() {
    // A 3-element input against a 4-element output has no broadcast, so it is
    // rejected rather than folded to whichever length happens to fit.
    let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let b_data = vec![1.0f32, 2.0, 3.0];
    let tile_a = Tile::new(
        DataType::F32,
        vec![2, 2],
        Layout::RowMajor,
        Residency::Register,
    );
    let tile_b = Tile::new(
        DataType::F32,
        vec![3],
        Layout::RowMajor,
        Residency::Register,
    );

    let program = Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(4),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(3),
            BufferDecl::output("out", 2, DataType::F32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::tile_load(
                "t_a",
                tile_a,
                "a",
                vec![Expr::u32(0), Expr::u32(0)],
                Layout::RowMajor,
            ),
            Node::tile_load("t_b", tile_b, "b", vec![Expr::u32(0)], Layout::RowMajor),
            Node::tile_elementwise(
                "sum",
                vec![Ident::from("t_a"), Ident::from("t_b")],
                vec![Node::let_bind(
                    "sum",
                    Expr::add(Expr::var("t_a"), Expr::var("t_b")),
                )],
            ),
            Node::tile_store("out", vec![Expr::u32(0)], "sum"),
        ],
    );

    let err = reference_eval(
        &program,
        &[
            Value::from(encode_f32(&a_data)),
            Value::from(encode_f32(&b_data)),
        ],
    )
    .expect_err("reference_eval must reject 3-element input against 4-element output");

    let err_msg = err.to_string();
    assert!(
        err_msg.contains("does not divide output length 4"),
        "expected divisibility error, got: {err_msg}"
    );
}
