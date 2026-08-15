// The one owner of the random-program corpus and the fixed-point scaffold the
// optimizer contract suites run against.
//
// `adversarial_graph_canonical_laws` and `optimizer_idempotence_proptest` ask
// different questions -- one about graph_view and the algebraic law registry,
// the other about the whole registered pass set -- but both had to build the
// same thing first: a bounded pure-u32 program, run an optimizer entry point on
// it repeatedly, and compare the runs against each other and against the
// reference interpreter. They carried a copy each, down to identical
// `prop_recursive` depth and branch weights, with `store_program` and
// `output_only_store` as two names for one function.
//
// Two copies of a generator is not a duplicated helper, it is two corpora. The
// property a suite proves is only as wide as the programs it draws, so a
// generator that drifts in one file silently narrows one suite's claim while
// both stay green.
//
// What is NOT here is which entry point to run or which malformed graph to
// reject. That is each suite's own subject.

use proptest::prelude::*;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::value::Value;

/// The single-element output buffer every generated program writes to.
pub(crate) fn test_output_buffer() -> BufferDecl {
    BufferDecl::read_write("out", 0, DataType::U32).with_count(1)
}

/// A program holding `body`, wrapped in the root region the IR requires.
pub(crate) fn program_with_body(body: Vec<Node>) -> Program {
    Program::wrapped(vec![test_output_buffer()], [1, 1, 1], body)
}

/// A program whose whole body is one store of `expr` to `out[0]`.
pub(crate) fn output_only_store(expr: Expr) -> Program {
    program_with_body(vec![Node::store("out", Expr::u32(0), expr)])
}

/// Run `program` on the reference interpreter with a single zero input.
pub(crate) fn run_reference(
    program: &Program,
) -> Result<Vec<Value>, vyre_reference::ReferenceError> {
    vyre_reference::reference_eval(program, &[Value::U32(0)])
}

fn leaf_expr() -> impl Strategy<Value = Expr> {
    prop_oneof![
        (0_u16..=1024).prop_map(|value| Expr::u32(u32::from(value))),
        Just(Expr::gid_x()),
    ]
}

fn non_zero_literal() -> impl Strategy<Value = Expr> {
    (1_u16..=1024).prop_map(|value| Expr::u32(u32::from(value)))
}

fn shift_amount() -> impl Strategy<Value = Expr> {
    (0_u8..=31).prop_map(|value| Expr::u32(u32::from(value)))
}

/// Bounded pure u32 expression surface: every arithmetic, bitwise and shift
/// `BinOp` an optimizer pass may rewrite, with divisors and shift amounts
/// constrained so the reference interpreter has a defined answer.
pub(crate) fn u32_expr() -> impl Strategy<Value = Expr> {
    leaf_expr().prop_recursive(5, 64, 4, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::add(left, right)),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::mul(left, right)),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::bitand(left, right)),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::bitor(left, right)),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::bitxor(left, right)),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::sub(left, right)),
            (inner.clone(), non_zero_literal()).prop_map(|(left, right)| Expr::div(left, right)),
            (inner.clone(), shift_amount()).prop_map(|(left, right)| Expr::shl(left, right)),
            (inner.clone(), shift_amount()).prop_map(|(left, right)| Expr::shr(left, right)),
        ]
    })
}

/// A straight-line program of up to 16 let bindings feeding one store.
pub(crate) fn program_strategy() -> impl Strategy<Value = Program> {
    prop::collection::vec(u32_expr(), 1..16).prop_map(|exprs| {
        let mut body = Vec::with_capacity(exprs.len() + 1);
        for (index, expr) in exprs.into_iter().enumerate() {
            body.push(Node::let_bind(format!("v{index}"), expr));
        }
        body.push(Node::store(
            "out",
            Expr::u32(0),
            Expr::var(format!("v{}", body.len().saturating_sub(1))),
        ));
        program_with_body(body)
    })
}

/// One optimizer entry point a suite holds to the fixed-point contract.
#[derive(Debug, Clone, Copy)]
pub(crate) struct OptimizerEntryPoint {
    /// Module path of the entry point, for the failure message.
    pub(crate) label: &'static str,
    /// Run it once. Panics rather than returning an error, because an entry
    /// point that cannot converge on a generated program is a suite failure,
    /// not a case the property is allowed to skip.
    pub(crate) run: fn(Program) -> Program,
}

/// The canonical wire encoding of `program`.
///
/// # Panics
/// Panics when the program does not encode, which means the entry point
/// produced IR the wire codec does not model.
pub(crate) fn canonical_wire(label: &str, program: &Program) -> Vec<u8> {
    program
        .to_wire()
        .unwrap_or_else(|error| panic!("Fix: `{label}` output must encode: {error}"))
}

/// Assert `entry` reaches a fixed point on `program` and preserves what the
/// reference interpreter computes.
///
/// Three runs, not two: the second run proves the entry point converged and the
/// third proves it HOLDS the fixed point rather than oscillating with period
/// two. Both the structural `Program` and its wire encoding are compared,
/// because a rewrite that survives serialization is not the same claim as one
/// that survives equality.
///
/// # Errors
/// Returns the proptest failure when a run differs from its predecessor or from
/// the original program's reference output.
pub(crate) fn assert_fixed_point_and_semantics(
    entry: &OptimizerEntryPoint,
    program: Program,
) -> Result<(), TestCaseError> {
    let label = entry.label;
    let expected = run_reference(&program).unwrap_or_else(|error| {
        panic!("Fix: generated program must run on the reference interpreter: {error}")
    });

    let mut previous = program;
    for run_index in 1..=3_u32 {
        let next = (entry.run)(previous.clone());
        let observed = run_reference(&next).unwrap_or_else(|error| {
            panic!("Fix: `{label}` output must run on the reference interpreter: {error}")
        });
        prop_assert_eq!(
            &expected,
            &observed,
            "`{}` run {} changed what the program computes",
            label,
            run_index
        );
        if run_index > 1 {
            prop_assert_eq!(
                canonical_wire(label, &previous),
                canonical_wire(label, &next),
                "`{}` run {} perturbed the canonical wire form of its own output",
                label,
                run_index
            );
            prop_assert_eq!(
                &previous,
                &next,
                "`{}` run {} perturbed its own output",
                label,
                run_index
            );
        }
        previous = next;
    }
    Ok(())
}
