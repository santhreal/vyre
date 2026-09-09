//! Adversarial closure contract for the cpu-ref backend.
//!
//! The obligations themselves live in `vyre_driver::hostile_input_closure`,
//! because they are the same obligations for every backend: a hostile byte
//! slice either dispatches or fails with an actionable message, and extra
//! trailing input buffers are rejected rather than ignored. This target says
//! which backend owes them.

#![forbid(unsafe_code)]
use vyre_driver::DispatchConfig;
use vyre_driver_reference::CpuRefEvaluator;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn hostile_input_closure_rejects_missing_and_extra_inputs() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("in", 0, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(0)),
        )],
    );

    // Missing input
    let err = evaluator
        .evaluate(&program, &[], &DispatchConfig::default())
        .expect_err("Fix: missing inputs must fail without synthesizing zero buffers");
    assert!(
        err.to_string().contains("missing an input buffer"),
        "Fix: error must report missing input buffer: {err}"
    );

    // Extra input
    let input = 10u32.to_le_bytes();
    let extra = 20u32.to_le_bytes();
    let err = evaluator
        .evaluate(
            &program,
            &[input.as_slice(), extra.as_slice()],
            &DispatchConfig::default(),
        )
        .expect_err("Fix: extra trailing inputs must be rejected");
    assert!(
        err.to_string().contains("extra input buffer"),
        "Fix: error must report extra input buffer: {err}"
    );
}
