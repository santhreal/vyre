//! The reference interpreter is reached through a named evaluator, not a
//! registration.
//!
//! WHY: this file used to loop over the backend registry and reject ids whose
//! text contained `ref` or `cpu`. That loop ran in a binary linking no driver
//! crate, so it enumerated an empty registry, and its substring test admitted
//! any host path named something else. The registry closure now lives in
//! `production_registry_execution_domain.rs`, which links every declared driver
//! and decides each entry through an exhaustive match. What is left here is the
//! evaluator seam itself: it computes reference values from a `Program` and it
//! has no dispatch identity.
use vyre_driver_reference::CpuRefEvaluator;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use crate::dispatch_fixtures::u32_out_buffer;

#[test]
fn cpu_ref_evaluator_evaluates_pure_reference_program() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![u32_out_buffer("out", 0)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    );
    let outputs = evaluator
        .evaluate(&program, &[], &vyre_driver::DispatchConfig::default())
        .expect("Fix: CpuRefEvaluator must evaluate a minimal Program.");

    assert_eq!(
        outputs,
        vec![42u32.to_le_bytes().to_vec()],
        "Fix: CpuRefEvaluator output must match reference interpreter bytes."
    );
}
#[test]
fn cpu_ref_evaluator_executes_a_fenced_program() {
    let evaluator = CpuRefEvaluator;
    let program = Program::wrapped(
        vec![
            BufferDecl::read("seed", 0, DataType::U32),
            u32_out_buffer("out", 1),
        ],
        [1, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::load("seed", Expr::u32(0))),
            Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::GridSync),
            Node::store(
                "out",
                Expr::u32(0),
                Expr::add(Expr::load("out", Expr::u32(0)), Expr::u32(1)),
            ),
        ],
    );
    let seed = 41u32.to_le_bytes().to_vec();
    let outputs = evaluator
        .evaluate(
            &program,
            &[seed.as_slice()],
            &vyre_driver::DispatchConfig::default(),
        )
        .expect("Fix: CpuRefEvaluator must evaluate a whole-grid fence.");
    assert_eq!(
        outputs,
        vec![42u32.to_le_bytes().to_vec()],
        "Fix: the segment after the fence must read what the segment before it wrote."
    );
}
