/// Registration contract: the reference oracle is not registered as a VyreBackend.

use vyre_driver::{acquire, registered_backends};
use vyre_driver_reference::CpuRefEvaluator;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use crate::dispatch_fixtures::u32_out_buffer;

#[test]
fn cpu_ref_does_not_register_as_vyre_backend() {
    if let Ok(registrations) = registered_backends() {
        for reg in registrations {
            assert_ne!(
                reg.id, "cpu-ref",
                "Fix: reference interpreter (cpu-ref) must not appear in the VyreBackend registry"
            );
            assert_ne!(
                reg.id, "reference",
                "Fix: reference interpreter must not appear in the VyreBackend registry"
            );
            assert!(
                !reg.id.contains("ref") && !reg.id.contains("cpu"),
                "Fix: reference interpreter must not appear in the VyreBackend registry, found `{}`",
                reg.id
            );
            assert!(
                !reg.reference_oracle,
                "Fix: reference oracle flag must not be set in the production VyreBackend registry"
            );
        }
    }
    assert!(
        acquire("cpu-ref").is_err(),
        "Fix: acquire('cpu-ref') must fail because cpu-ref is an oracle session, not a VyreBackend"
    );
    assert!(
        acquire("reference").is_err(),
        "Fix: acquire('reference') must fail because reference is an oracle session, not a VyreBackend"
    );
}

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
