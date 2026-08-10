//! Production conformance must exercise compiler, payload, materializer, ABI, and submission.

#![cfg(feature = "gpu")]

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::ProductionSession;
use vyre_driver::backend::registered_backends;
use vyre_driver_wgpu as _;

#[test]
fn wgpu_production_route_executes_canonical_artifact() {
    let registration = registered_backends()
        .iter()
        .copied()
        .find(|registration| registration.id == "wgpu")
        .expect("Fix: the gpu feature must link the wgpu registration");
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );

    let session = ProductionSession::compile(&program, registration)
        .expect("Fix: production conformance compilation and materialization must succeed");
    let outputs = session
        .submit(&[])
        .expect("Fix: typed artifact submission must succeed");

    assert_eq!(outputs, vec![7_u32.to_le_bytes().to_vec()]);
    assert_ne!(session.artifact_digest().0, [0; 32]);
    assert_ne!(session.payload_digest().0, [0; 32]);
}
