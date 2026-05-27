//! Test: public facade.
use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};

#[test]
fn public_facade_builds_and_serializes_program() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );

    let wire = program
        .to_wire()
        .expect("public facade must expose serializable Program construction");
    let decoded = Program::from_wire(&wire).expect("public facade must expose wire decode");

    assert_eq!(decoded.buffers().len(), 1);
    assert_eq!(decoded.entry().len(), 1);
}
