//! Const-fold test suite, one file per rule section so no single file
//! exceeds the 1000-LOC hygiene cap.

mod binop_identity;
mod early;
mod structural;
mod unary;

#[test]
fn analyze_skips_program_with_no_expression_bearing_nodes() {
    use crate::ir::{Node, Program};
    use crate::optimizer::passes::algebraic::const_fold::ConstFold;
    use crate::optimizer::PassAnalysis;

    let program = Program::wrapped(Vec::new(), [1, 1, 1], vec![Node::Return]);
    match crate::optimizer::ProgramPass::analyze(&ConstFold, &program) {
        PassAnalysis::SKIP => {}
        other => panic!("expected SKIP for expression-free program, got {other:?}"),
    }
}
