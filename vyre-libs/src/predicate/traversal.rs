use vyre_foundation::ir::Program;

use crate::graph::csr_frontier_step::{csr_frontier_step_program, CsrFrontierStepKind};
use crate::graph::program_graph::ProgramGraphShape;

pub(crate) fn forward_edge_program(
    op_id: &'static str,
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_mask: u32,
) -> Program {
    csr_frontier_step_program(
        op_id,
        CsrFrontierStepKind::Forward,
        shape,
        frontier_in,
        frontier_out,
        edge_mask,
    )
}

pub(crate) fn backward_edge_program(
    op_id: &'static str,
    shape: ProgramGraphShape,
    frontier_in: &str,
    frontier_out: &str,
    edge_mask: u32,
) -> Program {
    csr_frontier_step_program(
        op_id,
        CsrFrontierStepKind::Backward,
        shape,
        frontier_in,
        frontier_out,
        edge_mask,
    )
}

#[cfg(test)]
pub(crate) fn assert_region_op_id(program: &Program, expected: &'static str, label: &str) {
    let generator = match &program.entry[0] {
        vyre_foundation::ir::Node::Region { generator, .. } => generator.to_string(),
        other => panic!("Fix: {label} must build a Region entry, got {other:?}."),
    };
    assert_eq!(generator, expected);
}
