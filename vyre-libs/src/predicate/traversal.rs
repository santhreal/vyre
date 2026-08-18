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

pub(crate) fn tag_family_fixture_inputs(fixture_tags: &[u32]) -> Vec<Vec<Vec<u8>>> {
    let to_bytes = crate::predicate::inventory_u32_le_bytes;
    vec![vec![to_bytes(fixture_tags), to_bytes(&[0])]]
}

pub(crate) fn forward_edge_fixture_inputs(
    offsets: &[u32],
    targets: &[u32],
    masks: &[u32],
) -> Vec<Vec<Vec<u8>>> {
    let b = crate::predicate::inventory_u32_le_bytes;
    vec![vec![
        b(&[2, 1, 1, 1]),
        b(offsets),
        b(targets),
        b(masks),
        b(&[0, 0, 0, 0]),
        b(&[0b0001]),
        b(&[0]),
    ]]
}

pub(crate) fn single_output_fixture_expected(expected_bytes: &[u8]) -> Vec<Vec<Vec<u8>>> {
    vec![vec![expected_bytes.to_vec()]]
}

#[cfg(test)]
pub(crate) fn assert_region_op_id(program: &Program, expected: &'static str, label: &str) {
    let generator = match &program.entry[0] {
        vyre_foundation::ir::Node::Region { generator, .. } => generator.to_string(),
        other => panic!("Fix: {label} must build a Region entry, got {other:?}."),
    };
    assert_eq!(generator, expected);
}
