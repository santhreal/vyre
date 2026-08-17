//! Reverse CSR frontier expansion over an in-place accumulator bitset.

use vyre_foundation::ir::Program;

use crate::graph::program_graph::ProgramGraphShape;
use vyre_primitives::lane_grid;

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::graph::csr_backward_or_changed";

/// Workgroup size for the reverse CSR frontier kernel (one thread per source node).
pub const CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE: [u32; 3] = [lane_grid::PORTABLE_WARP_WIDTH, 1, 1];

/// Compute the dispatch grid shape for reverse CSR frontier expansion.
#[must_use]
pub const fn csr_backward_or_changed_parallel_grid(node_count: u32) -> [u32; 3] {
    let block = CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE[0];
    let groups = if node_count == 0 {
        1
    } else {
        (node_count + block - 1) / block
    };
    [groups, 1, 1]
}

/// Build a Program: reverse CSR frontier expansion over an in-place accumulator bitset.
///
/// Dispatches `shape.node_count` parallel invocations (one per source node).
/// Each thread reads the outgoing edges for its source node and tests whether
/// any destination reached along an edge whose kind passes `edge_kind_mask` is
/// present in the frontier accumulator. If so, it sets the source bit in
/// `frontier_out` and sets the `changed` flag.
#[must_use]
pub fn csr_backward_or_changed_parallel(
    shape: ProgramGraphShape,
    frontier_out: &str,
    changed: &str,
    edge_kind_mask: u32,
) -> Program {
    crate::builder::csr::CsrTraversalComposer::backward(
        OP_ID,
        shape.node_count,
        shape.edge_count,
        edge_kind_mask,
    )
    .with_workgroup_size(CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE)
    .build_parallel_backward_or_changed(frontier_out, changed)
}


inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            let shape = ProgramGraphShape { node_count: 4, edge_count: 4 };
            csr_backward_or_changed_parallel(shape, "frontier", "changed", 0xFFFF_FFFF)
        },
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![
                to_bytes(&[0, 1, 2, 3, 4]),
                to_bytes(&[1, 2, 3, 0]),
                to_bytes(&[1, 1, 1, 1]),
                to_bytes(&[0b0000_1000]),
                to_bytes(&[0]),
            ]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![
                to_bytes(&[0, 1, 2, 3, 4]),
                to_bytes(&[1, 2, 3, 0]),
                to_bytes(&[1, 1, 1, 1]),
                to_bytes(&[0b0000_1100]),
                to_bytes(&[1]),
            ]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_frontier_and_changed_bindings() {
        let program = csr_backward_or_changed_parallel(
            ProgramGraphShape::new(4, 3),
            "frontier",
            "changed",
            u32::MAX,
        );
        let names = program
            .buffers()
            .iter()
            .map(|buffer| buffer.name())
            .collect::<Vec<_>>();

        assert!(names.contains(&"frontier"));
        assert!(names.contains(&"changed"));
        assert_eq!(
            program.workgroup_size(),
            CSR_BACKWARD_OR_CHANGED_WORKGROUP_SIZE
        );
    }

    #[test]
    fn parallel_grid_packs_source_lanes_into_blocks() {
        assert_eq!(csr_backward_or_changed_parallel_grid(0), [1, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(1), [1, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(256), [1, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(257), [2, 1, 1]);
        assert_eq!(csr_backward_or_changed_parallel_grid(513), [3, 1, 1]);
    }
}
