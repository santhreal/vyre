//! Catalog entry for the motif operation.

use crate::graph::program_graph::ProgramGraphShape;

use super::pattern::MotifEdge;
use super::program::motif;
use super::OP_ID;

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || motif(ProgramGraphShape::new(4, 4), &[MotifEdge { from: 0, to: 1, kind_mask: 1 }], "witness"),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0, 0, 0]),          // pg_nodes
                to_bytes(&[0, 2, 3, 4, 4]),       // pg_edge_offsets
                to_bytes(&[1, 2, 3, 3]),          // pg_edge_targets
                to_bytes(&[1, 1, 1, 1]),          // pg_edge_kind_mask
                to_bytes(&[0, 0, 0, 0]),          // pg_node_tags
                to_bytes(&[0, 0, 0, 0]),          // motif_hits
                to_bytes(&[0, 0, 0, 0]),          // witness
            ]]
        }),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[1, 1, 0, 0]),          // expected motif_hits
                to_bytes(&[1, 1, 0, 0]),          // expected witness
            ]]
        }),
    )
}
