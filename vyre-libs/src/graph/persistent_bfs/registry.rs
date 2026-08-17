const OP_ID: &str = "vyre-libs::graph::persistent_bfs";
use super::program::persistent_bfs;
use crate::graph::program_graph::ProgramGraphShape;

const EXPECTED_PERSISTENT_BFS_FRONTIER_BYTES: [u8; 4] = [15, 0, 0, 0];
const EXPECTED_PERSISTENT_BFS_CHANGED_BYTES: [u8; 4] = [1, 0, 0, 0];
const EXPECTED_PERSISTENT_BFS_CONVERGED_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || persistent_bfs(ProgramGraphShape::new(4, 4), "fin", "fout", 0xFFFF_FFFF, 4),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0, 0, 0]),          // pg_nodes
                to_bytes(&[0, 2, 3, 4, 4]),       // pg_edge_offsets
                to_bytes(&[1, 2, 3, 3]),          // pg_edge_targets
                to_bytes(&[1, 1, 1, 1]),          // pg_edge_kind_mask
                to_bytes(&[0, 0, 0, 0]),          // pg_node_tags
                to_bytes(&[0b0001]),              // frontier_in = {0}
                to_bytes(&[0]),                   // frontier_out
                to_bytes(&[0]),                   // changed
                to_bytes(&[0]),                   // converged
            ]]
        }),
        Some(|| {
            // After 4 iterations the graph 0→1,0→2,1→3,2→3 is fully closed. The
            // fixpoint is reached at step 2 (no new nodes), one step inside the
            // max_iters=4 budget, so the converged readback is 1.
            vec![vec![
                EXPECTED_PERSISTENT_BFS_FRONTIER_BYTES.to_vec(),
                EXPECTED_PERSISTENT_BFS_CHANGED_BYTES.to_vec(),
                EXPECTED_PERSISTENT_BFS_CONVERGED_BYTES.to_vec(),
            ]]
        }),
    )
}
