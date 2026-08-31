//! Catalog entry for the motif operation.

use crate::graph::program_graph::ProgramGraphShape;

use super::pattern::MotifEdge;
use super::program::motif;
const OP_ID: &str = "vyre-libs::graph::motif";

const EXPECTED_MOTIF_HITS_BYTES: [u8; 16] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const EXPECTED_MOTIF_WITNESS_BYTES: [u8; 16] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || motif(ProgramGraphShape::new(4, 4), &[MotifEdge { from: 0, to: 1, kind_mask: 1 }], "witness"),
        Some(|| {
            // Both results are pipeline-live-out, so the backend allocates them
            // and the reference takes no seed Value for either.
            vec![crate::graph::program_graph::sample_program_graph_inputs(
                &[],
            )]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_MOTIF_HITS_BYTES.to_vec(),
                EXPECTED_MOTIF_WITNESS_BYTES.to_vec(),
            ]]
        }),
    )
}
