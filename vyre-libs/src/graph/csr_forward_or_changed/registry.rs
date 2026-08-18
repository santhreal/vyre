const OP_ID: &str = "vyre-libs::graph::csr_forward_or_changed";
use super::program_serial::csr_forward_or_changed;
use crate::graph::program_graph::ProgramGraphShape;

const EXPECTED_CSR_FOC_FRONTIER_BYTES: [u8; 4] = [15, 0, 0, 0];
const EXPECTED_CSR_FOC_CHANGED_BYTES: [u8; 4] = [1, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || csr_forward_or_changed(ProgramGraphShape::new(4, 4), "frontier", "changed", 1),
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![crate::graph::program_graph::sample_program_graph_inputs(&[
                to_bytes(&[0b0001]),
                to_bytes(&[0]),
            ])]
        }),
        Some(|| {
            vec![vec![
                EXPECTED_CSR_FOC_FRONTIER_BYTES.to_vec(),
                EXPECTED_CSR_FOC_CHANGED_BYTES.to_vec(),
            ]]
        }),
    )
}
