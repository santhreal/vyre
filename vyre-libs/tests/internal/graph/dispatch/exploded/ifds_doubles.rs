use super::*;
use crate::graph::exploded::{build_cpu_reference, canonicalize_csr_within_rows_in_place};
use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_megakernel::{
    SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionRequest, SemanticExecutor,
};

pub(super) fn canonical_expected(
    num_procs: u32,
    blocks_per_proc: u32,
    facts_per_proc: u32,
    intra: &[(u32, u32, u32)],
    inter: &[(u32, u32, u32, u32)],
    gen_edges: &[(u32, u32, u32)],
    kill: &[(u32, u32, u32)],
) -> (Vec<u32>, Vec<u32>) {
    let (row_ptr, mut col_idx) = build_cpu_reference(
        num_procs,
        blocks_per_proc,
        facts_per_proc,
        intra,
        inter,
        gen_edges,
        kill,
    );
    canonicalize_csr_within_rows_in_place(&row_ptr, &mut col_idx)
        .expect("canonical_expected row sorting failed");
    (row_ptr, col_idx)
}

pub(super) struct RecordingIfdsOracle {
    pub(super) inner: ReferenceSemanticExecutor,
    pub(super) intra_src_blocks: Mutex<Vec<Vec<u32>>>,
}

impl SemanticExecutor for RecordingIfdsOracle {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
        if inputs.len() >= 2 {
            self.intra_src_blocks
                .lock()
                .expect("Fix: IFDS recording mutex should not be poisoned")
                .push(crate::dispatch_buffers::read_u32s(&inputs[1]));
        }
        self.inner.execute(request)
    }
}
