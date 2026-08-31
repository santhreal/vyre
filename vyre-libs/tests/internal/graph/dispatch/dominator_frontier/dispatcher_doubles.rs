use super::*;
use crate::test_parity_oracles::{canonical_inputs, semantic_output};
use vyre_megakernel::{
    SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionRequest, SemanticExecutor,
};

pub(super) struct DominatorInputShapeDispatcher;

impl SemanticExecutor for DominatorInputShapeDispatcher {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = canonical_inputs(request)?;
        assert_eq!(inputs.len(), 5);
        assert_eq!(
            inputs[1].len(),
            4,
            "Fix: empty dominance targets must be padded to one u32 from the primitive plan"
        );
        assert_eq!(
            inputs[3].len(),
            4,
            "Fix: empty predecessor targets must be padded to one u32 from the primitive plan"
        );
        semantic_output(request, vec![u32_slice_to_le_bytes(&[0])])
    }
}

pub(super) struct RecordingDominatorDispatcher {
    pub(super) calls: Mutex<Vec<Vec<Vec<u8>>>>,
    pub(super) output: Vec<u8>,
}

impl SemanticExecutor for RecordingDominatorDispatcher {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = canonical_inputs(request)?;
        self.calls
            .lock()
            .expect("Fix: recording semantic executor calls lock should not be poisoned")
            .push(inputs);
        semantic_output(request, vec![self.output.clone()])
    }
}
