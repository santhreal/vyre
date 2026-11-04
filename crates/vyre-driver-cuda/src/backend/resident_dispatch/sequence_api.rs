use vyre_driver::BackendError;

use crate::backend::dispatch::CudaBackend;
use crate::backend::output_range::CudaOutputReadback;
use crate::backend::resident::CudaResidentBuffer;
use crate::backend::resident_dispatch_accounting::CudaResidentDispatchStep;

impl CudaBackend {
    pub(crate) fn upload_resident_many_sequence_read_ranges_borrowed_into(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        steps: &[CudaResidentDispatchStep<'_>],
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
            &[],
            uploads,
            steps,
            &[],
            0,
            read_handles,
            readbacks,
            outputs,
        )
    }

    pub(crate) fn upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
        prefix_steps: &[CudaResidentDispatchStep<'_>],
        repeated_steps: &[CudaResidentDispatchStep<'_>],
        repeat_count: usize,
        read_handles: &[CudaResidentBuffer],
        readbacks: &[CudaOutputReadback],
        outputs: &mut [&mut Vec<u8>],
    ) -> Result<(), BackendError> {
        self.fill_upload_resident_many_repeated_sequence_read_ranges_borrowed_into(
            &[],
            uploads,
            prefix_steps,
            repeated_steps,
            repeat_count,
            read_handles,
            readbacks,
            outputs,
        )
    }
}
