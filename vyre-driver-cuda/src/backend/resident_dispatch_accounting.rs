//! Shared resident-dispatch contracts and checked accounting helpers.

use smallvec::SmallVec;
use vyre_driver::transfer_accounting::TransferAccountingPolicy;
use vyre_driver::{BackendError, DispatchConfig};
use vyre_foundation::ir::Program;

use super::output_range::CudaOutputReadback;
use super::resident::CudaResidentBuffer;

const CUDA_RESIDENT_DISPATCH_ACCOUNTING: TransferAccountingPolicy =
    TransferAccountingPolicy::new("CUDA resident", "split the resident dispatch");

pub(crate) struct CudaResidentDispatchStep<'a> {
    pub(crate) program: &'a Program,
    pub(crate) handles: &'a [CudaResidentBuffer],
    pub(crate) config: DispatchConfig,
}

pub(crate) struct CudaResidentDispatch {
    pub(crate) pending: crate::stream::CudaPendingDispatch,
    /// One entry per program output in output order. `None` marks an output
    /// bound to a borrowed buffer, whose bytes are staged for this dispatch
    /// only and are therefore reachable through `pending`, never through a
    /// device handle that outlives the call.
    pub(crate) output_handles: SmallVec<[Option<CudaResidentBuffer>; 8]>,
    pub(crate) output_readbacks: SmallVec<[CudaOutputReadback; 8]>,
}

impl CudaResidentDispatch {
    /// Resident handle for every output, refusing a dispatch whose output was
    /// staged from a borrowed buffer.
    ///
    /// Device-side readback and output chaining need memory that outlives the
    /// call; a per-dispatch staging allocation is recycled the moment the
    /// dispatch ends, so handing one back would alias whatever the pool
    /// serves next.
    pub(crate) fn resident_output_handles(
        &self,
        context: &'static str,
    ) -> Result<SmallVec<[CudaResidentBuffer; 8]>, BackendError> {
        let mut handles = SmallVec::with_capacity(self.output_handles.len());
        for (output_index, handle) in self.output_handles.iter().enumerate() {
            let handle = handle.ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} needs a resident output buffer for output {output_index}, but that binding was given a borrowed resource that only lives for one dispatch. Bind a resident buffer for outputs the caller reads back from the device."
                ),
            })?;
            handles.push(handle);
        }
        Ok(handles)
    }
}

pub(crate) struct CudaResidentBatchDispatch {
    pub(crate) pending: crate::stream::CudaPendingDispatch,
    pub(crate) output_handles: SmallVec<[SmallVec<[CudaResidentBuffer; 8]>; 8]>,
    pub(crate) output_readbacks: SmallVec<[SmallVec<[CudaOutputReadback; 8]>; 8]>,
}

pub(crate) fn checked_resident_dispatch_capacity_mul(
    lhs: usize,
    rhs: usize,
    label: &str,
) -> Result<usize, BackendError> {
    CUDA_RESIDENT_DISPATCH_ACCOUNTING.mul_usize_capacity(lhs, rhs, label)
}

pub(crate) fn checked_resident_dispatch_capacity_add(
    lhs: usize,
    rhs: usize,
    label: &str,
) -> Result<usize, BackendError> {
    CUDA_RESIDENT_DISPATCH_ACCOUNTING.add_usize_capacity(lhs, rhs, label)
}

pub(crate) fn add_resident_dispatch_bytes(
    total: &mut u64,
    bytes: usize,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_RESIDENT_DISPATCH_ACCOUNTING.add_bytes(total, bytes, label)
}

pub(crate) fn add_resident_dispatch_usize_count(
    total: &mut usize,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_RESIDENT_DISPATCH_ACCOUNTING.add_usize_counter(total, 1, label, "count")
}

pub(crate) fn add_resident_dispatch_u64_count(
    total: &mut u64,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_RESIDENT_DISPATCH_ACCOUNTING.add_u64_counter(total, 1, label, "operation count")
}
