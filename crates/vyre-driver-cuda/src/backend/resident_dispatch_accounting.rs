//! Shared resident-dispatch contracts and checked accounting helpers.

use smallvec::SmallVec;
use vyre_runtime::transfer_accounting::TransferAccountingPolicy;
use vyre_runtime::{BackendError, DispatchConfig};
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

pub(crate) struct CudaResidentBatchDispatch {
    pub(crate) pending: crate::stream::CudaPendingDispatch,
    pub(crate) output_handles: SmallVec<[SmallVec<[CudaResidentBuffer; 8]>; 8]>,
    pub(crate) output_readbacks: SmallVec<[SmallVec<[CudaOutputReadback; 8]>; 8]>,
    /// One timing-event pair per submitted item, in submission order.
    ///
    /// Empty when the caller asked for no per-item timing. Every item's launch
    /// is enqueued on ONE stream before any of them is awaited, so an item's
    /// start event retires when the previous item's kernel ends and its own
    /// kernel is already submitted: the pair spans the kernel and not the host
    /// latency of submitting it. Reading the elapsed time requires the whole
    /// submission to have completed, which only the pending handle can prove.
    pub(crate) item_timing: SmallVec<[(crate::stream::CudaEvent, crate::stream::CudaEvent); 8]>,
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

pub(crate) fn add_resident_dispatch_u64_count(
    total: &mut u64,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_RESIDENT_DISPATCH_ACCOUNTING.add_u64_counter(total, 1, label, "operation count")
}
