//! Staging host bytes and enqueuing the host-to-device copies of a resident
//! dispatch onto its stream.

use std::ffi::c_void;

use vyre_driver::BackendError;

use crate::backend::pinned_allocations::HostTransferAllocations;
use crate::backend::resident_upload_fusion::ResidentUploadCopy;
use crate::backend::staging_reserve::reserve_vec;

pub(crate) fn stage_resident_fill_payload(
    payload: &mut Vec<u8>,
    value: u8,
    byte_len: usize,
) -> Result<&[u8], BackendError> {
    reserve_vec(payload, byte_len, "resident fallback fill byte")?;
    payload.clear();
    payload.resize(byte_len, value);
    Ok(payload.as_slice())
}

pub(crate) fn enqueue_resident_h2d_copy(
    dst_ptr: u64,
    host_ptr: *const c_void,
    byte_len: usize,
    stream_raw: cudarc::driver::sys::CUstream,
) -> Result<(), BackendError> {
    // SAFETY: The caller owns the stream ordering and guarantees that the
    // pinned host allocation and resident destination remain live until the
    // stream reaches this copy. The shared copy helper validates null pointers
    // for non-empty copies and treats zero-byte copies as no-ops.
    unsafe { crate::backend::copy::h2d_async_checked(dst_ptr, host_ptr, byte_len, stream_raw) }
}

pub(crate) fn enqueue_optional_resident_h2d_copy(
    upload: Option<(u64, *const c_void, usize)>,
    stream_raw: cudarc::driver::sys::CUstream,
) -> Result<(), BackendError> {
    if let Some((dst_ptr, host_ptr, byte_len)) = upload {
        enqueue_resident_h2d_copy(dst_ptr, host_ptr, byte_len, stream_raw)?;
    }
    Ok(())
}

pub(crate) fn enqueue_resident_upload_copies_on_stream(
    copies: &[ResidentUploadCopy<'_>],
    host_transfers: &mut HostTransferAllocations,
    stream_raw: cudarc::driver::sys::CUstream,
) -> Result<(), BackendError> {
    for copy in copies {
        let bytes = copy.bytes.as_slice();
        let host_ptr = host_transfers.push_upload(bytes)?;
        enqueue_resident_h2d_copy(copy.dst_ptr, host_ptr, bytes.len(), stream_raw)?;
    }
    Ok(())
}
