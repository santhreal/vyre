//! Host and device copies for CUDA-resident buffers.

use vyre_driver::transfer_accounting::TransferAccountingPolicy;
use vyre_driver::BackendError;

use super::dispatch::CudaBackend;
use super::pinned_allocations::HostTransferAllocations;
use super::resident::{CudaResidentBuffer, ResidentViewCache};
use super::resident_upload_fusion::{
    fuse_resident_upload_copies, push_resident_upload_copy, ResidentUploadCopy,
};
use super::staging_reserve::reserve_smallvec;
use super::transient_memory_budget::cuda_live_free_memory_bytes;
use crate::numeric::CUDA_NUMERIC;
use smallvec::SmallVec;

const CUDA_RESIDENT_BUDGET_NUMERATOR: u64 = 9;
const CUDA_RESIDENT_BUDGET_DENOMINATOR: u64 = 10;
const CUDA_RESIDENT_TRANSFER_ACCOUNTING: TransferAccountingPolicy =
    TransferAccountingPolicy::new("CUDA resident", "split the transfer into bounded chunks");

pub(crate) enum ResidentStreamFailure {
    Completed(BackendError),
    CompletionUnproven(BackendError),
}

impl ResidentStreamFailure {
    fn into_error(self) -> BackendError {
        match self {
            Self::Completed(error) | Self::CompletionUnproven(error) => error,
        }
    }
}

fn cuda_resident_total_budget_bytes(total_memory: u64) -> u64 {
    let budget = (u128::from(total_memory) * u128::from(CUDA_RESIDENT_BUDGET_NUMERATOR))
        / u128::from(CUDA_RESIDENT_BUDGET_DENOMINATOR);
    budget as u64
}

fn cuda_resident_live_budget_bytes(
    total_memory: u64,
    live_free_memory: u64,
    resident_bytes: u64,
) -> u64 {
    let total_budget = cuda_resident_total_budget_bytes(total_memory);
    if resident_bytes >= total_budget {
        return resident_bytes;
    }
    let accounted_available = total_budget - resident_bytes;
    let live_available = cuda_resident_total_budget_bytes(live_free_memory);
    resident_bytes + accounted_available.min(live_available)
}

impl CudaBackend {
    fn with_resident_stream<T>(
        &self,
        operation: impl FnOnce(&crate::stream::CudaStream) -> Result<T, BackendError>,
    ) -> Result<T, BackendError> {
        self.with_resident_stream_classified(operation)
            .map_err(ResidentStreamFailure::into_error)
    }

    pub(crate) fn with_resident_stream_classified<T>(
        &self,
        operation: impl FnOnce(&crate::stream::CudaStream) -> Result<T, BackendError>,
    ) -> Result<T, ResidentStreamFailure> {
        let stream = self
            .launch_resources
            .acquire_stream()
            .map_err(ResidentStreamFailure::Completed)?;
        let result = operation(&stream);
        if result.is_err() {
            match stream.synchronize() {
                Ok(()) => self.telemetry.record_sync_point(),
                Err(error) => {
                    tracing::error!(
                        "Fix: failed to synchronize CUDA resident I/O stream after operation error: {error}. In-flight resident I/O resources will not be recycled."
                    );
                    std::mem::forget(stream);
                    return result.map_err(ResidentStreamFailure::CompletionUnproven);
                }
            }
        }
        self.launch_resources.release_stream(stream);
        result.map_err(ResidentStreamFailure::Completed)
    }
}

fn add_resident_transfer_bytes(
    total: &mut u64,
    bytes: usize,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_RESIDENT_TRANSFER_ACCOUNTING.add_bytes(total, bytes, label)
}

pub(crate) fn add_resident_copy_count(total: &mut usize, label: &str) -> Result<(), BackendError> {
    CUDA_RESIDENT_TRANSFER_ACCOUNTING.add_copy_count(total, label)
}

pub(crate) fn add_resident_copy_slots(
    total: &mut usize,
    slots: usize,
    label: &str,
) -> Result<(), BackendError> {
    CUDA_RESIDENT_TRANSFER_ACCOUNTING.add_copy_slots(total, slots, label)
}

fn resident_upload_staging<'a>(
    upload_count: usize,
    copy_label: &'static str,
    view_label: &'static str,
) -> Result<(SmallVec<[ResidentUploadCopy<'a>; 8]>, ResidentViewCache), BackendError> {
    let mut copies = SmallVec::<[ResidentUploadCopy<'a>; 8]>::new();
    reserve_smallvec(&mut copies, upload_count, copy_label)?;
    let mut resident_view_cache = ResidentViewCache::new();
    reserve_smallvec(&mut resident_view_cache, upload_count, view_label)?;
    Ok((copies, resident_view_cache))
}

pub(crate) use super::resident_io_download::*;

impl CudaBackend {
    /// Allocate a CUDA-resident buffer owned by this backend.
    pub fn allocate_resident(&self, byte_len: usize) -> Result<CudaResidentBuffer, BackendError> {
        if byte_len == 0 {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: CUDA resident buffers must have a non-zero byte length.".to_string(),
            });
        }
        self.warmup()?;
        let resident_budget = self.cuda_resident_budget_bytes()?;
        let handle = self.resident_store.allocate(byte_len, resident_budget)?;
        self.telemetry.record_resident_allocation_bytes(
            CUDA_NUMERIC.usize_to_u64(byte_len, "resident allocation byte count")?,
        );
        Ok(handle)
    }

    /// Upload bytes into an existing CUDA-resident buffer.
    pub fn upload_resident(
        &self,
        handle: CudaResidentBuffer,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        self.upload_resident_many(&[(handle, bytes)])
    }

    /// Upload several full CUDA-resident buffers with one stream synchronization.
    pub fn upload_resident_many(
        &self,
        uploads: &[(CudaResidentBuffer, &[u8])],
    ) -> Result<(), BackendError> {
        if uploads.is_empty() {
            return Ok(());
        }
        let mut uploaded_bytes = 0_u64;
        let (mut copies, mut resident_view_cache) =
            resident_upload_staging(uploads.len(), "upload copy", "resident upload view cache")?;
        for &(handle, bytes) in uploads {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident upload view cache",
            )?;
            if bytes.len() != buffer.byte_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident upload for handle {} expected {} bytes but received {}.",
                        handle.handle,
                        buffer.byte_len,
                        bytes.len()
                    ),
                });
            }
            push_resident_upload_copy(
                &mut copies,
                &mut uploaded_bytes,
                handle.handle.id(),
                buffer.ptr,
                bytes,
                "upload",
            )?;
        }
        let (copies, uploaded_bytes) = fuse_resident_upload_copies(copies)?;
        self.copy_resident_uploads(&copies, uploaded_bytes)
    }

    /// Free a CUDA-resident buffer handle.
    pub fn free_resident(&self, handle: CudaResidentBuffer) -> Result<(), BackendError> {
        self.resident_store.free(handle)
    }

    /// Upload a partial byte slice into a CUDA-resident buffer at a byte offset.
    pub fn upload_resident_at(
        &self,
        handle: CudaResidentBuffer,
        dst_offset_bytes: usize,
        bytes: &[u8],
    ) -> Result<(), BackendError> {
        self.upload_resident_at_many(&[(handle, dst_offset_bytes, bytes)])
    }

    /// Upload several partial byte slices into CUDA-resident buffers with one stream fence.
    pub fn upload_resident_at_many(
        &self,
        uploads: &[(CudaResidentBuffer, usize, &[u8])],
    ) -> Result<(), BackendError> {
        if uploads.is_empty() {
            return Ok(());
        }
        let mut uploaded_bytes = 0_u64;
        let (mut copies, mut resident_view_cache) = resident_upload_staging(
            uploads.len(),
            "offset upload copy",
            "resident offset-upload view cache",
        )?;
        for &(handle, dst_offset_bytes, bytes) in uploads {
            let buffer = self.resident_store.view_cached(
                handle,
                &mut resident_view_cache,
                "resident offset-upload view cache",
            )?;
            let dst_ptr = checked_resident_dst(
                handle,
                buffer.ptr,
                buffer.byte_len,
                dst_offset_bytes,
                bytes.len(),
            )?;
            push_resident_upload_copy(
                &mut copies,
                &mut uploaded_bytes,
                handle.handle.id(),
                dst_ptr,
                bytes,
                "offset upload",
            )?;
        }
        let (copies, uploaded_bytes) = fuse_resident_upload_copies(copies)?;
        self.copy_resident_uploads(&copies, uploaded_bytes)
    }

    fn copy_resident_uploads(
        &self,
        copies: &[ResidentUploadCopy<'_>],
        uploaded_bytes: u64,
    ) -> Result<(), BackendError> {
        if copies.is_empty() {
            return Ok(());
        }
        self.warmup()?;
        let mut host_transfers = HostTransferAllocations::with_capacity(
            std::sync::Arc::clone(&self.host_pool),
            copies.len(),
            0,
        )?;
        match self.with_resident_stream_classified(|stream| {
            for copy in copies {
                let bytes = copy.bytes.as_slice();
                let host_ptr = host_transfers.push_upload(bytes)?;
                // SAFETY: FFI to libcuda.so. Pointer args were validated by the
                // matching alloc / store API; lifetimes are documented in the
                // surrounding function. cuda_check (or matching CUresult guard)
                // propagates non-success codes as BackendError.
                unsafe {
                    super::copy::h2d_async_checked(
                        copy.dst_ptr,
                        host_ptr,
                        bytes.len(),
                        stream.raw(),
                    )?;
                }
            }
            stream.synchronize()
        }) {
            Ok(()) => {}
            Err(ResidentStreamFailure::Completed(error)) => return Err(error),
            Err(ResidentStreamFailure::CompletionUnproven(error)) => {
                std::mem::forget(host_transfers);
                return Err(error);
            }
        }
        self.telemetry.record_sync_point();
        self.telemetry.record_host_to_device_bytes(uploaded_bytes);
        self.telemetry.record_host_upload_operations(
            CUDA_NUMERIC.usize_to_u64(copies.len(), "resident upload operation count")?,
        );
        drop(host_transfers);
        Ok(())
    }

    /// Return the raw CUDA device pointer for a resident buffer.
    pub fn resident_device_ptr(&self, handle: CudaResidentBuffer) -> Result<u64, BackendError> {
        self.with_resident(handle, |buffer| Ok(buffer.ptr))
    }

    /// Bytes currently held by CUDA resident buffers.
    #[must_use]
    pub fn resident_allocated_bytes(&self) -> u64 {
        self.resident_store.allocated_bytes()
    }

    fn cuda_resident_budget_bytes(&self) -> Result<u64, BackendError> {
        Ok(cuda_resident_live_budget_bytes(
            self.caps.total_memory,
            cuda_live_free_memory_bytes()?,
            self.resident_store.allocated_bytes(),
        ))
    }

    /// Pin a pre-allocated host buffer as page-locked for fast async H2D.
    ///
    /// # Safety
    ///
    /// The caller asserts `ptr..ptr+byte_len` is a uniquely owned, mapped
    /// host region that lives at least until [`Self::unpin_host_buffer`] is called.
    pub unsafe fn pin_host_buffer(&self, ptr: u64, byte_len: usize) -> Result<(), BackendError> {
        if byte_len == 0 {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: pin_host_buffer requires a non-zero byte length.".to_string(),
            });
        }
        self.warmup()?;
        // SAFETY: The caller provided the host range lifetime and uniqueness
        // guarantees documented on this unsafe public API.
        unsafe { super::host_memory::register_host_buffer(ptr, byte_len, "cuMemHostRegister_v2") }
    }

    /// Unregister a previously [`Self::pin_host_buffer`]d host region.
    ///
    /// # Safety
    ///
    /// The caller asserts there are no in-flight async copies sourcing from
    /// this region.
    pub unsafe fn unpin_host_buffer(&self, ptr: u64) -> Result<(), BackendError> {
        self.warmup()?;
        // SAFETY: The caller guarantees no in-flight async copies still use
        // this host range, as documented on this unsafe public API.
        unsafe { super::host_memory::unregister_host_buffer(ptr, "cuMemHostUnregister") }
    }

    /// Async H2D copy from a pinned host pointer into a CUDA-resident buffer.
    ///
    /// # Safety
    ///
    /// The caller asserts `src_ptr..src_ptr+byte_count` is page-locked and
    /// remains uniquely borrowed until [`Self::synchronize_uploads`] returns.
    pub unsafe fn upload_resident_async_at(
        &self,
        handle: CudaResidentBuffer,
        dst_offset_bytes: usize,
        src_ptr: u64,
        byte_count: usize,
    ) -> Result<(), BackendError> {
        if byte_count == 0 {
            return Ok(());
        }
        self.with_resident(handle, |buffer| {
            let dst_ptr = checked_resident_dst(handle, buffer.ptr, buffer.byte_len, dst_offset_bytes, byte_count)?;
            let mut pending_stream = self.async_upload_stream.lock().map_err(|_| {
                BackendError::new("CUDA async upload stream lock was poisoned. Fix: recreate the backend before queueing more resident uploads.")
            })?;
            let created_stream = pending_stream.is_none();
            if created_stream {
                *pending_stream = Some(self.launch_resources.acquire_stream()?);
            }
            let stream = pending_stream.as_ref().ok_or_else(|| {
                BackendError::new("CUDA async upload stream allocation failed. Fix: recreate the backend or lower concurrent upload pressure.")
            })?;
            // SAFETY: FFI to libcuda.so. Pointer args were validated by the
            // matching alloc / store API; lifetimes are documented in the
            // surrounding function. cuda_check (or matching CUresult guard)
            // propagates non-success codes as BackendError.
            unsafe {
                let copy_result = super::copy::h2d_async_checked(
                    dst_ptr,
                    src_ptr as *const std::ffi::c_void,
                    byte_count,
                    stream.raw(),
                );
                if let Err(error) = copy_result {
                    if created_stream {
                        if let Some(stream) = pending_stream.take() {
                            self.launch_resources.release_stream(stream);
                        }
                    }
                    return Err(error);
                }
            }
            self.telemetry
                .record_host_to_device_bytes(CUDA_NUMERIC.usize_to_u64(
                    byte_count,
                    "resident byte upload count",
                )?);
            self.telemetry.record_host_upload_operations(1);
            Ok(())
        })
    }

    /// Block until every queued async H2D copy on this backend's upload stream completes.
    pub fn synchronize_uploads(&self) -> Result<(), BackendError> {
        self.warmup()?;
        let stream = self
            .async_upload_stream
            .lock()
            .map_err(|_| {
                BackendError::new("CUDA async upload stream lock was poisoned. Fix: recreate the backend before synchronizing resident uploads.")
            })?
            .take();
        let Some(stream) = stream else {
            return Ok(());
        };
        if let Err(error) = stream.synchronize() {
            tracing::error!(
                "Fix: failed to synchronize CUDA async resident upload stream: {error}. In-flight async resident upload stream will not be recycled."
            );
            std::mem::forget(stream);
            return Err(error);
        }
        self.telemetry.record_sync_point();
        self.launch_resources.release_stream(stream);
        Ok(())
    }
}

// Inline: covers `cuda_resident_live_budget_bytes`, `cuda_resident_total_budget_bytes`, which no
// integration test can name.
#[cfg(test)]
mod resident_budget_tests {
    use super::{cuda_resident_live_budget_bytes, cuda_resident_total_budget_bytes};

    #[test]
    fn resident_budget_caps_new_allocations_against_live_free_vram() {
        assert_eq!(cuda_resident_total_budget_bytes(10_000), 9_000);
        assert_eq!(
            cuda_resident_live_budget_bytes(10_000, 1_000, 0),
            900,
            "Fix: resident allocation budget must respect live free VRAM, not only total board memory."
        );
        assert_eq!(
            cuda_resident_live_budget_bytes(10_000, 8_000, 2_000),
            9_000,
            "Fix: resident allocation budget must preserve already-owned resident bytes while capping only additional allocation headroom."
        );
        assert_eq!(
            cuda_resident_live_budget_bytes(10_000, 0, 2_000),
            2_000,
            "Fix: zero live free VRAM must allow no additional resident allocation beyond already-owned handles."
        );
    }
}

fn checked_resident_dst(
    handle: CudaResidentBuffer,
    base_ptr: u64,
    buffer_len: usize,
    dst_offset_bytes: usize,
    byte_count: usize,
) -> Result<u64, BackendError> {
    let _end = vyre_driver::accounting::checked_usize_byte_range_end_lazy(
        dst_offset_bytes,
        byte_count,
        buffer_len,
        || {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident upload at offset {dst_offset_bytes} for handle {} would overflow usize.",
                handle.handle
            ),
        }
        },
        |end| {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident upload for handle {} writes [{dst_offset_bytes}..{end}) but buffer is only {buffer_len} bytes; resize the resident slot or trim the source slice.",
                handle.handle
            ),
        }
        },
    )?;
    vyre_driver::accounting::checked_add_u64_usize_offset_lazy(
        base_ptr,
        dst_offset_bytes,
        || {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident upload offset {dst_offset_bytes} does not fit CUdeviceptr arithmetic for handle {}.",
                handle.handle
            ),
        }
        },
        || {
            BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident upload pointer arithmetic overflowed for handle {} at offset {dst_offset_bytes}.",
                handle.handle
            ),
        }
        },
    )
}
