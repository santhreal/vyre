//! Public persistent GPU buffer handle.

mod align;
mod pending;
mod resident;
#[cfg(test)]
mod tests;

use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use rustc_hash::FxHasher;
use vyre_driver::{BackendError, ResidentHandle};

pub(crate) use self::align::{aligned_len, aligned_len_u64, write_padded};
pub(crate) use self::pending::PendingGpuBufferReadback;
pub(crate) use self::resident::check_resident_owner;
use super::pool::PoolReturn;
use super::staging::StagingBufferPool;

fn pointer_identity_key<T>(ptr: *const T) -> u64 {
    let mut hasher = FxHasher::default();
    ptr.addr().hash(&mut hasher);
    hasher.finish()
}

/// Cheaply cloneable handle for a GPU-resident buffer.
///
/// The handle records the byte length originally requested by the caller,
/// the backing allocation length, the logical element count, and the actual
/// usage flags used to create the underlying `wgpu::Buffer`.
#[derive(Clone)]
pub struct GpuBufferHandle {
    inner: Arc<GpuBufferInner>,
}

struct GpuBufferInner {
    id: u64,
    buffer: Arc<wgpu::Buffer>,
    byte_len: u64,
    allocation_len: u64,
    element_count: usize,
    usage: wgpu::BufferUsages,
    pool_return: Option<PoolReturn>,
}

impl GpuBufferHandle {
    /// Upload `bytes` into a new GPU buffer.
    ///
    /// The created buffer always includes `COPY_DST` so the upload is legal.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the requested allocation length cannot fit
    /// `u64`.
    pub fn upload(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Result<Self, BackendError> {
        let allocation_len = aligned_len(bytes.len())?;
        let final_usage = usage | wgpu::BufferUsages::COPY_DST;
        // Fast path: create the buffer already mapped and memcpy host bytes
        // DIRECTLY into its host-visible / BAR backing store, then unmap. This
        // is ONE host copy with no wgpu-internal staging buffer and no GPU copy
        // command, the slow `queue.write_buffer` path routes every large upload
        // through wgpu's `StagingBelt`, which on Vulkan allocates + maps a fresh
        // staging buffer per write (the ~90 MB/s catalog-upload bottleneck on
        // the ~1 GB megakernel DFA catalog). `mapped_at_creation` works for ANY
        // usage flags (it does not require MAP_WRITE) and is correct for ALL
        // sizes, so it replaces the staged path unconditionally for non-empty
        // uploads. Zero-length buffers cannot be mapped at creation, so they
        // take the (no-op) `write_padded` path below: `aligned_len(0) == 0`,
        // and wgpu rejects a 0-byte `mapped_at_creation` buffer.
        if allocation_len > 0 {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vyre persistent upload"),
                size: allocation_len,
                usage: final_usage,
                mapped_at_creation: true,
            });
            {
                let mut mapped = buffer.slice(..).get_mapped_range_mut();
                crate::padded_upload::write_padded_into_mapped(&mut mapped, bytes)?;
            }
            buffer.unmap();
            let logical_len = u64::try_from(bytes.len()).map_err(|source| {
                BackendError::new(format!(
                    "GPU upload logical byte length cannot fit u64: {source}. Fix: split the dispatch input."
                ))
            })?;
            return Ok(Self::from_parts(
                Arc::new(buffer),
                logical_len,
                allocation_len,
                bytes.len(),
                final_usage,
                None,
            ));
        }
        // Zero-length upload: allocate a minimal buffer (wgpu forbids both a
        // 0-byte allocation and a 0-byte mapped_at_creation buffer). `write_padded`
        // is a no-op here; the handle reports a logical length of 0.
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vyre persistent upload"),
            size: allocation_len,
            usage: final_usage,
            mapped_at_creation: false,
        });
        write_padded(queue, &buffer, bytes, allocation_len)?;
        Ok(Self::from_parts(
            Arc::new(buffer),
            0,
            allocation_len,
            bytes.len(),
            final_usage,
            None,
        ))
    }

    /// Allocate a GPU-resident buffer without uploading host contents.
    ///
    /// # Errors
    ///
    /// Returns a backend error when `len` cannot be represented as a valid
    /// wgpu buffer size.
    pub fn alloc(
        device: &wgpu::Device,
        len: u64,
        usage: wgpu::BufferUsages,
    ) -> Result<Self, BackendError> {
        let allocation_len = aligned_len_u64(len, "persistent GPU allocation length")?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vyre persistent alloc"),
            size: allocation_len,
            usage,
            mapped_at_creation: false,
        });
        let host_len = usize::try_from(len).map_err(|error| {
            BackendError::new(format!(
                "GpuBufferHandle::alloc received logical byte length {len} that does not fit usize on this host: {error}. Fix: shard the GPU buffer before allocating or run on a host with a wide enough address space."
            ))
        })?;
        Ok(Self::from_parts(
            Arc::new(buffer),
            len,
            allocation_len,
            host_len,
            usage,
            None,
        ))
    }

    /// Download this GPU buffer into `out`.
    ///
    /// This is intended for terminal output and test assertions, not hot-loop
    /// dispatch. The buffer must have `COPY_SRC` usage.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the handle is not copy-readable or the GPU
    /// mapping fails.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        self.readback_until(device, None, queue, out, None)
    }

    /// Download the first `len` logical bytes of this GPU buffer into `out`.
    ///
    /// Hot paths that publish a device-side count should read back only the
    /// counted prefix instead of the whole capacity-sized buffer. The copy is
    /// rounded up to wgpu's 4-byte copy granularity internally, then truncated
    /// back to exactly `len` bytes before returning.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the handle is not copy-readable, `len`
    /// exceeds the logical buffer length, or the GPU mapping fails.
    pub fn readback_prefix(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        len: u64,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        self.readback_prefix_until(device, None, queue, len, out, None)
    }

    /// Download `len` logical bytes starting at `byte_offset` into `out`.
    ///
    /// The internal GPU copy is alignment-expanded when necessary, then the
    /// returned host slice is trimmed back to exactly the requested range.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the handle is not copy-readable, the range
    /// exceeds the logical buffer length, or the GPU mapping fails.
    pub fn readback_range(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        byte_offset: u64,
        len: u64,
        out: &mut Vec<u8>,
    ) -> Result<(), BackendError> {
        self.readback_range_until(device, None, queue, byte_offset, len, out, None)
    }

    pub(crate) fn readback_until(
        &self,
        device: &wgpu::Device,
        pool: Option<&StagingBufferPool>,
        queue: &wgpu::Queue,
        out: &mut Vec<u8>,
        deadline: Option<Instant>,
    ) -> Result<(), BackendError> {
        self.readback_prefix_until(device, pool, queue, self.byte_len(), out, deadline)
    }

    pub(crate) fn readback_prefix_until(
        &self,
        device: &wgpu::Device,
        pool: Option<&StagingBufferPool>,
        queue: &wgpu::Queue,
        len: u64,
        out: &mut Vec<u8>,
        deadline: Option<Instant>,
    ) -> Result<(), BackendError> {
        self.readback_range_until(device, pool, queue, 0, len, out, deadline)
    }

    pub(crate) fn readback_range_until(
        &self,
        device: &wgpu::Device,
        pool: Option<&StagingBufferPool>,
        queue: &wgpu::Queue,
        byte_offset: u64,
        len: u64,
        out: &mut Vec<u8>,
        deadline: Option<Instant>,
    ) -> Result<(), BackendError> {
        self.readback_range_async(device, pool, queue, byte_offset, len)?
            .await_into(device, deadline, out)
    }

    pub(crate) fn readback_range_async(
        &self,
        device: &wgpu::Device,
        pool: Option<&StagingBufferPool>,
        queue: &wgpu::Queue,
        byte_offset: u64,
        len: u64,
    ) -> Result<PendingGpuBufferReadback, BackendError> {
        if !self.usage().contains(wgpu::BufferUsages::COPY_SRC) {
            return Err(BackendError::new(
                "GpuBufferHandle readback requires COPY_SRC usage. Fix: allocate terminal-output buffers with COPY_SRC.",
            ));
        }
        let logical_end = byte_offset.checked_add(len).ok_or_else(|| {
            BackendError::new(format!(
                "GpuBufferHandle range readback overflows u64 at offset {byte_offset} len {len}. Fix: split the readback range before dispatch."
            ))
        })?;
        if logical_end > self.byte_len() {
            return Err(BackendError::new(format!(
                "GpuBufferHandle range readback requested bytes [{byte_offset}..{logical_end}) from a {} byte buffer. Fix: clamp the requested range to the device-published count.",
                self.byte_len()
            )));
        }
        if len == 0 {
            return Ok(PendingGpuBufferReadback::Ready);
        }
        let copy_start = byte_offset & !3;
        let trim_start = byte_offset - copy_start;
        let visible_copy_len = trim_start.checked_add(len).ok_or_else(|| {
            BackendError::new(format!(
                "GpuBufferHandle range readback copy length overflows u64 at trim {trim_start} len {len}. Fix: split the readback range before dispatch."
            ))
        })?;
        let read_len = aligned_len_u64(visible_copy_len, "GPU readback visible copy length")?;
        let copy_end = copy_start.checked_add(read_len).ok_or_else(|| {
            BackendError::new(format!(
                "GpuBufferHandle range readback aligned copy overflows u64 at start {copy_start} len {read_len}. Fix: split the readback range before dispatch."
            ))
        })?;
        if copy_end > self.inner.allocation_len {
            return Err(BackendError::new(format!(
                "GpuBufferHandle range readback rounded bytes [{byte_offset}..{logical_end}) to aligned bytes [{copy_start}..{copy_end}), beyond allocation length {}. Fix: allocate buffers with 4-byte padding.",
                self.inner.allocation_len
            )));
        }
        let visible_len = usize::try_from(len).map_err(|source| {
            BackendError::new(format!(
                "persistent buffer prefix length {len} cannot fit usize: {source}. Fix: split the buffer before readback."
            ))
        })?;
        let trim_start = usize::try_from(trim_start).map_err(|source| {
            BackendError::new(format!(
                "persistent buffer range trim offset {trim_start} cannot fit usize: {source}. Fix: split the buffer before readback."
            ))
        })?;
        let readback_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        let readback = if let Some(pool) = pool {
            pool.acquire(device, read_len, readback_usage)
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vyre persistent handle readback"),
                size: read_len,
                usage: readback_usage,
                mapped_at_creation: false,
            })
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vyre persistent handle readback encoder"),
        });
        encoder.copy_buffer_to_buffer(self.buffer(), copy_start, &readback, 0, read_len);
        let submission = queue.submit(std::iter::once(encoder.finish()));
        let slice = readback.slice(0..read_len);
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let ready = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ready_callback = Arc::clone(&ready);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            if let Err(error) = sender.send(result) {
                tracing::error!(
                    ?error,
                    "persistent buffer readback map_async result was lost because the receiver dropped"
                );
            }
            ready_callback.store(true, Ordering::Release);
        });
        Ok(PendingGpuBufferReadback::Mapping {
            readback,
            read_len,
            readback_usage,
            pool: pool.cloned(),
            submission,
            receiver,
            ready,
            trim_start,
            visible_len,
        })
    }

    /// Stable process-local handle id used for cache signatures.
    #[must_use]
    pub fn id(&self) -> u64 {
        self.inner.id
    }

    /// Stable process-local identity for the backing GPU allocation.
    ///
    /// Unlike [`Self::id`], this survives pool release/reacquire cycles for the
    /// same underlying `wgpu::Buffer`. Bind-group caches must key on this value
    /// plus the logical binding range; otherwise hot dispatches miss every time
    /// a pooled allocation is wrapped in a fresh handle.
    #[must_use]
    pub(crate) fn allocation_identity(&self) -> u64 {
        pointer_identity_key(Arc::as_ptr(&self.inner.buffer))
    }

    /// Resolve a process-local resident buffer id back into a live GPU handle.
    #[must_use]
    pub fn from_resident_id(id: u64) -> Option<Self> {
        resident::resolve_resident_id(id)
    }

    /// Resident handle naming this buffer, owner included.
    ///
    /// # Errors
    ///
    /// Returns a backend error when this process could not mint an identity
    /// for the WGPU resident namespace.
    pub fn resident_handle(&self) -> Result<ResidentHandle, BackendError> {
        Ok(resident::resident_owner()?.handle(self.inner.id))
    }

    /// Resolve a resident handle back into a live GPU handle.
    ///
    /// `Ok(None)` means the handle named a WGPU buffer that is no longer live.
    ///
    /// # Errors
    ///
    /// Returns a backend error when `handle` was minted by a different backend
    /// instance, which is refused rather than resolved: its id would name an
    /// unrelated buffer here.
    pub fn from_resident_handle(
        handle: ResidentHandle,
        context: &str,
    ) -> Result<Option<Self>, BackendError> {
        let id = resident::resident_owner()?.resolve(handle, context)?;
        Ok(Self::from_resident_id(id))
    }

    /// Underlying `wgpu::Buffer`.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.inner.buffer
    }

    /// Clone the internal `Arc<wgpu::Buffer>`  -  cheap, reference-
    /// count only. Used by the indirect dispatch path (C-B4) which
    /// needs to stash the buffer alongside other args.
    #[must_use]
    pub fn buffer_arc(&self) -> Arc<wgpu::Buffer> {
        Arc::clone(&self.inner.buffer)
    }

    /// Logical byte length requested by the caller.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.inner.byte_len
    }

    /// Backing allocation length.
    #[must_use]
    pub fn allocation_len(&self) -> u64 {
        self.inner.allocation_len
    }

    /// Logical element count. Byte buffers report one element per byte.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.inner.element_count
    }

    /// Actual usage flags on the underlying GPU allocation.
    #[must_use]
    pub fn usage(&self) -> wgpu::BufferUsages {
        self.inner.usage
    }

    pub(crate) fn from_parts(
        buffer: Arc<wgpu::Buffer>,
        byte_len: u64,
        allocation_len: u64,
        element_count: usize,
        usage: wgpu::BufferUsages,
        pool_return: Option<PoolReturn>,
    ) -> Self {
        let inner = Arc::new(GpuBufferInner {
            id: resident::next_buffer_id(),
            buffer,
            byte_len,
            allocation_len,
            element_count,
            usage,
            pool_return,
        });
        resident::register_resident_buffer(inner.id, &inner);
        Self { inner }
    }
}

impl std::fmt::Debug for GpuBufferHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GpuBufferHandle")
            .field("id", &self.id())
            .field("byte_len", &self.byte_len())
            .field("allocation_len", &self.allocation_len())
            .field("element_count", &self.element_count())
            .field("usage", &self.usage())
            .finish()
    }
}

impl Drop for GpuBufferInner {
    fn drop(&mut self) {
        resident::remove_resident_buffer(self.id);
        if let Some(pool_return) = self.pool_return.take() {
            pool_return.release(
                Arc::clone(&self.buffer),
                self.byte_len,
                self.allocation_len,
                self.usage,
            );
        }
    }
}
