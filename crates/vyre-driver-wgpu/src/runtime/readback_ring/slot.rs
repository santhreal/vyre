//! One ring slot, its lifecycle codes, and the ticket a submitted copy yields.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Slot is available for new writes.
pub(super) const SLOT_FREE: u8 = 0;
/// Copy has been submitted; data is ready after the fence.
pub(super) const SLOT_PENDING: u8 = 1;
/// Map has completed and data is visible to the host.
pub(super) const SLOT_READY: u8 = 2;
/// Mapping failed and the slot must be collected as an error.
pub(super) const SLOT_ERROR: u8 = 3;

/// Result type produced by one `map_async` callback.
pub type MapResult = Result<(), wgpu::BufferAsyncError>;

/// GPU-aware ring slot.
pub struct GpuSlot {
    /// Underlying wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// Atomic lifecycle state (0: Free, 1: Pending, 2: Ready).
    pub state: Arc<std::sync::atomic::AtomicU8>,
    pub(super) byte_len: AtomicU64,
    pub(super) mapped_len: AtomicU64,
    pub(super) capacity: u64,
}

/// Submitted copy ticket for one readback-ring slot.
pub struct ReadbackTicket {
    pub(super) idx: usize,
    pub(super) byte_len: u64,
    pub(super) mapped_len: u64,
}
