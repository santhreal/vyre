//! The ring itself: slot recycling, copy submission, and host collection.

use crossbeam_channel::Receiver;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use vyre_driver::accounting::rebasing_atomic_next_u64;
use vyre_driver::backend::BackendError;

use super::capacity::{aligned_copy_len, staging_capacity, MAX_RING_SIZE, MIN_RING_SIZE};
use super::slot::{
    GpuSlot, MapResult, ReadbackTicket, SLOT_ERROR, SLOT_FREE, SLOT_PENDING, SLOT_READY,
};
use super::stats::RingStats;
use crate::staging_reserve::reserve_backend_vec;

/// Async readback ring buffer with GPU-resident staging buffers.
pub struct ReadbackRing {
    slots: Vec<GpuSlot>,
    stats: Arc<RingStats>,
    next_idx: AtomicU64,
}

impl ReadbackRing {
    /// Construct a ring with N staging buffers.
    #[must_use]
    pub fn new(device: &wgpu::Device, size: usize, buffer_size: u64) -> Result<Self, BackendError> {
        let size = size.clamp(MIN_RING_SIZE, MAX_RING_SIZE);
        let capacity = staging_capacity(buffer_size)?;
        let mut slots = Vec::new();
        reserve_backend_vec(&mut slots, size, "readback ring slot table")?;
        for i in 0..size {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("vyre readback ring slot {i}")),
                size: capacity,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            slots.push(GpuSlot {
                buffer,
                state: Arc::new(std::sync::atomic::AtomicU8::new(SLOT_FREE)),
                byte_len: AtomicU64::new(0),
                mapped_len: AtomicU64::new(0),
                capacity,
            });
        }
        Ok(Self {
            slots,
            stats: Arc::new(RingStats::default()),
            next_idx: AtomicU64::new(0),
        })
    }

    /// Ensure slot `idx` is reusable for a fresh readback: either already
    /// `SLOT_FREE`, or `SLOT_PENDING` that completes to `SLOT_FREE` after one
    /// device poll. Any other terminal state is a caller contract violation and
    /// is reported as a distinct, fail-closed error.
    ///
    /// VYRE-WGPU-002: earlier code conflated `SLOT_READY` / `SLOT_ERROR` with a
    /// single misleading wrap-overflow message. We now name each
    /// state (but we DO NOT silently recycle an uncollected `SLOT_READY` slot).
    /// Recycling would unmap and discard a completed-but-uncollected readback
    /// a silent recall loss (Law 10). The caller MUST collect every readback
    /// before the ring wraps back to its slot; if it has not, we fail closed so
    /// the data loss is impossible to miss.
    fn ensure_slot_reusable(
        &self,
        idx: usize,
        slot: &GpuSlot,
        device: &wgpu::Device,
    ) -> Result<(), BackendError> {
        let mut state = slot.state.load(Ordering::Acquire);
        if state == SLOT_PENDING {
            self.stats.record_stall();
            crate::runtime::device::poll_device_once(device)?;
            state = slot.state.load(Ordering::Acquire);
        }
        match state {
            SLOT_FREE => Ok(()),
            SLOT_READY => Err(BackendError::new(format!(
                "readback ring slot {idx} holds an uncollected completed readback (SLOT_READY). Fix: collect every ReadbackTicket via collect_slot_into before the ring wraps back to this slot (recycling it would silently drop the prior result (a recall loss))."
            ))),
            SLOT_ERROR => Err(BackendError::new(format!(
                "readback ring slot {idx} is in SLOT_ERROR (prior map_async failed) and was not collected before reuse. Fix: collect error slots via collect_slot_into before submitting new readbacks to the same slot."
            ))),
            SLOT_PENDING => Err(BackendError::new(format!(
                "readback ring slot {idx} is still SLOT_PENDING after a device poll, the prior readback has not completed. Fix: increase ring depth (more slots) or collect outstanding readbacks before submitting more."
            ))),
            other => Err(BackendError::new(format!(
                "readback ring slot {idx} has unexpected state {other}. Fix: do not modify readback ring slot state outside the ring API."
            ))),
        }
    }

    /// Record a readback copy into the next available ring slot.
    ///
    /// The caller must submit the encoder and then arm the returned ticket with
    /// [`Self::arm_ticket`]. This path lets the main dispatch encoder copy into
    /// preallocated ring slots instead of allocating a fresh staging buffer per
    /// output.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the byte range cannot be represented, the
    /// ring slot is not reusable (an uncollected `SLOT_READY`/`SLOT_ERROR` slot
    /// or a still-pending slot after a device poll), or the requested readback
    /// exceeds slot capacity.
    pub fn record_copy(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        src_buffer: &wgpu::Buffer,
        src_offset: u64,
        byte_len: u64,
    ) -> Result<ReadbackTicket, BackendError> {
        let idx = self.next_slot_index()?;
        let slot = &self.slots[idx];
        let mapped_len = aligned_copy_len(byte_len)?;
        if mapped_len > slot.capacity {
            return Err(BackendError::new(format!(
                "readback request of {byte_len} bytes ({} bytes after wgpu copy alignment) exceeds ring slot capacity {} bytes. Fix: construct ReadbackRing with a buffer_size at least as large as the largest readback.",
                mapped_len, slot.capacity
            )));
        }

        self.ensure_slot_reusable(idx, slot, device)?;

        slot.byte_len.store(byte_len, Ordering::Release);
        slot.mapped_len.store(mapped_len, Ordering::Release);
        slot.state.store(SLOT_PENDING, Ordering::Release);
        if mapped_len != 0 {
            encoder.copy_buffer_to_buffer(src_buffer, src_offset, &slot.buffer, 0, mapped_len);
        } else {
            slot.state.store(SLOT_READY, Ordering::Release);
        }
        self.stats.record_dispatch();
        Ok(ReadbackTicket {
            idx,
            byte_len,
            mapped_len,
        })
    }

    /// Arm a submitted ticket by registering its `map_async` callback.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when `ticket` does not reference a live slot.
    pub fn arm_ticket(
        &self,
        ticket: &ReadbackTicket,
    ) -> Result<(Receiver<MapResult>, Arc<AtomicBool>), BackendError> {
        let Some(slot) = self.slots.get(ticket.idx) else {
            return Err(BackendError::new(format!(
                "readback ring ticket slot {} is out of bounds for {} slots. Fix: keep tickets paired with their originating ring.",
                ticket.idx,
                self.slots.len()
            )));
        };
        let (sender, receiver) = crossbeam_channel::bounded(1);
        let ready = Arc::new(AtomicBool::new(false));
        if ticket.mapped_len == 0 {
            if let Err(error) = sender.send(Ok(())) {
                tracing::error!(
                    ?error,
                    "readback ring zero-length callback result was lost because the receiver dropped"
                );
            }
            ready.store(true, Ordering::Release);
            return Ok((receiver, ready));
        }

        let state = Arc::clone(&slot.state);
        let ready_cb = Arc::clone(&ready);
        slot.buffer
            .slice(0..ticket.mapped_len)
            .map_async(wgpu::MapMode::Read, move |result| {
                match &result {
                    Ok(()) => state.store(SLOT_READY, Ordering::Release),
                    Err(error) => {
                        tracing::error!(
                            "readback ring map_async failed: {error:?}. Fix: inspect device health and readback buffer usage."
                        );
                        state.store(SLOT_ERROR, Ordering::Release);
                    }
                }
                if let Err(error) = sender.send(result) {
                    tracing::error!(
                        ?error,
                        "readback ring callback result was lost because the receiver dropped"
                    );
                }
                ready_cb.store(true, Ordering::Release);
            });
        Ok((receiver, ready))
    }

    /// Expose a ready ticket's mapped bytes to `visitor`, then free the slot.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the ticket is stale, the slot is not ready,
    /// or mapped length metadata is inconsistent.
    pub fn with_mapped_ticket<R>(
        &self,
        ticket: &ReadbackTicket,
        visitor: impl FnOnce(&[u8]) -> Result<R, BackendError>,
    ) -> Result<R, BackendError> {
        let Some(slot) = self.slots.get(ticket.idx) else {
            return Err(BackendError::new(format!(
                "readback ring ticket slot {} is out of bounds for {} slots. Fix: keep tickets paired with their originating ring.",
                ticket.idx,
                self.slots.len()
            )));
        };
        match slot.state.load(Ordering::Acquire) {
            SLOT_READY => {}
            SLOT_ERROR => {
                slot.byte_len.store(0, Ordering::Release);
                slot.mapped_len.store(0, Ordering::Release);
                slot.state.store(SLOT_FREE, Ordering::Release);
                return Err(BackendError::new(
                    "readback ring map_async failed. Fix: inspect GPU device health and ensure the slot buffer has MAP_READ usage.",
                ));
            }
            _ => {
                return Err(BackendError::new(
                    "readback ring ticket was collected before its map callback completed. Fix: poll the device or wait for the submitted GPU work before collection.",
                ));
            }
        }

        let len = usize::try_from(ticket.byte_len).map_err(|source| {
            BackendError::new(format!(
                "readback ring byte length {} cannot fit usize: {source}. Fix: split the readback before collecting it.",
                ticket.byte_len
            ))
        })?;
        if ticket.mapped_len == 0 {
            slot.byte_len.store(0, Ordering::Release);
            slot.mapped_len.store(0, Ordering::Release);
            slot.state.store(SLOT_FREE, Ordering::Release);
            return visitor(&[]);
        }
        let view = slot.buffer.slice(0..ticket.mapped_len).get_mapped_range();
        if len > view.len() {
            let mapped_len = view.len();
            drop(view);
            slot.buffer.unmap();
            slot.byte_len.store(0, Ordering::Release);
            slot.mapped_len.store(0, Ordering::Release);
            slot.state.store(SLOT_FREE, Ordering::Release);
            return Err(BackendError::new(format!(
                "readback ring mapped length {mapped_len} is shorter than requested length {len}. Fix: keep ticket and slot byte lengths synchronized."
            )));
        }
        let result = visitor(&view[..len]);
        drop(view);
        slot.buffer.unmap();
        slot.byte_len.store(0, Ordering::Release);
        slot.mapped_len.store(0, Ordering::Release);
        slot.state.store(SLOT_FREE, Ordering::Release);
        result
    }

    /// Submit a copy from `src_buffer` at `src_offset` and mark the slot pending.
    ///
    /// `src_offset` is the byte offset within `src_buffer` to copy from. Pass
    /// `0` to read from the start of the buffer. This mirrors the `src_offset`
    /// parameter accepted by `record_copy`; callers that need a sub-range of
    /// the source buffer must supply a non-zero offset here rather than
    /// wrapping a slice (the wgpu copy API requires aligned buffer offsets).
    ///
    /// # Errors
    /// Returns `BackendError` if encoder or queue submission fails.
    pub fn submit_readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        src_offset: u64,
        byte_len: u64,
    ) -> Result<usize, BackendError> {
        let idx = self.next_slot_index()?;
        let slot = &self.slots[idx];
        let mapped_len = aligned_copy_len(byte_len)?;
        if mapped_len > slot.capacity {
            return Err(BackendError::new(format!(
                "readback request of {byte_len} bytes ({} bytes after wgpu copy alignment) exceeds ring slot capacity {} bytes. Fix: construct ReadbackRing with a buffer_size at least as large as the largest readback.",
                mapped_len, slot.capacity
            )));
        }

        self.ensure_slot_reusable(idx, slot, device)?;

        let state_clone = Arc::clone(&slot.state);
        slot.byte_len.store(byte_len, Ordering::Release);
        slot.mapped_len.store(mapped_len, Ordering::Release);
        state_clone.store(SLOT_PENDING, Ordering::Release);

        if mapped_len == 0 {
            state_clone.store(SLOT_READY, Ordering::Release);
            self.stats.record_dispatch();
            return Ok(idx);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("vyre readback ring copy"),
        });
        encoder.copy_buffer_to_buffer(src_buffer, src_offset, &slot.buffer, 0, mapped_len);
        queue.submit(std::iter::once(encoder.finish()));

        slot.buffer
            .slice(0..mapped_len)
            .map_async(wgpu::MapMode::Read, move |result| {
                match result {
                    Ok(()) => state_clone.store(SLOT_READY, Ordering::Release),
                    Err(error) => {
                        tracing::error!(
                            "readback ring map_async failed: {error:?}. Fix: inspect device health and readback buffer usage."
                        );
                        state_clone.store(SLOT_ERROR, Ordering::Release);
                    }
                }
            });

        self.stats.record_dispatch();

        Ok(idx)
    }

    /// Try to collect data from a specific slot.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when `idx` is out of bounds or `map_async`
    /// failed for the slot.
    pub fn collect_slot(
        &self,
        device: &wgpu::Device,
        idx: usize,
    ) -> Result<Option<Vec<u8>>, BackendError> {
        let mut data = Vec::new();
        if self.collect_slot_into(device, idx, &mut data)?.is_some() {
            Ok(Some(data))
        } else {
            Ok(None)
        }
    }

    /// Try to collect data from a specific slot into a caller-owned buffer.
    ///
    /// Reusing `out` avoids an allocation on every ready readback. The buffer is
    /// cleared before bytes are appended.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when `idx` is out of bounds or `map_async`
    /// failed for the slot.
    pub fn collect_slot_into(
        &self,
        device: &wgpu::Device,
        idx: usize,
        out: &mut Vec<u8>,
    ) -> Result<Option<usize>, BackendError> {
        let Some(slot) = self.slots.get(idx) else {
            return Err(BackendError::new(format!(
                "readback ring slot index {idx} is out of bounds for {} slots. Fix: collect only indices returned by submit_readback.",
                self.slots.len()
            )));
        };
        match slot.state.load(Ordering::Acquire) {
            SLOT_READY => {
                let len = self.copy_ready_slot_into(idx, out)?;
                Ok(Some(len))
            }
            SLOT_ERROR => {
                slot.byte_len.store(0, Ordering::Release);
                slot.mapped_len.store(0, Ordering::Release);
                slot.state.store(SLOT_FREE, Ordering::Release);
                Err(BackendError::new(
                    "readback ring map_async failed. Fix: inspect GPU device health and ensure the slot buffer has MAP_READ usage.",
                ))
            }
            _ => {
                crate::runtime::device::poll_device_once(device)?;
                Ok(None)
            }
        }
    }

    fn copy_ready_slot_into(&self, idx: usize, out: &mut Vec<u8>) -> Result<usize, BackendError> {
        let slot = &self.slots[idx];
        let byte_len = slot.byte_len.load(Ordering::Acquire);
        let mapped_len = slot.mapped_len.load(Ordering::Acquire);
        let len = usize::try_from(byte_len).map_err(|source| {
            BackendError::new(format!(
                "readback ring byte length {byte_len} cannot fit usize: {source}. Fix: split the readback before collecting it."
            ))
        })?;
        if mapped_len != 0 {
            let view = slot.buffer.slice(0..mapped_len).get_mapped_range();
            let bytes = &view[..len];
            if out.len() == len {
                out.copy_from_slice(bytes);
            } else {
                if len > out.capacity() {
                    let additional = len - out.capacity();
                    out.try_reserve_exact(additional).map_err(|source| {
                        BackendError::new(format!(
                            "readback ring collection could not reserve {len} output bytes exactly: {source}. Fix: lower max_output_bytes or collect readback in smaller shards."
                        ))
                    })?;
                }
                out.clear();
                out.extend_from_slice(bytes);
            }
            drop(view);
            slot.buffer.unmap();
        } else {
            out.clear();
        }
        slot.byte_len.store(0, Ordering::Release);
        slot.mapped_len.store(0, Ordering::Release);
        slot.state.store(SLOT_FREE, Ordering::Release);
        Ok(len)
    }

    #[inline]
    fn next_slot_index(&self) -> Result<usize, BackendError> {
        let slot_len = u64::try_from(self.slots.len()).map_err(|source| {
            BackendError::new(format!(
                "readback ring slot count {} cannot fit u64: {source}. Fix: reduce readback ring slot count.",
                self.slots.len()
            ))
        })?;
        if slot_len == 0 {
            return Err(BackendError::new(
                "readback ring has zero slots. Fix: construct rings with at least two slots.",
            ));
        }
        let next = rebasing_atomic_next_u64(
            &self.next_idx,
            0,
            Ordering::Relaxed,
            Ordering::Relaxed,
            Ordering::Relaxed,
            |_, _| {
                tracing::error!(
                    "readback ring slot counter reached u64::MAX and was rebased to zero. Fix: shard readback rings or scrape counters before wrap."
                );
            },
        );
        usize::try_from(next % slot_len).map_err(|source| {
            BackendError::new(format!(
                "readback ring slot index cannot fit usize: {source}. Fix: reduce readback ring slot count."
            ))
        })
    }
}
