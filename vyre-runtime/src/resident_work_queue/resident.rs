//! Host mirrors for resident work-queue runtime buffers.

use super::io;
use super::planner::ResidentWorkItem;
use super::protocol;
use super::protocol_api::{validate_control_bytes, validate_debug_log_bytes};
use super::readback::ResidentQueueReadback;
use super::scheduler::write_default_priority_offsets;
use super::ResidentWorkQueue;
use crate::PipelineError;

/// Host-side mirror of the four buffers kept resident by the persistent
/// megakernel runtime: control, ring, debug log, and IO queue.
#[derive(Debug)]
pub struct ResidentQueueBuffers {
    control_bytes: Vec<u8>,
    ring_bytes: Vec<u8>,
    debug_log_bytes: Vec<u8>,
    io_queue_bytes: Vec<u8>,
    slot_count: u32,
}

impl Clone for ResidentQueueBuffers {
    fn clone(&self) -> Self {
        Self {
            control_bytes: self.control_bytes.clone(),
            ring_bytes: self.ring_bytes.clone(),
            debug_log_bytes: self.debug_log_bytes.clone(),
            io_queue_bytes: self.io_queue_bytes.clone(),
            slot_count: self.slot_count,
        }
    }
}

impl PartialEq for ResidentQueueBuffers {
    fn eq(&self, other: &Self) -> bool {
        self.control_bytes == other.control_bytes
            && self.ring_bytes == other.ring_bytes
            && self.debug_log_bytes == other.debug_log_bytes
            && self.io_queue_bytes == other.io_queue_bytes
            && self.slot_count == other.slot_count
    }
}

impl Eq for ResidentQueueBuffers {}

impl ResidentQueueBuffers {
    /// Allocate a fresh host mirror for a megakernel's resident buffers.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when any runtime buffer size overflows.
    pub fn new(
        slot_count: u32,
        tenant_count: u32,
        observable_slots: u32,
    ) -> Result<Self, PipelineError> {
        let control_capacity = protocol::control_byte_len(observable_slots).ok_or_else(|| {
            PipelineError::Backend(
                "megakernel resident control byte length overflowed usize. Fix: shard observable resident buffers before allocation."
                    .to_string(),
            )
        })?;
        let ring_capacity = protocol::ring_byte_len(slot_count).ok_or_else(|| {
            PipelineError::Backend(
                "megakernel resident ring byte length overflowed usize. Fix: shard resident rings before allocation."
                    .to_string(),
            )
        })?;
        let debug_log_capacity =
            protocol::debug_log_byte_len(protocol::debug::RECORD_CAPACITY).ok_or_else(|| {
                PipelineError::Backend(
                    "megakernel resident debug-log byte length overflowed usize. Fix: reduce debug record capacity before allocation."
                        .to_string(),
                )
        })?;
        let io_queue_capacity = io::empty_io_queue_byte_len(io::IO_SLOT_COUNT)?;
        let mut control_bytes = Vec::new();
        reserve_resident_bytes(
            &mut control_bytes,
            control_capacity,
            "control",
            "shard observable resident buffers before allocation",
        )?;
        let mut ring_bytes = Vec::new();
        reserve_resident_bytes(
            &mut ring_bytes,
            ring_capacity,
            "ring",
            "shard resident rings before allocation",
        )?;
        let mut debug_log_bytes = Vec::new();
        reserve_resident_bytes(
            &mut debug_log_bytes,
            debug_log_capacity,
            "debug-log",
            "reduce debug record capacity before allocation",
        )?;
        let mut io_queue_bytes = Vec::new();
        reserve_resident_bytes(
            &mut io_queue_bytes,
            io_queue_capacity,
            "io-queue",
            "reduce resident IO queue capacity before allocation",
        )?;
        let mut buffers = Self {
            control_bytes,
            ring_bytes,
            debug_log_bytes,
            io_queue_bytes,
            slot_count,
        };
        buffers.reset(tenant_count, observable_slots)?;
        Ok(buffers)
    }

    /// Reinitialize this host mirror in place for the same resident geometry.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when any runtime buffer size overflows.
    pub fn reset(&mut self, tenant_count: u32, observable_slots: u32) -> Result<(), PipelineError> {
        ResidentWorkQueue::try_encode_control_into(
            false,
            tenant_count,
            observable_slots,
            &mut self.control_bytes,
        )?;
        write_default_priority_offsets(&mut self.control_bytes, self.slot_count)?;
        ResidentWorkQueue::try_encode_empty_ring_into(self.slot_count, &mut self.ring_bytes)?;
        ResidentWorkQueue::try_encode_empty_debug_log_into(
            protocol::debug::RECORD_CAPACITY,
            &mut self.debug_log_bytes,
        )?;
        io::try_encode_empty_io_queue_into(io::IO_SLOT_COUNT, &mut self.io_queue_bytes)?;
        Ok(())
    }

    /// Build a resident-buffer mirror from caller-owned byte buffers.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when any buffer violates the megakernel ABI.
    pub fn from_parts(
        slot_count: u32,
        control_bytes: Vec<u8>,
        ring_bytes: Vec<u8>,
        debug_log_bytes: Vec<u8>,
        io_queue_bytes: Vec<u8>,
    ) -> Result<Self, PipelineError> {
        validate_control_bytes(&control_bytes)?;
        validate_debug_log_bytes(&debug_log_bytes)?;
        io::validate_io_queue_bytes(&io_queue_bytes)?;
        let expected_ring_bytes = protocol::ring_byte_len(slot_count).ok_or_else(|| {
            PipelineError::Backend(
                "megakernel resident ring byte length overflowed usize. Fix: shard resident rings before allocation."
                    .to_string(),
            )
        })?;
        if ring_bytes.len() != expected_ring_bytes {
            return Err(PipelineError::Backend(format!(
                "megakernel resident ring has {} bytes, expected {expected_ring_bytes}. Fix: build resident rings with the same slot_count as the Megakernel handle.",
                ring_bytes.len()
            )));
        }
        Ok(Self {
            control_bytes,
            ring_bytes,
            debug_log_bytes,
            io_queue_bytes,
            slot_count,
        })
    }

    /// Publish one work slot into the resident ring mirror.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the slot is out of bounds or
    /// still in flight.
    pub fn publish_slot(
        &mut self,
        slot_idx: u32,
        tenant_id: u32,
        opcode: u32,
        args: &[u32],
    ) -> Result<(), PipelineError> {
        ResidentWorkQueue::publish_slot(&mut self.ring_bytes, slot_idx, tenant_id, opcode, args)
    }

    /// Publish a contiguous fixed-ABI work-item window into the resident ring
    /// mirror without resetting unrelated slots.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::QueueFull`] when the target slots are outside
    /// the resident ring, still in flight, or contain an unpublished opcode.
    pub fn publish_work_items(
        &mut self,
        start_slot: u32,
        tenant_id: u32,
        items: &[ResidentWorkItem],
    ) -> Result<u32, PipelineError> {
        ResidentWorkQueue::publish_work_items(&mut self.ring_bytes, start_slot, tenant_id, items)
    }

    /// Apply a strict dispatch readback to the resident host mirror.
    pub fn apply_readback(&mut self, readback: ResidentQueueReadback) {
        self.control_bytes = readback.control_bytes;
        self.ring_bytes = readback.ring_bytes;
        self.debug_log_bytes = readback.debug_log_bytes;
        self.io_queue_bytes = readback.io_queue_bytes;
    }

    /// Clone the current host mirror into a strict readback record.
    #[must_use]
    pub fn snapshot_readback(&self) -> ResidentQueueReadback {
        ResidentQueueReadback {
            control_bytes: self.control_bytes.clone(),
            ring_bytes: self.ring_bytes.clone(),
            debug_log_bytes: self.debug_log_bytes.clone(),
            io_queue_bytes: self.io_queue_bytes.clone(),
        }
    }

    /// Clone the current host mirror into caller-owned readback storage.
    pub fn snapshot_readback_into(&self, out: &mut ResidentQueueReadback) {
        out.control_bytes.clone_from(&self.control_bytes);
        out.ring_bytes.clone_from(&self.ring_bytes);
        out.debug_log_bytes.clone_from(&self.debug_log_bytes);
        out.io_queue_bytes.clone_from(&self.io_queue_bytes);
    }

    /// Control-buffer mirror bytes.
    #[must_use]
    pub fn control_bytes(&self) -> &[u8] {
        &self.control_bytes
    }

    /// Ring-buffer mirror bytes.
    #[must_use]
    pub fn ring_bytes(&self) -> &[u8] {
        &self.ring_bytes
    }

    /// Mutable ring-buffer mirror bytes.
    #[must_use]
    pub fn ring_bytes_mut(&mut self) -> &mut [u8] {
        &mut self.ring_bytes
    }

    /// Debug-log mirror bytes.
    #[must_use]
    pub fn debug_log_bytes(&self) -> &[u8] {
        &self.debug_log_bytes
    }

    /// IO-queue mirror bytes.
    #[must_use]
    pub fn io_queue_bytes(&self) -> &[u8] {
        &self.io_queue_bytes
    }

    /// Resident ring slot count.
    #[must_use]
    pub const fn slot_count(&self) -> u32 {
        self.slot_count
    }
}

fn reserve_resident_bytes(
    bytes: &mut Vec<u8>,
    capacity: usize,
    label: &'static str,
    fix: &'static str,
) -> Result<(), PipelineError> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(bytes, capacity).map_err(|error| {
        PipelineError::Backend(format!(
            "megakernel resident {label} byte reservation failed for {capacity} bytes: {error}. Fix: {fix}."
        ))
    })
}
