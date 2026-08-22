//! Staging-capacity size classes and the configured slot count.

use vyre_driver::BackendError;

pub(super) const MIN_RING_SIZE: usize = 2;
pub(super) const MAX_RING_SIZE: usize = 256;
const DEFAULT_RING_SLOTS: usize = 256;
const RING_CAPACITY_GRANULARITY: u64 = 4096;

#[inline]
pub(super) fn staging_capacity(byte_len: u64) -> Result<u64, BackendError> {
    aligned_copy_len(byte_len).map_err(|error| {
        tracing::warn!(
            "readback ring staging capacity overflowed for {byte_len} bytes: {error}. Fix: shard the readback buffer before constructing the ring."
        );
        error
    }).map(|len| len.max(4))
}

#[inline]
pub(super) fn ring_capacity_class(byte_len: u64) -> Result<u64, BackendError> {
    let aligned = aligned_copy_len(byte_len)?.max(4);
    aligned
        .checked_add(RING_CAPACITY_GRANULARITY - 1)
        .map(|len| len & !(RING_CAPACITY_GRANULARITY - 1))
        .ok_or_else(|| {
            BackendError::new(
                "readback ring capacity class overflows u64. Fix: split the readback before submitting it to the ring.",
            )
        })
}

#[inline]
pub(super) fn aligned_copy_len(byte_len: u64) -> Result<u64, BackendError> {
    crate::numeric::WGPU_NUMERIC.align_up_u64(byte_len, 4, 0, "readback byte length")
}

pub(super) fn readback_ring_slots_from_env() -> usize {
    let raw = std::env::var("VYRE_WGPU_READBACK_RING_SLOTS").ok();
    readback_ring_slots_from_raw(raw.as_deref())
}

pub(super) fn readback_ring_slots_from_raw(raw: Option<&str>) -> usize {
    let Some(raw) = raw else {
        return DEFAULT_RING_SLOTS;
    };
    let slots = match raw.parse::<usize>() {
        Ok(0) => {
            tracing::warn!(
                "VYRE_WGPU_READBACK_RING_SLOTS=0 is invalid for GPU readback rings; defaulting to {MIN_RING_SIZE}. Fix: set it to a positive integer between {MIN_RING_SIZE} and {MAX_RING_SIZE}, or unset it."
            );
            MIN_RING_SIZE
        }
        Ok(value) if value > MAX_RING_SIZE => {
            tracing::warn!(
                "VYRE_WGPU_READBACK_RING_SLOTS={value} exceeds the safe cap of {MAX_RING_SIZE}; clamping.
                Fix: set it to an integer between {MIN_RING_SIZE} and {MAX_RING_SIZE}, or unset it."
            );
            MAX_RING_SIZE
        }
        Ok(value) => value,
        Err(error) => {
            tracing::warn!(
                "VYRE_WGPU_READBACK_RING_SLOTS={raw:?} is invalid ({error:?}); defaulting to {DEFAULT_RING_SLOTS}. Fix: set it to a positive integer between {MIN_RING_SIZE} and {MAX_RING_SIZE}, or unset it."
            );
            DEFAULT_RING_SLOTS
        }
    };
    slots.clamp(MIN_RING_SIZE, MAX_RING_SIZE)
}
