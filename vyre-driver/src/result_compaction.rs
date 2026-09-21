//! Backend-neutral compact result readback planning.

use crate::accounting::{
    checked_add_u64_count as checked_add, checked_add_usize_count as checked_add_usize,
    checked_sub_u64_count as checked_sub, ArithmeticOverflow,
};
use crate::numeric::BackendNumericPolicy;
use crate::reservation_policy::{
    caller_owned_index_scratch, reserved_typed_vec as reserved_vec,
    storage_reserve_failure_adapter, ReservationPolicy,
};

const RESULT_COMPACTION_RESERVATION: ReservationPolicy = ReservationPolicy::new(
    "result compaction",
    "shard result readback planning before launch",
);

const RESULT_COMPACTION_NUMERIC: BackendNumericPolicy =
    BackendNumericPolicy::new("result compaction");

/// One output slot before result compaction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultSlot {
    /// Stable output slot id.
    pub slot: u32,
    /// Meaningful bytes produced by the kernel.
    pub meaningful_bytes: u64,
    /// Allocated/readback capacity for the output slot.
    pub capacity_bytes: u64,
}

/// One compact readback record.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompactResultRecord {
    /// Source output slot id.
    pub slot: u32,
    /// Offset in the compact readback slab.
    pub compact_offset: u64,
    /// Meaningful bytes copied into the slab.
    pub bytes: u64,
}

/// Compact result readback plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultCompactionPlan {
    /// Records copied into the compact slab.
    pub compact_records: Vec<CompactResultRecord>,
    /// Output slots left as direct readback ranges.
    pub direct_slots: Vec<u32>,
    /// Total allocated/readback capacity across all output slots.
    pub full_capacity_bytes: u64,
    /// Total compact slab bytes.
    pub compact_bytes: u64,
    /// Total direct readback bytes.
    pub direct_bytes: u64,
    /// Total bytes actually selected for readback after compaction planning.
    pub selected_readback_bytes: u64,
    /// Bytes avoided compared with reading full output capacities.
    pub avoided_readback_bytes: u64,
    /// Avoided readback as floor basis points of full capacity.
    pub avoided_readback_basis_points: u32,
}

caller_owned_index_scratch! {
    /// Caller-owned scratch for repeated result-compaction planning.
    ResultCompactionScratch {
        key: u32,
        error: ResultCompactionError,
        reservation: RESULT_COMPACTION_RESERVATION,
        reserve_failed: storage_reserve_failed,
        seen_item: "scratch.ids",
        subject: "compaction",
        counted: "output-slot",
        ordering: "slot-ordering",
        reserve: try_reserve_slots,
        capacity: id_capacity,
    }
}

/// Result compaction errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResultCompactionError {
    /// Duplicate output slot id.
    DuplicateSlot {
        /// Duplicate slot.
        slot: u32,
    },
    /// Meaningful bytes exceed allocated slot capacity.
    MeaningfulExceedsCapacity {
        /// Output slot.
        slot: u32,
        /// Meaningful bytes.
        meaningful_bytes: u64,
        /// Slot capacity.
        capacity_bytes: u64,
    },
    /// Byte arithmetic overflowed.
    ByteCountOverflow {
        /// Field being computed.
        field: &'static str,
    },
    /// Scratch or result-vector storage reservation failed before launch planning.
    StorageReserveFailed {
        /// Field being reserved.
        field: &'static str,
        /// Requested total capacity.
        requested: usize,
        /// Allocator failure details.
        message: String,
    },
}

impl ArithmeticOverflow for ResultCompactionError {
    fn arithmetic_overflow(field: &'static str) -> Self {
        Self::ByteCountOverflow { field }
    }
}

impl std::fmt::Display for ResultCompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateSlot { slot } => write!(
                f,
                "result compaction received duplicate output slot {slot}. Fix: assign unique output slots before readback planning."
            ),
            Self::MeaningfulExceedsCapacity {
                slot,
                meaningful_bytes,
                capacity_bytes,
            } => write!(
                f,
                "result slot {slot} has meaningful_bytes={meaningful_bytes} above capacity_bytes={capacity_bytes}. Fix: compute compact result sizes before dispatch readback."
            ),
            Self::ByteCountOverflow { field } => write!(
                f,
                "result compaction overflowed while computing {field}. Fix: shard compact result readback before launch."
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "result compaction failed to reserve {field} for {requested} entries: {message}. Fix: shard result readback planning before launch."
            ),
        }
    }
}

impl std::error::Error for ResultCompactionError {}

/// Plan compact readback for small outputs.
///
/// # Errors
///
/// Returns [`ResultCompactionError`] when slots are invalid, byte accounting
/// overflows, or result storage cannot be reserved.
pub fn plan_result_compaction(
    slots: &[ResultSlot],
    max_compact_record_bytes: u64,
) -> Result<ResultCompactionPlan, ResultCompactionError> {
    let mut scratch = ResultCompactionScratch::try_with_capacity(slots.len())?;
    plan_result_compaction_with_scratch(slots, max_compact_record_bytes, &mut scratch)
}

/// Plan compact readback using caller-owned temporary storage.
///
/// # Errors
///
/// Returns [`ResultCompactionError`] when slots are invalid, byte accounting
/// overflows, or result storage cannot be reserved.
pub fn plan_result_compaction_with_scratch(
    slots: &[ResultSlot],
    max_compact_record_bytes: u64,
    scratch: &mut ResultCompactionScratch,
) -> Result<ResultCompactionPlan, ResultCompactionError> {
    scratch.index_scratch.clear();
    scratch.try_reserve_slots(slots.len())?;
    let mut full_capacity_bytes = 0_u64;
    let mut compact_record_count = 0usize;
    let mut direct_slot_count = 0usize;

    for (index, slot) in slots.iter().copied().enumerate() {
        if !scratch.index_scratch.insert_seen(slot.slot) {
            return Err(ResultCompactionError::DuplicateSlot { slot: slot.slot });
        }
        if slot.meaningful_bytes > slot.capacity_bytes {
            return Err(ResultCompactionError::MeaningfulExceedsCapacity {
                slot: slot.slot,
                meaningful_bytes: slot.meaningful_bytes,
                capacity_bytes: slot.capacity_bytes,
            });
        }
        full_capacity_bytes = checked_add(
            full_capacity_bytes,
            slot.capacity_bytes,
            "full capacity bytes",
        )?;
        if slot.meaningful_bytes != 0 {
            if slot.meaningful_bytes <= max_compact_record_bytes {
                compact_record_count =
                    checked_add_usize(compact_record_count, 1, "compact record count")?;
            } else {
                direct_slot_count = checked_add_usize(direct_slot_count, 1, "direct slot count")?;
            }
        }
        scratch.index_scratch.push_index(index);
    }
    scratch
        .index_scratch
        .sort_indices_unstable_by_key_if_needed(|index| slots[index].slot);

    let mut compact_records = reserved_result_vec(compact_record_count, "compact_records")?;
    let mut direct_slots = reserved_result_vec(direct_slot_count, "direct_slots")?;
    let mut compact_bytes = 0_u64;
    let mut direct_bytes = 0_u64;

    for &index in scratch.index_scratch.ordered_indices() {
        let slot = slots[index];
        if slot.meaningful_bytes == 0 {
            continue;
        }
        if slot.meaningful_bytes <= max_compact_record_bytes {
            compact_records.push(CompactResultRecord {
                slot: slot.slot,
                compact_offset: compact_bytes,
                bytes: slot.meaningful_bytes,
            });
            compact_bytes = checked_add(compact_bytes, slot.meaningful_bytes, "compact bytes")?;
        } else {
            direct_slots.push(slot.slot);
            direct_bytes = checked_add(direct_bytes, slot.meaningful_bytes, "direct bytes")?;
        }
    }

    let selected_readback_bytes =
        checked_add(compact_bytes, direct_bytes, "selected readback bytes")?;
    let avoided_readback_bytes = checked_sub(
        full_capacity_bytes,
        selected_readback_bytes,
        "avoided readback bytes",
    )?;

    Ok(ResultCompactionPlan {
        compact_records,
        direct_slots,
        full_capacity_bytes,
        compact_bytes,
        direct_bytes,
        selected_readback_bytes,
        avoided_readback_bytes,
        avoided_readback_basis_points: RESULT_COMPACTION_NUMERIC.ratio_basis_points_u64(
            avoided_readback_bytes,
            full_capacity_bytes,
            0,
            "result-compaction avoided-readback",
        ),
    })
}

fn reserved_result_vec<T>(
    capacity: usize,
    field: &'static str,
) -> Result<Vec<T>, ResultCompactionError> {
    reserved_vec(
        RESULT_COMPACTION_RESERVATION,
        capacity,
        field,
        storage_reserve_failed,
    )
}

storage_reserve_failure_adapter!(ResultCompactionError);
