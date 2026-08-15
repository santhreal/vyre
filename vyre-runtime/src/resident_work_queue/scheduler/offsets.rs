use super::{priority, PRIORITY_LEVELS, PRIORITY_OFFSETS_BASE};
use crate::PipelineError;

const PRIORITY_LEVELS_USIZE: usize = 5;
const PRIORITY_OFFSETS_WITH_SENTINEL: usize = PRIORITY_LEVELS_USIZE + 1;

/// Encode default priority partition offsets into a fixed array without allocation.
#[must_use]
pub fn default_priority_offsets_array(total_slots: u32) -> [u32; PRIORITY_OFFSETS_WITH_SENTINEL] {
    let mut offsets = [0u32; PRIORITY_OFFSETS_WITH_SENTINEL];
    write_default_priority_offsets_array(total_slots, &mut offsets);
    offsets
}

fn write_default_priority_offsets_array(
    total_slots: u32,
    offsets: &mut [u32; PRIORITY_OFFSETS_WITH_SENTINEL],
) {
    let base_per_pri = total_slots / PRIORITY_LEVELS;
    let remainder = total_slots % PRIORITY_LEVELS;
    let mut cursor = 0u32;
    for pri in 0..PRIORITY_LEVELS_USIZE {
        offsets[pri] = cursor;
        let pri_u32 = pri as u32;
        let size = base_per_pri
            + if pri_u32 == priority::NORMAL {
                remainder
            } else {
                0
            };
        cursor = cursor.saturating_add(size);
    }
    offsets[PRIORITY_LEVELS_USIZE] = cursor;
}

/// Write default priority partition offsets into an encoded control buffer.
///
/// # Errors
///
/// Returns [`PipelineError::QueueFull`] when the provided control buffer is too
/// short or not aligned to u32 words.
pub fn write_default_priority_offsets(
    control_bytes: &mut [u8],
    total_slots: u32,
) -> Result<(), PipelineError> {
    if control_bytes.len() % 4 != 0 {
        return Err(PipelineError::QueueFull {
            queue: "submission",
            fix: "control buffer byte length is not 4-byte aligned; rebuild it with Megakernel::encode_control",
        });
    }
    let mut offsets = [0u32; PRIORITY_OFFSETS_WITH_SENTINEL];
    write_default_priority_offsets_array(total_slots, &mut offsets);
    for (i, value) in offsets.iter().enumerate() {
        let word_idx = priority_offsets_base_usize()?.checked_add(i).ok_or(
            PipelineError::QueueFull {
                queue: "submission",
                fix: "priority-offset control word index overflowed usize; keep control ABI constants bounded",
            },
        )?;
        let start = word_idx.checked_mul(4).ok_or(PipelineError::QueueFull {
            queue: "submission",
            fix: "priority-offset byte index overflowed usize; keep control ABI constants bounded",
        })?;
        let end = start.checked_add(4).ok_or(PipelineError::QueueFull {
            queue: "submission",
            fix: "priority-offset byte index overflowed usize; keep control ABI constants bounded",
        })?;
        let dst = control_bytes.get_mut(start..end).ok_or(PipelineError::QueueFull {
            queue: "submission",
            fix: "control buffer is too small for priority partition offsets; rebuild it with Megakernel::encode_control",
        })?;
        dst.copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn priority_offsets_base_usize() -> Result<usize, PipelineError> {
    usize::try_from(PRIORITY_OFFSETS_BASE).map_err(|_| PipelineError::QueueFull {
        queue: "submission",
        fix: "priority-offset base word cannot fit host usize; keep control ABI constants bounded",
    })
}
