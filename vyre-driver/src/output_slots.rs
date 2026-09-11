//! Backend-neutral fallible output-slot vector management.
//!
//! Every device backend resizes caller-owned output slot vectors on hot
//! dispatch paths. The policy is identical: preserve existing
//! slots where possible, grow fallibly, initialize new slots from a caller
//! factory, and truncate stale slots. Keeping that policy here prevents
//! backend-local allocation drift.

use crate::BackendError;

/// Reserve enough capacity for a target vector length without mutating length.
///
/// # Errors
///
/// Returns [`BackendError`] when the vector cannot reserve the additional
/// capacity required for `target_len`.
pub fn reserve_vec_exact_for_len<T>(
    vec: &mut Vec<T>,
    target_len: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError> {
    if vec.len() < target_len {
        let additional = target_len - vec.len();
        vec.try_reserve_exact(additional).map_err(|source| {
            BackendError::new(format!(
                "{context} could not reserve {additional} additional {item}(s) for target length {target_len}: {source}. Fix: {fix}."
            ))
        })?;
    }
    Ok(())
}

/// Resize a vector while preserving the existing prefix and growing fallibly.
///
/// # Errors
///
/// Returns [`BackendError`] when growth to `len` cannot reserve memory.
pub fn resize_vec_with<T, F>(
    vec: &mut Vec<T>,
    len: usize,
    make: F,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError>
where
    F: FnMut() -> T,
{
    if vec.len() < len {
        reserve_vec_exact_for_len(vec, len, context, item, fix)?;
        vec.resize_with(len, make);
    } else {
        vec.truncate(len);
    }
    Ok(())
}

/// Ensure a `Vec<Vec<T>>` has at least `slot_count` output slots.
///
/// Existing slot buffers are preserved; new slots are empty vectors.
///
/// # Errors
///
/// Returns [`BackendError`] when the outer slot vector cannot grow.
pub fn ensure_vec_slots_at_least<T>(
    slots: &mut Vec<Vec<T>>,
    slot_count: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError> {
    if slots.len() < slot_count {
        reserve_vec_exact_for_len(slots, slot_count, context, item, fix)?;
        slots.resize_with(slot_count, Vec::new);
    }
    Ok(())
}

/// Resize a `Vec<Vec<T>>` to exactly `slot_count` output slots.
///
/// Existing slot buffers are preserved up to the new length; stale trailing
/// slots are dropped.
///
/// # Errors
///
/// Returns [`BackendError`] when the outer slot vector cannot grow.
pub fn resize_vec_slots<T>(
    slots: &mut Vec<Vec<T>>,
    slot_count: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError> {
    resize_vec_with(slots, slot_count, Vec::new, context, item, fix)
}

/// Clear every inner output buffer without changing the slot count.
pub fn clear_vec_slots<T>(slots: &mut [Vec<T>]) {
    for slot in slots {
        slot.clear();
    }
}
