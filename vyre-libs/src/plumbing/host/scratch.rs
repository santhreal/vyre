//! Shared fallible scratch reservation helpers.
//!
//! Two families live here, distinguished by what a caller can do with the
//! failure. Dispatch release paths reuse caller-owned buffers heavily and
//! surface a [`DispatchError`], so they use `reserve_vec`, `reserve_vec_capacity`
//! and friends. Host builders report the owning kernel and the scratch role as
//! a message the caller maps into its own error type, so they use `reserve_items`
//! and `reserve_items_with`.
//! Keeping both families here prevents each domain from growing its own
//! unchecked `Vec::reserve` variant and keeps allocation failures actionable.
//! Nothing here truncates or saturates on overflow.

#[cfg(all(test, feature = "device"))]
use std::collections::HashSet;
#[cfg(all(test, feature = "device"))]
use std::hash::{BuildHasher, Hash};
#[cfg(feature = "device")]
use vyre_foundation::program_dispatch::DispatchError;

/// Reserve additional items in a scratch vector with a standard, actionable
/// allocation diagnostic.
///
/// # Errors
///
/// Returns a message naming `owner`, `context`, and the allocator failure.
#[cfg(any(feature = "graph", feature = "math-kernels"))]
pub(crate) fn reserve_items<T>(
    buffer: &mut Vec<T>,
    additional: usize,
    owner: &str,
    context: &str,
) -> Result<(), String> {
    buffer.try_reserve(additional).map_err(|error| {
        format!(
            "Fix: {owner} could not reserve {additional} item(s) for {context}: {error}. Split the batch or reuse a smaller scratch buffer."
        )
    })
}

/// Reserve scratch and map the shared diagnostic into a domain-specific error
/// type.
///
/// # Errors
///
/// Returns the mapped allocation error when `Vec::try_reserve` fails.
#[cfg(any(feature = "graph", feature = "math-kernels"))]
pub(crate) fn reserve_items_with<T, E>(
    buffer: &mut Vec<T>,
    additional: usize,
    owner: &str,
    context: &str,
    map: impl FnOnce(String) -> E,
) -> Result<(), E> {
    reserve_items(buffer, additional, owner, context).map_err(map)
}

/// Grow `buffer` to hold at least `capacity` items.
///
/// # Errors
/// Returns the allocator's refusal rendered as a message.
#[cfg(feature = "device")]
pub(crate) fn try_reserve_vec_capacity<T>(
    buffer: &mut Vec<T>,
    capacity: usize,
) -> Result<(), String> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(buffer, capacity)
        .map_err(|error| error.to_string())
}

/// Reserve room for `additional` more items in `buffer`.
///
/// # Errors
/// Returns a [`DispatchError::BackendError`] naming `context` and the count.
#[cfg(feature = "device")]
pub(crate) fn reserve_vec<T>(
    buffer: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), DispatchError> {
    if additional == 0 {
        return Ok(());
    }
    buffer.try_reserve_exact(additional).map_err(|error| {
        DispatchError::BackendError(format!(
            "Fix: {context} could not reserve {additional} additional scratch slot(s): {error}. Split the dispatch window before retrying."
        ))
    })
}

/// Grow `buffer` to hold at least `capacity` items.
///
/// # Errors
/// Returns a [`DispatchError::BackendError`] naming `context` and the capacity.
#[cfg(feature = "device")]
pub(crate) fn reserve_vec_capacity<T>(
    buffer: &mut Vec<T>,
    capacity: usize,
    context: &'static str,
) -> Result<(), DispatchError> {
    try_reserve_vec_capacity(buffer, capacity).map_err(|message| {
        DispatchError::BackendError(format!(
            "Fix: {context} could not reserve scratch capacity for {capacity} item(s): {message}. Split the dispatch window before retrying."
        ))
    })
}

/// Reserve scratch capacity for `capacity` items, failing closed when the allocation is refused.
///
/// # Panics
/// Panics when the reservation fails. Continuing with a short buffer would let a pass
/// write past the scratch it believes it owns.
#[cfg(all(test, feature = "device"))]
pub(crate) fn reserve_vec_capacity_or_panic<T>(
    buffer: &mut Vec<T>,
    capacity: usize,
    context: &'static str,
) {
    // The name promises a panic on failure; the old body did `let _ = …`,
    // silently swallowing the reservation error (and discarding `context`)
    // a name/behavior incoherence and a silent fallback (Law 10). Honor the
    // contract: fail loud with context.
    if let Err(message) = try_reserve_vec_capacity(buffer, capacity) {
        panic!("{context} could not reserve scratch capacity for {capacity} item(s): {message}");
    }
}

/// Reserve room for `additional` more entries in `set`.
///
/// # Errors
/// Returns a [`DispatchError::BackendError`] when the target capacity overflows
/// or the allocator refuses it.
#[cfg(all(test, feature = "device"))]
pub(crate) fn reserve_hash_set<T, S>(
    set: &mut HashSet<T, S>,
    additional: usize,
    context: &'static str,
) -> Result<(), DispatchError>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    if additional == 0 {
        return Ok(());
    }
    let target_capacity = set.len().checked_add(additional).ok_or_else(|| {
        DispatchError::BackendError(format!(
            "Fix: {context} hash scratch reservation overflowed for {additional} additional slot(s). Split the dispatch window before retrying."
        ))
    })?;
    vyre_foundation::allocation::try_reserve_hash_set_to_capacity(set, target_capacity).map_err(|error| {
        DispatchError::BackendError(format!(
            "Fix: {context} could not reserve {additional} additional hash slot(s): {error}. Split the dispatch window before retrying."
        ))
    })
}

#[cfg(all(test, feature = "device"))]
mod dispatch_tests {
    use super::*;

    #[test]
    fn reserve_vec_capacity_reuses_existing_allocation() {
        let mut scratch = Vec::<u32>::with_capacity(8);
        reserve_vec_capacity(&mut scratch, 4, "frontier seed")
            .expect("Fix: scratch grow must reuse capacity; fall back to allocate on hostile zero-cap - existing capacity should be reused");
        assert_eq!(scratch.capacity(), 8);
    }

    #[test]
    fn reserve_vec_capacity_reports_context_on_overflow() {
        let mut scratch = Vec::<u8>::new();
        let err = reserve_vec_capacity(&mut scratch, usize::MAX, "huge frontier")
            .expect_err("oversized reservation should fail");
        let message = err.to_string();
        assert!(message.contains("huge frontier"));
        assert!(message.contains("Fix:"));
    }

    #[test]
    fn reserve_vec_additional_reports_context_on_overflow() {
        let mut scratch = Vec::<u8>::new();
        let err = reserve_vec(&mut scratch, usize::MAX, "huge additional frontier")
            .expect_err("oversized reservation should fail");
        let message = err.to_string();
        assert!(message.contains("huge additional frontier"));
        assert!(message.contains("Fix:"));
    }
}

#[cfg(all(test, any(feature = "graph", feature = "math-kernels")))]
mod owner_reported_tests {
    use super::{reserve_items, reserve_items_with};

    #[test]
    fn reserve_items_reuses_existing_capacity() {
        let mut scratch = Vec::<u32>::with_capacity(8);

        reserve_items(&mut scratch, 4, "test kernel", "frontier")
            .expect("existing capacity should satisfy the reservation without allocating");

        assert_eq!(scratch.capacity(), 8);
        assert!(scratch.is_empty());
    }

    #[test]
    fn reserve_items_reports_owner_and_context_on_capacity_overflow() {
        let mut scratch = Vec::<u8>::new();

        let err = reserve_items(
            &mut scratch,
            usize::MAX,
            "test kernel",
            "adversarial huge scratch",
        )
        .expect_err("usize::MAX reservation must fail without allocating");

        assert!(err.contains("test kernel"));
        assert!(err.contains("adversarial huge scratch"));
        assert!(err.contains("usize::MAX") || err.contains("capacity"));
    }

    #[test]
    fn reserve_items_with_preserves_domain_error_mapping() {
        #[derive(Debug, PartialEq, Eq)]
        struct DomainError(String);

        let mut scratch = Vec::<u8>::new();
        let err = reserve_items_with(
            &mut scratch,
            usize::MAX,
            "mapped kernel",
            "mapped scratch",
            DomainError,
        )
        .expect_err("usize::MAX reservation must fail without allocating");

        assert!(err.0.contains("mapped kernel"));
        assert!(err.0.contains("mapped scratch"));
    }
}
