//! Shared fallible scratch reservation helpers.
//!
//! Semantic execution paths reuse caller-owned buffers heavily and surface a
//! [`SemanticExecutionError`], so they use `reserve_vec`,
//! `reserve_vec_capacity`, and related helpers. Host builders map the owning
//! kernel and scratch role through `reserve_items` and `reserve_items_with`.
//! Keeping both families here prevents each domain from growing its own
//! unchecked `Vec::reserve` variant and keeps allocation failures actionable.
//! Nothing here truncates or saturates on overflow.

#[cfg(feature = "device")]
use vyre_megakernel::SemanticExecutionError;

/// Reserve additional items in a scratch vector with a standard, actionable
/// allocation diagnostic.
///
/// Gated to match its callers: the `graph` dispatch plans reserve through this
/// on the production path, and the `math-kernels` CPU parity oracles reserve
/// through it only in a test build.
///
/// # Errors
///
/// Returns a message naming `owner`, `context`, and the allocator failure.
#[cfg(any(feature = "graph", all(test, feature = "math-kernels")))]
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
#[cfg(any(feature = "graph", all(test, feature = "math-kernels")))]
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
// The callers are `analysis`, `encoding` and `solvers` builders, plus the
// panicking wrapper below under `scheduling`. `device` admitted this on a
// feature set that compiles none of them, where it was dead code.
#[cfg(any(
    feature = "analysis",
    feature = "encoding",
    feature = "solvers",
    all(test, feature = "scheduling")
))]
pub(crate) fn try_reserve_vec_capacity<T>(
    buffer: &mut Vec<T>,
    capacity: usize,
) -> Result<(), String> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(buffer, capacity)
        .map_err(|error| error.to_string())
}

/// Reserve room for `additional` more items in `buffer`.
///
/// `graph::dispatch::csr_forward_or_changed` is the sole caller, so this takes
/// `graph-dispatch` rather than the `graph` gate its `reserve_items` siblings
/// carry.
///
/// # Errors
/// Returns a [`SemanticExecutionError::Backend`] naming `context` and the count.
#[cfg(feature = "graph-dispatch")]
pub(crate) fn reserve_vec<T>(
    buffer: &mut Vec<T>,
    additional: usize,
    context: &'static str,
) -> Result<(), SemanticExecutionError> {
    if additional == 0 {
        return Ok(());
    }
    buffer.try_reserve_exact(additional).map_err(|error| {
        SemanticExecutionError::Backend(format!(
            "Fix: {context} could not reserve {additional} additional scratch slot(s): {error}. Split the dispatch window before retrying."
        ))
    })
}

/// Grow `buffer` to hold at least `capacity` items.
///
/// # Errors
/// Returns a [`SemanticExecutionError::Backend`] naming `context` and the capacity.
#[cfg(any(feature = "analysis", feature = "encoding", feature = "solvers"))]
pub(crate) fn reserve_vec_capacity<T>(
    buffer: &mut Vec<T>,
    capacity: usize,
    context: &'static str,
) -> Result<(), SemanticExecutionError> {
    try_reserve_vec_capacity(buffer, capacity).map_err(|message| {
        SemanticExecutionError::Backend(format!(
            "Fix: {context} could not reserve scratch capacity for {capacity} item(s): {message}. Split the dispatch window before retrying."
        ))
    })
}

/// Reserve scratch capacity for `capacity` items, failing closed when the allocation is refused.
///
/// # Panics
/// Panics when the reservation fails. Continuing with a short buffer would let a pass
/// write past the scratch it believes it owns.
// `scheduling` rather than `device`: the eviction-set inversion in
// `scheduling::submodular_cache_eviction` is the only caller that wants a panic.
#[cfg(all(test, feature = "scheduling"))]
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

#[cfg(all(
    test,
    any(feature = "analysis", feature = "encoding", feature = "solvers")
))]
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

    /// `reserve_vec` rides `graph-dispatch`, so this case does too; without the
    /// predicate a `device` build without `graph-dispatch` cannot compile it.
    #[cfg(feature = "graph-dispatch")]
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
