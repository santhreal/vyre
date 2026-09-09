//! Shared fallible scratch reservation helpers.

use vyre_megakernel::SemanticExecutionError;

/// Reserve additional items in a scratch vector with a standard, actionable
/// allocation diagnostic.
pub fn reserve_items<T>(
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
pub fn reserve_items_with<T, E>(
    buffer: &mut Vec<T>,
    additional: usize,
    owner: &str,
    context: &str,
    map: impl FnOnce(String) -> E,
) -> Result<(), E> {
    reserve_items(buffer, additional, owner, context).map_err(map)
}

/// Grow `buffer` to hold at least `capacity` items.
pub fn try_reserve_vec_capacity<T>(buffer: &mut Vec<T>, capacity: usize) -> Result<(), String> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(buffer, capacity)
        .map_err(|error| error.to_string())
}

/// Reserve room for `additional` more items in `buffer`.
pub fn reserve_vec<T>(
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
pub fn reserve_vec_capacity<T>(
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
pub fn reserve_vec_capacity_or_panic<T>(
    buffer: &mut Vec<T>,
    capacity: usize,
    context: &'static str,
) {
    if let Err(message) = try_reserve_vec_capacity(buffer, capacity) {
        panic!("{context} could not reserve scratch capacity for {capacity} item(s): {message}");
    }
}
