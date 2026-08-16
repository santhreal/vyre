//! Backend-neutral fallible allocation reservation helpers.
//!
//! Concrete backends still own domain wording, but the arithmetic for
//! "reserve additional" and "reserve up to target capacity" must not drift.

use std::collections::{HashMap, HashSet, TryReserveError};
use std::hash::{BuildHasher, Hash};

use smallvec::{Array, SmallVec};

use crate::BackendError;

fn reserve_error(
    context: &'static str,
    requested: usize,
    item: &'static str,
    source: impl std::fmt::Display,
    fix: &'static str,
) -> BackendError {
    BackendError::new(format!(
        "{context} could not reserve {requested} {item}(s): {source}. Fix: {fix}."
    ))
}

/// Reserve additional capacity for a [`Vec`] without changing its length.
///
/// # Errors
///
/// Returns [`BackendError`] when allocation fails.
pub fn reserve_vec_additional<T>(
    vec: &mut Vec<T>,
    additional: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError> {
    vec.try_reserve(additional)
        .map_err(|source| reserve_error(context, additional, item, source, fix))
}

/// Ensure a [`Vec`] can hold `target_capacity` items without changing length,
/// returning the standard allocation error for domain-specific adapters.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_vec_to_capacity<T>(
    vec: &mut Vec<T>,
    target_capacity: usize,
) -> Result<(), TryReserveError> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(vec, target_capacity)
}

/// Ensure a [`Vec`] can hold `target_capacity` items without changing length.
///
/// Uses `target_capacity - len`, not `target_capacity - capacity`, so a vector
/// that was cleared after holding many elements still grows to the requested
/// target if its retained capacity is too small.
///
/// # Errors
///
/// Returns [`BackendError`] when allocation fails.
pub fn reserve_vec_to_capacity<T>(
    vec: &mut Vec<T>,
    target_capacity: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError> {
    try_reserve_vec_to_capacity(vec, target_capacity)
        .map_err(|source| reserve_error(context, target_capacity, item, source, fix))
}

/// Reserve additional capacity for a [`SmallVec`] without changing its length.
///
/// # Errors
///
/// Returns [`BackendError`] when allocation fails.
pub fn reserve_smallvec_additional<A>(
    vec: &mut SmallVec<A>,
    additional: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError>
where
    A: Array,
{
    vec.try_reserve(additional)
        .map_err(|source| reserve_error(context, additional, item, source, fix))
}

/// Ensure a [`SmallVec`] can hold `target_capacity` items without changing
/// length.
///
/// # Errors
///
/// Returns [`BackendError`] when allocation fails.
pub fn reserve_smallvec_to_capacity<A>(
    vec: &mut SmallVec<A>,
    target_capacity: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError>
where
    A: Array,
{
    vyre_foundation::allocation::try_reserve_smallvec_to_capacity(vec, target_capacity)
        .map_err(|source| reserve_error(context, target_capacity, item, source, fix))
}

/// Ensure a [`HashMap`] can hold `target_capacity` entries without changing
/// length, returning the standard allocation error for domain-specific
/// adapters.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_hash_map_to_capacity<K, V, S>(
    map: &mut HashMap<K, V, S>,
    target_capacity: usize,
) -> Result<(), TryReserveError>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    vyre_foundation::allocation::try_reserve_hash_map_to_capacity(map, target_capacity)
}

/// Ensure a [`HashSet`] can hold `target_capacity` entries without changing
/// length, returning the standard allocation error for domain-specific
/// adapters.
///
/// # Errors
///
/// Returns [`TryReserveError`] when allocation fails.
pub fn try_reserve_hash_set_to_capacity<T, S>(
    set: &mut HashSet<T, S>,
    target_capacity: usize,
) -> Result<(), TryReserveError>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    vyre_foundation::allocation::try_reserve_hash_set_to_capacity(set, target_capacity)
}

/// Ensure a [`HashMap`] can hold `target_capacity` entries without changing
/// length.
///
/// # Errors
///
/// Returns [`BackendError`] when allocation fails.
pub fn reserve_hash_map_to_capacity<K, V, S>(
    map: &mut HashMap<K, V, S>,
    target_capacity: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    try_reserve_hash_map_to_capacity(map, target_capacity)
        .map_err(|source| reserve_error(context, target_capacity, item, source, fix))
}

/// Ensure a [`HashSet`] can hold `target_capacity` entries without changing
/// length.
///
/// # Errors
///
/// Returns [`BackendError`] when allocation fails.
pub fn reserve_hash_set_to_capacity<T, S>(
    set: &mut HashSet<T, S>,
    target_capacity: usize,
    context: &'static str,
    item: &'static str,
    fix: &'static str,
) -> Result<(), BackendError>
where
    T: Eq + Hash,
    S: BuildHasher,
{
    try_reserve_hash_set_to_capacity(set, target_capacity)
        .map_err(|source| reserve_error(context, target_capacity, item, source, fix))
}
