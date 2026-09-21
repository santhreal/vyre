//! Resident input and output slot indexes must form a dense permutation of
//! `0..len`. This rejects the duplicate, sparse, and length-mismatch cases with
//! the message the resident path needs.

use smallvec::SmallVec;
use vyre_driver::BackendError;

use crate::backend::ordering::{classify_dense_permutation, DensePermutationDefect};
use crate::backend::staging_reserve::reserve_smallvec;

fn validate_dense_resident_indices<I>(
    indices: I,
    expected_len: usize,
    context: &'static str,
    index_kind: &'static str,
    rebuild_action: &'static str,
) -> Result<(), BackendError>
where
    I: IntoIterator<Item = usize>,
{
    // Callers sort before validating (resident_dispatch::{borrowed,batch,
    // async_dispatch} all `sort_unstable_by_key_if_needed` first); the shared
    // classifier is defined on sorted slot order. Collect into the fallibly
    // reserved staging buffer, then delegate the dense-permutation invariant to
    // the single backend-neutral owner and format the resident-specific message
    // from the classified defect (one algorithm, no per-subsystem fork).
    let iter = indices.into_iter();
    let mut sorted = SmallVec::<[usize; 8]>::new();
    reserve_smallvec(
        &mut sorted,
        iter.size_hint().0,
        "CUDA resident dense index validation",
    )?;
    sorted.extend(iter);
    match classify_dense_permutation(&sorted, expected_len) {
        Ok(()) => Ok(()),
        Err(DensePermutationDefect::Duplicate { index, slot }) => {
            Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} found a duplicate {index_kind} index {index} at sorted {index_kind} slot {slot}; duplicate {index_kind} indexes alias one logical slot onto two descriptors. Rebuild the binding plan with dense unique {index_kind} indexes 0..{expected_len} before {rebuild_action}.",
                ),
            })
        }
        Err(DensePermutationDefect::Sparse { index, slot }) => {
            Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} resolved sparse {index_kind} index {index} at sorted {index_kind} slot {slot}; expected dense {index_kind} indexes 0..{expected_len}. Rebuild the binding plan before {rebuild_action}.",
                ),
            })
        }
        Err(DensePermutationDefect::LengthMismatch { resolved, expected }) => {
            Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA {context} resolved {resolved} {index_kind} index(es); expected {expected}. Rebuild the binding plan before {rebuild_action}.",
                ),
            })
        }
    }
}

pub(crate) fn validate_dense_resident_output_indices<I>(
    output_indices: I,
    expected_len: usize,
    context: &'static str,
) -> Result<(), BackendError>
where
    I: IntoIterator<Item = usize>,
{
    validate_dense_resident_indices(
        output_indices,
        expected_len,
        context,
        "output",
        "resident readback",
    )
}

pub(crate) fn validate_dense_resident_input_indices<I>(
    input_indices: I,
    expected_len: usize,
    context: &'static str,
) -> Result<(), BackendError>
where
    I: IntoIterator<Item = usize>,
{
    validate_dense_resident_indices(
        input_indices,
        expected_len,
        context,
        "input",
        "borrowed fallback launch",
    )
}
