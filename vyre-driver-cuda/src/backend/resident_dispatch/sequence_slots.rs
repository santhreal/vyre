//! Per-step slot bookkeeping for a resident dispatch sequence: borrowing the
//! caller's output vectors, and coalescing the fills an upload already covers.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use vyre_driver::BackendError;

use crate::backend::resident::CudaResidentBuffer;
use crate::backend::staging_reserve::{reserve_hash_set, reserve_smallvec, resize_vec_slots};

pub(crate) fn borrow_resident_sequence_output_slots(
    outputs: &mut Vec<Vec<u8>>,
    slot_count: usize,
) -> Result<SmallVec<[&mut Vec<u8>; 8]>, BackendError> {
    resize_vec_slots(outputs, slot_count, "resident sequence output slots")?;
    let mut borrowed_outputs = SmallVec::<[&mut Vec<u8>; 8]>::new();
    reserve_smallvec(
        &mut borrowed_outputs,
        outputs.len(),
        "resident sequence borrowed output slots",
    )?;
    borrowed_outputs.extend(outputs.iter_mut());
    Ok(borrowed_outputs)
}

pub(crate) fn prepare_resident_sequence_fills(
    fills: &[(CudaResidentBuffer, u8)],
    uploads: &[(CudaResidentBuffer, &[u8])],
) -> Result<SmallVec<[(CudaResidentBuffer, u8); 8]>, BackendError> {
    let mut uploaded_handles = FxHashSet::<CudaResidentBuffer>::default();
    if !uploads.is_empty() {
        reserve_hash_set(
            &mut uploaded_handles,
            uploads.len(),
            "resident sequence upload handle set",
        )?;
        uploaded_handles.extend(uploads.iter().map(|&(handle, _)| handle));
    }

    let mut effective = SmallVec::<[(CudaResidentBuffer, u8); 8]>::new();
    reserve_smallvec(
        &mut effective,
        fills.len(),
        "resident sequence effective fills",
    )?;

    let mut effective_indices = FxHashMap::<CudaResidentBuffer, usize>::default();
    effective_indices
        .try_reserve(fills.len())
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident sequence fill index could not reserve {} handle slot(s): {error}.",
                fills.len()
            ),
        })?;

    for &(handle, value) in fills {
        if !uploaded_handles.is_empty() && uploaded_handles.contains(&handle) {
            continue;
        }
        if let Some(&index) = effective_indices.get(&handle) {
            let Some(existing) = effective.get_mut(index) else {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident sequence fill index for handle {} pointed at stale effective fill slot {index} after {} slot(s) were prepared. Rebuild duplicate-fill coalescing before launching the resident sequence.",
                        handle.handle,
                        effective.len()
                    ),
                });
            };
            existing.1 = value;
            continue;
        }
        effective_indices.insert(handle, effective.len());
        effective.push((handle, value));
    }

    Ok(effective)
}
