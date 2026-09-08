//! The live buffers threaded between split segments: caller-borrowed inputs,
//! owned segment outputs, their refresh after each dispatch, and the
//! fingerprint that detects a converged accumulator.

use std::collections::HashMap;

use smallvec::SmallVec;
use vyre_foundation::ir::Ident;

use super::reserve_grid_sync_vec;
use super::segment_buffers::PlannedGridSyncSegment;
use crate::backend::{BackendError, OutputBuffers};

pub(super) enum GridSyncInput<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl GridSyncInput<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }

    fn refresh_from_output(&mut self, bytes: &mut Vec<u8>) -> Result<(), BackendError> {
        match self {
            Self::Borrowed(_) => {
                let mut owned = Vec::new();
                reserve_grid_sync_vec(&mut owned, bytes.len(), "grid-sync readwrite input")?;
                owned.extend_from_slice(bytes);
                *self = Self::Owned(owned);
            }
            Self::Owned(owned) => {
                std::mem::swap(owned, bytes);
            }
        }
        Ok(())
    }
}

pub(super) fn borrowed_grid_sync_inputs_by_name<'a>(
    segment: &PlannedGridSyncSegment,
    inputs: &'a HashMap<Ident, GridSyncInput<'a>>,
) -> Result<SmallVec<[&'a [u8]; 8]>, BackendError> {
    let mut borrowed = SmallVec::<[&[u8]; 8]>::new();
    borrowed
        .try_reserve(segment.input_names.len())
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve grid-sync borrowed input slices for {} segment input(s): {error}. Split the program into fewer grid-sync live buffers or run on a backend with native grid sync.",
                segment.input_names.len()
            ),
        })?;
    for name in &segment.input_names {
        let input = inputs.get(name).ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: grid-sync segment input `{name}` has no bytes from caller input or a prior segment output. Ensure every cross-segment read is written before the GridSync barrier."
            ),
        })?;
        borrowed.push(input.as_slice());
    }
    Ok(borrowed)
}

/// Exact state snapshot of evolving owned buffers threaded between grid-sync segments.
pub(super) fn snapshot_owned_accumulators(
    inputs: &HashMap<Ident, GridSyncInput<'_>>,
) -> HashMap<Ident, Vec<u8>> {
    let mut snapshot = HashMap::new();
    for (name, input) in inputs {
        if let GridSyncInput::Owned(bytes) = input {
            snapshot.insert(name.clone(), bytes.clone());
        }
    }
    snapshot
}

/// Exact byte comparison between previous and current evolving accumulator states.
pub(super) fn owned_accumulators_equal(
    prev: &HashMap<Ident, Vec<u8>>,
    current: &HashMap<Ident, GridSyncInput<'_>>,
) -> bool {
    let current_owned_count = current
        .values()
        .filter(|v| matches!(v, GridSyncInput::Owned(_)))
        .count();
    if prev.len() != current_owned_count {
        return false;
    }
    for (name, prev_bytes) in prev {
        match current.get(name) {
            Some(GridSyncInput::Owned(cur_bytes)) if cur_bytes == prev_bytes => {}
            _ => return false,
        }
    }
    true
}

pub(super) fn refresh_named_outputs<'a>(
    segment: &PlannedGridSyncSegment,
    outputs: &mut Vec<Vec<u8>>,
    inputs: &mut HashMap<Ident, GridSyncInput<'a>>,
) -> Result<(), BackendError> {
    if outputs.len() != segment.output_names.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: grid-sync split segment produced {} output slot(s) but the planned buffer map expected {}. Preserve segment output declaration order when dispatching split kernels.",
                outputs.len(),
                segment.output_names.len()
            ),
        });
    }
    for (name, bytes) in segment.output_names.iter().cloned().zip(outputs.iter_mut()) {
        match inputs.get_mut(&name) {
            Some(slot) => slot.refresh_from_output(bytes)?,
            None => {
                let mut owned = GridSyncInput::Owned(Vec::new());
                owned.refresh_from_output(bytes)?;
                inputs.insert(name, owned);
            }
        }
    }
    for output in outputs {
        output.clear();
    }
    Ok(())
}

pub(super) fn collect_final_named_outputs<'a>(
    final_output_names: &[Ident],
    inputs: &mut HashMap<Ident, GridSyncInput<'a>>,
    outputs: &mut OutputBuffers,
) -> Result<(), BackendError> {
    let mut final_outputs = Vec::new();
    reserve_grid_sync_vec(
        &mut final_outputs,
        final_output_names.len(),
        "grid-sync final named outputs",
    )?;
    for name in final_output_names {
        let output = inputs
            .remove(name)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: grid-sync final output `{name}` was not produced by any split segment."
                ),
            })?;
        match output {
            GridSyncInput::Owned(bytes) => final_outputs.push(bytes),
            GridSyncInput::Borrowed(bytes) => {
                let mut owned = Vec::new();
                reserve_grid_sync_vec(&mut owned, bytes.len(), "grid-sync borrowed final output")?;
                owned.extend_from_slice(bytes);
                final_outputs.push(owned);
            }
        }
    }
    crate::replace_output_buffers_preserving_slots(final_outputs, outputs);
    Ok(())
}

// Inline: covers `GridSyncInput` and the allocation reuse `refresh_named_outputs`
// depends on, neither of which an integration test can name.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_sync::test_programs::buffer;
    use vyre_foundation::ir::{Node, Program};

    /// The ReadWrite slot takes the backend's output allocation instead of copying.
    ///
    /// WHY: the segment loop refreshes every live buffer after each dispatch, so a
    /// copy per segment is a copy per iteration of the whole fixpoint. The first
    /// refresh promotes a borrowed slot to owned; every later one swaps the two
    /// allocations, which is why the assertions compare pointers rather than only
    /// bytes.
    #[test]
    fn refreshing_a_named_output_takes_the_backend_allocation_instead_of_copying() {
        let name = Ident::from("acc");
        let segment = PlannedGridSyncSegment {
            program: Program::wrapped(vec![buffer()], [1, 1, 1], vec![Node::Return]),
            input_names: vec![name.clone()],
            output_names: vec![name.clone()],
        };
        let initial = [1u8, 0, 0, 0];
        let mut inputs = HashMap::new();
        inputs.insert(name.clone(), GridSyncInput::Borrowed(initial.as_slice()));
        let mut outputs = vec![Vec::with_capacity(8)];
        let output_ptr = outputs[0].as_ptr() as usize;
        outputs[0].extend_from_slice(&[2, 0, 0, 0]);

        refresh_named_outputs(&segment, &mut outputs, &mut inputs)
            .expect("Fix: borrowed promotion storage must fit the refreshed output");

        let first_owned_ptr = match &inputs[&name] {
            GridSyncInput::Owned(bytes) => {
                assert_eq!(bytes, &[2, 0, 0, 0]);
                bytes.as_ptr() as usize
            }
            GridSyncInput::Borrowed(_) => panic!("ReadWrite input must become owned after refresh"),
        };
        assert_eq!(outputs[0].as_ptr() as usize, output_ptr);
        assert!(outputs[0].is_empty());

        outputs[0].extend_from_slice(&[3, 0, 0, 0]);
        let second_output_ptr = outputs[0].as_ptr() as usize;
        refresh_named_outputs(&segment, &mut outputs, &mut inputs)
            .expect("Fix: owned storage must be reused by the refreshed output");

        match &inputs[&name] {
            GridSyncInput::Owned(bytes) => {
                assert_eq!(bytes, &[3, 0, 0, 0]);
                assert_eq!(
                    bytes.as_ptr() as usize,
                    second_output_ptr,
                    "owned ReadWrite input should take the backend output allocation instead of copying"
                );
            }
            GridSyncInput::Borrowed(_) => panic!("ReadWrite input must remain owned"),
        }
        assert_eq!(
            outputs[0].as_ptr() as usize,
            first_owned_ptr,
            "backend output slot should receive the previous owned input allocation for reuse"
        );
    }
    #[test]
    fn exact_accumulator_state_comparison_covers_adversarial_cases() {
        let mut inputs_a = HashMap::new();
        inputs_a.insert(Ident::from("buf1"), GridSyncInput::Owned(vec![1, 2, 3, 4]));
        inputs_a.insert(
            Ident::from("buf2"),
            GridSyncInput::Owned(vec![0x7F, 0xC0, 0x00, 0x01]),
        ); // Specific NaN payload
        inputs_a.insert(Ident::from("zero_buf"), GridSyncInput::Owned(Vec::new()));

        let snapshot = snapshot_owned_accumulators(&inputs_a);
        assert!(
            owned_accumulators_equal(&snapshot, &inputs_a),
            "identical map must be equal"
        );

        // Reordered map insertion with identical bytes
        let mut inputs_reordered = HashMap::new();
        inputs_reordered.insert(Ident::from("zero_buf"), GridSyncInput::Owned(Vec::new()));
        inputs_reordered.insert(
            Ident::from("buf2"),
            GridSyncInput::Owned(vec![0x7F, 0xC0, 0x00, 0x01]),
        );
        inputs_reordered.insert(Ident::from("buf1"), GridSyncInput::Owned(vec![1, 2, 3, 4]));
        assert!(
            owned_accumulators_equal(&snapshot, &inputs_reordered),
            "reordered map with identical contents must be equal"
        );

        // Single bit change in NaN payload
        let mut inputs_nan_drift = HashMap::new();
        inputs_nan_drift.insert(Ident::from("buf1"), GridSyncInput::Owned(vec![1, 2, 3, 4]));
        inputs_nan_drift.insert(
            Ident::from("buf2"),
            GridSyncInput::Owned(vec![0x7F, 0xC0, 0x00, 0x02]),
        ); // 1 bit drift
        inputs_nan_drift.insert(Ident::from("zero_buf"), GridSyncInput::Owned(Vec::new()));
        assert!(
            !owned_accumulators_equal(&snapshot, &inputs_nan_drift),
            "NaN payload drift must be detected by exact comparison"
        );

        // Buffer length drift
        let mut inputs_len_drift = HashMap::new();
        inputs_len_drift.insert(
            Ident::from("buf1"),
            GridSyncInput::Owned(vec![1, 2, 3, 4, 5]),
        );
        inputs_len_drift.insert(
            Ident::from("buf2"),
            GridSyncInput::Owned(vec![0x7F, 0xC0, 0x00, 0x01]),
        );
        inputs_len_drift.insert(Ident::from("zero_buf"), GridSyncInput::Owned(Vec::new()));
        assert!(
            !owned_accumulators_equal(&snapshot, &inputs_len_drift),
            "length drift must be detected"
        );
    }
}
