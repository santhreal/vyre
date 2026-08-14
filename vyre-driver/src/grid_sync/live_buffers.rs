//! The live buffers threaded between split segments: caller-borrowed inputs,
//! owned segment outputs, their refresh after each dispatch, and the
//! fingerprint that detects a converged accumulator.

use std::collections::HashMap;

use smallvec::SmallVec;
use vyre_foundation::ir::{Ident, Program};

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

fn borrowed_grid_sync_inputs<'a>(
    inputs: &'a [GridSyncInput<'a>],
) -> Result<SmallVec<[&'a [u8]; 8]>, BackendError> {
    let mut borrowed = SmallVec::<[&[u8]; 8]>::new();
    borrowed.try_reserve(inputs.len()).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve grid-sync borrowed input slices for {} input(s): {error}. Split the program into fewer grid-sync live buffers or run on a backend with native grid sync.",
                inputs.len()
            ),
        }
    })?;
    borrowed.extend(inputs.iter().map(GridSyncInput::as_slice));
    Ok(borrowed)
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

/// Order-independent fingerprint of the EVOLVING accumulator state threaded
/// between grid-sync segments.
///
/// Only `Owned` entries are hashed: a `Borrowed` entry is a caller input that
/// is never written by any segment (constant for the whole split), so it cannot
/// change between passes and excluding it keeps the fingerprint cheap. Each
/// owned buffer mixes its NAME and its bytes (FNV-1a) so a value moving between
/// buffers is observed, and the per-buffer hashes are XOR-combined so map
/// iteration order does not affect the result. Two consecutive passes with an
/// identical fingerprint prove the deterministic segment sequence reached a
/// fixpoint (used to early-exit the outer iteration loop).
pub(super) fn owned_accumulator_fingerprint(inputs: &HashMap<Ident, GridSyncInput<'_>>) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut combined: u64 = 0;
    for (name, input) in inputs {
        let GridSyncInput::Owned(bytes) = input else {
            continue;
        };
        let mut hash = FNV_OFFSET;
        for byte in name.as_str().as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Separator so `name`+`bytes` cannot alias a different split.
        hash ^= 0xff;
        hash = hash.wrapping_mul(FNV_PRIME);
        for byte in bytes.iter() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        combined ^= hash;
    }
    combined
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

/// After each segment dispatch, overwrite every ReadWrite buffer's
/// slot in `inputs` with the freshly-read bytes from `outputs`. The
/// backend returns one Vec<u8> per ReadWrite buffer in declaration
/// order; this function locates each ReadWrite buffer's input-slot
/// index and overwrites it. ReadOnly buffers stay untouched between
/// segments.
fn refresh_readwrite_inputs(
    segment: &Program,
    outputs: &mut Vec<Vec<u8>>,
    inputs: &mut [GridSyncInput<'_>],
) -> Result<(), BackendError> {
    use vyre_foundation::ir::BufferAccess;
    // Walk the segment's buffer table twice in lockstep  -  once for the
    // input slice, once for the output readback. Both paths must
    // mirror the convention `dispatch_borrowed` uses: input position
    // skips Workgroup AND `is_output` buffers; output position emits
    // one slot per ReadWrite buffer (whether or not is_output).
    let mut input_idx = 0usize;
    let mut output_idx = 0usize;
    for buffer in segment.buffers() {
        if matches!(buffer.access(), BufferAccess::Workgroup) {
            continue;
        }
        let is_output_buffer = buffer.is_output();
        let is_readwrite = matches!(buffer.access(), BufferAccess::ReadWrite);

        // Refresh the input slot from the readback if this buffer
        // appears in BOTH input and output positions (i.e. ReadWrite
        // and NOT is_output  -  the rule scratch / `gets` case).
        if is_readwrite && !is_output_buffer {
            if let (Some(slot), Some(bytes)) =
                (inputs.get_mut(input_idx), outputs.get_mut(output_idx))
            {
                slot.refresh_from_output(bytes)?;
            }
        }

        // Advance the input cursor for every non-output buffer.
        if !is_output_buffer {
            input_idx += 1;
        }
        // Advance the output cursor for every ReadWrite buffer (output
        // or not  -  the backend includes them all in the readback).
        if is_readwrite {
            output_idx += 1;
        }
    }
    for output in outputs {
        output.clear();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_sync::test_programs::buffer;
    use vyre_foundation::ir::Node;

    #[test]
    fn refresh_readwrite_inputs_swaps_owned_buffers_after_first_segment() {
        let segment = Program::wrapped(vec![buffer()], [1, 1, 1], vec![Node::Return]);
        let initial = [1u8, 0, 0, 0];
        let mut inputs = [GridSyncInput::Borrowed(initial.as_slice())];
        let mut outputs = vec![Vec::with_capacity(8)];
        let output_ptr = outputs[0].as_ptr() as usize;
        outputs[0].extend_from_slice(&[2, 0, 0, 0]);

        refresh_readwrite_inputs(&segment, &mut outputs, &mut inputs)
            .expect("Fix: test readwrite refresh should fit borrowed promotion storage");

        let first_owned_ptr = match &inputs[0] {
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
        refresh_readwrite_inputs(&segment, &mut outputs, &mut inputs)
            .expect("Fix: test readwrite refresh should reuse owned storage");

        match &inputs[0] {
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
}
