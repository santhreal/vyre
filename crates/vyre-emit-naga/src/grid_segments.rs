//! Entry-point naming for the dispatch segments one descriptor emits.
//!
//! A whole-grid fence is a launch boundary without a cooperative launch, so a
//! fenced descriptor emits one compute entry point per segment. Two surfaces
//! read those names: the emitter, which stamps them into the module, and a
//! dispatch layer, which resolves them to submit the segments in order. Both
//! call [`segment_entry_name`] here, so a name the emitter writes and a name a
//! caller looks up cannot drift apart.
//!
//! Naming is one owned `String` per segment and is not on the per-dispatch
//! path: it runs once per descriptor, when a pipeline is built.

use vyre_lower::KernelDescriptor;

use crate::error::EmitError;
use crate::GRID_SEGMENT_ENTRY_PREFIX;

/// The entry-point name of dispatch segment `index`.
///
/// Segment zero keeps the name `main` that a fence-free descriptor has always
/// emitted, so a caller that never fences reads the same module it did before
/// segmentation existed.
pub(crate) fn segment_entry_name(index: usize) -> String {
    if index == 0 {
        "main".to_owned()
    } else {
        format!("{GRID_SEGMENT_ENTRY_PREFIX}{index}")
    }
}

/// Entry-point names of the dispatch segments `desc` emits, in submission
/// order.
///
/// # Errors
///
/// Returns [`EmitError`] when the descriptor's fence placement admits no
/// launch boundary.
pub fn grid_segment_entry_points(desc: &KernelDescriptor) -> Result<Vec<String>, EmitError> {
    let count = vyre_lower::dispatch_segments(desc)
        .map_err(|source| EmitError::InvalidDescriptor(source.to_string()))?
        .len();
    Ok((0..count).map(segment_entry_name).collect())
}
