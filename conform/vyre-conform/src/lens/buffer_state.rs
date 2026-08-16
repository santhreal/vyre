//! Carrying and narrowing the buffer state vector an iterative lens holds.
//!
//! An iterative lens keeps every buffer the program declares so it can copy
//! `next` into `current` between steps. The comparator is sized to the declared
//! outputs, so the state is projected down before any comparison.

use vyre_foundation::ir::{BufferAccess, Program};

/// Overwrite the read-write slots of `state` with `outputs`, in binding order.
pub fn merge_rw(state: &mut [Vec<u8>], outputs: &[Vec<u8>], program: &Program) {
    // Reference and production artifact execution return writable buffers in
    // canonical binding order. Walk the declarations in the same order.
    let mut out_iter = outputs.iter();
    for (slot, decl) in state.iter_mut().zip(program.buffers().iter()) {
        if matches!(decl.access(), BufferAccess::ReadWrite) {
            if let Some(next) = out_iter.next() {
                *slot = next.clone();
            }
        }
    }
}

/// Index of the buffer named `name` in the program buffer table.
pub fn index_of_buffer(program: &Program, name: &str) -> Option<usize> {
    program
        .buffers()
        .iter()
        .position(|decl| decl.name() == name)
}

/// Project a full convergence/fixpoint state vector down to just the
/// program's declared output buffers (`ReadWrite`/`WriteOnly`), in
/// `output_buffer_indices` order, the exact shape
/// [`compare_output_buffers`] requires (it asserts the comparison vectors
/// have one entry per declared output).
///
/// The iterative lenses carry the FULL buffer state across iterations
/// (every read-only input plus the read-write frontier) so they can copy
/// `next` → `current` between steps. The only buffers the backend
/// actually computes are the outputs; the read-only inputs are
/// host-managed and identical on both sides by construction, so comparing
/// them adds nothing. Returns an explicit `Err` (never a silently short
/// vector) if `state` is missing a declared output slot, so a malformed
/// fixture surfaces loudly instead of degrading into a confusing
/// length-mismatch downstream.
pub fn project_output_buffers(
    program: &Program,
    state: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>, String> {
    program
        .output_buffer_indices()
        .iter()
        .map(|&index| {
            let slot = index as usize;
            state.get(slot).cloned().ok_or_else(|| {
                format!(
                    "convergence state has {} buffer(s) but the program declares an output at \
                     index {slot}; the fixture must supply an initial value for every program buffer.",
                    state.len()
                )
            })
        })
        .collect()
}
