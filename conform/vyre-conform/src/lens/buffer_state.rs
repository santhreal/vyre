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

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::fp_parity::{compare_output_buffers, BufferParity};
    use vyre_foundation::ir::{BufferDecl, DataType};

    /// Mirrors the real `vyre-libs::security::flows_to` layout: many
    /// read-only graph/input buffers plus a single read-write frontier.
    /// The convergence loop carries the FULL state (7 buffers), but
    /// `compare_output_buffers` is contractually sized to the program's
    /// declared outputs. Before the projection fix the lens fed the
    /// 7-buffer state straight into the comparator, which rejected it with
    /// "program declares 1 output buffer(s), compared 7 result buffer(s)"
    /// even when CPU and GPU agreed byte-for-byte, a false parity
    /// failure. These two assertions pin both halves: the raw full state
    /// is rejected, the projected state is accepted.
    fn flows_to_shaped_program() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::storage("pg_nodes", 0, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("pg_edges", 1, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("edge_offsets", 2, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("sources", 3, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("dims", 4, BufferAccess::ReadOnly, DataType::U32).with_count(1),
                BufferDecl::storage("reach_in", 5, BufferAccess::ReadOnly, DataType::U32)
                    .with_count(4),
                BufferDecl::storage("reach_out", 6, BufferAccess::ReadWrite, DataType::U32)
                    .with_count(4),
            ],
            [1, 1, 1],
            vec![],
        )
    }

    #[test]
    fn project_output_buffers_selects_only_the_declared_output() {
        let program = flows_to_shaped_program();
        // Full convergence state: a distinct value per buffer so a wrong
        // projection is detectable, not coincidentally equal.
        let full_state: Vec<Vec<u8>> = (0..7u32)
            .map(|index| {
                let byte = (index + 1) as u8;
                vec![byte; if index == 4 { 4 } else { 16 }]
            })
            .collect();
        assert_eq!(program.output_buffer_indices(), &[6]);

        let projected =
            project_output_buffers(&program, &full_state).expect("Fix: projection must succeed");
        assert_eq!(
            projected.len(),
            1,
            "exactly one declared output (reach_out) must survive projection"
        );
        assert_eq!(
            projected[0], full_state[6],
            "the surviving buffer must be reach_out (the RW frontier), byte-for-byte"
        );
    }

    #[test]
    fn full_state_breaks_comparator_but_projection_restores_parity() {
        let program = flows_to_shaped_program();
        let full_state: Vec<Vec<u8>> = (0..7u32)
            .map(|index| vec![(index + 1) as u8; if index == 4 { 4 } else { 16 }])
            .collect();

        // The raw full state (CPU == GPU, both 7 buffers) is STILL rejected
        // by the comparator because it is sized to declared outputs (1).
        // This is the exact false failure the GPU parity test hit.
        match compare_output_buffers(&program, &full_state, &full_state) {
            BufferParity::Mismatch(detail) => assert!(
                detail.contains("declares 1 output buffer(s), compared 7 result buffer(s)"),
                "expected the output-count mismatch, got: {detail}"
            ),
            BufferParity::Ok => {
                panic!("full 7-buffer state must NOT satisfy a 1-output comparator")
            }
        }

        // Projecting to declared outputs first makes identical states pass.
        let cpu = project_output_buffers(&program, &full_state).expect("Fix: projection");
        let gpu = project_output_buffers(&program, &full_state).expect("Fix: projection");
        assert!(
            matches!(
                compare_output_buffers(&program, &cpu, &gpu),
                BufferParity::Ok
            ),
            "projected identical outputs must compare equal"
        );
    }

    #[test]
    fn project_output_buffers_errs_loudly_on_missing_output_slot() {
        let program = flows_to_shaped_program();
        // State truncated before the RW output index (6), a malformed
        // fixture. Projection must surface this, never silently drop it.
        let short_state: Vec<Vec<u8>> = vec![vec![0u8; 16]; 3];
        let err = project_output_buffers(&program, &short_state)
            .expect_err("missing output slot must be an error, not a silent short vector");
        assert!(
            err.contains("declares an output at index 6"),
            "error must name the missing output index: {err}"
        );
    }
}
