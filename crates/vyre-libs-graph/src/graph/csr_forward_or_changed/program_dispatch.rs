//! The one place a program key turns into a program.
//!
//! The key records everything the choice depends on, so a cache can answer
//! from the key alone and only a miss reaches this builder.

use vyre_foundation::ir::Program;

use super::layout::CsrForwardOrChangedProgramKey;
use super::program_parallel::csr_forward_or_changed_parallel;
use super::program_parallel_batch_global::try_csr_forward_or_changed_parallel_batch_global_dynamic_slot;
use crate::graph::program_graph::ProgramGraphShape;

/// Build the primitive program selected by a launch-plan key.
///
/// # Errors
///
/// Returns an actionable diagnostic when the changed-history program cannot be
/// represented.
pub fn build_csr_forward_or_changed_dispatch_program(
    key: CsrForwardOrChangedProgramKey,
) -> Result<Program, String> {
    let layout = key.layout();
    let shape = ProgramGraphShape::new(layout.node_count, layout.shape_edge_count);
    if key.uses_changed_history() {
        try_csr_forward_or_changed_parallel_batch_global_dynamic_slot(
            shape,
            "frontier_out",
            "changed",
            "changed_slot",
            key.allow_mask(),
            1,
            key.changed_slots(),
        )
    } else {
        Ok(csr_forward_or_changed_parallel(
            shape,
            "frontier_out",
            "changed",
            key.allow_mask(),
        ))
    }
}
