//! Which buffers a fused Program still publishes to the host.
//!
//! Fusing two passes concatenates their buffer declarations, so every
//! intermediate the first pass declared as an output stays an output of the
//! fused Program. The host then reads back scratch it never asked for, and the
//! dispatcher must allocate a staging slot for it. One owner rewrites those
//! roles instead of each fusing composition repeating the walk.

use vyre_foundation::ir::Program;

/// Keep `final_output` an output buffer and demote every other output of
/// `program` to a pipeline-live-out intermediate.
///
/// A demoted buffer is still written and still visible to a later fused stage.
/// It is no longer read back to the host.
pub(crate) fn demote_intermediate_outputs(program: Program, final_output: &str) -> Program {
    let buffers = program
        .buffers()
        .iter()
        .map(|buffer| {
            let mut buffer = buffer.clone();
            if buffer.name() != final_output && buffer.is_output() {
                buffer.is_output = false;
                buffer.pipeline_live_out = true;
            }
            buffer
        })
        .collect();
    program.with_rewritten_buffers(buffers)
}
