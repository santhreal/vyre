//! Which buffers a fused Program still publishes to the host.
//!
//! Fusing two passes concatenates their buffer declarations, so every
//! intermediate the first pass declared as an output stays an output of the
//! fused Program. The host then reads back scratch it never asked for, and the
//! dispatcher must allocate a staging slot for it. One owner rewrites those
//! roles instead of each fusing composition repeating the walk.

use vyre_foundation::ir::Program;

/// Make `final_output` the program's output buffer and demote every other
/// output to a pipeline-live-out intermediate.
///
/// A demoted buffer is still written and still visible to a later fused stage.
/// It is no longer read back to the host.
///
/// The final buffer is promoted, not merely kept. A composition whose last
/// stage declares its result as plain `ReadWrite` storage publishes no output
/// at all: the buffer then consumes a host input slot, so the reference oracle
/// demands a value for it while every backend allocates it, and the two
/// disagree on what the program returns.
pub fn demote_intermediate_outputs(program: Program, final_output: &str) -> Program {
    let buffers = program
        .buffers()
        .iter()
        .map(|buffer| {
            let mut buffer = buffer.clone();
            if buffer.name() == final_output {
                buffer.is_output = true;
                buffer.pipeline_live_out = true;
            } else if buffer.is_output() {
                buffer.is_output = false;
                buffer.pipeline_live_out = true;
            }
            buffer
        })
        .collect();
    program.with_rewritten_buffers(buffers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::composition::wrap_anonymous_region;
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node};

    fn program(buffers: Vec<BufferDecl>) -> Program {
        Program::wrapped(
            buffers,
            [1, 1, 1],
            vec![wrap_anonymous_region(
                "test::outputs",
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            )],
        )
    }

    /// A final stage that declares its result as plain `ReadWrite` storage
    /// still publishes it. Otherwise the buffer consumes a host input slot,
    /// the reference oracle demands a value for it, and every backend
    /// allocates it instead.
    #[test]
    fn a_final_output_declared_as_plain_storage_is_promoted() {
        let fused = program(vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ]);
        let rewritten = demote_intermediate_outputs(fused, "out");
        let out = rewritten
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "out")
            .expect("Fix: the rewrite must keep the final buffer");

        assert!(out.is_output(), "the final buffer must be an output");
        assert!(
            out.is_backend_allocated_output(),
            "a backend must allocate the final output"
        );
        assert!(
            !out.consumes_host_input(),
            "the final output must not also consume a host input slot"
        );
    }

    /// Exactly one buffer is read back: the named one. Every other output of a
    /// fused stage stays written and visible downstream without reaching the
    /// host.
    #[test]
    fn only_the_named_buffer_is_read_back() {
        let fused = program(vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("scratch", 1, DataType::U32).with_count(1),
            BufferDecl::output("out", 2, DataType::U32).with_count(1),
        ]);
        let rewritten = demote_intermediate_outputs(fused, "out");

        let outputs: Vec<&str> = rewritten
            .buffers()
            .iter()
            .filter(|buffer| buffer.is_output())
            .map(BufferDecl::name)
            .collect();
        assert_eq!(outputs, ["out"], "only the named buffer is read back");

        let scratch = rewritten
            .buffers()
            .iter()
            .find(|buffer| buffer.name() == "scratch")
            .expect("Fix: a demoted buffer is still declared");
        assert!(
            scratch.is_pipeline_live_out(),
            "a demoted intermediate stays live for a later fused stage"
        );
        assert!(
            !scratch.consumes_host_input(),
            "a demoted intermediate is written, not uploaded"
        );
    }
}
