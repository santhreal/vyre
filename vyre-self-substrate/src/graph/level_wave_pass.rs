//! Interprocedural callee-before-caller pass dispatch via #74 level_wave (#74 self-consumer).
//!
//! Closes the recursion thesis for #74  -  `level_wave_program` ships to
//! user dialects (whole-schema migrations, BFS layering, breadth-first
//! graph rewrites) AND drives vyre's interprocedural pass dispatch
//! when callees must finish before callers start.
//!
//! # The self-use
//!
//! Every interprocedural pass that walks a call graph has the same
//! shape: visit each function in callee-before-caller order, run a
//! per-function body, barrier between depth waves. Without
//! `level_wave_program`, each backend hand-codes that loop on the
//! host (one dispatch per depth, host-side termination check). With
//! it, the entire BFS becomes one Program and one dispatch.
//!
//! # Algorithm
//!
//! ```text
//! 1. Caller computes per-function depth in the call graph (leaves at
//!    depth 0, increasing toward main).
//! 2. Caller hands `step_body` (the per-function rewrite/analysis body)
//!    plus the depth array to `build_callee_before_caller_program`.
//! 3. Returned Program runs the body for every function at depth `d`,
//!    barriers, then advances to depth `d+1`  -  all in one dispatch.
//! ```
//!
//! P-DRIVER-10: every interprocedural callee-before-caller pass should
//! consume this rather than hand-rolling a host depth loop.

use vyre_foundation::ir::{BufferDecl, Node, Program};
use vyre_primitives::graph::level_wave::{level_wave_program, level_wave_program_with_buffers};

/// Build a Program that visits every function in callee-before-caller
/// order using GPU-side level-wave dispatch.
///
/// `step_body`: per-function body. Reads/writes any caller-declared
/// buffer via `Expr::InvocationId { axis: 0 }` to address the function
/// being visited.
///
/// `depth_buf`: name of the buffer containing per-function depth in the
/// call graph (leaves at 0).
///
/// `max_depth`: number of waves (i.e., `max(depth) + 1`).
///
/// `function_count`: total functions in the dispatch grid.
#[must_use]
pub fn build_callee_before_caller_program(
    step_body: Vec<Node>,
    depth_buf: &str,
    max_depth: u32,
    function_count: u32,
) -> Program {
    use crate::observability::{bump, level_wave_pass_calls};
    bump(&level_wave_pass_calls);
    level_wave_program(step_body, depth_buf, max_depth, function_count)
}

/// Like [`build_callee_before_caller_program`], but declares the pass's own
/// per-function DATA buffers after `depth_buf` so the `step_body` can read/write
/// them.
///
/// The no-argument form declares ONLY `depth_buf` (binding 0), so a `step_body`
/// that touches any per-function buffer emits IR referencing an undeclared name
/// (the no-shadowing/undeclared validator, and the CUDA backend, reject it). A
/// real interprocedural pass reads the function under visit and writes its
/// analysis/rewrite result, so it MUST declare those buffers: pass them as
/// `extra_buffers`, each with a distinct binding index `>= 1` that the `step_body`
/// references via `Expr::InvocationId { axis: 0 }` for the function being visited.
///
/// This is the ONE-PLACE composition point over
/// [`level_wave_program_with_buffers`]; the buffer-free
/// [`build_callee_before_caller_program`] is the `extra_buffers = []` case.
#[must_use]
pub fn build_callee_before_caller_program_with_buffers(
    step_body: Vec<Node>,
    depth_buf: &str,
    extra_buffers: Vec<BufferDecl>,
    max_depth: u32,
    function_count: u32,
) -> Program {
    use crate::observability::{bump, level_wave_pass_calls};
    bump(&level_wave_pass_calls);
    level_wave_program_with_buffers(
        step_body,
        depth_buf,
        extra_buffers,
        max_depth,
        function_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_nonempty_program() {
        let body = vec![Node::barrier()];
        let program = build_callee_before_caller_program(body, "depths", 4, 16);
        assert_ne!(program.entry().len(), 0);
    }

    #[test]
    fn zero_depth_still_builds() {
        let body = vec![Node::barrier()];
        let program = build_callee_before_caller_program(body, "depths", 0, 1);
        // Even at depth=0 the wrapper builds a valid (empty-loop) Program.
        // Workgroup size matches the primitive's [256,1,1] standard tile.
        assert_eq!(program.workgroup_size(), [256, 1, 1]);
        assert!(!program.buffers().is_empty());
    }

    /// Real VALUE test of the callee-before-caller contract (not just program shape): a 4-function
    /// call chain fn0←fn1←fn2←fn3 (each `fn_i` calls `fn_{i-1}`) with the per-function body
    /// `out[t] = 1 + out[callee[t]]` MUST evaluate to `[1, 2, 3, 4]`: every caller reads its
    /// callee's COMMITTED value because the depth-wave barrier makes level-`d` writes visible before
    /// level-`d+1` runs. A broken ordering (no barrier / caller before callee) would have `fn3` read
    /// the uncommitted `out[2] = 0` and yield `1`. This exercises the `_with_buffers` form (the
    /// buffer-free wrapper cannot declare `callee`/`out`, so a real pass body is unrunnable through
    /// it) and drives the harness through `reference_eval`, which honors the barrier.
    #[test]
    fn callee_before_caller_commits_children_before_parents() {
        use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr};
        use vyre_reference::reference_eval;
        use vyre_reference::value::Value;

        let t = Expr::InvocationId { axis: 0 };
        // out[t] = 1 + out[callee[t]]
        let step_body = vec![
            Node::let_bind("c", Expr::load("callee", t.clone())),
            Node::store(
                "out",
                t.clone(),
                Expr::add(Expr::u32(1), Expr::load("out", Expr::var("c"))),
            ),
        ];
        let extra_buffers = vec![
            BufferDecl::storage("callee", 1, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ];
        let program = build_callee_before_caller_program_with_buffers(
            step_body,
            "depths",
            extra_buffers,
            4, // max_depth
            4, // function_count
        );

        let pack = |data: &[u32]| Value::from(vyre_primitives::wire::pack_u32_slice(data));
        // Binding order: depths(0), callee(1), out(2). leaf fn0's callee is itself (0).
        let inputs = vec![
            pack(&[0, 1, 2, 3]), // depths: fn0..fn3 at increasing call-graph depth
            pack(&[0, 0, 1, 2]), // callee[t]: fn1→fn0, fn2→fn1, fn3→fn2 (fn0 self)
            pack(&[0, 0, 0, 0]), // out seed
        ];
        let results = reference_eval(&program, &inputs).expect("Fix: level-wave pass eval failed");
        // Sole ReadWrite buffer `out` is the returned output.
        let out: Vec<u32> = results[0]
            .to_bytes()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            out,
            vec![1, 2, 3, 4],
            "each caller must read its callee's committed value: fn0=1, fn1=1+1, fn2=1+2, fn3=1+3"
        );
    }
}
