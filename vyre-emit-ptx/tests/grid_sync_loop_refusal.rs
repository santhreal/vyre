//! A `MemoryOrdering::GridSync` barrier nested inside a `Node::Loop` must be
//! REFUSED at emit time, not lowered.
//!
//! The monotonic-counter grid barrier fixes its release target at emit time to
//! `(barrier_index + 1) * gridSize`. A loop emits its body ONCE, so every
//! iteration after the first finds the counter already at or past that target
//! and the barrier silently becomes a no-op, leaving the grid unsynchronized
//! with no error and no wrong-looking output. Silent desynchronization is the
//! worst failure mode available here, so the emitter refuses the shape instead.

use vyre_emit_ptx::PtxEmitOptions;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::ir::MemoryOrdering;

#[path = "emit_probe/probe.rs"]
mod emit_probe;
use emit_probe::{lower_and_emit, region_program};

/// One store plus one barrier, with `nested` deciding whether the barrier sits
/// inside the loop body or at dispatch level. Everything else is identical, so a
/// difference in emit outcome is attributable to nesting alone.
fn barrier_program(nested: bool) -> Program {
    let step = Node::store("state", Expr::gid_x(), Expr::u32(1));
    let fence = Node::barrier_with_ordering(MemoryOrdering::GridSync);
    let body = if nested {
        vec![Node::loop_for(
            "iter",
            Expr::u32(0),
            Expr::u32(4),
            vec![step, fence],
        )]
    } else {
        vec![
            Node::loop_for("iter", Expr::u32(0), Expr::u32(4), vec![step]),
            fence,
        ]
    };
    region_program(
        "grid-sync-loop-refusal-probe",
        vec![BufferDecl::read_write("state", 0, DataType::U32).with_count(256)],
        body,
    )
}

/// Emit on the COOPERATIVE path. `vyre-driver-cuda` forces
/// `cuLaunchCooperativeKernel` for grid-sync programs, so this is the path a
/// grid-synced program actually takes on this box. With
/// `cooperative_grid_sync` left false the emitter refuses every GridSync
/// barrier outright, nested or not, and the loop-nesting rule below would never
/// be reached.
fn emit(program: &Program) -> Result<String, String> {
    let mut options = PtxEmitOptions::default();
    options.cooperative_grid_sync = true;
    lower_and_emit(program, options)
}

/// A GridSync inside a loop body is refused, and the refusal names the
/// mechanism rather than failing generically.
///
/// The defect this locks out is the silent one: emitting the nested barrier
/// produces PTX whose second and later iterations do not synchronize at all,
/// because the release target was burned in on the single emitted copy. A
/// generic "unsupported" error would also stop the miscompile, but the message
/// is what stops the next person from re-introducing the shape, so it is
/// asserted here alongside the refusal.
#[test]
fn grid_sync_barrier_inside_a_loop_is_refused_at_emit_time() {
    let error = emit(&barrier_program(true))
        .expect_err("a GridSync barrier inside a loop body must not lower to PTX");
    assert!(
        error.contains("InvalidDescriptor"),
        "the nested GridSync must be refused as an invalid descriptor; got: {error}"
    );
    for fragment in [
        "GridSync",
        "inside a loop",
        "release target",
        "emit",
        "no-op",
    ] {
        assert!(
            error.contains(fragment),
            "the refusal must explain the fixed-release-target mechanism, missing {fragment:?}; got: {error}"
        );
    }
}

/// The same program with the fence hoisted to dispatch level emits cleanly.
///
/// Without this half the test above would pass just as well if the emitter had
/// started refusing every GridSync barrier, or every program containing a loop,
/// which would be a much larger regression wearing the same green checkmark.
/// This pins the refusal to nesting specifically.
#[test]
fn the_same_grid_sync_barrier_at_dispatch_level_emits() {
    let ptx = emit(&barrier_program(false))
        .expect("a dispatch-level GridSync barrier must still lower to PTX");
    assert!(
        ptx.contains("bar.sync"),
        "a lowered grid barrier must emit its per-CTA bar.sync bracket; got PTX without one"
    );
    assert!(
        ptx.contains("membar.gl"),
        "a lowered grid barrier must emit the global memory fence that publishes cross-CTA writes"
    );
}
