use super::*;
use vyre_foundation::ir::MemoryOrdering;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_wo, lit, op};

fn barrier_kernel(ordering: MemoryOrdering) -> KernelDescriptor {
    descriptor("barrier_scope")
        .body(body().op(effect(KernelOpKind::Barrier { ordering }, [])))
        .build()
}

#[test]
fn workgroup_barrier_emits_cta_barrier() {
    let ptx = emit(&barrier_kernel(MemoryOrdering::SeqCst))
        .expect("Fix: workgroup-scope barriers must remain PTX-emittable.");

    assert!(
        ptx.contains("bar.sync 0;"),
        "Fix: workgroup-scope barrier lowering must emit a CTA barrier."
    );
}

#[test]
fn grid_sync_barrier_is_not_silently_downgraded_to_cta_barrier() {
    match emit(&barrier_kernel(MemoryOrdering::GridSync)) {
        Err(EmitError::InvalidDescriptor(message)) => {
            assert!(
                message.contains("GridSync") && message.contains("bar.sync 0"),
                "Fix: GridSync rejection must name the semantic scope loss; got: {message}"
            );
        }
        Ok(ptx) => panic!(
            "Fix: PTX emitter silently accepted GridSync; this would downgrade cross-grid synchronization to CTA scope. PTX:\n{ptx}"
        ),
        Err(other) => panic!(
            "Fix: GridSync PTX rejection must be an actionable InvalidDescriptor, not {other:?}."
        ),
    }
}

#[test]
fn nested_barrier_kernel_keeps_lanes_live_and_predicates_global_store() {
    let kernel = descriptor("nested_barrier_store")
        .slot(global_wo(0, DataType::U32, "out"))
        .dispatch(256, 1, 1)
        .body(
            body()
                .ops([
                    effect(KernelOpKind::StructuredBlock, [0]),
                    op(KernelOpKind::LocalInvocationId, [0], 0),
                    lit(0, 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ])
                .children([body().ops([effect(
                    KernelOpKind::Barrier {
                        ordering: MemoryOrdering::SeqCst,
                    },
                    [],
                )])])
                .literal(LiteralValue::U32(7)),
        )
        .build();

    let ptx = emit(&kernel)
        .expect("Fix: nested workgroup barriers with global stores must remain PTX-emittable.");

    assert!(
        ptx.contains("Full-workgroup entry")
            && !ptx.contains("setp.ge.u32     %p0, %r3, %r26;")
            && !ptx.contains("@%p0 bra $L_exit;"),
        "barrier kernels must not exit lanes before all lanes reach shared/barrier code:\n{ptx}"
    );
    assert!(
        ptx.lines()
            .any(|line| line.contains("st.global.u32") && line.trim_start().starts_with("@%p")),
        "global stores in full-workgroup-entry kernels must be bounds-predicated:\n{ptx}"
    );
    assert!(
        ptx.contains("bar.sync 0;"),
        "nested barrier body must still lower to a CTA barrier:\n{ptx}"
    );
}
/// Emit options with native cooperative grid-sync lowering enabled, matching what
/// `vyre-driver-cuda/src/codegen/mod.rs:75` passes unconditionally.
fn cooperative_options() -> PtxEmitOptions {
    PtxEmitOptions {
        target: ComputeCapability::SM_70,
        subgroup_size: 32,
        ulp_budget: None,
        cooperative_grid_sync: true,
    }
}

/// A `GridSync` barrier nested in a loop body, plus a bare one at top level for
/// the control case. `lo = 0`, `hi = 4`, body index 0.
fn grid_sync_in_loop_kernel() -> KernelDescriptor {
    descriptor("grid_sync_in_loop")
        .dispatch(256, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(
                        KernelOpKind::StructuredForLoop {
                            loop_var: "iter".into(),
                        },
                        [0, 1, 0],
                    ),
                ])
                .children([body().ops([effect(
                    KernelOpKind::Barrier {
                        ordering: MemoryOrdering::GridSync,
                    },
                    [],
                )])])
                .literals([LiteralValue::U32(0), LiteralValue::U32(4)]),
        )
        .build()
}

/// A `GridSync` barrier inside a loop body MUST be refused, because the
/// monotonic-counter lowering cannot express a per-iteration barrier.
///
/// The defect this locks out is silent and survives every other check. The
/// release target is computed at EMIT time as `(barrier_index + 1) * gridSize`
/// and baked into the instruction stream, but a loop emits its body ONCE and
/// branches back. So on iteration 0 the counter climbs to `gridSize` and the
/// barrier releases correctly, and on every later iteration the counter is
/// ALREADY at or past that fixed target before any CTA arrives, the
/// `setp.lt.u32` guard is false immediately, and the spin never waits. The
/// barrier degrades into a no-op with no error, no diagnostic, and a barrier
/// still visibly present in both the IR and the PTX.
///
/// Nothing else in the stack catches this: `vyre_driver::grid_sync` deliberately
/// recurses into `Node::Loop { body }` when detecting barriers, so the IR is
/// accepted and routed to native lowering. This emitter refusal is the only
/// enforcement.
#[test]
fn grid_sync_barrier_inside_a_loop_is_refused_not_silently_degraded() {
    match emit_with_options(&grid_sync_in_loop_kernel(), cooperative_options()) {
        Err(EmitError::InvalidDescriptor(message)) => {
            assert!(
                message.contains("inside a loop body"),
                "Fix: the refusal must name loop nesting as the cause; got: {message}"
            );
            assert!(
                message.contains("no-op"),
                "Fix: the refusal must say the barrier would degrade to a no-op, since that \
                 silent degradation is the whole reason it is refused; got: {message}"
            );
            assert!(
                message.contains("unroll"),
                "Fix: the refusal must name the unroll remedy that persistent_fixpoint_grid \
                 uses; got: {message}"
            );
        }
        Ok(ptx) => panic!(
            "Fix: emitter accepted a GridSync barrier inside a loop. Every iteration after the \
             first would run unsynchronized because the release target is a compile-time \
             constant. PTX:\n{ptx}"
        ),
        Err(other) => {
            panic!("Fix: GridSync-in-loop must be an actionable InvalidDescriptor, not {other:?}.")
        }
    }
}

/// The control case: a top-level `GridSync` barrier still lowers. This keeps the
/// loop refusal from being implemented as a blanket rejection, which would break
/// `persistent_fixpoint_grid`'s unrolled waves and every cooperative dispatch.
#[test]
fn top_level_grid_sync_barrier_still_lowers_to_the_cooperative_counter_barrier() {
    let ptx = emit_with_options(
        &barrier_kernel(MemoryOrdering::GridSync),
        cooperative_options(),
    )
    .expect("Fix: a top-level GridSync barrier must lower under cooperative_grid_sync");

    assert!(
        ptx.contains("atom.global.add.u32"),
        "Fix: the cooperative barrier must record arrival with a global atomic add; PTX:\n{ptx}"
    );
    assert!(
        ptx.contains("ld.volatile.global.u32"),
        "Fix: the cooperative barrier must spin on a volatile load; PTX:\n{ptx}"
    );
    assert!(
        ptx.contains(".global .align 4 .u32 _vyre_grid_barrier[1];"),
        "Fix: the module-scope arrival counter must be declared; PTX:\n{ptx}"
    );
}
