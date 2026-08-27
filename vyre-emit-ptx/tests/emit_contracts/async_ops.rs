//! Test: async ops.
use super::*;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_rw, lit, op, shared_rw, SlotCount,
};

#[test]
fn async_load_emits_bounded_sync_copy() {
    let kernel = descriptor("async_load")
        .slots([
            global_ro(0, DataType::U32, "src"),
            shared_rw(1, DataType::U32, 64, "dst"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::async_load("tile0".into()), [0, 1, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_70).unwrap();
    assert!(s.contains("// async_load tag=tile0"));
    assert!(s.contains(".shared .align 4 .b8 shared_buf_1[256];"));
    assert!(s.contains("ld.global.u32"));
    assert!(s.contains("st.shared.u32"));
    assert!(
        s.contains("%tid.x") && s.contains("setp.ne.u32"),
        "must leader predicate scalar copy"
    );
    assert!(s.contains("lowered as bounded synchronous copy"));
}

#[test]
fn async_load_uses_cp_async_on_sm_80() {
    let kernel = descriptor("k")
        .slots([
            global_ro(0, DataType::U32, "src"),
            shared_rw(1, DataType::U32, 64, "dst"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::async_load("t".into()), [0, 1, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_80).unwrap();
    assert!(s.contains("// cp.async_load tag=t"));
    assert!(s.contains("cp.async.ca.shared.global"));
    assert!(s.contains("cp.async.commit_group"));
    assert!(s.contains("cp.async.wait_group 0"));
    assert!(
        s.contains("%tid.x") && s.contains("setp.ne.u32"),
        "must leader predicate cp.async issue"
    );
    assert!(
        s.contains("bar.sync 0"),
        "implicit drain must synchronize CTA"
    );
    assert!(
        !s.contains("lowered as bounded synchronous copy"),
        "sm_80 global-to-shared U32 AsyncLoad must use the native cp.async path"
    );
}

#[test]
fn cp_async_wait_is_deferred_until_async_wait_to_overlap_compute() {
    let kernel = descriptor("cp_async_overlap")
        .slots([
            global_ro(0, DataType::U32, "src"),
            shared_rw(1, DataType::U32, 64, "dst"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    effect(KernelOpKind::async_load("tile".into()), [0, 1, 0, 1]),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [2, 3], 4),
                    effect(KernelOpKind::async_wait("tile".into()), []),
                ])
                .literals([
                    LiteralValue::U32(0),
                    LiteralValue::U32(16),
                    LiteralValue::U32(7),
                    LiteralValue::U32(9),
                ]),
        )
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_80).unwrap();
    let commit = s
        .find("cp.async.commit_group;")
        .expect("native cp.async path must commit a group");
    let wait = s
        .find("cp.async.wait_group 0;")
        .expect("AsyncWait must drain the pending cp.async group");
    let overlapped_add = s[commit..wait]
        .find("add.u32")
        .expect("independent compute must remain between cp.async commit and wait");
    assert!(overlapped_add > 0);
}

#[test]
fn async_store_emits_bounded_sync_copy() {
    let kernel = descriptor("async_store")
        .slots([
            shared_rw(0, DataType::U32, 64, "src"),
            global_rw(1, DataType::U32, "dst").with_count(64),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::async_store("tile0".into()), [0, 1, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_70).unwrap();
    assert!(s.contains("// async_store tag=tile0"));
    assert!(s.contains("ld.shared.u32"));
    assert!(s.contains("st.global.u32"));
}

#[test]
fn async_wait_emits_workgroup_memory_barrier() {
    let kernel = descriptor("async_wait")
        .dispatch(64, 1, 1)
        .body(body().op(effect(KernelOpKind::async_wait("t".into()), [])))
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_80).unwrap();
    assert!(s.contains("membar.cta"));
    assert!(
        s.contains("bar.sync 0"),
        "AsyncWait must synchronize workgroup with bar.sync 0"
    );
}

/// One staged transfer of a `ring_slots`-deep ring.
fn staged_load(tag: &str, slot: u16, ring_slots: u16) -> KernelOpKind {
    KernelOpKind::AsyncLoad(Box::new(AsyncTransaction::staged(
        tag.into(),
        TransactionScope::Workgroup,
        StageSlot::new(slot, ring_slots),
    )))
}

/// A wait for one staged transfer.
fn staged_wait(tag: &str, slot: u16, ring_slots: u16) -> KernelOpKind {
    KernelOpKind::AsyncWait(Box::new(AsyncWaitSpec::new(AsyncTransaction::staged(
        tag.into(),
        TransactionScope::Workgroup,
        StageSlot::new(slot, ring_slots),
    ))))
}

/// Three staged transfers over one global source and one shared destination,
/// followed by the ops a caller appends.
fn ring_kernel(id: &str, tail: Vec<KernelOp>) -> KernelDescriptor {
    let mut ops = vec![
        lit(0, 0),
        lit(1, 1),
        effect(staged_load("slot0", 0, 3), [0, 1, 0, 1]),
        effect(staged_load("slot1", 1, 3), [0, 1, 0, 1]),
        effect(staged_load("slot2", 2, 3), [0, 1, 0, 1]),
    ];
    ops.extend(tail);
    descriptor(id)
        .slots([
            global_ro(0, DataType::U32, "src"),
            shared_rw(1, DataType::U32, 64, "dst"),
        ])
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops(ops)
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build()
}

/// WHY: a bounded stage ring exists so a wait completes one slot and leaves the
/// rest of the ring outstanding. The wait used to drain every committed group,
/// which is correct and buys no overlap: the transfers the ring was sized for
/// were already complete by the time the first consumer ran. The emitted bound
/// has to fall out of the declared ring depth and shrink as slots are consumed.
#[test]
fn a_wait_leaves_the_rest_of_the_declared_ring_in_flight() {
    let kernel = ring_kernel(
        "staged_ring",
        vec![
            effect(staged_wait("slot0", 0, 3), []),
            effect(staged_wait("slot1", 1, 3), []),
            effect(staged_wait("slot2", 2, 3), []),
        ],
    );
    let s = emit_with_target(&kernel, ComputeCapability::SM_80).unwrap();
    let waits: Vec<&str> = s
        .lines()
        .map(str::trim)
        .filter(|line| line.starts_with("cp.async.wait_group"))
        .collect();
    assert_eq!(
        waits,
        vec![
            "cp.async.wait_group 2;",
            "cp.async.wait_group 1;",
            "cp.async.wait_group 0;"
        ],
        "Fix: waiting on one slot of a three-slot ring leaves the two younger transfers in flight"
    );
}

/// WHY: the native transfer form completes committed groups oldest first, so a
/// wait that completes a younger transfer while an older one is outstanding has
/// no spelling in it. Draining the older one instead is the silent
/// approximation this rejection exists to prevent: it would discard exactly the
/// overlap the descriptor's ring states.
#[test]
fn a_wait_that_skips_an_older_transfer_is_rejected() {
    let kernel = ring_kernel(
        "out_of_order_wait",
        vec![effect(staged_wait("slot1", 1, 3), [])],
    );
    let error = emit_with_target(&kernel, ComputeCapability::SM_80)
        .expect_err("a wait the native form cannot express must be rejected");
    assert!(
        matches!(error, EmitError::UnsupportedOp(_)),
        "Fix: report the wait as unsupported instead of draining an older transfer; got {error:?}"
    );
}

/// WHY: the fence is stated by the descriptor, not chosen by the emitter. A
/// transfer only its issuing invocation reads needs no fence at all, and one
/// read from outside the workgroup needs a wider fence than the workgroup form
/// every wait used to emit. Emitting one fixed pair for every wait either
/// over-synchronizes the first case or under-synchronizes the last.
#[test]
fn the_stated_fence_selects_the_emitted_form() {
    let private = descriptor("private_wait")
        .dispatch(64, 1, 1)
        .body(body().op(effect(
            KernelOpKind::AsyncWait(Box::new(AsyncWaitSpec::fenced(
                AsyncTransaction::unstaged("t".into(), TransactionScope::Invocation),
                MemoryProxyFence::None,
            ))),
            [],
        )))
        .build();
    let s = emit_with_target(&private, ComputeCapability::SM_80).unwrap();
    assert!(
        !s.contains("membar") && !s.contains("bar.sync"),
        "Fix: a transfer only its issuing invocation reads needs no fence:\n{s}"
    );

    let device = descriptor("device_wait")
        .dispatch(64, 1, 1)
        .body(body().op(effect(
            KernelOpKind::AsyncWait(Box::new(AsyncWaitSpec::new(AsyncTransaction::unstaged(
                "t".into(),
                TransactionScope::Device,
            )))),
            [],
        )))
        .build();
    let s = emit_with_target(&device, ComputeCapability::SM_80).unwrap();
    assert!(
        s.contains("membar.gl"),
        "Fix: a transfer read from outside the workgroup needs the device fence:\n{s}"
    );
    assert!(s.contains("bar.sync 0"));
}
