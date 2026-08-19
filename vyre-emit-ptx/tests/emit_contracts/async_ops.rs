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
                    effect(
                        KernelOpKind::AsyncLoad {
                            tag: "tile0".into(),
                        },
                        [0, 1, 0, 1],
                    ),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_70).unwrap();
    assert!(s.contains("// async_load tag=tile0"));
    assert!(s.contains(".shared .align 4 .b8 shared_buf_1[256];"));
    assert!(s.contains("ld.global.u32"));
    assert!(s.contains("st.shared.u32"));
    assert!(s.contains("%tid.x") && s.contains("setp.ne.u32"), "must leader predicate scalar copy");
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
                    effect(KernelOpKind::AsyncLoad { tag: "t".into() }, [0, 1, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_80).unwrap();
    assert!(s.contains("// cp.async_load tag=t"));
    assert!(s.contains("cp.async.ca.shared.global"));
    assert!(s.contains("cp.async.commit_group"));
    assert!(s.contains("cp.async.wait_group 0"));
    assert!(s.contains("%tid.x") && s.contains("setp.ne.u32"), "must leader predicate cp.async issue");
    assert!(s.contains("bar.sync 0"), "implicit drain must synchronize CTA");
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
                    effect(KernelOpKind::AsyncLoad { tag: "tile".into() }, [0, 1, 0, 1]),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [2, 3], 4),
                    effect(KernelOpKind::AsyncWait { tag: "tile".into() }, []),
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
                    effect(
                        KernelOpKind::AsyncStore {
                            tag: "tile0".into(),
                        },
                        [0, 1, 0, 1],
                    ),
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
        .body(body().op(effect(KernelOpKind::AsyncWait { tag: "t".into() }, [])))
        .build();
    let s = emit_with_target(&kernel, ComputeCapability::SM_80).unwrap();
    assert!(s.contains("membar.cta"));
    assert!(s.contains("bar.sync 0"), "AsyncWait must synchronize workgroup with bar.sync 0");
}
