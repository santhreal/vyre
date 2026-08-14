//! Test: subgroup.
use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit, op};

/// Single-seed subgroup reduction kernel: one literal reduced across the
/// subgroup by `reduce_op`.
fn subgroup_reduce_kernel(
    id: &str,
    reduce_op: vyre_lower::SubgroupReduceOp,
    seed: LiteralValue,
) -> KernelDescriptor {
    descriptor(id)
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::SubgroupReduce { op: reduce_op }, [0], 1),
                ])
                .literal(seed),
        )
        .build()
}

/// Subgroup shuffle kernel: `value` is read from lane 3.
fn subgroup_shuffle_kernel(id: &str, value: LiteralValue) -> KernelDescriptor {
    descriptor(id)
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::SubgroupShuffle, [0, 1], 2),
                ])
                .literals([value, LiteralValue::U32(3)]),
        )
        .build()
}

/// A 32-lane XOR butterfly takes log2(32) = 5 exchange steps, each an
/// `xor.b32` partner-lane computation feeding an idx-mode shuffle.
fn assert_xor_butterfly_steps(s: &str) {
    assert!(
        s.matches("xor.b32").count() >= 5,
        "expected >=5 XOR-partner steps for a 32-lane warp, got: {s}"
    );
    assert!(
        s.matches("shfl.sync.idx.b32").count() >= 5,
        "expected >=5 idx-shuffle exchange steps for a 32-lane warp, got: {s}"
    );
}

#[test]
fn subgroup_ballot_emits_vote_sync_ballot() {
    let kernel = descriptor("ballot")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([lit(0, 0), op(KernelOpKind::SubgroupBallot, [0], 1)])
                .literal(LiteralValue::Bool(true)),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("activemask.b32"));
    assert!(s.contains("vote.sync.ballot.b32"));
}

#[test]
fn subgroup_shuffle_emits_shfl_sync_idx() {
    let kernel = subgroup_shuffle_kernel("shuffle", LiteralValue::U32(7));
    let s = emit(&kernel).unwrap();
    assert!(s.contains("shfl.sync.idx.b32"));
}

#[test]
fn f32_subgroup_shuffle_bitcasts_through_b32() {
    let kernel = subgroup_shuffle_kernel("shuffle_f32", LiteralValue::F32(7.0));
    let s = emit(&kernel).unwrap();
    assert!(s.contains("mov.b32"));
    assert!(s.contains("shfl.sync.idx.b32"));
}

#[test]
fn f32_subgroup_add_emits_shuffle_tree() {
    let kernel = subgroup_reduce_kernel(
        "add",
        vyre_lower::SubgroupReduceOp::Add,
        LiteralValue::F32(5.0),
    );
    let s = emit(&kernel).unwrap();
    assert!(s.contains("activemask.b32"));
    // All-lane broadcast contract: f32 reduction uses an XOR all-reduce so every
    // lane ends with the full result, NOT a shfl.down tree (which only feeds
    // lane 0). The exchange uses `.idx` mode with an explicit `laneid ^ offset`
    // source, mirroring the verified `subgroup_shuffle` lowering (idx-mode is
    // proven all-lane correct on sm_120; we standardize on it rather than `.bfly`
    // so one shuffle path is exercised everywhere). log2(32) = 5 exchange steps.
    assert!(s.contains("%laneid"));
    assert!(s.contains("shfl.sync.idx.b32"));
    assert!(!s.contains("shfl.sync.down.b32"));
    assert!(!s.contains("shfl.sync.bfly.b32"));
    assert_xor_butterfly_steps(&s);
    assert!(s.contains("add.f32"));
    assert!(!s.contains("redux.sync.add.f32"));
}

#[test]
fn u32_subgroup_add_emits_redux_sync() {
    let kernel = subgroup_reduce_kernel(
        "add_u32",
        vyre_lower::SubgroupReduceOp::Add,
        LiteralValue::U32(5),
    );
    let s = emit(&kernel).unwrap();
    assert!(s.contains("redux.sync.add.u32"));
}

#[test]
fn u32_subgroup_mul_emits_idx_butterfly_not_redux() {
    // Integer product has no `redux.sync`; it must reduce with the shfl.idx XOR
    // butterfly (laneid^offset source) and `mul.lo.u32`, all-lane-broadcast.
    let kernel = subgroup_reduce_kernel(
        "mul_u32",
        vyre_lower::SubgroupReduceOp::Mul,
        LiteralValue::U32(3),
    );
    let s = emit(&kernel).unwrap();
    assert!(s.contains("activemask.b32"));
    assert!(s.contains("%laneid"));
    assert!(s.contains("mul.lo.u32"));
    assert!(!s.contains("redux.sync"));
    assert_xor_butterfly_steps(&s);
    // Integer path shuffles the accumulator directly, no f32<->b32 bitcast pair
    // and no float combine.
    assert!(!s.contains(".f32"));
}

#[test]
fn f32_subgroup_mul_emits_mul_f32_butterfly_not_redux() {
    // f32 product has no `redux.sync` (redux is integer-only); it reduces with
    // the shared shfl.idx XOR butterfly and a `mul.f32` combine, bitcasting the
    // accumulator through b32 around each shuffle. All-lane broadcast.
    let kernel = subgroup_reduce_kernel(
        "mul_f32",
        vyre_lower::SubgroupReduceOp::Mul,
        LiteralValue::F32(3.0),
    );
    let s = emit(&kernel).unwrap();
    assert!(s.contains("activemask.b32"));
    assert!(s.contains("%laneid"));
    assert!(s.contains("mul.f32"));
    assert!(!s.contains("redux.sync"));
    // Float path shuffles via b32 bitcast and must NOT use the integer product.
    assert!(!s.contains("mul.lo"));
    assert!(s.contains("mov.b32"));
    assert_xor_butterfly_steps(&s);
}

#[test]
fn subgroup_local_id_emits_laneid() {
    let kernel = descriptor("lane")
        .dispatch(64, 1, 1)
        .body(body().op(op(KernelOpKind::SubgroupLocalId, [], 0)))
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("%laneid"));
}

#[test]
fn subgroup_size_emits_probed_width_literal() {
    let kernel = descriptor("wsz")
        .dispatch(64, 1, 1)
        .body(body().op(op(KernelOpKind::SubgroupSize, [], 0)))
        .build();
    let s = emit_with_options(
        &kernel,
        PtxEmitOptions {
            target: ComputeCapability::SM_70,
            subgroup_size: 16,
            ulp_budget: None,
            cooperative_grid_sync: false,
        },
    )
    .unwrap();
    assert!(s.contains("mov.u32") && s.contains(", 16;"));
}

#[test]
fn atomic_unsupported_op_returns_error() {
    let kernel = atomic_kernel(
        "k",
        global_rw(0, DataType::U32, "b"),
        1,
        AtomicOp::FetchNand,
        MemoryOrdering::SeqCst,
        1,
    );
    let r = emit(&kernel);
    assert!(matches!(r, Err(EmitError::UnsupportedOp(_))));
}

#[test]
fn for_loop_var_name_appears_in_comment() {
    let kernel = descriptor("named_loop")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(
                        KernelOpKind::StructuredForLoop {
                            loop_var: "row_idx".into(),
                        },
                        [0, 1, 0],
                    ),
                ])
                .child(empty_child_body())
                .literals([LiteralValue::U32(0), LiteralValue::U32(16)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("// for row_idx in"));
}
