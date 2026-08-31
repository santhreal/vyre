//! Test: scalar ops.
use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_wo, lit, op, SlotCount};

#[test]
fn emit_ends_with_return() {
    let s = emit(&one_store_kernel()).unwrap();
    // Last meaningful line is `ret;` followed by closing brace.
    assert!(s.contains("    ret;\n}"));
}

#[test]
fn empty_kernel_emits_just_preamble_and_ret() {
    let desc = descriptor("empty").dispatch(64, 1, 1).build();
    let s = emit(&desc).unwrap();
    assert!(s.contains(".visible .entry main(\n    .param .u64 params_buf\n)"));
    assert!(s.contains("ret;"));
}

#[test]
fn binop_add_emits_add_u32() {
    let kernel = descriptor("add")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 2),
                ])
                .literals([LiteralValue::U32(3), LiteralValue::U32(4)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("add.u32"));
}

#[test]
fn f32_canonicalization_uses_native_flush_to_zero_and_nan_selection() {
    let kernel = descriptor("canonical_f32_add")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 2),
                ])
                .literals([LiteralValue::F32(-0.0), LiteralValue::F32(f32::NAN)]),
        )
        .build();

    let ptx = emit(&kernel).expect("Fix: f32 canonicalization fixture must emit PTX.");
    assert!(
        ptx.contains("mul.ftz.f32")
            && ptx.contains("setp.nan.f32")
            && ptx.contains("selp.f32"),
        "Fix: f32 canonicalization must preserve signed zero, flush subnormals, and select the canonical NaN with the compact native sequence:\n{ptx}"
    );
    assert!(
        !ptx.contains("0x00800000"),
        "Fix: f32 canonicalization must not reconstruct flush-to-zero through an eight-instruction integer mask sequence:\n{ptx}"
    );
}

#[test]
fn integer_single_use_mul_add_emits_mad_without_dead_mul() {
    let kernel = descriptor("int_mad")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 3),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [3, 2], 4),
                ])
                .literals([
                    LiteralValue::I32(-3),
                    LiteralValue::I32(7),
                    LiteralValue::I32(5),
                ]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("mad.lo.s32"), "{s}");
    assert!(!s.contains("mul.lo.s32"), "{s}");
    assert!(!s.contains("add.s32"), "{s}");
}

#[test]
fn integer_multi_use_mul_add_keeps_separate_mul() {
    let kernel = descriptor("int_mad_multi_use")
        .slot(global_wo(0, DataType::I32, "out").with_count(1))
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    op(KernelOpKind::BinOpKind(BinOp::Mul), [0, 1], 3),
                    op(KernelOpKind::BinOpKind(BinOp::Add), [3, 2], 4),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 3]),
                ])
                .literals([
                    LiteralValue::I32(-3),
                    LiteralValue::I32(7),
                    LiteralValue::I32(5),
                ]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("mul.lo.s32"), "{s}");
    assert!(!s.contains("mad.lo.s32"), "{s}");
}

#[test]
fn binop_lt_emits_setp_lt_to_pred_register() {
    let kernel = descriptor("lt")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Lt), [0, 1], 2),
                ])
                .literals([LiteralValue::U32(3), LiteralValue::U32(4)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("setp.lt.u32"));
    assert!(s.contains(".reg .pred"));
}

#[test]
fn integer_shift_masks_rhs_to_reference_width() {
    let kernel = descriptor("masked_shift")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Shl), [0, 1], 2),
                    op(KernelOpKind::BinOpKind(BinOp::Shr), [0, 1], 3),
                ])
                .literals([LiteralValue::U32(1), LiteralValue::U32(33)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("and.b32"),
        "Fix: PTX shift lowering must mask the RHS to five bits before shl/shr."
    );
    assert!(
        s.contains(", 31;"),
        "Fix: PTX shift lowering must match the reference `rhs & 31` contract."
    );
    assert!(s.contains("shl.b32"));
    assert!(s.contains("shr.u32"));
}

#[test]
fn u32_power_of_two_const_mod_emits_mask_without_rem() {
    let kernel = descriptor("mod_pow2")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    op(KernelOpKind::BinOpKind(BinOp::Mod), [0, 1], 2),
                ])
                .literals([LiteralValue::U32(37), LiteralValue::U32(8)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("and.b32"),
        "Fix: u32 `% power_of_two` must lower to an integer mask.\n{s}"
    );
    assert!(
        !s.contains("rem.u32"),
        "Fix: u32 `% power_of_two` must not emit slow total modulo control flow.\n{s}"
    );
}

#[test]
fn unop_negate_emits_neg() {
    let kernel = descriptor("neg")
        .body(
            body()
                .ops([lit(0, 0), op(KernelOpKind::UnOpKind(UnOp::Negate), [0], 1)])
                .literal(LiteralValue::I32(-5)),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("neg.s32"));
}

#[test]
fn unop_reciprocal_emits_strict_or_approx_rcp() {
    let kernel = descriptor("reciprocal")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::UnOpKind(UnOp::Reciprocal), [0], 1),
                ])
                .literal(LiteralValue::F32(4.0)),
        )
        .build();
    let strict = emit(&kernel).unwrap();
    assert!(strict.contains("rcp.rn.f32"));
    let approx = emit_with_options(
        &kernel,
        PtxEmitOptions {
            target: ComputeCapability::SM_70,
            subgroup_size: 32,
            ulp_budget: Some(4),
            cooperative_grid_sync: false,
        },
    )
    .unwrap();
    assert!(approx.contains("rcp.approx.f32"));
}

#[test]
fn local_invocation_id_emits_tid_x() {
    let kernel = descriptor("tid")
        .dispatch(64, 1, 1)
        .body(body().op(op(KernelOpKind::LocalInvocationId, [0], 0)))
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("%tid.x"));
}

#[test]
fn workgroup_id_emits_ctaid() {
    let kernel = descriptor("wid")
        .dispatch(64, 1, 1)
        .body(body().ops([KernelOp {
            kind: KernelOpKind::WorkgroupId,
            operands: vec![1], // y axis
            result: Some(0),
        }]))
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("%ctaid.y"));
}

#[test]
fn trap_emits_lane_exit() {
    // Trap is genuinely unsupported in PTX phase 1.
    let kernel = descriptor("k")
        .body(
            body()
                .ops([
                    lit(0, 0),
                    effect(KernelOpKind::Trap { tag: "t".into() }, [0]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build();
    let r = emit(&kernel);
    let s = r.unwrap();
    assert!(s.contains("// vyre trap tag 1 t"));
    assert!(s.contains("bra $L_exit;"));
}

#[test]
fn add_of_two_single_use_muls_keeps_one_mul_available_for_mad() {
    let kernel = two_slot_u32_kernel(
        "mul_mul_add",
        vec![
            lit(0, 0),
            lit(1, 1),
            lit(2, 2),
            lit(3, 3),
            lit(4, 4),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [1, 2], 5),
            op(KernelOpKind::BinOpKind(BinOp::Mul), [3, 4], 6),
            op(KernelOpKind::BinOpKind(BinOp::Add), [5, 6], 7),
            effect(KernelOpKind::StoreGlobal, [1, 0, 7]),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
            LiteralValue::U32(5),
            LiteralValue::U32(7),
        ],
    );

    let s =
        emit(&kernel).expect("Fix: MAD fusion must not defer both Mul operands feeding one Add.");

    assert!(
        s.contains("mad.lo.u32") && s.contains("st.global.u32"),
        "one product must remain materialized as the MAD addend while the other Mul fuses:\n{s}"
    );
}
