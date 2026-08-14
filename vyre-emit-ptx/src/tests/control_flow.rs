//! Test: control flow.
use super::*;
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_rw, lit, KernelDescriptorBuilder, SlotCount,
};

/// 64-thread kernel writing into one read-write `out` slot of `count` u32s.
/// The caller supplies the body.
fn predication_kernel(id: &str, count: u32) -> KernelDescriptorBuilder {
    descriptor(id)
        .slot(global_rw(0, DataType::U32, "out").with_count(count))
        .dispatch(64, 1, 1)
}

#[test]
fn region_op_passes_through_with_comment() {
    let kernel = descriptor("region")
        .body(
            body()
                .ops([effect(
                    KernelOpKind::Region {
                        generator: "vyre.libs.test".into(),
                    },
                    [0],
                )])
                .child(body()),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("// region: vyre.libs.test"));
}

// ============== Structured control flow + composite ops (parity push) ==============

#[test]
fn structured_if_then_emits_branch_and_label() {
    let kernel = descriptor("if_then")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([lit(0, 0), effect(KernelOpKind::StructuredIfThen, [0, 0])])
                .child(empty_child_body())
                .literal(LiteralValue::Bool(true)),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("@!"), "must emit a negated-pred branch");
    assert!(s.contains("bra "));
    assert!(s.contains("$L_if_end_"), "must emit an if_end label");
}

#[test]
fn structured_if_then_else_emits_else_label_and_unconditional_jump() {
    let kernel = descriptor("if_else")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    effect(KernelOpKind::StructuredIfThenElse, [0, 0, 1]),
                ])
                .children([empty_child_body(), empty_child_body()])
                .literal(LiteralValue::Bool(false)),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("$L_if_else_"), "must emit an else label");
    assert!(s.contains("$L_if_end_"));
    // Jump from end of then-body to end label.
    assert!(s.matches("bra ").count() >= 2, "if-else needs ≥ 2 bra ops");
}

#[test]
fn short_if_then_store_is_predicated_without_branch() {
    let kernel = predication_kernel("if_store", 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    effect(KernelOpKind::StructuredIfThen, [0, 0]),
                ])
                .child(body().op(effect(KernelOpKind::StoreGlobal, [0, 1, 2])))
                .literals([
                    LiteralValue::Bool(true),
                    LiteralValue::U32(0),
                    LiteralValue::U32(7),
                ]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("@%p"), "store must be guarded by the condition");
    assert!(s.contains(" st.global.u32"));
    assert!(
        !s.contains("$L_if_end_"),
        "single-store if must avoid branch/label divergence"
    );
}

#[test]
fn short_if_then_literal_store_body_is_predicated_without_branch() {
    let kernel = predication_kernel("if_literal_store", 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StructuredIfThen, [0, 0]),
                ])
                .children([body()
                    .ops([lit(0, 20), effect(KernelOpKind::StoreGlobal, [0, 1, 20])])
                    .literal(LiteralValue::U32(13))])
                .literals([LiteralValue::Bool(true), LiteralValue::U32(0)]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("@%p"));
    assert!(s.contains(" st.global.u32"));
    assert!(
        !s.contains("$L_if_end_"),
        "short pure-prefix store body must use predication instead of branch/reconvergence"
    );
}

#[test]
fn short_if_then_two_store_body_is_fully_predicated_without_branch() {
    let kernel = predication_kernel("if_two_stores", 2)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StructuredIfThen, [0, 0]),
                ])
                .children([body()
                    .ops([
                        lit(0, 20),
                        effect(KernelOpKind::StoreGlobal, [0, 1, 20]),
                        lit(1, 21),
                        effect(KernelOpKind::StoreGlobal, [0, 1, 21]),
                    ])
                    .literals([LiteralValue::U32(13), LiteralValue::U32(17)])])
                .literals([LiteralValue::Bool(true), LiteralValue::U32(0)]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("@%p"));
    assert_eq!(s.matches(" st.global.u32").count(), 2);
    assert!(
        !s.contains("$L_if_end_"),
        "short multi-store body must use predication instead of branch/reconvergence"
    );
}

#[test]
fn short_if_else_stores_are_dual_predicated_without_reconvergence_branch() {
    let kernel = predication_kernel("if_else_store", 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    lit(2, 2),
                    lit(3, 3),
                    effect(KernelOpKind::StructuredIfThenElse, [0, 0, 1]),
                ])
                .children([
                    body().op(effect(KernelOpKind::StoreGlobal, [0, 1, 2])),
                    body().op(effect(KernelOpKind::StoreGlobal, [0, 1, 3])),
                ])
                .literals([
                    LiteralValue::Bool(true),
                    LiteralValue::U32(0),
                    LiteralValue::U32(7),
                    LiteralValue::U32(9),
                ]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("@%p"));
    assert!(s.contains("@!%p"));
    assert_eq!(s.matches(" st.global.u32").count(), 2);
    assert!(
        !s.contains("$L_if_else_") && !s.contains("$L_if_end_"),
        "dual predicated store arms must avoid SIMT branch and reconvergence labels"
    );
}

#[test]
fn short_if_else_literal_store_bodies_are_dual_predicated_without_branch() {
    let kernel = predication_kernel("if_else_literal_store", 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StructuredIfThenElse, [0, 0, 1]),
                ])
                .children([
                    body()
                        .ops([lit(0, 20), effect(KernelOpKind::StoreGlobal, [0, 1, 20])])
                        .literal(LiteralValue::U32(21)),
                    body()
                        .ops([lit(0, 21), effect(KernelOpKind::StoreGlobal, [0, 1, 21])])
                        .literal(LiteralValue::U32(34)),
                ])
                .literals([LiteralValue::Bool(true), LiteralValue::U32(0)]),
        )
        .build();

    let s = emit(&kernel).unwrap();

    assert!(s.contains("@%p"));
    assert!(s.contains("@!%p"));
    assert_eq!(s.matches(" st.global.u32").count(), 2);
    assert!(
        !s.contains("$L_if_else_") && !s.contains("$L_if_end_"),
        "short pure-prefix store arms must avoid SIMT branch and reconvergence labels"
    );
}

#[test]
fn structured_for_loop_emits_head_label_setp_and_jump_back() {
    let kernel = descriptor("for_loop")
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(
                        KernelOpKind::StructuredForLoop {
                            loop_var: "i".into(),
                        },
                        [0, 1, 0],
                    ),
                ])
                .child(empty_child_body())
                .literals([LiteralValue::U32(0), LiteralValue::U32(64)]),
        )
        .build();
    let s = emit(&kernel).unwrap();
    assert!(s.contains("$L_for_head_"), "must emit head label");
    assert!(s.contains("$L_for_exit_"), "must emit exit label");
    assert!(s.contains("setp.ge.u32"), "must emit loop-bound predicate");
    assert!(s.contains("// for i in"));
}
