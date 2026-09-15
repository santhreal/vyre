//! `pipeline_prewarm` pattern analysis contracts.

use vyre_emit_naga::patterns::pipeline_prewarm::*;
use vyre_foundation::ir::DataType;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_rw, lit};
use vyre_lower::{BindingSlot, KernelDescriptor, KernelOpKind, LiteralValue};

fn binding(slot: u32) -> BindingSlot {
    global_rw(slot, DataType::U32, &format!("b{slot}"))
}

fn small_kernel() -> KernelDescriptor {
    descriptor("small")
        .slot(binding(0))
        .dispatch(64, 1, 1)
        .body(
            body()
                .ops([lit(0, 0), effect(KernelOpKind::Return, [])])
                .literal(LiteralValue::U32(0)),
        )
        .build()
}

#[test]
fn small_kernel_does_not_warrant_prewarm() {
    let h = analyze(&small_kernel());
    assert!(!h.should_prewarm);
    assert!(h.reason.contains("not worth it"));
}

#[test]
fn many_op_kernel_warrants_prewarm() {
    let mut ops = Vec::with_capacity(60);
    for i in 0..60 {
        ops.push(lit(0, i));
    }
    let kernel = descriptor("big")
        .body(body().ops(ops).literal(LiteralValue::U32(0)))
        .build();
    let h = analyze(&kernel);
    assert!(h.should_prewarm);
    assert!(h.reason.contains("op-count"));
}

#[test]
fn many_binding_kernel_warrants_prewarm() {
    let kernel = descriptor("many_bindings")
        .slots((0..6).map(binding))
        .build();
    let h = analyze(&kernel);
    assert!(h.should_prewarm);
    assert!(h.reason.contains("binding-count"));
}

#[test]
fn estimated_us_grows_with_op_and_binding_counts() {
    let small = analyze(&small_kernel());
    // 10 baseline + 2 ops + 50 * 1 binding = 62us
    assert_eq!(small.estimated_first_dispatch_us, 62);
}

#[test]
fn empty_kernel_estimated_at_baseline() {
    let kernel = descriptor("empty").dispatch(64, 1, 1).build();
    let h = analyze(&kernel);
    assert_eq!(h.estimated_first_dispatch_us, 10); // baseline only
    assert!(!h.should_prewarm);
}

#[test]
fn threshold_constant_is_documented_value() {
    assert_eq!(PREWARM_OP_THRESHOLD, 50);
}

#[test]
fn kernel_id_echoed_in_hint() {
    let h = analyze(&small_kernel());
    assert_eq!(h.kernel_id, "small");
}
