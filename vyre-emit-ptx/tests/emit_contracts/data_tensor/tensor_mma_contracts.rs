use super::*;
use vyre_lower::descriptor_builder::{body, descriptor, effect, global_wo, lit, SlotCount};

#[test]
fn matrix_mma_emits_real_mma_sync_and_binds_all_four_results() {
    let mut ops = Vec::new();
    let mut literals = Vec::new();
    for id in 0..6 {
        literals.push(LiteralValue::U32(id));
        ops.push(lit(id, id));
    }
    for id in 6..10 {
        literals.push(LiteralValue::F32(0.0));
        ops.push(lit(id, id));
    }
    ops.push(KernelOp {
        kind: mma_f16_m16n8k16(),
        operands: (0..10).collect(),
        result: Some(10),
    });
    ops.push(lit(10, 14));
    literals.push(LiteralValue::U32(0));
    ops.push(effect(KernelOpKind::StoreGlobal, [0, 14, 13]));

    let kernel = descriptor("mma")
        .slot(global_wo(0, DataType::F32, "out").with_count(1))
        .dispatch(32, 1, 1)
        .body(body().ops(ops).literals(literals))
        .build();

    vyre_lower::verify(&kernel)
        .expect("MatrixMma must publish result ids base..base+4 to verifier");
    let s = emit_with_target(&kernel, ComputeCapability::SM_70).unwrap();
    assert!(s.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"));
    assert!(
        s.contains("st.global.f32"),
        "fourth MatrixMma result id must be usable by later ops"
    );
}
