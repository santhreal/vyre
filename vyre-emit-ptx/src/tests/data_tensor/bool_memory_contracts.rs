use super::*;
use vyre_lower::descriptor_builder::{
    SlotCount,
    body,
    descriptor,
    effect,
    global_ro,
    global_wo,
    lit,
    op,
};

#[test]
fn bool_global_load_uses_word_load_then_predicate_set() {
    let kernel = descriptor("bool_load")
        .slots([
            global_ro(0, DataType::Bool, "input").with_count(1),
            global_wo(1, DataType::U32, "out").with_count(1),
        ])
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::LoadGlobal, [0, 0], 1),
                    op(KernelOpKind::Cast {
                        target: DataType::U32,
                    }, [1], 2),
                    effect(KernelOpKind::StoreGlobal, [1, 0, 2]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let s = emit(&kernel).unwrap();
    assert!(
        !s.contains("ld.global.pred"),
        "PTX cannot load predicate registers from memory:\n{s}"
    );
    assert!(
        s.contains("ld.global.u32"),
        "Bool memory load must use the physical word ABI:\n{s}"
    );
    assert!(
        s.contains("setp.ne.u32"),
        "Bool memory load must canonicalize non-zero words to predicates:\n{s}"
    );
}

#[test]
fn bool_global_store_materializes_predicate_word() {
    let kernel = descriptor("bool_store")
        .slot(global_wo(0, DataType::Bool, "out").with_count(1))
        .body(
            body()
                .ops([
                    lit(0, 0),
                    lit(1, 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::Bool(true)]),
        )
        .build();

    let s = emit(&kernel).unwrap();
    assert!(
        !s.contains("st.global.pred"),
        "PTX cannot store predicate registers to memory:\n{s}"
    );
    assert!(
        s.contains("selp.u32"),
        "Bool memory store must materialize a 0/1 word:\n{s}"
    );
    assert!(
        s.contains("st.global.u32"),
        "Bool memory store must use the physical word ABI:\n{s}"
    );
}
