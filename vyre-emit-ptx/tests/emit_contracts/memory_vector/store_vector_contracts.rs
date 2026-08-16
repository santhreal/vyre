use super::*;
use vyre_lower::descriptor_builder::{effect, lit, op};

#[test]
fn emit_fuses_four_adjacent_u32_stores_to_ptx_vector_store() {
    let s = emit(&two_slot_u32_kernel(
        "vec_store",
        vec![
            lit(0, 0),
            lit(1, 1),
            lit(2, 2),
            lit(3, 3),
            lit(4, 4),
            lit(5, 5),
            effect(KernelOpKind::StoreGlobal, [1, 0, 2]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 6),
            effect(KernelOpKind::StoreGlobal, [1, 6, 3]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [6, 1], 7),
            effect(KernelOpKind::StoreGlobal, [1, 7, 4]),
            op(KernelOpKind::BinOpKind(BinOp::Add), [7, 1], 8),
            effect(KernelOpKind::StoreGlobal, [1, 8, 5]),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(1),
            LiteralValue::U32(10),
            LiteralValue::U32(11),
            LiteralValue::U32(12),
            LiteralValue::U32(13),
        ],
    ))
    .unwrap();
    assert!(s.contains("st.global.v4.u32"));
    assert!(!s.contains("st.global.u32"));
    assert!(
        !s.contains("add.u32"),
        "fused vector store must not leave dead scalar index-increment adds:\n{s}"
    );
}

#[test]
fn generated_dynamic_reassociated_store_indices_fuse_to_v4() {
    for seed in 0..1024 {
        let s = emit(&dynamic_reassociated_vector_store_kernel(seed))
            .unwrap_or_else(|error| panic!("seed {seed} failed to emit: {error}"));
        assert!(
            s.contains("st.global.v4.u32"),
            "seed {seed} must recover v4 store fusion after affine reassociation:\n{s}"
        );
        assert_eq!(
            s.matches("st.global.u32").count(),
            0,
            "seed {seed} must not leave scalar stores after v4 store fusion:\n{s}"
        );
    }
}

#[test]
fn emit_fuses_vector_store_across_folded_literal_index_gaps() {
    let s = emit(&two_slot_u32_kernel(
        "folded_literal_vec_store",
        vec![
            lit(0, 0),
            lit(1, 1),
            lit(2, 2),
            lit(3, 3),
            lit(4, 4),
            effect(KernelOpKind::StoreGlobal, [1, 0, 1]),
            lit(5, 5),
            effect(KernelOpKind::StoreGlobal, [1, 5, 2]),
            lit(6, 6),
            effect(KernelOpKind::StoreGlobal, [1, 6, 3]),
            lit(7, 7),
            effect(KernelOpKind::StoreGlobal, [1, 7, 4]),
        ],
        vec![
            LiteralValue::U32(0),
            LiteralValue::U32(10),
            LiteralValue::U32(11),
            LiteralValue::U32(12),
            LiteralValue::U32(13),
            LiteralValue::U32(1),
            LiteralValue::U32(2),
            LiteralValue::U32(3),
        ],
    ))
    .unwrap();

    assert!(
        s.contains("st.global.v4.u32"),
        "Fix: folded adjacent store indices must still fuse into a vector store.\n{s}"
    );
    assert!(
        !s.contains("st.global.u32"),
        "Fix: folded-index vector store fusion must not leave scalar stores behind.\n{s}"
    );
}
