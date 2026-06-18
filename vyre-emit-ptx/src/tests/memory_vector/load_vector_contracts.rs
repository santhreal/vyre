use super::*;

#[test]
fn emit_fuses_four_adjacent_u32_loads_to_ptx_vector_load() {
    let s = emit(&two_slot_u32_kernel(
        "vec_load",
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![0, 0],
                result: Some(2),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![0, 1],
                result: Some(3),
            },
            KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![0, 3],
                result: Some(4),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![3, 1],
                result: Some(5),
            },
            KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![0, 5],
                result: Some(6),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![5, 1],
                result: Some(7),
            },
            KernelOp {
                kind: KernelOpKind::LoadGlobal,
                operands: vec![0, 7],
                result: Some(8),
            },
            KernelOp {
                kind: KernelOpKind::StoreGlobal,
                operands: vec![1, 0, 8],
                result: None,
            },
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(1)],
    ))
    .unwrap();
    assert!(s.contains("ld.global.nc.v4.u32"));
    assert_eq!(s.matches("ld.global.u32").count(), 0);
    assert!(
        !s.contains("add.u32"),
        "fused vector load must not leave dead scalar index-increment adds:\n{s}"
    );
}

#[test]
fn generated_dynamic_reassociated_load_indices_fuse_to_v4() {
    for seed in 0..1024 {
        let s = emit(&dynamic_reassociated_vector_load_kernel(seed))
            .unwrap_or_else(|error| panic!("seed {seed} failed to emit: {error}"));
        assert!(
            s.contains("ld.global.nc.v4.u32"),
            "seed {seed} must recover v4 load fusion after affine reassociation:\n{s}"
        );
        assert_eq!(
            s.matches("ld.global.u32").count() + s.matches("ld.global.nc.u32").count(),
            0,
            "seed {seed} must not leave scalar data loads after v4 load fusion:\n{s}"
        );
    }
}

#[test]
fn misaligned_scalar_gather_values_do_not_fuse_to_vector_store() {
    let s = emit(&dynamic_misaligned_gather_to_vector_store_kernel()).unwrap();
    assert!(
        !s.contains("ld.global.nc.v4.u32") && !s.contains("ld.global.v4.u32"),
        "Fix: misaligned dynamic gather must stay scalar on the load side.\n{s}"
    );
    assert!(
        !s.contains("st.global.v2.u32") && !s.contains("st.global.v4.u32"),
        "Fix: values produced by a scalarized misaligned gather must not be repacked into a vector store on live CUDA.\n{s}"
    );
    assert!(
        s.matches("ld.global.u32").count() + s.matches("ld.global.nc.u32").count() >= 4,
        "Fix: misaligned gather fixture must emit scalar loads.\n{s}"
    );
    assert!(
        s.matches("st.global.u32").count() >= 4,
        "Fix: misaligned gather fixture must emit scalar stores.\n{s}"
    );
}

