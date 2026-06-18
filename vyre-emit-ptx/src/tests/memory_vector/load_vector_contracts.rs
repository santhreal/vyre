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

/// When adjacent `LoadGlobal` ops exist at the same slot but the base index is
/// not provably aligned (runtime value with unknown modulo), `align_vector_chain`
/// returns `None` and the fallback to scalar loads must be announced via a PTX
/// comment. Before this fix, the fallback was silent: the operator had no way to
/// detect that a hot scan kernel was running in scalar mode after vector-fusion
/// analysis completed.
#[test]
fn vector_fusion_alignment_fallback_emits_diagnostic_comment() {
    // Base index: `LocalInvocationId * 3`, which is NOT divisible by 4 for
    // most values (tid % 4 != 0 in general), so alignment cannot be proved
    // for a v4 or v2 load. The emitter must fall back to scalar loads AND emit
    // `// vyre: vector-fusion-skipped` in the PTX text.
    let desc = KernelDescriptor {
        id: "unaligned_fusion_fallback".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: None,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "input".into(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(1),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::WriteOnly,
                    name: "output".into(),
                },
            ],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
                // tid = LocalInvocationId[0]
                KernelOp { kind: KernelOpKind::LocalInvocationId, operands: vec![0], result: Some(0) },
                // stride = 3  (not a power of 2 aligned multiplier)
                KernelOp { kind: KernelOpKind::Literal, operands: vec![0], result: Some(1) },
                // base = tid * 3
                KernelOp { kind: KernelOpKind::BinOpKind(BinOp::Mul), operands: vec![0, 1], result: Some(2) },
                // inc1 = 1
                KernelOp { kind: KernelOpKind::Literal, operands: vec![1], result: Some(3) },
                // base+1 = base + 1
                KernelOp { kind: KernelOpKind::BinOpKind(BinOp::Add), operands: vec![2, 3], result: Some(4) },
                // Four adjacent loads at base, base+1, base+2, base+3.
                // Alignment of `tid * 3` mod 4 is unknown → fusion must be skipped.
                KernelOp { kind: KernelOpKind::LoadGlobal, operands: vec![0, 2], result: Some(5) },
                KernelOp { kind: KernelOpKind::LoadGlobal, operands: vec![0, 4], result: Some(6) },
                // Store just one result to keep the kernel non-trivial.
                KernelOp { kind: KernelOpKind::Literal, operands: vec![2], result: Some(7) },
                KernelOp { kind: KernelOpKind::StoreGlobal, operands: vec![1, 7, 5], result: None },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(3), LiteralValue::U32(1), LiteralValue::U32(0)],
        },
    };
    let s = emit(&desc).expect("Fix: unaligned adjacent load kernel must emit PTX without error.");
    // Must NOT fuse into a vector load (alignment unknown for tid*3).
    assert!(
        !s.contains("ld.global.nc.v2.u32") && !s.contains("ld.global.nc.v4.u32")
            && !s.contains("ld.global.v2.u32") && !s.contains("ld.global.v4.u32"),
        "Fix: loads at a non-provably-aligned base must NOT be fused into a vector load; \
         got:\n{}", &s[..s.len().min(600)]
    );
    // Must emit the Law-10 diagnostic comment announcing the skipped fusion.
    assert!(
        s.contains("vyre: vector-fusion-skipped"),
        "Fix: when vector fusion is skipped due to unknown alignment, the emitter must \
         write a `// vyre: vector-fusion-skipped` comment in the PTX text so the operator \
         can detect scalar fallback. Got PTX:\n{}", &s[..s.len().min(600)]
    );
}

/// Constant-memory `LoadConstant` ops must NOT be fused into a `ld.global.nc.vN`
/// vector load. Before this fix, `is_vector_load_op` included `LoadConstant`,
/// causing four adjacent constant loads to attempt vector fusion via the global
/// address-space path. Since constant-memory bindings use `ld.global` (their
/// pointer is converted to global via `cvta.to.global`), the `.nc` (non-coherent)
/// cache suffix and vector wrapper were both wrong — `.nc` implies an independent
/// read-only bypass that is semantically different from the constant-cache path.
///
/// After this fix, `LoadConstant` is excluded from vector fusion and falls through
/// to scalar `ld.global.u32` loads.
#[test]
fn constant_binding_loads_are_not_vectorised_as_global_nc() {
    let desc = KernelDescriptor {
        id: "const_no_vec".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: Some(16),
                    memory_class: MemoryClass::Constant,
                    visibility: BindingVisibility::ReadOnly,
                    name: "const_input".into(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::WriteOnly,
                    name: "output".into(),
                },
            ],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            // Four consecutive LoadConstant ops at indices 0, 1, 2, 3.
            ops: vec![
                KernelOp { kind: KernelOpKind::Literal, operands: vec![0], result: Some(0) },
                KernelOp { kind: KernelOpKind::Literal, operands: vec![1], result: Some(1) },
                KernelOp { kind: KernelOpKind::Literal, operands: vec![2], result: Some(2) },
                KernelOp { kind: KernelOpKind::Literal, operands: vec![3], result: Some(3) },
                KernelOp { kind: KernelOpKind::LoadConstant, operands: vec![0, 0], result: Some(4) },
                KernelOp { kind: KernelOpKind::LoadConstant, operands: vec![0, 1], result: Some(5) },
                KernelOp { kind: KernelOpKind::LoadConstant, operands: vec![0, 2], result: Some(6) },
                KernelOp { kind: KernelOpKind::LoadConstant, operands: vec![0, 3], result: Some(7) },
                KernelOp { kind: KernelOpKind::StoreGlobal, operands: vec![1, 0, 4], result: None },
                KernelOp { kind: KernelOpKind::StoreGlobal, operands: vec![1, 1, 5], result: None },
                KernelOp { kind: KernelOpKind::StoreGlobal, operands: vec![1, 2, 6], result: None },
                KernelOp { kind: KernelOpKind::StoreGlobal, operands: vec![1, 3, 7], result: None },
            ],
            child_bodies: vec![],
            literals: vec![
                LiteralValue::U32(0),
                LiteralValue::U32(1),
                LiteralValue::U32(2),
                LiteralValue::U32(3),
            ],
        },
    };
    let s = emit(&desc).expect("Fix: constant-memory kernel must emit PTX without error.");
    assert!(
        !s.contains("ld.global.nc.v4.u32") && !s.contains("ld.global.nc.v2.u32"),
        "Fix: LoadConstant ops must NOT be fused into ld.global.nc vector loads; \
         constant bindings use a different cache path. PTX:\n{}",
        &s[..s.len().min(600)]
    );
    // Constant loads must fall through to scalar ld.global loads (since vyre
    // uses cvta.to.global for all buffer parameters including constant bindings).
    assert!(
        s.matches("ld.global.u32").count() >= 4,
        "Fix: constant-memory loads must emit scalar ld.global.u32, got:\n{}",
        &s[..s.len().min(600)]
    );
}

