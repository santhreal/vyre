//! Cross-emitter parity: every cleaned descriptor in the corpus must emit
//! through Naga, PTX, and SPIR-V. Failure in any target breaks the
//! substrate-neutral `KernelDescriptor` contract.
//!
//! Lives in `vyre-emit-spirv` because it depends on both
//! `vyre-emit-naga` (always) and `vyre-emit-ptx` (added as dev-dep
//! for this test). Putting it here avoids the in-flight `vyre-libs`
//! Codex hold that blocks `vyre-bench`.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};

fn rw_slot(id: u32, name: &str) -> BindingSlot {
    BindingSlot {
        slot: id,
        element_type: DataType::U32,
        element_count: None,
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: name.into(),
    }
}

fn descriptor_corpus() -> Vec<KernelDescriptor> {
    vec![
        // (1) Empty kernel.
        KernelDescriptor {
            id: "empty".into(),
            bindings: BindingLayout { slots: vec![] },
            dispatch: Dispatch::new(1, 1, 1),
            body: KernelBody {
                ops: vec![],
                child_bodies: vec![],
                literals: vec![],
            },
        },
        // (2) Single store.
        KernelDescriptor {
            id: "single_store".into(),
            bindings: BindingLayout {
                slots: vec![rw_slot(0, "out")],
            },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
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
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
            },
        },
        // (3) Add and store.
        KernelDescriptor {
            id: "add_store".into(),
            bindings: BindingLayout {
                slots: vec![rw_slot(0, "out")],
            },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    }, // 3
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    }, // 4
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![2],
                        result: Some(2),
                    }, // 0 (idx)
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![0, 1],
                        result: Some(3),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 2, 3],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![
                    LiteralValue::U32(3),
                    LiteralValue::U32(4),
                    LiteralValue::U32(0),
                ],
            },
        },
        // (4) Identity arithmetic that the rewrite stack will eliminate
        //     before any of the emitters even sees it.
        KernelDescriptor {
            id: "identity_heavy".into(),
            bindings: BindingLayout {
                slots: vec![rw_slot(0, "out")],
            },
            dispatch: Dispatch::new(64, 1, 1),
            body: KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(0),
                    }, // 0
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(1),
                    }, // 99
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![1, 0],
                        result: Some(2),
                    }, // identity
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Mul),
                        operands: vec![1, 0],
                        result: Some(3),
                    }, // absorbing zero
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 1],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(99)],
            },
        },
        // (5) Store-load-store pattern that load_forwarding+dead_store
        //     should collapse.
        KernelDescriptor {
            id: "stl".into(),
            bindings: BindingLayout {
                slots: vec![rw_slot(0, "buf")],
            },
            dispatch: Dispatch::new(1, 1, 1),
            body: KernelBody {
                ops: vec![
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
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 1],
                        result: None,
                    },
                    KernelOp {
                        kind: KernelOpKind::LoadGlobal,
                        operands: vec![0, 0],
                        result: Some(2),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 2],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
            },
        },
    ]
}

#[test]
fn every_descriptor_lowers_through_all_three_emitters() {
    for desc in descriptor_corpus() {
        let id = desc.id.clone();
        let cleaned = vyre_lower::verify_then_optimize(&desc)
            .expect("verified descriptor cleanup")
            .0;

        let naga_module = vyre_emit_naga::emit(&cleaned)
            .unwrap_or_else(|error| panic!("Naga emit failed for `{id}`: {error:?}"));
        assert!(
            !naga_module.entry_points.is_empty(),
            "naga module for `{id}` must expose an entry point"
        );

        let ptx = vyre_emit_ptx::emit(&cleaned)
            .unwrap_or_else(|error| panic!("PTX emit failed for `{id}`: {error:?}"));
        assert!(
            ptx.contains(".version"),
            "ptx for `{id}` must include a version directive"
        );

        let spirv_words = vyre_emit_spirv::emit(&cleaned)
            .unwrap_or_else(|error| panic!("SPIR-V emit failed for `{id}`: {error:?}"));
        assert_eq!(
            spirv_words.first().copied(),
            Some(vyre_emit_spirv::SPIRV_MAGIC),
            "spirv for `{id}` must start with the SPIR-V magic word"
        );
    }
}

#[test]
fn naga_and_spirv_main_entry_points_match() {
    // Naga and SPIR-V come from the same Naga module; their entry
    // point names + workgroup sizes must be identical.
    for desc in descriptor_corpus() {
        let naga_module = vyre_emit_naga::emit(
            &vyre_lower::verify_then_optimize(&desc)
                .expect("verified descriptor cleanup")
                .0,
        )
        .unwrap();
        let spirv_words = vyre_emit_spirv::emit(
            &vyre_lower::verify_then_optimize(&desc)
                .expect("verified descriptor cleanup")
                .0,
        )
        .unwrap();

        // Naga module entry point matches descriptor's dispatch.
        let entry = &naga_module.entry_points[0];
        assert_eq!(entry.name, "main");
        assert_eq!(entry.workgroup_size, desc.dispatch.workgroup_size);

        // SPIR-V starts with the magic word.
        assert_eq!(spirv_words[0], vyre_emit_spirv::SPIRV_MAGIC);
    }
}

#[test]
fn ptx_output_contains_required_directives_for_every_kernel() {
    for desc in descriptor_corpus() {
        if desc.body.ops.is_empty() {
            // PTX skipped on empty kernels  -  nothing to emit.
            continue;
        }
        let ptx = vyre_emit_ptx::emit(
            &vyre_lower::verify_then_optimize(&desc)
                .expect("verified descriptor cleanup")
                .0,
        )
        .unwrap();
        assert!(
            ptx.contains(".version"),
            "PTX for `{}` missing .version",
            desc.id
        );
        assert!(
            ptx.contains(".target"),
            "PTX for `{}` missing .target",
            desc.id
        );
    }
}

#[test]
fn shared_cleanup_preserves_naga_emit_acceptance() {
    for desc in descriptor_corpus() {
        let cleaned = vyre_lower::verify_then_optimize(&desc)
            .expect("descriptor corpus must pass shared cleanup")
            .0;
        assert!(
            vyre_emit_naga::emit(&cleaned).is_ok(),
            "cleaned descriptor `{}` must emit through Naga",
            desc.id
        );
    }
}

#[test]
fn every_audit_layer_succeeds_without_panic_on_corpus() {
    // The audit family must be robust across realistic shapes: each
    // layer's audit() function takes a descriptor and produces a
    // typed report. None should panic, even on edge cases (empty
    // kernel, identity-only arithmetic, etc.).
    use vyre_emit_ptx::ComputeCapability;
    for desc in descriptor_corpus() {
        // Substrate-neutral.
        let lower_report = vyre_lower::audit::audit(&desc);
        assert_eq!(lower_report.kernel_id, desc.id);

        // Naga-specific.
        let naga_report = vyre_emit_naga::patterns::audit(&desc);
        assert_eq!(naga_report.kernel_id, desc.id);

        // PTX-specific.
        let ptx_report = vyre_emit_ptx::patterns::audit(&desc, ComputeCapability::SM_80);
        assert_eq!(ptx_report.kernel_id, desc.id);
        assert_eq!(ptx_report.target, ComputeCapability::SM_80);

        // SPIR-V-specific.
        let spirv_report = vyre_emit_spirv::patterns::audit(&desc);
        assert_eq!(spirv_report.kernel_id, desc.id);
    }
}

#[test]
fn verify_then_optimize_succeeds_on_corpus() {
    // The production-grade entry point should succeed for every shape
    // in the corpus (every shape is well-formed by construction; the
    // rewrite stack is fuzz-verified to produce well-formed output).
    for desc in descriptor_corpus() {
        let r = vyre_lower::verify_then_optimize(&desc);
        match r {
            Ok((optimized, stats)) => {
                assert_eq!(optimized.id, desc.id, "id round-trips");
                assert!(stats.iterations >= 1);
            }
            Err(f) => panic!(
                "verify_then_optimize failed on `{}`: {:?}",
                desc.id,
                f.errors()
            ),
        }
    }
}

#[test]
fn audit_optimized_doesnt_panic_across_corpus() {
    // Mirror of every_audit_layer_succeeds_without_panic_on_corpus
    // but for the _optimized variants  -  runs run_all first, then
    // audit. None should panic; kernel_id should round-trip.
    use vyre_emit_ptx::ComputeCapability;
    for desc in descriptor_corpus() {
        let lower = vyre_lower::audit::audit_optimized(&desc);
        assert_eq!(lower.kernel_id, desc.id);
        let naga = vyre_emit_naga::patterns::audit_optimized(&desc);
        assert_eq!(naga.kernel_id, desc.id);
        let ptx = vyre_emit_ptx::patterns::audit_optimized(&desc, ComputeCapability::SM_80);
        assert_eq!(ptx.kernel_id, desc.id);
        let spirv = vyre_emit_spirv::patterns::audit_optimized(&desc);
        assert_eq!(spirv.kernel_id, desc.id);
    }
}

#[test]
fn audit_carries_kernel_id_through_every_layer() {
    // For a kernel with a distinct id, the id should survive into all
    // four audit reports unchanged.
    let desc = KernelDescriptor {
        id: "named_kernel_42".into(),
        bindings: BindingLayout {
            slots: vec![rw_slot(0, "buf")],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![
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
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 1],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
        },
    };
    use vyre_emit_ptx::ComputeCapability;
    assert_eq!(vyre_lower::audit::audit(&desc).kernel_id, "named_kernel_42");
    assert_eq!(
        vyre_emit_naga::patterns::audit(&desc).kernel_id,
        "named_kernel_42"
    );
    assert_eq!(
        vyre_emit_ptx::patterns::audit(&desc, ComputeCapability::SM_70).kernel_id,
        "named_kernel_42"
    );
    assert_eq!(
        vyre_emit_spirv::patterns::audit(&desc).kernel_id,
        "named_kernel_42"
    );
}
