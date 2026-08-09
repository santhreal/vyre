//! Test: atomics.
use super::*;

#[test]
fn nested_if_inside_for_emits_correct_label_nesting() {
    // for { if { ... } }
    let kernel = KernelDescriptor {
        id: "nested".into(),
        bindings: BindingLayout { slots: vec![] },
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
                    kind: KernelOpKind::StructuredForLoop {
                        loop_var: "i".into(),
                    },
                    operands: vec![0, 1, 0],
                    result: None,
                },
            ],
            child_bodies: vec![KernelBody {
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![0],
                        result: Some(10),
                    },
                    KernelOp {
                        kind: KernelOpKind::StructuredIfThen,
                        operands: vec![10, 0],
                        result: None,
                    },
                ],
                child_bodies: vec![empty_child_body()],
                literals: vec![LiteralValue::Bool(true)],
            }],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(8)],
        },
    };
    let s = emit(&kernel).unwrap();
    assert!(s.contains("$L_for_head_"));
    assert!(s.contains("$L_if_end_"));
}

#[test]
fn atomic_add_emits_atom_global_add_u32() {
    use vyre_foundation::ir::AtomicOp;
    let kernel = KernelDescriptor {
        id: "atomic_add".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: None,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "counter".into(),
            }],
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
                    kind: KernelOpKind::Atomic {
                        op: AtomicOp::Add,
                        ordering: vyre_foundation::memory_model::MemoryOrdering::SeqCst,
                    },
                    operands: vec![0, 0, 1],
                    result: Some(2),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    };
    let s = emit(&kernel).unwrap();
    assert!(s.contains("atom.global.add.u32"));
}

#[test]
fn atomic_exchange_emits_atom_global_exch_b32() {
    use vyre_foundation::ir::AtomicOp;
    let kernel = KernelDescriptor {
        id: "atomic_exchange".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: None,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "slot".into(),
            }],
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
                    kind: KernelOpKind::Atomic {
                        op: AtomicOp::Exchange,
                        ordering: vyre_foundation::memory_model::MemoryOrdering::SeqCst,
                    },
                    operands: vec![0, 0, 1],
                    result: Some(2),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    };
    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("atom.global.exch.b32"),
        "PTX exch must use bit-size suffix, not .u32:\n{s}"
    );
    assert!(
        !s.contains("atom.global.exch.u32"),
        "ptxas rejects atom.global.exch.u32:\n{s}"
    );
}

#[test]
fn atomic_bitwise_emits_atom_global_b32_suffix() {
    use vyre_foundation::ir::AtomicOp;
    for (op, mnemonic) in [
        (AtomicOp::And, "and"),
        (AtomicOp::Or, "or"),
        (AtomicOp::Xor, "xor"),
    ] {
        let kernel = KernelDescriptor {
            id: "atomic_bitwise".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: None,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "slot".into(),
                }],
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
                        kind: KernelOpKind::Atomic {
                            op,
                            ordering: vyre_foundation::memory_model::MemoryOrdering::Relaxed,
                        },
                        operands: vec![0, 0, 1],
                        result: Some(2),
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
            },
        };
        let s = emit(&kernel).unwrap();
        assert!(
            s.contains(&format!("atom.global.{mnemonic}.b32")),
            "PTX atom.{mnemonic} must use .b32, not .u32/.s32:\n{s}"
        );
    }
}

#[test]
fn atomic_bitwise_bool_operand_materializes_u32_before_atom() {
    use vyre_foundation::ir::AtomicOp;
    let kernel = KernelDescriptor {
        id: "atomic_bool_to_b32".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: None,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "slot".into(),
            }],
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
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Eq),
                    operands: vec![1, 2],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::Atomic {
                        op: AtomicOp::Or,
                        ordering: vyre_foundation::memory_model::MemoryOrdering::Relaxed,
                    },
                    operands: vec![0, 0, 3],
                    result: Some(4),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
        },
    };
    let s = emit(&kernel).unwrap();
    let atom_line = s
        .lines()
        .find(|line| line.contains("atom.global.or.b32"))
        .expect("atomic OR must emit .b32");
    assert!(
        s.contains("selp.u32"),
        "bool atomic operand must be materialized as 0/1 before atom.global.or.b32:\n{s}"
    );
    assert!(
        !atom_line.contains("], %p"),
        "ptxas rejects predicate operands for atom.global.or.b32; got:\n{atom_line}\n{s}"
    );
}

#[test]
fn atomic_min_max_emit_correct_mnemonic() {
    use vyre_foundation::ir::AtomicOp;
    for (op, mnemonic) in [(AtomicOp::Min, "min"), (AtomicOp::Max, "max")] {
        let kernel = KernelDescriptor {
            id: "atomic_minmax".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: None,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "b".into(),
                }],
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
                        kind: KernelOpKind::Atomic {
                            op,
                            ordering: vyre_foundation::memory_model::MemoryOrdering::Relaxed,
                        },
                        operands: vec![0, 0, 1],
                        result: Some(2),
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
            },
        };
        let s = emit(&kernel).unwrap();
        assert!(s.contains(&format!("atom.global.{mnemonic}.u32")));
    }
}

/// Build a kernel whose only atomic targets a SHARED (workgroup) binding.
///
/// `slot 0` is the shared bin array, indexed by the literal `0`, incremented by
/// the literal `1`.
fn shared_atomic_kernel(op: vyre_foundation::ir::AtomicOp) -> KernelDescriptor {
    KernelDescriptor {
        id: "shared_atomic".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: Some(256),
                memory_class: MemoryClass::Shared,
                visibility: BindingVisibility::ReadWrite,
                name: "wg_bins".into(),
            }],
        },
        dispatch: Dispatch::new(256, 1, 1),
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
                    kind: KernelOpKind::Atomic {
                        op,
                        ordering: vyre_foundation::memory_model::MemoryOrdering::SeqCst,
                    },
                    operands: vec![0, 0, 1],
                    result: Some(2),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    }
}

/// An atomic on a workgroup-shared binding must lower to `atom.shared.*`
/// against the shared symbol, never to `atom.global.*`.
///
/// Defect this locks out: `emit_atomic` resolving its address through
/// `slot_to_ptr` (which is populated for global bindings only) and hardcoding
/// the `.global` state space. Shared bindings are absent from `slot_to_ptr`, so
/// the old code failed with the misleading `global pointer not preloaded`
/// binding error even though the IR is well formed and `BufferDecl::workgroup`
/// plus `Expr::atomic_add` compose to exactly this shape. Had the pointer been
/// present, `atom.global` on a shared address is an illegal-address fault or a
/// silently wrong read of unrelated global memory.
///
/// This is the enabler for workgroup-privatized histograms: privatizing 256 bins
/// into shared memory requires an atomic increment on shared memory.
#[test]
fn atomic_add_on_shared_binding_emits_atom_shared_not_atom_global() {
    use vyre_foundation::ir::AtomicOp;
    let s = emit(&shared_atomic_kernel(AtomicOp::Add))
        .expect("Fix: an atomic on a workgroup-shared binding must emit, not error");

    assert!(
        s.contains("atom.shared.add.u32"),
        "Fix: shared-binding atomic add must lower to atom.shared.add.u32; emitted PTX:\n{s}"
    );
    assert!(
        !s.contains("atom.global"),
        "Fix: no atom.global may be emitted for a shared-only binding; a global-space \
         atomic on a shared address faults or corrupts unrelated memory. Emitted PTX:\n{s}"
    );
    assert!(
        s.contains(".shared .align 4 .b8 shared_buf_0[1024];"),
        "Fix: the 256-element u32 shared bin array must be declared as 1024 bytes; \
         emitted PTX:\n{s}"
    );
}

/// The shared atomic's address must be a 32-bit shared-window offset derived
/// from the shared symbol, not a 64-bit global address.
///
/// Defect this locks out: reusing the global `mul.wide.u32` + `add.u64` address
/// arithmetic for a shared operand. PTX shared addresses live in a distinct
/// 32-bit window, so a `.u64` register there is the wrong operand width and
/// addresses the wrong location.
#[test]
fn shared_atomic_address_is_a_shared_window_offset_from_the_shared_symbol() {
    use vyre_foundation::ir::AtomicOp;
    let s = emit(&shared_atomic_kernel(AtomicOp::Add))
        .expect("Fix: an atomic on a workgroup-shared binding must emit, not error");

    let atom_line = s
        .lines()
        .find(|line| line.contains("atom.shared.add.u32"))
        .expect("Fix: shared atomic add must be emitted");
    let addr_reg = atom_line
        .split_once('[')
        .and_then(|(_, rest)| rest.split_once(']'))
        .map(|(inner, _)| inner.trim().to_string())
        .expect("Fix: shared atomic must address through [reg]");

    assert!(
        s.contains(&format!("mov.u32    {addr_reg}, shared_buf_0;"))
            || s.contains(&format!("add.u32    {addr_reg},")),
        "Fix: the shared atomic address register {addr_reg} must be produced by 32-bit \
         shared-symbol arithmetic; emitted PTX:\n{s}"
    );
    assert!(
        s.contains("mov.u32") && s.contains("shared_buf_0"),
        "Fix: the shared atomic must take its base from the shared symbol \
         shared_buf_0; emitted PTX:\n{s}"
    );
}

/// Every RMW mnemonic must honor the shared state space, not just add. A
/// per-mnemonic `.global` literal is exactly how one op gets fixed and the rest
/// keep faulting.
#[test]
fn every_shared_atomic_rmw_mnemonic_uses_the_shared_state_space() {
    use vyre_foundation::ir::AtomicOp;
    for (op, expected) in [
        (AtomicOp::Add, "atom.shared.add.u32"),
        (AtomicOp::Or, "atom.shared.or.b32"),
        (AtomicOp::And, "atom.shared.and.b32"),
        (AtomicOp::Xor, "atom.shared.xor.b32"),
        (AtomicOp::Min, "atom.shared.min.u32"),
        (AtomicOp::Max, "atom.shared.max.u32"),
        (AtomicOp::Exchange, "atom.shared.exch.b32"),
    ] {
        let s = emit(&shared_atomic_kernel(op))
            .unwrap_or_else(|error| panic!("Fix: shared atomic {op:?} must emit: {error:?}"));
        assert!(
            s.contains(expected),
            "Fix: shared atomic {op:?} must emit `{expected}`; emitted PTX:\n{s}"
        );
        assert!(
            !s.contains("atom.global"),
            "Fix: shared atomic {op:?} must not emit an atom.global; emitted PTX:\n{s}"
        );
    }
}
