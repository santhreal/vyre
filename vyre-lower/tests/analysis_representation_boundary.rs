//! Tests for analysis representation boundary and fact isolation (Section 188.2).
//!
//! Verifies that `vyre-lower` owns lowered `KernelDescriptor` facts (ResultId namespace,
//! body-local op sequences, binding slots, def-use chains, and dead-op analysis)
//! and preserves their semantic completeness without conflating them with pre-lowering Program facts.

use vyre_foundation::ir::BinOp;
use vyre_lower::analyses::alias_facts::{AliasFactSet, NoAliasFact};
use vyre_lower::analyses::{analyze_dead_op, analyze_def_use};
use vyre_lower::{BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind};

#[test]
fn descriptor_def_use_chains_track_result_ids_and_uses() {
    let mut ops = Vec::new();
    // Op 0: produces result %1 (e.g. Add)
    ops.push(KernelOp {
        kind: KernelOpKind::BinOpKind(BinOp::Add),
        operands: vec![100, 200],
        result: Some(1),
    });
    // Op 1: uses %1 in operand position 0
    ops.push(KernelOp {
        kind: KernelOpKind::BinOpKind(BinOp::Mul),
        operands: vec![1, 300],
        result: Some(2),
    });
    // Op 2: side-effecting store using %2 as operand 2
    ops.push(KernelOp {
        kind: KernelOpKind::StoreGlobal,
        operands: vec![0, 0, 2],
        result: None,
    });

    let desc = KernelDescriptor {
        id: "test_def_use_kernel".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals: vec![],
        },
    };

    let report = analyze_def_use(&desc);
    assert!(!report.bodies.is_empty());

    let top_chains = &report.bodies[0];
    let uses_of_1 = top_chains.uses.get(&1).expect("result 1 must have uses");
    assert_eq!(uses_of_1.len(), 1);
    assert_eq!(uses_of_1[0].op_index, 1);
    assert_eq!(uses_of_1[0].operand_pos, 0);

    let uses_of_2 = top_chains.uses.get(&2).expect("result 2 must have uses");
    assert_eq!(uses_of_2.len(), 1);
    assert_eq!(uses_of_2[0].op_index, 2);
}

#[test]
fn descriptor_dead_op_analysis_never_flags_side_effects() {
    let mut ops = Vec::new();
    // Op 0: pure add whose result %1 is unused -> dead
    ops.push(KernelOp {
        kind: KernelOpKind::BinOpKind(BinOp::Add),
        operands: vec![10, 20],
        result: Some(1),
    });
    // Op 1: barrier (side effect) -> never dead
    ops.push(KernelOp {
        kind: KernelOpKind::Barrier {
            ordering: vyre_foundation::ir::MemoryOrdering::SeqCst,
        },
        operands: vec![],
        result: None,
    });
    // Op 2: store (side effect) -> never dead
    ops.push(KernelOp {
        kind: KernelOpKind::StoreGlobal,
        operands: vec![0, 0, 99],
        result: None,
    });

    let desc = KernelDescriptor {
        id: "test_dead_op_kernel".to_string(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals: vec![],
        },
    };

    let report = analyze_dead_op(&desc);
    assert_eq!(
        report.dead_op_indices,
        vec![0],
        "only op 0 should be flagged dead"
    );
}

#[test]
fn descriptor_alias_facts_enforce_bidirectional_slot_isolation() {
    let mut alias_set = AliasFactSet::default();
    alias_set.insert_no_alias(NoAliasFact {
        left_binding: 0,
        left_index: 10,
        right_binding: 1,
        right_index: 20,
    });

    assert!(alias_set.proves_no_alias(0, 10, 1, 20));
    assert!(alias_set.proves_no_alias(1, 20, 0, 10));
    assert!(!alias_set.proves_no_alias(0, 10, 0, 10));
}
