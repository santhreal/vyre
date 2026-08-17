//! Dead operation elimination for KernelDescriptor.
//!
//! Strips pure operations whose results are never read by any op in the kernel
//! body or its child bodies. Preserves all effectful operations and all SSA result IDs.

use rustc_hash::FxHashSet;

use crate::op_facts::kernel_op_kind_is_dce_pure;
use crate::operand_class::{classify_operand, OperandClass};
use crate::{KernelBody, KernelDescriptor};

/// Strip dead pure operations from the descriptor.
#[must_use]
pub fn rewrite_dead_ops(descriptor: &KernelDescriptor) -> KernelDescriptor {
    let mut referenced_results = FxHashSet::default();
    collect_referenced_results(&descriptor.body, &mut referenced_results);
    let mut output = descriptor.clone();
    strip_dead_in_body(&mut output.body, &referenced_results);
    output
}

fn strip_dead_in_body(body: &mut KernelBody, referenced_results: &FxHashSet<u32>) {
    for child in &mut body.child_bodies {
        strip_dead_in_body(child, referenced_results);
    }

    body.ops.retain(|op| {
        if let Some(result) = op.result {
            if kernel_op_kind_is_dce_pure(&op.kind) && !referenced_results.contains(&result) {
                return false;
            }
        }
        true
    });
}

fn collect_referenced_results(body: &KernelBody, referenced: &mut FxHashSet<u32>) {
    for op in &body.ops {
        for (pos, &operand) in op.operands.iter().enumerate() {
            if classify_operand(&op.kind, pos) == OperandClass::ResultRef {
                referenced.insert(operand);
            }
        }
    }
    for child in &body.child_bodies {
        collect_referenced_results(child, referenced);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor_builder::lit;
    use crate::{
        BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelOp, KernelOpKind,
        LiteralValue, MemoryClass,
    };
    use vyre_foundation::ir::{BinOp, DataType};

    #[test]
    fn unreferenced_pure_op_is_stripped() {
        let desc = KernelDescriptor {
            id: "dead_op_test".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    name: "out".into(),
                    element_type: DataType::U32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    element_count: Some(64),
                }],
            },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops: vec![
                    lit(0, 0), // result 0, used by store
                    lit(0, 1), // result 1, dead!
                    lit(0, 2), // result 2, dead!
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 0], // slot 0, index 0, value 0
                        result: None,
                    },
                ],
                literals: vec![LiteralValue::U32(42)],
                child_bodies: vec![],
            },
        };

        let rewritten = rewrite_dead_ops(&desc);

        assert_eq!(rewritten.body.ops.len(), 2);
        assert_eq!(rewritten.body.ops[0].result, Some(0));
        assert_eq!(rewritten.body.ops[1].kind, KernelOpKind::StoreGlobal);
        assert!(crate::verify(&rewritten).is_ok());
    }

    #[test]
    fn effectful_op_without_result_is_preserved() {
        let desc = KernelDescriptor {
            id: "effectful_test".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    name: "out".into(),
                    element_type: DataType::U32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    element_count: Some(64),
                }],
            },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops: vec![
                    lit(0, 0),
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 0],
                        result: None,
                    },
                    KernelOp {
                        kind: KernelOpKind::Return,
                        operands: vec![],
                        result: None,
                    },
                ],
                literals: vec![LiteralValue::U32(0)],
                child_bodies: vec![],
            },
        };

        let rewritten = rewrite_dead_ops(&desc);
        assert_eq!(rewritten.body.ops.len(), 3);
    }

    #[test]
    fn referenced_pure_op_is_preserved() {
        let desc = KernelDescriptor {
            id: "used_pure_test".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    name: "out".into(),
                    element_type: DataType::U32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    element_count: Some(64),
                }],
            },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops: vec![
                    lit(0, 0),
                    lit(0, 1),
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(BinOp::Add),
                        operands: vec![0, 1],
                        result: Some(2),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 0, 2],
                        result: None,
                    },
                ],
                literals: vec![LiteralValue::U32(10)],
                child_bodies: vec![],
            },
        };

        let rewritten = rewrite_dead_ops(&desc);
        assert_eq!(rewritten.body.ops.len(), 4);
    }
    #[test]
    fn child_body_dead_ops_stripped_while_preserving_cross_body_references() {
        let desc = KernelDescriptor {
            id: "child_body_dead_op_test".into(),
            bindings: BindingLayout {
                slots: vec![BindingSlot {
                    slot: 0,
                    name: "out".into(),
                    element_type: DataType::U32,
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    element_count: Some(64),
                }],
            },
            dispatch: Dispatch {
                workgroup_size: [64, 1, 1],
            },
            body: KernelBody {
                ops: vec![
                    lit(0, 0), // parent result 0 (used in child)
                    lit(0, 1), // parent result 1 (dead in parent)
                    KernelOp {
                        kind: KernelOpKind::StructuredIfThenElse,
                        operands: vec![0, 0, 1], // cond 0, then 0, else 1
                        result: None,
                    },
                ],
                literals: vec![LiteralValue::U32(1)],
                child_bodies: vec![
                    KernelBody {
                        ops: vec![
                            lit(0, 2), // child result 2 (dead in child)
                            KernelOp {
                                kind: KernelOpKind::StoreGlobal,
                                operands: vec![0, 0, 0], // uses parent result 0
                                result: None,
                            },
                        ],
                        literals: vec![LiteralValue::U32(2)],
                        child_bodies: vec![],
                    },
                    KernelBody {
                        ops: vec![KernelOp {
                            kind: KernelOpKind::StoreGlobal,
                            operands: vec![0, 0, 0],
                            result: None,
                        }],
                        literals: vec![],
                        child_bodies: vec![],
                    },
                ],
            },
        };

        let rewritten = rewrite_dead_ops(&desc);

        // Parent result 1 stripped, result 0 kept.
        assert_eq!(rewritten.body.ops.len(), 2);
        assert_eq!(rewritten.body.ops[0].result, Some(0));

        // Child 0 result 2 stripped, store kept.
        assert_eq!(rewritten.body.child_bodies[0].ops.len(), 1);
        assert_eq!(
            rewritten.body.child_bodies[0].ops[0].kind,
            KernelOpKind::StoreGlobal
        );

        assert!(crate::verify(&rewritten).is_ok());
    }
}
