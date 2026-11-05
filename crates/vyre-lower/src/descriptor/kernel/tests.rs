use super::*;
use crate::descriptor::test_descriptors::build;
use crate::descriptor::{BindingLayout, BindingSlot, BindingVisibility, LiteralValue, MemoryClass};
use crate::descriptor_builder::{effect, for_loop, if_then, lit, op};
use vyre_foundation::ir::DataType;
use vyre_foundation::ir::MemoryOrdering;

fn binding(slot: u32, element: DataType, mc: MemoryClass) -> BindingSlot {
    BindingSlot {
        slot,
        element_type: element,
        element_count: None,
        memory_class: mc,
        visibility: BindingVisibility::ReadWrite,
        name: format!("b{slot}"),
    }
}

#[test]
fn summary_includes_all_counts() {
    let d = build(vec![], vec![]);
    let s = d.summary();
    assert!(s.contains("k:"));
    assert!(s.contains("0 ops"));
    assert!(s.contains("1 bindings"));
    assert!(s.contains("0 child bodies"));
    assert!(s.contains("1 literals"));
    assert!(s.contains("[64, 1, 1]"));
}

#[test]
fn summary_compact_terser_form() {
    let d = build(vec![lit(0, 0)], vec![]);
    let s = d.summary_compact();
    assert_eq!(s, "k(1 ops, 1 bindings)");
}

#[test]
fn unnamed_descriptor_uses_unnamed_label() {
    let mut d = build(vec![], vec![]);
    d.id = String::new();
    let s = d.summary();
    assert!(s.contains("<unnamed>"));
}

#[test]
fn total_ops_recurses_into_child_bodies() {
    let child = KernelBody {
        ops: vec![lit(0, 0), lit(0, 1)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(5)],
    };
    let parent_ops = vec![lit(0, 0)];
    let d = build(parent_ops, vec![child]);
    assert_eq!(d.body.ops.len(), 1); // shallow
    assert_eq!(d.total_ops(), 3); // 1 parent + 2 child
}

#[test]
fn body_at_empty_path_returns_parent() {
    let d = build(vec![lit(0, 7)], vec![]);
    let body = d.body_at(&[]).unwrap();
    assert_eq!(body.ops.len(), 1);
    assert_eq!(body.ops[0].result, Some(7));
}

#[test]
fn body_at_descends_into_children() {
    let grandchild = KernelBody {
        ops: vec![lit(0, 99)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(7)],
    };
    let child = KernelBody {
        ops: vec![],
        child_bodies: vec![grandchild],
        literals: vec![],
    };
    let d = build(vec![], vec![child]);
    // Path [0]: first child of parent  -  empty body with one grandchild.
    let b = d.body_at(&[0]).unwrap();
    assert!(b.ops.is_empty());
    // Path [0, 0]: grandchild  -  has the Literal with result 99.
    let b = d.body_at(&[0, 0]).unwrap();
    assert_eq!(b.ops[0].result, Some(99));
}

#[test]
fn body_at_out_of_range_returns_none() {
    let d = build(vec![], vec![]);
    assert!(d.body_at(&[5]).is_none());
    assert!(d.body_at(&[0, 0]).is_none());
}

#[test]
fn body_count_includes_parent_plus_recursive_children() {
    let nested = KernelBody {
        ops: vec![],
        child_bodies: vec![KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        }],
        literals: vec![],
    };
    let d = build(vec![], vec![nested]);
    // Parent (1) + first child (1) + grandchild (1) = 3.
    assert_eq!(d.body_count(), 3);
}

#[test]
fn body_count_flat_kernel_is_one() {
    let d = build(vec![], vec![]);
    assert_eq!(d.body_count(), 1);
}

#[test]
fn max_body_depth_flat_is_zero() {
    let d = build(vec![], vec![]);
    assert_eq!(d.max_body_depth(), 0);
}

#[test]
fn max_body_depth_one_if_is_one() {
    let child = KernelBody {
        ops: vec![],
        child_bodies: vec![],
        literals: vec![],
    };
    let d = build(vec![], vec![child]);
    assert_eq!(d.max_body_depth(), 1);
}

#[test]
fn max_body_depth_two_levels() {
    let grandchild = KernelBody {
        ops: vec![],
        child_bodies: vec![],
        literals: vec![],
    };
    let child = KernelBody {
        ops: vec![],
        child_bodies: vec![grandchild],
        literals: vec![],
    };
    let d = build(vec![], vec![child]);
    assert_eq!(d.max_body_depth(), 2);
}

#[test]
fn total_ops_zero_for_empty_kernel() {
    let d = build(vec![], vec![]);
    assert_eq!(d.total_ops(), 0);
}

#[test]
fn is_empty_true_when_no_ops() {
    let d = build(vec![], vec![]);
    assert!(d.is_empty());
}

#[test]
fn is_empty_false_when_parent_has_ops() {
    let d = build(vec![lit(0, 0)], vec![]);
    assert!(!d.is_empty());
    assert_eq!(d.total_ops(), 1);
}

#[test]
fn is_empty_false_when_child_has_ops() {
    let child = KernelBody {
        ops: vec![lit(0, 0)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let d = build(vec![], vec![child]);
    assert!(!d.is_empty());
    assert_eq!(d.total_ops(), 1);
}

#[test]
fn has_side_effects_true_with_store() {
    let d = build(
        vec![lit(0, 0), effect(KernelOpKind::StoreGlobal, [0, 0, 0])],
        vec![],
    );
    assert!(d.has_side_effects());
}

#[test]
fn has_side_effects_false_with_only_arithmetic() {
    let d = build(
        vec![
            lit(0, 0),
            op(
                KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                [0, 0],
                1,
            ),
        ],
        vec![],
    );
    assert!(!d.has_side_effects());
}

#[test]
fn has_side_effects_true_for_async_and_indirect_dispatch_ops() {
    // Regression: AsyncLoad writes shared memory other threads read,
    // AsyncWait is a sync point, and IndirectDispatch reconfigures the grid
    //: all cross-thread/dispatch effects (like the already-listed Barrier /
    // AsyncStore), so a descriptor containing one is NOT droppable. They
    // were omitted from the side-effecting set before the exhaustive-match
    // change, which would have let a "drop pure descriptor" caller drop one.
    for kind in [
        KernelOpKind::async_load("t".into()),
        KernelOpKind::async_wait("t".into()),
        KernelOpKind::IndirectDispatch { count_offset: 0 },
    ] {
        let d = build(vec![effect(kind.clone(), [0])], vec![]);
        assert!(
            d.has_side_effects(),
            "{kind:?} is a cross-thread/dispatch effect and must not be droppable"
        );
        assert!(!d.is_pure(), "{kind:?} must not be classified pure");
    }
}

#[test]
fn ops_iter_visits_parent_then_children_in_order() {
    let child0 = KernelBody {
        ops: vec![lit(0, 10), lit(0, 11)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let child1 = KernelBody {
        ops: vec![lit(0, 20)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(2)],
    };
    let d = build(vec![lit(0, 0), lit(0, 1)], vec![child0, child1]);
    let visited: Vec<u32> = d.ops_iter().map(|o| o.result.unwrap()).collect();
    // Parent ops (0, 1) first, then child0 (10, 11), then child1 (20).
    assert_eq!(visited, vec![0, 1, 10, 11, 20]);
}

#[test]
fn ops_iter_count_matches_total_ops() {
    let child = KernelBody {
        ops: vec![lit(0, 0), lit(0, 1)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(7)],
    };
    let d = build(vec![lit(0, 0)], vec![child]);
    assert_eq!(d.ops_iter().count(), d.total_ops());
}

#[test]
fn dispatch_total_threads_multiplies_dims() {
    let d = build(vec![], vec![]);
    assert_eq!(d.dispatch_total_threads(), 64); // build() uses Dispatch::new(64, 1, 1)

    let mut d2 = build(vec![], vec![]);
    d2.dispatch = Dispatch::new(8, 8, 4);
    assert_eq!(d2.dispatch_total_threads(), 256);
}

#[test]
fn with_id_preserves_everything_else() {
    let d = build(vec![lit(0, 0)], vec![]);
    let renamed = d.with_id("renamed");
    assert_eq!(renamed.id, "renamed");
    assert_eq!(d.id, "k"); // original unchanged
    assert_eq!(renamed.body.ops.len(), d.body.ops.len());
    assert_eq!(renamed.bindings, d.bindings);
    assert_eq!(renamed.dispatch, d.dispatch);
}

#[test]
fn dispatch_total_threads_saturates_on_overflow() {
    let mut d = build(vec![], vec![]);
    d.dispatch = Dispatch::new(u32::MAX, u32::MAX, u32::MAX);
    // Saturating multiplication means we get u32::MAX rather than wrap.
    assert_eq!(d.dispatch_total_threads(), u32::MAX);
}

#[test]
fn find_op_by_id_in_parent() {
    let d = build(vec![lit(0, 7), lit(0, 42)], vec![]);
    let op = d.find_op_by_id(42).expect("Fix: found");
    assert_eq!(op.result, Some(42));
    assert!(d.find_op_by_id(99).is_none());
}

#[test]
fn find_op_by_id_finds_in_child() {
    let child = KernelBody {
        ops: vec![lit(0, 100)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(7)],
    };
    let d = build(vec![], vec![child]);
    assert!(d.find_op_by_id(100).is_some());
}

#[test]
fn ops_iter_empty_descriptor_yields_none() {
    let d = build(vec![], vec![]);
    assert!(d.ops_iter().next().is_none());
}

#[test]
fn is_pure_inverse_of_has_side_effects() {
    let pure_kernel = build(vec![lit(0, 0)], vec![]);
    assert!(pure_kernel.is_pure());
    assert!(!pure_kernel.has_side_effects());

    let impure = build(
        vec![lit(0, 0), effect(KernelOpKind::StoreGlobal, [0, 0, 0])],
        vec![],
    );
    assert!(!impure.is_pure());
    assert!(impure.has_side_effects());
}

#[test]
fn empty_descriptor_round_trips_serde_byte_stable() {
    let k = KernelDescriptor {
        id: "test".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    let json1 = serde_json::to_string(&k).unwrap();
    let parsed: KernelDescriptor = serde_json::from_str(&json1).unwrap();
    let json2 = serde_json::to_string(&parsed).unwrap();
    assert_eq!(json1, json2);
    assert_eq!(k, parsed);
}

#[test]
fn one_store_kernel_round_trips_byte_stable() {
    let k = KernelDescriptor {
        id: "store_one".into(),
        bindings: BindingLayout {
            slots: vec![binding(0, DataType::U32, MemoryClass::Global)],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                lit(0, 0),
                lit(1, 1),
                effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
        },
    };
    let json1 = serde_json::to_string(&k).unwrap();
    let parsed: KernelDescriptor = serde_json::from_str(&json1).unwrap();
    let json2 = serde_json::to_string(&parsed).unwrap();
    assert_eq!(json1, json2);
}

#[test]
fn nested_if_then_body_round_trips() {
    let inner = KernelBody {
        ops: vec![effect(
            KernelOpKind::Barrier {
                ordering: MemoryOrdering::SeqCst,
            },
            [],
        )],
        child_bodies: vec![],
        literals: vec![],
    };
    let outer = KernelBody {
        ops: vec![lit(0, 0), if_then(0, 0)],
        child_bodies: vec![inner],
        literals: vec![LiteralValue::Bool(true)],
    };
    let k = KernelDescriptor {
        id: "if_then".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: outer,
    };
    let json1 = serde_json::to_string(&k).unwrap();
    let parsed: KernelDescriptor = serde_json::from_str(&json1).unwrap();
    let json2 = serde_json::to_string(&parsed).unwrap();
    assert_eq!(json1, json2);
}

#[test]
fn for_loop_with_var_name_round_trips() {
    let body = KernelBody {
        ops: vec![],
        child_bodies: vec![],
        literals: vec![],
    };
    let outer = KernelBody {
        ops: vec![lit(0, 0), lit(1, 1), for_loop("i", 0, 1, 0)],
        child_bodies: vec![body],
        literals: vec![LiteralValue::U32(0), LiteralValue::U32(64)],
    };
    let k = KernelDescriptor {
        id: "for_i".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: outer,
    };
    let json = serde_json::to_string(&k).unwrap();
    let parsed: KernelDescriptor = serde_json::from_str(&json).unwrap();
    assert_eq!(k, parsed);
}

#[test]
fn async_load_wait_carry_tag() {
    let body = KernelBody {
        ops: vec![
            lit(0, 0),
            lit(1, 1),
            effect(KernelOpKind::async_load("chunk-0".into()), [0, 1, 0, 1]),
            effect(KernelOpKind::async_wait("chunk-0".into()), []),
        ],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(0), LiteralValue::U32(16)],
    };
    let k = KernelDescriptor {
        id: "async".into(),
        bindings: BindingLayout {
            slots: vec![
                binding(0, DataType::U32, MemoryClass::Global),
                binding(1, DataType::U32, MemoryClass::Shared),
            ],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body,
    };
    let json = serde_json::to_string(&k).unwrap();
    let parsed: KernelDescriptor = serde_json::from_str(&json).unwrap();
    assert_eq!(k, parsed);
}

#[test]
fn dispatch_constructor_preserves_axes() {
    let d = Dispatch::new(64, 4, 2);
    assert_eq!(d.workgroup_size, [64, 4, 2]);
}
