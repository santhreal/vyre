//! Fixture-builder contracts.
//!
//! The builders are what every other descriptor test is written against, so a
//! builder that drifts from the struct literal it replaces moves every suite
//! that uses it at once.

use vyre_foundation::ir::DataType;
use vyre_lower::analyses::child_body_operands;
use vyre_lower::descriptor_builder::*;
use vyre_lower::*;

#[test]
fn builder_output_matches_the_struct_literal_it_replaces() {
    let built = descriptor("k")
        .slot(global_rw(0, DataType::U32, "buf"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)])
                .op(lit(0, 0))
                .op(lit(1, 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1])),
        )
        .build();

    let literal = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout {
            slots: vec![BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: None,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "buf".into(),
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
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 1],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(7)],
        },
    };

    assert_eq!(built, literal);
}

#[test]
fn child_body_index_is_the_append_order() {
    let built = descriptor("nested")
        .body(
            body()
                .op(effect(KernelOpKind::StructuredIfThenElse, [0, 0, 1]))
                .child(body().op(lit(0, 10)))
                .child(body().op(lit(0, 20))),
        )
        .build();
    assert_eq!(built.body.child_bodies.len(), 2);
    assert_eq!(built.body.child_bodies[0].ops[0].result, Some(10));
    assert_eq!(built.body.child_bodies[1].ops[0].result, Some(20));
}

/// The child-body operand positions the structured constructors write must
/// be the positions `analyses::child_body_operands` reads.
///
/// WHY: fixtures used to spell the operand vector out by hand, so the
/// answer to "which operand is a child index" lived in every test module
/// that built a branch. Moving a child index on one side only produced a
/// fixture no analysis descended into, and a test that then proved nothing.
/// This goes red if either side moves a position.
///
/// It says nothing about kinds no constructor covers; `descent_contract`
/// in `analyses` owns those.
#[test]
fn the_structured_constructors_name_the_child_indices_the_walk_reads() {
    for (op, expected) in [
        (if_then(7, 3), vec![3]),
        (if_then_else(7, 3, 4), vec![3, 4]),
        (for_loop("i", 1, 2, 5), vec![5]),
    ] {
        assert_eq!(
            child_body_operands(&op.kind, &op.operands).collect::<Vec<_>>(),
            expected,
            "Fix: {:?} writes its child indices where child_body_operands does not read them.",
            op.kind
        );
    }
}

#[test]
fn defaults_are_empty_body_no_bindings_single_invocation() {
    let built = descriptor("empty").build();
    assert!(built.bindings.slots.is_empty());
    assert!(built.body.ops.is_empty());
    assert!(built.body.child_bodies.is_empty());
    assert!(built.body.literals.is_empty());
    assert_eq!(built.dispatch, Dispatch::new(1, 1, 1));
}

#[test]
fn shared_slot_carries_its_element_count() {
    let s = shared_rw(1, DataType::F32, 64, "tile");
    assert_eq!(s.element_count, Some(64));
    assert_eq!(s.memory_class, MemoryClass::Shared);
    assert_eq!(global_ro(0, DataType::F32, "g").element_count, None);
}

/// The two capability fixtures must sit on opposite sides of the admission
/// gate for every feature, or a rejection test written against them proves
/// nothing about the gate.
#[test]
fn the_capability_fixtures_bracket_the_subgroup_admission_gate() {
    let all = all_subgroup_capabilities();
    assert_eq!(all.count(), 4);
    assert!(all.first_missing(all).is_none());

    let none = target_without_subgroups().subgroup;
    assert!(!none.any());
    assert_eq!(none.first_missing(all), Some("subgroup.basic"));
}

#[test]
fn permissive_limits_admit_a_full_invocation_workgroup_and_reject_one_over() {
    let limits = permissive_workgroup_limits();
    assert!(validate_workgroup_size([1024, 1, 1], limits).is_empty());
    assert!(!validate_workgroup_size([1025, 1, 1], limits).is_empty());
}
