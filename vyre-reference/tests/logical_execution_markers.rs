//! Reference semantics for schedule-free logical execution markers.
//!
//! WHY: reference parity evaluates semantic library IR before device schedule
//! lowering. Logical coordinates must match their corresponding invocation
//! coordinates, and logical barriers must follow the same ordering contract as
//! physical barriers.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};
use vyre_reference::value::Value;
use vyre_reference::workgroup::{Invocation, InvocationIds, Memory};
use vyre_reference::{expr as eval_expr, reference_eval};

#[test]
fn logical_coordinates_read_the_semantic_invocation_coordinates() {
    let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
    let ids = InvocationIds {
        global: [10, 11, 12],
        workgroup: [20, 21, 22],
        local: [30, 31, 32],
    };

    for (expr, expected) in [
        (Expr::logical_index(0), 10),
        (Expr::logical_index(1), 11),
        (Expr::logical_index(2), 12),
        (Expr::logical_tile_index(0), 20),
        (Expr::logical_tile_index(1), 21),
        (Expr::logical_tile_index(2), 22),
        (Expr::logical_within_tile_index(0), 30),
        (Expr::logical_within_tile_index(1), 31),
        (Expr::logical_within_tile_index(2), 32),
    ] {
        let value = eval_expr::eval(
            &expr,
            &mut Invocation::new(ids, program.entry()),
            &mut Memory::empty(),
            &program,
        )
        .unwrap();
        assert_eq!(value, Value::U32(expected), "{expr:?}");
    }
}

#[test]
fn every_logical_barrier_ordering_is_executed_or_rejected_by_contract() {
    let orderings: Vec<_> = (0..=u8::MAX)
        .filter_map(|tag| MemoryOrdering::from_wire_tag(tag).ok())
        .collect();
    assert!(!orderings.is_empty());

    for ordering in orderings {
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::logical_barrier(ordering),
                Node::store("out", Expr::u32(0), Expr::u32(7)),
            ],
        );
        let result = reference_eval(&program, &[]);
        if ordering == MemoryOrdering::Relaxed {
            let error = result.expect_err("Relaxed barriers must fail semantic validation");
            assert!(error.to_string().contains("V043"));
        } else {
            assert_eq!(
                result.unwrap(),
                vec![Value::Bytes(7u32.to_le_bytes().to_vec().into())]
            );
        }
    }
}
