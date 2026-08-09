//! Naga entry and parallel emission contracts.
use super::*;

#[test]
fn empty_descriptor_emits_compute_entrypoint() {
    let module = emit(&empty_desc()).unwrap();
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].name, "main");
    assert_eq!(module.entry_points[0].workgroup_size, [1, 1, 1]);
}

#[test]
fn identical_descriptor_emits_equal_entry_metadata() {
    let desc = empty_desc_with_workgroup("deterministic", 1);
    let first = emit(&desc).unwrap();
    let second = emit(&desc).unwrap();
    assert_eq!(first.entry_points[0].name, second.entry_points[0].name);
    assert_eq!(
        first.entry_points[0].workgroup_size,
        second.entry_points[0].workgroup_size
    );
}

#[test]
fn emit_many_preserves_input_order_for_independent_descriptors() {
    let descs = vec![
        empty_desc_with_workgroup("a", 1),
        empty_desc_with_workgroup("b", 2),
        empty_desc_with_workgroup("c", 3),
        empty_desc_with_workgroup("d", 4),
    ];

    let modules = emit_many(&descs)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .expect("parallel emit should succeed for independent descriptors");

    let workgroups: Vec<[u32; 3]> = modules
        .iter()
        .map(|module| module.entry_points[0].workgroup_size)
        .collect();
    assert_eq!(workgroups, vec![[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1]]);
}

#[test]
fn scalar_store_descriptor_emits_globals_and_statements() {
    let desc = KernelDescriptor {
        id: "store".into(),
        bindings: BindingLayout {
            slots: vec![vyre_lower::BindingSlot {
                slot: 0,
                element_type: DataType::U32,
                element_count: None,
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "out".into(),
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
    let module = emit(&desc).unwrap();
    assert_eq!(module.global_variables.len(), 1);
    assert_eq!(module.entry_points.len(), 1);
    assert!(!module.entry_points[0].function.body.is_empty());
}
