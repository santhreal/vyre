//! What the lowering boundary promises about observable effects.
//!
//! Descriptor verification proves the lowered form is well shaped and says
//! nothing about whether it still does what the program said. A descriptor with
//! one store removed verifies exactly as well as one with it, so the missing
//! store reaches a device as a buffer nobody wrote. The cases below pin the
//! comparison that catches it, in both directions and for the read-modify-write
//! case, and pin the two exclusions the comparison depends on: workgroup-scoped
//! storage has no semantic counterpart, and a dead read may legitimately
//! disappear.

use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{AtomicOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_rw, lit, op, shared_rw,
};
use vyre_lower::{
    check_effects, lower, lower_physical, AsyncTransaction, BindingSlot, EffectSignature,
    EquivalenceError, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
    TransactionScope, TRAP_SIDECAR_NAME, WORKGROUP_SLOT_BASE,
};

/// Program that reads one binding and stores into another.
fn copy_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(0)),
        )],
    )
}

/// Hand-built descriptor that loads slot 0 and stores into slot 1.
fn copy_descriptor(extra: Vec<KernelOp>, slots: Vec<BindingSlot>) -> KernelDescriptor {
    let mut ops = vec![
        lit(0, 0),
        op(KernelOpKind::LoadGlobal, [0, 0], 1),
        effect(KernelOpKind::StoreGlobal, [1, 0, 1]),
    ];
    ops.extend(extra);
    let mut built = body().literal(LiteralValue::U32(0));
    for one in ops {
        built = built.op(one);
    }
    let mut all = vec![
        global_ro(0, DataType::U32, "in"),
        global_rw(1, DataType::U32, "out"),
    ];
    all.extend(slots);
    descriptor("copy").slots(all).body(built).build()
}

fn drop_stores(body: &KernelBody) -> KernelBody {
    KernelBody {
        ops: body
            .ops
            .iter()
            .filter(|op| {
                !matches!(
                    op.kind,
                    KernelOpKind::StoreGlobal | KernelOpKind::VectorStoreGlobal { .. }
                )
            })
            .cloned()
            .collect(),
        child_bodies: body.child_bodies.iter().map(drop_stores).collect(),
        literals: body.literals.clone(),
    }
}

#[test]
fn a_kernel_that_lowers_states_the_same_effects_on_both_sides() {
    let program = copy_program();
    let lowering = lower_physical(&program).expect("the copy program lowers");

    let stated = EffectSignature::from_program(&lowering.program);
    let performed = EffectSignature::from_descriptor(lowering.kernel.descriptor());

    let read = stated.binding("in").expect("the program reads `in`");
    assert!(read.read && !read.written);
    let written = stated.binding("out").expect("the program writes `out`");
    assert!(written.written);
    assert_eq!(performed.binding("in"), stated.binding("in"));
    assert_eq!(performed.binding("out"), stated.binding("out"));
    check_effects(&stated, &performed, &[TRAP_SIDECAR_NAME])
        .expect("a faithful lowering states the same effects");
}

#[test]
fn a_store_the_lowering_dropped_is_reported_as_a_lost_write() {
    let program = copy_program();
    let mut descriptor = lower(&program).expect("the copy program lowers");
    descriptor.body = drop_stores(&descriptor.body);

    let errors = check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect_err("a kernel that stores nothing is not the program");

    assert_eq!(
        errors,
        vec![EquivalenceError::WriteLost {
            binding: "out".to_owned()
        }]
    );
}

#[test]
fn a_store_no_statement_performs_is_reported_as_invented() {
    let program = copy_program();
    let descriptor = copy_descriptor(vec![effect(KernelOpKind::StoreGlobal, [0, 0, 1])], vec![]);

    let errors = check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect_err("writing a read-only input is not the program");

    assert_eq!(
        errors,
        vec![EquivalenceError::WriteInvented {
            binding: "in".to_owned()
        }]
    );
}

#[test]
fn a_read_modify_write_only_the_physical_side_performs_is_reported() {
    let program = copy_program();
    let descriptor = copy_descriptor(
        vec![op(
            KernelOpKind::Atomic {
                op: AtomicOp::Add,
                ordering: MemoryOrdering::default(),
            },
            [1, 0, 1],
            2,
        )],
        vec![],
    );

    let errors = check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect_err("an invented atomic changes what other invocations observe");

    assert_eq!(
        errors,
        vec![
            EquivalenceError::AtomicDisagreement {
                binding: "out".to_owned(),
                semantic: false
            },
            EquivalenceError::ReadInvented {
                binding: "out".to_owned()
            },
        ],
        "an invented read-modify-write also reads storage the program only writes"
    );
}

#[test]
fn a_read_modify_write_lowered_as_a_plain_store_is_reported() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::read_write("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(1),
            Expr::atomic_add("out", Expr::u32(0), Expr::load("in", Expr::u32(0))),
        )],
    );
    let descriptor = copy_descriptor(vec![], vec![]);

    let errors = check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect_err("a dropped read-modify-write races where the program did not");

    assert_eq!(
        errors,
        vec![EquivalenceError::AtomicDisagreement {
            binding: "out".to_owned(),
            semantic: true
        }]
    );
}

#[test]
fn a_load_no_expression_performs_is_reported_as_an_invented_read() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7))],
    );
    let descriptor = copy_descriptor(vec![], vec![]);

    let errors = check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect_err("reading storage the program never reads is not the program");

    assert_eq!(
        errors,
        vec![EquivalenceError::ReadInvented {
            binding: "in".to_owned()
        }]
    );
}

#[test]
fn a_read_the_program_performs_may_disappear() {
    let program = copy_program();
    let descriptor = descriptor("store_only")
        .slots(vec![
            global_ro(0, DataType::U32, "in"),
            global_rw(1, DataType::U32, "out"),
        ])
        .body(
            body()
                .op(lit(0, 0))
                .op(effect(KernelOpKind::StoreGlobal, [1, 0, 0]))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect("a value nothing consumes is eliminable");
}

#[test]
fn the_diagnostic_sidecar_is_compared_only_when_it_is_not_excluded() {
    let program = copy_program();
    let sidecar = global_rw(2, DataType::U32, TRAP_SIDECAR_NAME);
    let descriptor = copy_descriptor(
        vec![effect(KernelOpKind::StoreGlobal, [2, 0, 1])],
        vec![sidecar],
    );
    let stated = EffectSignature::from_program(&program);
    let performed = EffectSignature::from_descriptor(&descriptor);

    check_effects(&stated, &performed, &[TRAP_SIDECAR_NAME])
        .expect("the sidecar is a binding lowering adds, so no program declares it");

    let errors = check_effects(&stated, &performed, &[])
        .expect_err("without the exclusion the sidecar is an invented write");
    assert_eq!(
        errors,
        vec![EquivalenceError::WriteInvented {
            binding: TRAP_SIDECAR_NAME.to_owned()
        }]
    );
}

#[test]
fn workgroup_storage_is_outside_the_boundary_comparison() {
    let tile = WORKGROUP_SLOT_BASE;
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(4),
            BufferDecl::workgroup("tile", 4, DataType::U32),
        ],
        [4, 1, 1],
        vec![
            Node::store("tile", Expr::u32(0), Expr::u32(1)),
            Node::store("out", Expr::u32(0), Expr::load("tile", Expr::u32(0))),
        ],
    );
    let descriptor = descriptor("shared")
        .slots(vec![
            global_rw(0, DataType::U32, "out"),
            shared_rw(tile, DataType::U32, 4, "tile"),
        ])
        .body(
            body()
                .op(lit(0, 0))
                .op(effect(KernelOpKind::StoreShared, [tile, 0, 0]))
                .op(op(KernelOpKind::LoadShared, [tile, 0], 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let stated = EffectSignature::from_program(&program);
    let performed = EffectSignature::from_descriptor(&descriptor);

    assert_eq!(stated.binding("tile"), None);
    assert_eq!(performed.binding("tile"), None);
    assert_eq!(stated.len(), 1);
    assert_eq!(performed.len(), 1);
    check_effects(&stated, &performed, &[TRAP_SIDECAR_NAME])
        .expect("shared storage has no caller-visible counterpart");
}

#[test]
fn a_dispatch_count_buffer_is_read_and_not_written() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("counts", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![
            Node::indirect_dispatch("counts", 0),
            Node::store("out", Expr::u32(0), Expr::u32(1)),
        ],
    );

    let stated = EffectSignature::from_program(&program);

    let counts = stated
        .binding("counts")
        .expect("the dispatch reads `counts`");
    assert!(counts.read, "the dispatch consumes the count");
    assert!(
        !counts.written && !counts.atomic,
        "a count buffer read is not a write, whatever dependency ordering assumes"
    );
}

#[test]
fn every_disagreement_is_reported_at_once() {
    let program = copy_program();
    let descriptor = descriptor("wrong")
        .slots(vec![
            global_ro(0, DataType::U32, "in"),
            global_rw(1, DataType::U32, "out"),
        ])
        .body(
            body()
                .op(lit(0, 0))
                .op(op(KernelOpKind::LoadGlobal, [0, 0], 1))
                .op(effect(KernelOpKind::StoreGlobal, [0, 0, 1]))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let errors = check_effects(
        &EffectSignature::from_program(&program),
        &EffectSignature::from_descriptor(&descriptor),
        &[TRAP_SIDECAR_NAME],
    )
    .expect_err("two bindings disagree");

    assert_eq!(
        errors,
        vec![
            EquivalenceError::WriteInvented {
                binding: "in".to_owned()
            },
            EquivalenceError::WriteLost {
                binding: "out".to_owned()
            },
        ],
        "one report names the whole difference, in binding-name order"
    );
}

#[test]
fn an_asynchronous_transfer_reads_its_source_and_writes_its_destination() {
    let transfer = KernelOpKind::AsyncLoad(Box::new(AsyncTransaction::unstaged(
        "dma".into(),
        TransactionScope::Workgroup,
    )));
    let descriptor = descriptor("transfer")
        .slots(vec![
            global_ro(0, DataType::U32, "in"),
            global_rw(1, DataType::U32, "out"),
        ])
        .body(
            body()
                .op(lit(0, 0))
                .op(effect(transfer, [0, 1, 0, 0]))
                .literal(LiteralValue::U32(0)),
        )
        .build();

    let performed = EffectSignature::from_descriptor(&descriptor);

    let source = performed.binding("in").expect("the transfer reads `in`");
    assert!(source.read && !source.written);
    let destination = performed.binding("out").expect("the transfer writes `out`");
    assert!(destination.written && !destination.read);
}
