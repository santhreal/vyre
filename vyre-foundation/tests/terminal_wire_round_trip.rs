//! Terminal enum variant wire-format round trips.
//!
//! Which operator variants exist is owned by
//! `tests/support/spec_variant_tables.rs`, and what a round trip has to
//! preserve is owned by `tests/support/wire_round_trip.rs`. That every variant
//! survives one is this suite's contract. The `DataType` list below stays
//! local: a cast target may be `Handle`, `Vec`, `TensorShaped` or `Opaque`,
//! none of which a buffer element table describes.

#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;
#[path = "../../tests/support/wire_round_trip.rs"]
mod wire_round_trip;

use smallvec::smallvec;
use spec_variant_tables::{builtin_atomic_ops, builtin_bin_ops, builtin_un_ops};
use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{AtomicOp, BinOp, BufferDecl, DataType, Expr, Node, Program, UnOp};
use vyre_spec::extension::{
    ExtensionAtomicOpId, ExtensionBinOpId, ExtensionDataTypeId, ExtensionUnOpId,
};
use vyre_spec::TypeId;
use wire_round_trip::assert_canonical_wire_round_trip;

#[test]
fn every_terminal_data_type_round_trips_in_cast_targets() {
    let variants = vec![
        DataType::U32,
        DataType::I32,
        DataType::U64,
        DataType::Vec2U32,
        DataType::Vec4U32,
        DataType::Bool,
        DataType::Bytes,
        DataType::Array { element_size: 16 },
        DataType::F16,
        DataType::BF16,
        DataType::F32,
        DataType::F64,
        DataType::Tensor,
        DataType::U8,
        DataType::U16,
        DataType::I8,
        DataType::I16,
        DataType::I64,
        DataType::Handle(TypeId(0x1357_2468)),
        DataType::Vec {
            element: Box::new(DataType::U16),
            count: 3,
        },
        DataType::TensorShaped {
            element: Box::new(DataType::F32),
            shape: smallvec![2, 3, 5, 7],
        },
        DataType::Opaque(ExtensionDataTypeId(0x8000_0001)),
    ];

    let nodes = variants
        .into_iter()
        .enumerate()
        .map(|(idx, target)| {
            Node::let_bind(
                format!("cast_{idx}"),
                Expr::Cast {
                    target,
                    value: Box::new(Expr::u32(idx as u32)),
                },
            )
        })
        .chain([Node::Return])
        .collect();

    assert_canonical_wire_round_trip(
        &Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            nodes,
        ),
        "every cast target",
    );
}

#[test]
fn every_terminal_binop_round_trips() {
    let mut variants = builtin_bin_ops();
    variants.push(BinOp::Opaque(ExtensionBinOpId(0x8000_0101)));

    let nodes = variants
        .into_iter()
        .enumerate()
        .map(|(idx, op)| {
            Node::let_bind(
                format!("bin_{idx}"),
                Expr::BinOp {
                    op,
                    left: Box::new(Expr::u32(1)),
                    right: Box::new(Expr::u32(2)),
                },
            )
        })
        .chain([Node::Return])
        .collect();

    assert_canonical_wire_round_trip(
        &Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            nodes,
        ),
        "every binop",
    );
}

#[test]
fn every_terminal_unop_round_trips() {
    let mut variants = builtin_un_ops();
    variants.push(UnOp::Opaque(ExtensionUnOpId(0x8000_0202)));

    let nodes = variants
        .into_iter()
        .enumerate()
        .map(|(idx, op)| {
            Node::let_bind(
                format!("un_{idx}"),
                Expr::UnOp {
                    op,
                    operand: Box::new(Expr::u32(1)),
                },
            )
        })
        .chain([Node::Return])
        .collect();

    assert_canonical_wire_round_trip(
        &Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            nodes,
        ),
        "every unop",
    );
}

#[test]
fn every_terminal_atomic_op_round_trips() {
    let mut variants = builtin_atomic_ops();
    variants.push(AtomicOp::Opaque(ExtensionAtomicOpId(0x8000_0303)));

    let nodes = variants
        .into_iter()
        .enumerate()
        .map(|(idx, op)| {
            Node::let_bind(
                format!("atomic_{idx}"),
                Expr::Atomic {
                    op,
                    buffer: "out".into(),
                    index: Box::new(Expr::u32(0)),
                    expected: matches!(
                        op,
                        AtomicOp::CompareExchange | AtomicOp::CompareExchangeWeak
                    )
                    .then(|| Box::new(Expr::u32(1))),
                    value: Box::new(Expr::u32(2)),
                    ordering: MemoryOrdering::SeqCst,
                },
            )
        })
        .chain([Node::Return])
        .collect();

    assert_canonical_wire_round_trip(
        &Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            nodes,
        ),
        "every atomic op",
    );
}
