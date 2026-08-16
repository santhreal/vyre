use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::{build_indexed_map, strided_accumulate_child};

const INDEXED_MAP_OP_ID: &str = "vyre-libs::builder::indexed_map";
const STRIDED_ACCUMULATE_OP_ID: &str = "vyre-libs::builder::strided_accumulate";

fn u32s(words: &[u32]) -> Vec<u8> {
    vyre_primitives::wire::pack_u32_slice(words)
}

fn indexed_map_program() -> Program {
    build_indexed_map(
        INDEXED_MAP_OP_ID,
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::output("out", 1, DataType::U32).with_count(4),
        ],
        "out",
        4,
        [4, 1, 1],
        |i| (i.clone(), Expr::add(Expr::load("input", i), Expr::u32(1))),
    )
}

fn strided_accumulate_program() -> Program {
    let tile = 4;
    let body = vec![
        Node::let_bind("local", Expr::LocalId { axis: 0 }),
        strided_accumulate_child(
            STRIDED_ACCUMULATE_OP_ID,
            tile,
            1,
            4,
            "acc",
            Expr::u32(0),
            "scratch",
            |idx, acc| Expr::add(acc, Expr::load("values", idx)),
        ),
        Node::barrier(),
        Node::if_then(
            Expr::eq(Expr::var("local"), Expr::u32(0)),
            vec![Node::store(
                "out",
                Expr::u32(0),
                Expr::load("scratch", Expr::u32(0)),
            )],
        ),
    ];
    Program::wrapped(
        vec![
            BufferDecl::storage("values", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::workgroup("scratch", tile, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [tile, 1, 1],
        vec![wrap_anonymous_region(STRIDED_ACCUMULATE_OP_ID, body)],
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        INDEXED_MAP_OP_ID,
        indexed_map_program,
        Some(|| vec![vec![
            u32s(&[1, 2, 3, 4]),
        ]]),
        Some(|| vec![vec![u32s(&[2, 3, 4, 5])]]),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        STRIDED_ACCUMULATE_OP_ID,
        strided_accumulate_program,
        Some(|| vec![vec![
            u32s(&[7, 11, 13, 17]),
        ]]),
        Some(|| vec![vec![u32s(&[7])]]),
    )
}
