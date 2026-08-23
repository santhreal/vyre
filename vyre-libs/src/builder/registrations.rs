//! Catalog registrations for the generic builder operations, with the witness
//! inputs and expected bytes each one is checked against.

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
        Node::let_bind("local", Expr::LogicalWithinTileId { axis: 0 }),
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
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
        // The result is one value, so exactly one workgroup may write it. A
        // dispatch is rounded up to whole workgroups, and a second workgroup
        // reduces lanes outside the input: its accumulator is the identity, and
        // an unguarded store would publish that instead of the answer.
        Node::if_then(
            Expr::and(
                Expr::is_first_logical_tile(),
                Expr::eq(Expr::var("local"), Expr::u32(0)),
            ),
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

const EXPECTED_INDEXED_MAP_OUTPUT_BYTES: [u8; 16] =
    [2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0];
const EXPECTED_STRIDED_ACCUMULATE_OUTPUT_BYTES: [u8; 4] = [7, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        INDEXED_MAP_OP_ID,
        indexed_map_program,
        Some(|| vec![vec![
            u32s(&[1, 2, 3, 4]),
        ]]),
        Some(|| vec![vec![EXPECTED_INDEXED_MAP_OUTPUT_BYTES.to_vec()]]),
    )
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        STRIDED_ACCUMULATE_OP_ID,
        strided_accumulate_program,
        Some(|| vec![vec![
            u32s(&[7, 11, 13, 17]),
        ]]),
        Some(|| vec![vec![EXPECTED_STRIDED_ACCUMULATE_OUTPUT_BYTES.to_vec()]]),
    )
}
