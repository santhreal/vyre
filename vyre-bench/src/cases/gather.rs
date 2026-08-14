//! Indexed u32 gather over a million lanes.

use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) static GATHER: MicroCase = MicroCase {
    id: "foundation.gather.u32.1m",
    name: "Gather U32 1M",
    summary: "Indexed u32 gather over 1M lanes",
    tags: &["memory-bound", "indexed"],
    contract: None,
    program,
    fixture,
    reference,
    work: MicroWork::Flops(1_000_000),
};

fn program() -> Program {
    let count = 1_000_000u32;
    Program::wrapped(
        vec![
            BufferDecl::storage("values", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::storage("indices", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::output("out", 2, DataType::U32).with_count(count),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(count)),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::load("values", Expr::load("indices", Expr::var("idx"))),
                )],
            ),
        ],
    )
}

fn fixture() -> Vec<Vec<u8>> {
    let count = 1_000_000usize;
    let mut values = vec![0u8; count * 4];
    let mut indices = vec![0u8; count * 4];
    for i in 0..count {
        values[i * 4..i * 4 + 4].copy_from_slice(&((i as u32).wrapping_mul(17)).to_le_bytes());
        indices[i * 4..i * 4 + 4].copy_from_slice(&((count - 1 - i) as u32).to_le_bytes());
    }
    vec![values, indices]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::gather_u32_bytes(
        &inputs[0], &inputs[1],
    )]
}
