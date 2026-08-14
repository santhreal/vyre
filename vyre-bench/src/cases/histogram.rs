//! Atomic 256-bin histogram over a million u32 values.

use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) static HISTOGRAM: MicroCase = MicroCase {
    id: "foundation.histogram.u32_256.1m",
    name: "Histogram U32 256-bin 1M",
    summary: "Atomic 256-bin histogram over 1M u32 values",
    tags: &["memory-bound", "atomics", "histogram"],
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
            BufferDecl::output("bins", 0, DataType::U32).with_count(256),
            BufferDecl::storage("values", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::lt(Expr::var("idx"), Expr::u32(count)),
                vec![Node::let_bind(
                    "old",
                    Expr::atomic_add(
                        "bins",
                        Expr::bitand(Expr::load("values", Expr::var("idx")), Expr::u32(255)),
                        Expr::u32(1),
                    ),
                )],
            ),
        ],
    )
}

fn fixture() -> Vec<Vec<u8>> {
    let count = 1_000_000usize;
    let mut values = vec![0u8; count * 4];
    for i in 0..count {
        values[i * 4..i * 4 + 4].copy_from_slice(&((i * 31 % 256) as u32).to_le_bytes());
    }
    vec![values]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::histogram_u32_256_bytes(
        &inputs[0],
    )]
}
