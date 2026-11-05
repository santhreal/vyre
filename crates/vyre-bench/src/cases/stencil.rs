//! Three-point u32 stencil over a million elements.

use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) static STENCIL3: MicroCase = MicroCase {
    id: "foundation.stencil3.u32.1m",
    name: "Stencil3 U32 1M",
    summary: "Three-point u32 stencil over 1M elements",
    tags: &["convolution", "memory-bound"],
    contract: None,
    program,
    fixture,
    reference,
    work: MicroWork::Flops(2_000_000),
};

fn program() -> Program {
    let count = 1_000_000u32;
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::output("out", 1, DataType::U32).with_count(count),
        ],
        [256, 1, 1],
        vec![
            Node::let_bind("idx", Expr::gid_x()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::u32(0), Expr::var("idx")),
                    Expr::lt(Expr::var("idx"), Expr::u32(count - 1)),
                ),
                vec![Node::store(
                    "out",
                    Expr::var("idx"),
                    Expr::add(
                        Expr::add(
                            Expr::load("input", Expr::sub(Expr::var("idx"), Expr::u32(1))),
                            Expr::load("input", Expr::var("idx")),
                        ),
                        Expr::load("input", Expr::add(Expr::var("idx"), Expr::u32(1))),
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
        values[i * 4..i * 4 + 4].copy_from_slice(&((i % 997) as u32).to_le_bytes());
    }
    vec![values]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::stencil3_u32_bytes(&inputs[0])]
}
