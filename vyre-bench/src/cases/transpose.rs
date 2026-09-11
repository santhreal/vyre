//! Dense f32 matrix transpose with coalesced reads and strided writes.

use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) static TRANSPOSE: MicroCase = MicroCase {
    id: "foundation.transpose.512",
    name: "Transpose 512x512",
    summary: "Dense f32 matrix transpose with coalesced reads and strided writes",
    tags: &["memory-bound", "layout"],
    contract: None,
    program,
    fixture,
    reference,
    work: MicroWork::Flops(512 * 512),
};

fn program() -> Program {
    let rows = 512u32;
    let cols = 512u32;
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(rows * cols),
            BufferDecl::output("out", 1, DataType::F32).with_count(rows * cols),
        ],
        [16, 16, 1],
        vec![
            Node::let_bind("row", Expr::gid_x()),
            Node::let_bind("col", Expr::gid_y()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("row"), Expr::u32(rows)),
                    Expr::lt(Expr::var("col"), Expr::u32(cols)),
                ),
                vec![Node::store(
                    "out",
                    Expr::add(
                        Expr::mul(Expr::var("col"), Expr::u32(rows)),
                        Expr::var("row"),
                    ),
                    Expr::load(
                        "input",
                        Expr::add(
                            Expr::mul(Expr::var("row"), Expr::u32(cols)),
                            Expr::var("col"),
                        ),
                    ),
                )],
            ),
        ],
    )
}

fn fixture() -> Vec<Vec<u8>> {
    let rows = 512usize;
    let cols = 512usize;
    let mut input = vec![0u8; rows * cols * 4];
    for i in 0..rows * cols {
        input[i * 4..i * 4 + 4].copy_from_slice(&((i % 251) as f32).to_le_bytes());
    }
    vec![input]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::transpose_f32_bytes(
        &inputs[0], 512, 512,
    )]
}
