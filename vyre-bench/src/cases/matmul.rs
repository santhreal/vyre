//! Dense 256x256 f32 matrix multiplication.

use crate::api::case::BaselineClass;
use crate::cases::harness::ContractDescription;
use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) static MATMUL: MicroCase = MicroCase {
    id: "foundation.matmul.256",
    name: "MatMul 256x256",
    summary: "Dense matrix multiplication 256x256 floats",
    tags: &["compute", "compute-bound"],
    contract: Some(ContractDescription {
        primitive: "f32 matmul 256x256",
        baseline_crate: "faer",
        baseline_name: "faer CPU matrix multiply baseline",
        baseline_class: BaselineClass::CpuSota,
        min_speedup_x: 3.0,
    }),
    program,
    fixture,
    reference,
    work: MicroWork::Flops(2 * 256 * 256 * 256),
};

fn program() -> Program {
    let m = 256;
    let n = 256;
    let k = 256;

    Program::wrapped(
        vec![
            BufferDecl::storage("A", 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count((m * k) as u32),
            BufferDecl::storage("B", 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count((k * n) as u32),
            BufferDecl::output("C", 2, DataType::F32).with_count((m * n) as u32),
        ],
        [16, 16, 1],
        vec![
            Node::let_bind("row", Expr::gid_x()),
            Node::let_bind("col", Expr::gid_y()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("row"), Expr::u32(m)),
                    Expr::lt(Expr::var("col"), Expr::u32(n)),
                ),
                vec![
                    Node::let_bind("sum", Expr::f32(0.0)),
                    Node::loop_(
                        "k",
                        Expr::u32(0),
                        Expr::u32(k),
                        vec![
                            Node::let_bind(
                                "a_idx",
                                Expr::add(
                                    Expr::mul(Expr::var("row"), Expr::u32(k)),
                                    Expr::var("k"),
                                ),
                            ),
                            Node::let_bind(
                                "b_idx",
                                Expr::add(
                                    Expr::mul(Expr::var("k"), Expr::u32(n)),
                                    Expr::var("col"),
                                ),
                            ),
                            Node::assign(
                                "sum",
                                Expr::fma(
                                    Expr::load("A", Expr::var("a_idx")),
                                    Expr::load("B", Expr::var("b_idx")),
                                    Expr::var("sum"),
                                ),
                            ),
                        ],
                    ),
                    Node::store(
                        "C",
                        Expr::add(Expr::mul(Expr::var("row"), Expr::u32(n)), Expr::var("col")),
                        Expr::var("sum"),
                    ),
                ],
            ),
        ],
    )
}

fn fixture() -> Vec<Vec<u8>> {
    let m = 256;
    let n = 256;
    let k = 256;

    let mut a_bytes = vec![0u8; m * k * 4];
    let mut b_bytes = vec![0u8; k * n * 4];
    for i in 0..m * k {
        let value = (i % 7) as f32;
        a_bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    for i in 0..k * n {
        let value = (i % 5) as f32;
        b_bytes[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }

    vec![a_bytes, b_bytes]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::matmul_f32_bytes(
        &inputs[0], &inputs[1], 256, 256, 256,
    )]
}
