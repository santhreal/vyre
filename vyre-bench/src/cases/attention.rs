//! Self-attention QKV block over a 64-sequence, 64-dimension tile.
//!
//! Real attention needs a full softmax across K before multiplying V. This is a
//! simplified proxy that does Q * K^T * V sequentially for the cell, which is
//! what the CPU reference computes as well.

use crate::cases::harness::ContractDescription;
use crate::cases::micro::{MicroCase, MicroWork};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) static ATTENTION: MicroCase = MicroCase {
    id: "foundation.attention.64",
    name: "Attention 64x64",
    summary: "Self-Attention QKV block (64 seq, 64 dim)",
    tags: &["compute", "memory-bound"],
    contract: Some(ContractDescription {
        primitive: "attention proxy 64x64",
        baseline_crate: "rayon",
        baseline_name: "rayon CPU attention baseline",
        min_speedup_x: 1.5,
    }),
    program,
    fixture,
    reference,
    work: MicroWork::Flops(3 * 64 * 64 * 64),
};

fn program() -> Program {
    let seq = 64;
    let dim = 64;

    Program::wrapped(
        vec![
            BufferDecl::storage("Q", 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count((seq * dim) as u32),
            BufferDecl::storage("K", 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count((seq * dim) as u32),
            BufferDecl::storage("V", 2, BufferAccess::ReadOnly, DataType::F32)
                .with_count((seq * dim) as u32),
            BufferDecl::output("out", 3, DataType::F32).with_count((seq * dim) as u32),
        ],
        [16, 16, 1],
        vec![
            Node::let_bind("row", Expr::gid_x()),
            Node::let_bind("col", Expr::gid_y()),
            Node::if_then(
                Expr::and(
                    Expr::lt(Expr::var("row"), Expr::u32(seq)),
                    Expr::lt(Expr::var("col"), Expr::u32(dim)),
                ),
                vec![
                    Node::let_bind("acc", Expr::f32(0.0)),
                    Node::loop_(
                        "k",
                        Expr::u32(0),
                        Expr::u32(seq),
                        vec![
                            Node::let_bind(
                                "q_val",
                                Expr::load(
                                    "Q",
                                    Expr::add(
                                        Expr::mul(Expr::var("row"), Expr::u32(dim)),
                                        Expr::var("col"),
                                    ),
                                ),
                            ),
                            Node::let_bind(
                                "k_val",
                                Expr::load(
                                    "K",
                                    Expr::add(
                                        Expr::mul(Expr::var("k"), Expr::u32(dim)),
                                        Expr::var("col"),
                                    ),
                                ),
                            ),
                            Node::let_bind(
                                "v_val",
                                Expr::load(
                                    "V",
                                    Expr::add(
                                        Expr::mul(Expr::var("k"), Expr::u32(dim)),
                                        Expr::var("col"),
                                    ),
                                ),
                            ),
                            Node::assign(
                                "acc",
                                Expr::add(
                                    Expr::var("acc"),
                                    Expr::mul(
                                        Expr::mul(Expr::var("q_val"), Expr::var("k_val")),
                                        Expr::var("v_val"),
                                    ),
                                ),
                            ),
                        ],
                    ),
                    Node::store(
                        "out",
                        Expr::add(
                            Expr::mul(Expr::var("row"), Expr::u32(dim)),
                            Expr::var("col"),
                        ),
                        Expr::var("acc"),
                    ),
                ],
            ),
        ],
    )
}

fn fixture() -> Vec<Vec<u8>> {
    let seq = 64;
    let dim = 64;

    let mut q_bytes = vec![0u8; seq * dim * 4];
    let mut k_bytes = vec![0u8; seq * dim * 4];
    let mut v_bytes = vec![0u8; seq * dim * 4];
    for i in 0..seq * dim {
        let q = (i % 3) as f32;
        let k = (i % 5) as f32;
        let v = (i % 7) as f32;
        q_bytes[i * 4..i * 4 + 4].copy_from_slice(&q.to_le_bytes());
        k_bytes[i * 4..i * 4 + 4].copy_from_slice(&k.to_le_bytes());
        v_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }

    vec![q_bytes, k_bytes, v_bytes]
}

fn reference(inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    vec![crate::cases::cpu_baselines::attention_proxy_f32_bytes(
        &inputs[0], &inputs[1], &inputs[2], 64, 64,
    )]
}
