//! The element-wise programs the SPIR-V dispatch test and the Vulkan probe run.
//!
//! Shared the same way as `tests/support/preferred_dispatch_backend_contract.rs`:
//! each consumer includes this file with `#[path]`.
//!
//! One owner matters here because the two consumers ask the same question from
//! opposite sides. The test dispatches the program and compares it against the
//! CPU reference; the probe dispatches it and prints what the device returned so
//! an operator can see whether Vulkan works at all. A probe that succeeds while
//! the test fails is only informative while both ran the same program and the
//! same binding layout.

#![allow(dead_code)]

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

/// `out[i] = a[i] + b[i]`, written once so the two binding orders below cannot
/// diverge in what they compute.
fn add_body() -> Vec<Node> {
    vec![Node::store(
        "out",
        Expr::gid_x(),
        Expr::add(
            Expr::load("a", Expr::gid_x()),
            Expr::load("b", Expr::gid_x()),
        ),
    )]
}

/// `out[i] = a[i] + b[i]` over `count` u32 lanes, inputs at bindings 0 and 1.
pub(crate) fn elementwise_add_program(count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(count),
            BufferDecl::read("b", 1, DataType::U32).with_count(count),
            BufferDecl::output("out", 2, DataType::U32).with_count(count),
        ],
        [1, 1, 1],
        add_body(),
    )
}

/// The same computation with the output at binding 0.
///
/// A backend that binds host inputs by raw binding order rather than through the
/// binding plan feeds the first input buffer into the output slot here, so the
/// answer differs from [`elementwise_add_program`] only when that bug is present.
pub(crate) fn output_first_elementwise_add_program(count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(count),
            BufferDecl::read("a", 1, DataType::U32).with_count(count),
            BufferDecl::read("b", 2, DataType::U32).with_count(count),
        ],
        [1, 1, 1],
        add_body(),
    )
}

/// `out[i] = a[i] * 2 + 1` over `count` u32 lanes.
pub(crate) fn elementwise_fma_program(count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(count),
            BufferDecl::output("out", 1, DataType::U32).with_count(count),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::gid_x(),
            Expr::add(
                Expr::mul(Expr::load("a", Expr::gid_x()), Expr::u32(2)),
                Expr::u32(1),
            ),
        )],
    )
}

/// Little-endian bytes for `values`.
pub(crate) fn u32_values_to_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// `bytes` decoded as little-endian u32 lanes. A trailing partial word is
/// dropped, which is why every caller also asserts the buffer lengths agree.
pub(crate) fn bytes_to_u32_values(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}
