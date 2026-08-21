//! Shared program and dispatch fixtures for WGPU binary-operation parity suites.

#![cfg(feature = "device-tests")]

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) fn program(length: u32, build: fn(Expr, Expr) -> Expr) -> Program {
    let mut body = Vec::new();
    for index in 0..length {
        body.push(Node::store(
            "out",
            Expr::u32(index),
            build(
                Expr::load("a", Expr::u32(index)),
                Expr::load("b", Expr::u32(index)),
            ),
        ));
    }
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(length),
            BufferDecl::storage("a", 1, BufferAccess::ReadOnly, DataType::U32).with_count(length),
            BufferDecl::storage("b", 2, BufferAccess::ReadOnly, DataType::U32).with_count(length),
        ],
        [1, 1, 1],
        body,
    )
}

pub(crate) fn dispatch(
    backend: &WgpuBackend,
    program: &Program,
    pairs: &[(u32, u32)],
    contract: &str,
) -> Vec<u32> {
    let pack = |words: &[u32]| vyre_primitives::wire::pack_u32_slice(words);
    let left = pack(&pairs.iter().map(|&(left, _)| left).collect::<Vec<_>>());
    let right = pack(&pairs.iter().map(|&(_, right)| right).collect::<Vec<_>>());
    let output = pack(&vec![0; pairs.len()]);
    let outputs = backend
        .dispatch_borrowed(
            program,
            &[output.as_slice(), left.as_slice(), right.as_slice()],
            &DispatchConfig::default(),
        )
        .unwrap_or_else(|error| panic!("Fix: WGPU must dispatch the {contract}: {error}"));
    outputs[0]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four-byte output chunk")))
        .collect()
}
