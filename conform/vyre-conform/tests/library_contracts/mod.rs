//! Contracts for the vyre-conform library surface, one module per concern.

mod bundle_cert;
mod cert;
mod prover;
mod witness_plan;
use vyre::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

pub(crate) fn scratch_readwrite_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::storage(
            "scratch",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [1, 1, 1],
        Vec::<Node>::new(),
    )
}

pub(crate) fn input_scratch_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        Vec::<Node>::new(),
    )
}
