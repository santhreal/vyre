//! Shared program graph test fixtures.
#![allow(dead_code)]

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ShapeDim, ValueContract, ValueLifetime,
};

pub(crate) fn contract(access: BufferAccess, lifetime: ValueLifetime) -> ValueContract {
    ValueContract {
        dtype: DataType::F32,
        shape: vec![ShapeDim::Symbol("tokens".into()), ShapeDim::Known(8)],
        access,
        lifetime,
    }
}

pub(crate) fn copy_program(input: &str, output: &str) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32),
            BufferDecl::storage(output, 1, BufferAccess::ReadWrite, DataType::F32),
        ],
        [1, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            Expr::load(input, Expr::u32(0)),
        )],
    )
}
