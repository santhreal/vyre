//! Program fixtures shared by the `vyre` facade wire tests.
//!
//! The round-trip suite and the malformed-wire suite must agree on what a
//! valid program is: the first proves such a program survives an encode and a
//! decode byte for byte, the second proves hostile bytes never decode into
//! one. Two copies of the fixture let the two halves of that claim drift.

#![allow(dead_code, unreachable_pub)]

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};

/// A program with no buffers and no body, at the smallest workgroup size.
pub fn empty_program() -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], Vec::new())
}

/// The smallest program that exercises a buffer, an index and a literal:
/// one store of 42 into a read-write `u32` buffer named `out`.
pub fn one_store_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    )
}
