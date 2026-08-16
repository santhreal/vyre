//! Program fixtures shared by the `vyre-debug` test targets.
//!
//! Three targets each carried a byte-identical `minimal_program`, and four more
//! sites restated the same output buffer declaration inline. The descriptor a
//! dump renders, the descriptor a diff compares, and the WGSL a backend emits
//! are only comparable while those copies agree, so the declaration lives here
//! once.

#![allow(dead_code, unreachable_pub)]

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};

/// The one output buffer every debug fixture writes to: sixteen read-write
/// `u32` at binding 0, named `out`.
pub fn out_buffer() -> BufferDecl {
    BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(16)
}

/// A program over [`out_buffer`] with the given workgroup size and body.
pub fn program_over_out(workgroup: [u32; 3], body: Vec<Node>) -> Program {
    Program::wrapped(vec![out_buffer()], workgroup, body)
}

/// The smallest well-formed program: one store of a literal at the invocation
/// id, over [`out_buffer`], at a workgroup size every backend accepts.
pub fn minimal_program() -> Program {
    program_over_out(
        [64, 1, 1],
        vec![Node::Store {
            buffer: Ident::from("out"),
            index: Expr::InvocationId { axis: 0 },
            value: Expr::LitU32(7),
        }],
    )
}
