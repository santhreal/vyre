//! Programs the native Metal tests dispatch.
//!
//! The buffer declaration is what the readback assertions depend on: a declared
//! element count that disagrees with the output byte range changes what the
//! backend collects, so every test asserting `vec![n.to_le_bytes().to_vec()]` is
//! asserting against one shape. That shape is declared here once rather than
//! restated per test, where a single edited count would silently retarget the
//! assertion at a different number of bytes.

#![cfg(feature = "device-tests")]

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// One WriteOnly `u32` word named `out` at binding 0, collected as bytes 0..4.
pub(super) fn one_word_output(workgroups: [u32; 3], body: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        workgroups,
        body,
    )
}

/// Stores `value` into the single output word from one thread.
pub(super) fn stores_word(value: u32) -> Program {
    one_word_output(
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(value))],
    )
}

/// A ReadOnly `u32` word at binding 0 and a WriteOnly `u32` word at binding 1,
/// storing `value` applied to the loaded input word.
///
/// Binding order is the contract the resident dispatch tests assert: resources
/// are bound in declaration order, so the input must stay at 0 and the output at
/// 1.
pub(super) fn word_to_word(
    input: &'static str,
    output: &'static str,
    value: impl FnOnce(Expr) -> Expr,
) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage(output, 1, BufferAccess::WriteOnly, DataType::U32)
                .with_count(1)
                .with_output_byte_range(0..4),
        ],
        [1, 1, 1],
        vec![Node::store(
            output,
            Expr::u32(0),
            value(Expr::load(input, Expr::u32(0))),
        )],
    )
}
