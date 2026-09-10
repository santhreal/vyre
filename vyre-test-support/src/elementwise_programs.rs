//! The element-wise programs a device dispatch is checked against.
//!
//! A dispatch proof needs a program small enough that a wrong answer names one
//! defect. These are that program, in the three shapes a dispatch path can get
//! wrong: the ordinary binding order, the order that catches a backend binding
//! by raw index, and a multiply-add.
//!
//! One owner is what makes a probe and a test comparable. A probe dispatches a
//! program and prints what the device returned, a test dispatches it and
//! compares against the reference, and a probe that succeeds while the test
//! fails is only informative while both ran the same program and the same
//! binding layout.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

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
#[must_use]
pub fn elementwise_add_program(count: u32) -> Program {
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
#[must_use]
pub fn output_first_elementwise_add_program(count: u32) -> Program {
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
#[must_use]
pub fn elementwise_fma_program(count: u32) -> Program {
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

/// `out[0] = in[0]` over `count`-element `u32` bindings named `in` and `out`.
///
/// The smallest program with one read binding and one written binding, which is
/// what a case about identity rather than arithmetic needs: a lowering
/// equivalence proof, a pipeline fingerprint, an ABI shape. Those cases each
/// built it, so the shape a fingerprint was taken over and the shape a lowering
/// was compared against were separate programs that only happened to agree.
#[must_use]
pub fn single_element_copy_program(count: u32) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::output("out", 1, DataType::U32).with_count(count),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(0)),
        )],
    )
}
