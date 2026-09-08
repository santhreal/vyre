//! The f32 programs the strict-IEEE lowering contract is asserted against.
//!
//! # Why this has one owner
//!
//! Strict lowering is a claim about two shapes: an approximable transcendental
//! that must be replaced by its exact expansion, and a multiply-add the target
//! may not contract. Suites in `vyre-foundation`, `vyre-driver-wgpu` and
//! `vyre-registry-link` each assert part of it, and each one had written the
//! buffer layout out again. Two spellings of one layout hash to two different
//! programs, so a suite can report on a shape a sibling suite never ran while
//! both stay green, and no file states which program the mode was measured on.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

/// `out[i] = value(i)` over `count` f32 lanes, reading `in` and writing `out`.
///
/// `value` receives the lane index, so a caller states the arithmetic and
/// nothing else. The store is guarded by the lane bound, which is what lets one
/// program serve a single-lane witness and a swept domain.
#[must_use]
pub fn f32_lane_program(count: u32, value: impl FnOnce(&Expr) -> Expr) -> Program {
    let index = Expr::gid_x();
    let stored = value(&index);
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::F32).with_count(count),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::F32).with_count(count),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(index.clone(), Expr::u32(count)),
            vec![Node::store("out", index, stored)],
        )],
    )
}

/// `out[i] = first(a[i]) * b[i] + c[i]` over `count` f32 lanes.
///
/// `first` wraps the `a` load, so `None` is the bare multiply-add that measures
/// contraction and `Some(UnOp::Sin)` adds the transcendental strict lowering
/// has to expand. Both halves of the mode then run on one buffer layout.
#[must_use]
pub fn f32_multiply_add_program(count: u32, first: Option<UnOp>) -> Program {
    let index = Expr::gid_x();
    let load_a = Expr::load("a", index.clone());
    let factor = match first {
        Some(op) => Expr::UnOp {
            op,
            operand: Box::new(load_a),
        },
        None => load_a,
    };
    let stored = Expr::add(
        Expr::mul(factor, Expr::load("b", index.clone())),
        Expr::load("c", index.clone()),
    );
    Program::wrapped(
        vec![
            BufferDecl::storage("a", 0, BufferAccess::ReadOnly, DataType::F32).with_count(count),
            BufferDecl::storage("b", 1, BufferAccess::ReadOnly, DataType::F32).with_count(count),
            BufferDecl::storage("c", 2, BufferAccess::ReadOnly, DataType::F32).with_count(count),
            BufferDecl::output("out", 3, DataType::F32).with_count(count),
        ],
        [64, 1, 1],
        vec![Node::if_then(
            Expr::lt(index.clone(), Expr::u32(count)),
            vec![Node::store("out", index, stored)],
        )],
    )
}
