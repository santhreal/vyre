//! The grid-sync split programs the split contracts drive.
//!
//! A whole-grid barrier is what makes the host split a program into segments,
//! so every suite that checks the split needs the same two-region program and
//! the same region constructor. The driver's own unit tests reach a `cfg(test)`
//! module for them; an integration test binary cannot, so it wrote its own
//! copy. Two copies of the fixture is two definitions of what a segment
//! boundary is, and a copy that changes a buffer or an index makes the two
//! suites judge different programs under the same name.

use std::sync::Arc;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Ident, MemoryOrdering, Node, Program};

/// A returning region named `generator` around `body`.
#[must_use]
pub fn region(generator: &str, body: Vec<Node>) -> Node {
    Node::Region {
        generator: Ident::from(generator),
        source_region: None,
        body: Arc::new(body),
    }
}

/// Two grid-sync segments writing different slots of one four-element output.
///
/// The cross-segment accumulator regression: arm A stores element 0 in segment
/// 0 and arm B stores element 2 in the final segment, so a split that hands the
/// final segment a fresh write-only `out` drops arm A's slot entirely.
#[must_use]
pub fn cross_segment_store_program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        vec![
            region("a", vec![Node::store("out", Expr::u32(0), Expr::u32(0xAA))]),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
            region("b", vec![Node::store("out", Expr::u32(2), Expr::u32(0xBB))]),
        ],
    )
}
