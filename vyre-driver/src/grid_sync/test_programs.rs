//! Program fixtures shared by the grid-sync split tests.

use std::sync::Arc;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::memory_model::MemoryOrdering;

pub(super) fn buffer() -> BufferDecl {
    BufferDecl::storage("buf", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4)
}

pub(super) fn region(generator: &str, body: Vec<Node>) -> Node {
    Node::Region {
        generator: Ident::from(generator),
        source_region: None,
        body: Arc::new(body),
    }
}

/// One returning region per name in `regions`, separated by `ordering` barriers,
/// over the single read-write `buf` declaration and `workgroup_size` threads.
///
/// This is the shape every split test drives: the region count fixes the
/// expected segment count, and the ordering decides whether the program splits
/// at all. Stating it once keeps a new segment count one argument instead of
/// four more lines that can disagree about the buffer or the barrier.
pub(super) fn barrier_chain(
    regions: &[&str],
    ordering: MemoryOrdering,
    workgroup_size: [u32; 3],
) -> Program {
    let mut nodes = Vec::with_capacity(regions.len() * 2);
    for (index, name) in regions.iter().enumerate() {
        if index > 0 {
            nodes.push(Node::barrier_with_ordering(ordering));
        }
        nodes.push(region(name, vec![Node::Return]));
    }
    Program::wrapped(vec![buffer()], workgroup_size, nodes)
}

/// [`barrier_chain`] with grid-sync barriers and a single-thread workgroup,
/// which is what every test that does not assert on workgroup size wants.
pub(super) fn grid_sync_chain(regions: &[&str]) -> Program {
    barrier_chain(regions, MemoryOrdering::GridSync, [1, 1, 1])
}

/// Two grid-sync segments writing different slots of one four-element output.
///
/// The cross-segment accumulator regression: arm A stores element 0 in segment
/// 0 and arm B stores element 2 in the final segment, so a split that hands the
/// final segment a fresh write-only `out` drops arm A's slot entirely.
pub(super) fn cross_segment_store_program() -> Program {
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

/// Apply every literal-index literal-value store to `out` in `nodes` onto
/// `state`, one byte per four-byte element, recursing into nested bodies.
///
/// The test backends that stand in for a device all need the same reading of a
/// segment body, and a nesting form this walk forgets makes a store invisible
/// and the split look correct when it dropped a write.
pub(super) fn apply_out_stores(nodes: &[Node], state: &mut [u8]) {
    for node in nodes {
        match node {
            Node::Store {
                buffer,
                index: Expr::LitU32(index),
                value: Expr::LitU32(value),
            } if buffer.as_str() == "out" => {
                state[(*index as usize) * 4] = (*value & 0xff) as u8;
            }
            Node::Region { body, .. } => apply_out_stores(body, state),
            Node::Block(body) => apply_out_stores(body, state),
            Node::If {
                then, otherwise, ..
            } => {
                apply_out_stores(then, state);
                apply_out_stores(otherwise, state);
            }
            Node::Loop { body, .. } => apply_out_stores(body, state),
            _ => {}
        }
    }
}
