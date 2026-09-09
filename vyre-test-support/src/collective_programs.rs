//! The RFC-0004 collective node set every collective suite states.
//!
//! Four nodes cover the four collective variants, and each suite that asserts
//! over them declared its own copy of the same list. A copy that dropped a
//! variant would assert a narrower contract under the same name, so the list
//! has one owner and each suite chooses only the group its scoped nodes carry.

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Node,
};

/// The three buffers [`collective_nodes`] addresses, eight `u32` elements each.
///
/// `a` is reduced in place, `input` is read and `out` is written.
#[must_use]
pub fn collective_buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage("a", 0, BufferAccess::ReadWrite, DataType::U32).with_count(8),
        BufferDecl::read("input", 1, DataType::U32).with_count(8),
        BufferDecl::storage("out", 2, BufferAccess::ReadWrite, DataType::U32).with_count(8),
    ]
}

/// One node per collective variant over [`collective_buffers`].
///
/// The all-reduce and all-gather name the world group. The reduce-scatter and
/// broadcast name `scoped`, so a suite proves a non-world group id survives
/// whatever it validates while the world group is still exercised.
///
/// `broadcast_root` is separate from the group because single-rank lowering
/// treats only root 0 as local, so a suite asserting that rewrite states 0
/// while a suite asserting the field survives a round trip states something
/// else.
#[must_use]
pub fn collective_nodes(scoped: CommGroup, broadcast_root: u32) -> [Node; 4] {
    [
        Node::AllReduce {
            buffer: "a".into(),
            op: CollectiveOp::Sum,
            group: CommGroup::WORLD,
        },
        Node::AllGather {
            input: "input".into(),
            output: "out".into(),
            group: CommGroup::WORLD,
        },
        Node::ReduceScatter {
            input: "input".into(),
            output: "out".into(),
            op: CollectiveOp::Max,
            group: scoped,
        },
        Node::Broadcast {
            buffer: "a".into(),
            root: broadcast_root,
            group: scoped,
        },
    ]
}
