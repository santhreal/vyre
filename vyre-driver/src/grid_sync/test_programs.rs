//! Program fixtures shared by the grid-sync split tests.

use std::sync::Arc;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Ident, Node};

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
