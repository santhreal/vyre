//! Contracts for `vyre_runtime::resident_work_queue::advanced::zero_copy_io`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_foundation::ir::{Expr, Node};
use vyre_runtime::resident_work_queue::advanced::zero_copy_io::pull_file_async_direct;
use vyre_runtime::resident_work_queue::io::{
    IO_DESTINATION_CAPABILITY_TABLE, IO_QUEUE_DMA_TAG, IO_SOURCE_CAPABILITY_TABLE,
};

#[test]
fn direct_io_uses_extended_async_load_fields() {
    let node = pull_file_async_direct();
    let Node::AsyncLoad {
        source,
        destination,
        offset,
        size,
        tag,
    } = node
    else {
        panic!("direct IO must emit AsyncLoad");
    };
    assert_eq!(source.as_str(), IO_SOURCE_CAPABILITY_TABLE);
    assert_eq!(destination.as_str(), IO_DESTINATION_CAPABILITY_TABLE);
    assert_eq!(tag.as_str(), IO_QUEUE_DMA_TAG);
    assert!(matches!(*offset, Expr::Var(_)));
    assert!(matches!(*size, Expr::BinOp { .. }));
}
