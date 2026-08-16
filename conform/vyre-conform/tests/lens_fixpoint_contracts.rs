//! Contracts for `vyre_conform::lens::fixpoint`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_conform::lens::fixpoint::infer_fixpoint_buffers;
use vyre_foundation::ir::{BufferAccess, Program};

use vyre_foundation::ir::{BufferDecl, DataType};

#[test]
fn infer_fixpoint_buffers_rejects_no_rw() {
    let program = Program::wrapped(
        vec![BufferDecl::storage("x", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![],
    );
    assert!(infer_fixpoint_buffers(&program).is_err());
}

#[test]
fn infer_fixpoint_buffers_matches_in_out_pair() {
    // Simulate the buffer layout of flows_to / sanitized_by.
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("pg_nodes", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(4),
            BufferDecl::storage("fin", 1, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::storage("fout", 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(1),
        ],
        [1, 1, 1],
        vec![],
    );
    let (current, next, count) = infer_fixpoint_buffers(&program).expect("Fix: inference");
    assert_eq!(current, "fin");
    assert_eq!(next, "fout");
    assert_eq!(count, 1);
}
