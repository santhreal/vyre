//! A permuted shared binding computes the values the unpermuted program states.
//!
//! The bank-conflict mitigation rewrites the element index of every access to a
//! permutable shared binding. A rewrite that is not one-to-one, or that is
//! applied at one access and not the other, reads back a different element than
//! it wrote, and the kernel still runs and still returns numbers. The reference
//! interpreter applies no permutation, so it is the unpermuted answer this
//! device run is held to.

#![cfg(feature = "device-tests")]

use crate::harness;
use harness::bytes_u32;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};

/// A column walk over a 32-row tile staged in workgroup memory.
///
/// Lane `t` writes and reads element `t * 32`, so on 32 four-byte banks every
/// lane addresses the same bank. That is the classifier's 32-way case, one
/// element of padding per row is the cheapest accepted candidate for it, and
/// the emitted kernel displaces every access by one element per row it crosses.
fn column_walk_tile_program(rows: u32) -> Program {
    let column = move || Expr::mul(Expr::gid_x(), Expr::u32(rows));
    Program::wrapped(
        vec![
            BufferDecl::workgroup("tile", rows * rows, DataType::U32),
            BufferDecl::output("out", 0, DataType::U32).with_count(rows),
        ],
        [rows, 1, 1],
        vec![
            Node::store("tile", column(), Expr::gid_x()),
            Node::Barrier {
                ordering: MemoryOrdering::SeqCst,
            },
            Node::store("out", Expr::gid_x(), Expr::load("tile", column())),
        ],
    )
}

#[test]
fn a_permuted_column_walk_reads_back_what_it_wrote() {
    let program = column_walk_tile_program(32);

    let expected = vyre_reference::ReferenceRequest::standard(&program, &[])
        .outputs()
        .expect("Fix: the reference interpreter must execute a workgroup column walk.");
    let expected = bytes_u32(&expected[0].to_bytes());

    let backend =
        CudaBackend::acquire().expect("Fix: CUDA backend acquire failed on a GPU-required host.");
    let outputs = backend
        .dispatch(&program, &[], &DispatchConfig::default())
        .expect("Fix: CUDA must execute a column walk over a padded workgroup tile.");

    assert_eq!(
        bytes_u32(&outputs[0]),
        expected,
        "Fix: apply the shared permutation at every access to the binding, \
         or at none; a rewrite applied at one site reads a different element \
         than it wrote."
    );
}
