//! Shared canonical artifact and unique-name fixtures for pipeline-cache tests.
//!
//! Every pipeline-cache test needs the same input: a Program compiled into a
//! neutral artifact under fixed facts and a bounded search. The unit tests reach
//! it as a crate module and the `pipeline_fingerprint_surface` and
//! `fingerprint_cross_host` integration tests include this file with `#[path]`,
//! the same way `tests/support/artifact_fixtures.rs` is shared. Those tests
//! compare artifact digests, so their fixtures only mean what they say while
//! they compile the same way, which is one function rather than three copies.

#![allow(dead_code)]

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{compile, Artifact, CompileRequest, DeviceFacts, Digest, ExternalFacts, SearchBudget};

pub(crate) fn tiny_artifact() -> Artifact {
    artifact_for_program(Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    ))
}

/// Compile `program` into its neutral artifact.
///
/// Every buffer the program declares becomes an external value carrying that
/// buffer's own element type, count, and access, so a fixture's graph contract
/// cannot disagree with the program it wraps.
pub(crate) fn artifact_for_program(program: Program) -> Artifact {
    let mut graph = ProgramGraph::new();
    for buffer in program.buffers() {
        graph
            .add_external_value(
                buffer.name(),
                ValueContract {
                    dtype: buffer.element(),
                    shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
                    access: buffer.access(),
                    lifetime: ValueLifetime::Invocation,
                },
            )
            .unwrap();
    }
    graph
        .add_node("main", program, Vec::new(), Vec::new())
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap();
    compile(&request).unwrap()
}

pub(crate) fn unique_u64() -> u64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos() as u64,
        Err(_) => 0,
    }
}
