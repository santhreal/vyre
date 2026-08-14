//! Shared canonical artifact and unique-name fixtures for pipeline-cache tests.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{compile, Artifact, CompileRequest, Digest, ExternalFacts, SearchBudget};

pub(in crate::pipeline_cache) fn tiny_artifact() -> Artifact {
    artifact_for_program(Program::wrapped(
        vec![BufferDecl::read_write("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(42))],
    ))
}

pub(in crate::pipeline_cache) fn artifact_for_program(program: Program) -> Artifact {
    let buffer = &program.buffers()[0];
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            buffer.name(),
            ValueContract {
                dtype: buffer.element(),
                shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();
    graph
        .add_node("main", program, Vec::new(), Vec::new())
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap();
    compile(&request).unwrap()
}

pub(in crate::pipeline_cache) fn unique_u64() -> u64 {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(duration) => duration.as_nanos() as u64,
        Err(_) => 0,
    }
}
