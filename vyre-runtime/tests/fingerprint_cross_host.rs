//! Cross-host cache identity follows canonical neutral artifact bytes.

use std::collections::BTreeMap;

use vyre::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{compile, CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::PipelineFingerprint;

fn artifact(program: Program) -> vyre_megakernel::Artifact {
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
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap();
    compile(&request).unwrap()
}

fn single_store() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("in", Expr::u32(0)),
        )],
    )
}

#[test]
fn repeated_artifact_identity_is_stable() {
    let artifact = artifact(single_store());
    let first = PipelineFingerprint::of(&artifact);
    let second = PipelineFingerprint::of(&artifact);
    assert_eq!(first, second);
    assert_eq!(first.0, artifact.digest().0);
}

#[test]
fn independently_compiled_identical_artifacts_share_identity() {
    let first = artifact(single_store());
    let second = artifact(single_store());
    assert_eq!(
        PipelineFingerprint::of(&first),
        PipelineFingerprint::of(&second)
    );
}

#[test]
fn distinct_artifacts_do_not_share_cache_identity() {
    let empty = artifact(Program::empty());
    let store = artifact(single_store());
    assert_ne!(
        PipelineFingerprint::of(&empty),
        PipelineFingerprint::of(&store)
    );
}
