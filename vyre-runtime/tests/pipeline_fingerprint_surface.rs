//! Pipeline-cache fingerprints are exact authenticated artifact identities.

use std::collections::BTreeMap;

use vyre::ir::{
    BufferDecl, DataType, Node, Program, ProgramGraph, ShapeDim, ValueContract, ValueLifetime,
};
use vyre_megakernel::{compile, CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::pipeline_cache::PipelineFingerprint;

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

#[test]
fn fingerprint_is_the_neutral_artifact_digest() {
    let artifact = artifact(Program::empty());
    let fingerprint = PipelineFingerprint::of(&artifact);
    assert_eq!(fingerprint.0, artifact.digest().0);
    assert_eq!(fingerprint.hex().len(), 64);
}

#[test]
fn fingerprint_is_deterministic() {
    let artifact = artifact(Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Return],
    ));
    assert_eq!(
        PipelineFingerprint::of(&artifact),
        PipelineFingerprint::of(&artifact)
    );
}

#[test]
fn distinct_artifacts_have_distinct_fingerprints() {
    let first = artifact(Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Return],
    ));
    let second = artifact(Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(2)],
        [1, 1, 1],
        vec![Node::Return],
    ));
    assert_ne!(
        PipelineFingerprint::of(&first),
        PipelineFingerprint::of(&second)
    );
}

#[test]
fn fingerprint_hex_is_lowercase() {
    let hex = PipelineFingerprint::of(&artifact(Program::empty())).hex();
    assert!(hex
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
}
