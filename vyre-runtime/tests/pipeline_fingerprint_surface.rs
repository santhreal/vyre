//! Pipeline-cache fingerprints are exact authenticated artifact identities.

use vyre_foundation::ir::{BufferDecl, DataType, Node, Program};
use vyre_runtime::pipeline_cache::PipelineFingerprint;

#[path = "../src/pipeline_cache/test_artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::artifact_for_program as artifact;

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
