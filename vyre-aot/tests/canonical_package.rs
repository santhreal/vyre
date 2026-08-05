//! Regression coverage for AOT packaging of the canonical megakernel envelope.

use std::collections::BTreeMap;

use vyre_aot::{
    package_artifact, read_bundle_artifact, target_payload_format, CompiledArtifact, Target,
    TargetEntryPoint, TargetPayload, TargetResourceAccess, TargetResourceBinding,
    TargetResourceMemory,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, TensorContract,
    ValueLifetime,
};
use vyre_megakernel::{
    compile, ArtifactRoute, CompileOptions, MegakernelArtifactEnvelope, ValidatedCompileRequest,
};

fn compiled_artifact() -> CompiledArtifact {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "params",
            TensorContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(16)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::ImmutableWeight,
            },
        )
        .expect("parameter resource must be valid");
    graph
        .add_external_value(
            "out",
            TensorContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(4)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
        )
        .expect("output resource must be valid");
    graph
        .add_node(
            "main",
            Program::wrapped(
                vec![
                    BufferDecl::read("params", 0, DataType::U32).with_count(16),
                    BufferDecl::output("out", 1, DataType::U32).with_count(4),
                ],
                [4, 1, 1],
                Vec::new(),
            ),
            Vec::new(),
            Vec::new(),
        )
        .expect("entry node must be valid");
    let request = ValidatedCompileRequest::new(
        graph,
        CompileOptions::new(ArtifactRoute::Static, BTreeMap::new(), 1_000_000),
    )
    .expect("neutral request must validate");
    let neutral = compile(&request).expect("neutral request must compile");
    let node = neutral.nodes()[0].id;
    let params = neutral
        .resources()
        .iter()
        .find(|resource| resource.name == "params")
        .expect("canonical parameter resource must exist")
        .value;
    let out = neutral
        .resources()
        .iter()
        .find(|resource| resource.name == "out")
        .expect("canonical output resource must exist")
        .value;
    let target_bytes = vec![0, 1, 2, 3, 5, 8, 13];
    let payload = TargetPayload::new(
        &neutral,
        target_payload_format(Target::Ptx).expect("target format must be valid"),
        vec![TargetEntryPoint {
            name: "main".into(),
            node,
            grid_size: [2, 1, 1],
            dynamic_shared_bytes: 0,
            resource_bindings: vec![
                TargetResourceBinding {
                    resource: params,
                    slot: 0,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadOnly,
                },
                TargetResourceBinding {
                    resource: out,
                    slot: 1,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::WriteOnly,
                },
            ],
        }],
        target_bytes,
    )
    .expect("target payload must bind to neutral artifact");
    let mut envelope = MegakernelArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .expect("target payload must attach");
    CompiledArtifact::new(
        Target::Ptx,
        envelope,
        "0.7.2",
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    )
    .expect("AOT package handle must admit matching payload")
}

/// Regression: AOT package/read must preserve canonical IDs, target bytes, and manifest identities.
#[test]
fn package_and_read_round_trip_the_canonical_envelope() {
    let artifact = compiled_artifact();
    let neutral_digest = artifact.envelope().neutral().digest();
    let payload_digest = artifact
        .target_payload()
        .expect("fixture payload must remain compatible")
        .digest();
    let target_bytes = artifact
        .target_payload()
        .expect("fixture payload must remain compatible")
        .bytes()
        .to_vec();
    let directory = tempfile::tempdir().expect("temporary package directory must exist");

    package_artifact(
        directory.path(),
        &artifact,
        &[9; 32],
        "canonical-package",
        "regression fixture",
    )
    .expect("canonical package must write");
    let (manifest, decoded) =
        read_bundle_artifact(directory.path()).expect("canonical package must read");

    assert_eq!(manifest.schema, "vyre-aot-manifest-v2");
    assert_eq!(manifest.artifact_name, "canonical-package");
    assert_eq!(decoded.target, Target::Ptx);
    assert_eq!(decoded.envelope().neutral().digest(), neutral_digest);
    assert_eq!(
        decoded
            .target_payload()
            .expect("decoded payload must be compatible")
            .digest(),
        payload_digest
    );
    assert_eq!(
        decoded
            .target_payload()
            .expect("decoded payload must be compatible")
            .bytes(),
        target_bytes
    );
    assert_eq!(decoded.envelope().neutral().resources()[0].value.0, 0);
    assert_eq!(decoded.envelope().neutral().resources()[1].value.0, 1);
    assert_eq!(decoded.envelope().neutral().geometry()[0].workgroup_size, [4, 1, 1]);
}
