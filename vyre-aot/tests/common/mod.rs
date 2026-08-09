use std::collections::BTreeMap;

use vyre_aot::{
    target_payload_format, CompiledArtifact, Target, TargetEntryPoint, TargetPayload,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{
    compile, ArtifactEnvelope, CompileRequest, Digest, ExternalFacts, SearchBudget,
};

pub(crate) fn compiled_artifact() -> CompiledArtifact {
    compiled_artifact_with_grid([1, 1, 1])
}

pub(crate) fn compiled_artifact_with_grid(grid_size: [u32; 3]) -> CompiledArtifact {
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "params",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(256)],
                access: BufferAccess::ReadOnly,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();
    graph
        .add_external_value(
            "out",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(64)],
                access: BufferAccess::WriteOnly,
                lifetime: ValueLifetime::Output,
            },
        )
        .unwrap();
    graph
        .add_node(
            "main",
            Program::wrapped(
                vec![
                    BufferDecl::read("params", 0, DataType::U32).with_count(256),
                    BufferDecl::output("out", 1, DataType::U32).with_count(64),
                ],
                [64, 1, 1],
                Vec::new(),
            ),
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        1_000_000,
    )
    .validate()
    .unwrap();
    let neutral = compile(&request).unwrap();
    let node = neutral.nodes()[0].id;
    let params = neutral
        .resources()
        .iter()
        .find(|resource| resource.name == "params")
        .unwrap()
        .value;
    let out = neutral
        .resources()
        .iter()
        .find(|resource| resource.name == "out")
        .unwrap()
        .value;
    let payload = TargetPayload::new(
        &neutral,
        target_payload_format(Target::Ptx).unwrap(),
        vec![TargetEntryPoint {
            name: "main".into(),
            node,
            grid_size,
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
        b"target-payload-fixture".to_vec(),
    )
    .unwrap();
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope.attach_target_payload(payload).unwrap();
    CompiledArtifact::new(
        Target::Ptx,
        envelope,
        vyre_aot::VERSION,
        vec![1, 2, 3, 4, 5, 6, 7, 8],
    )
    .unwrap()
}
