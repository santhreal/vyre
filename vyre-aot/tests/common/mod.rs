use std::collections::BTreeMap;

use vyre_aot::{
    ArtifactEnvelope, Target, TargetEntryPoint, TargetPayload, TargetPayloadFormat,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::target::{TargetModuleBundle, TargetModuleImage};
use vyre_megakernel::{compile, CompileRequest, Digest, ExternalFacts, SearchBudget};

pub(crate) fn compiled_artifact() -> ArtifactEnvelope {
    compiled_artifact_with_grid([1, 1, 1])
}

pub(crate) fn compiled_artifact_with_grid(grid_size: [u32; 3]) -> ArtifactEnvelope {
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
    let group = &neutral.fusion()[0];
    let module_bytes = TargetModuleBundle::new(vec![TargetModuleImage {
        group: group.id,
        stage: group.stage,
        entry_point: "main".into(),
        bytes: b"target-payload-fixture".to_vec(),
    }])
    .to_bytes()
    .unwrap();
    let payload = TargetPayload::new(
        &neutral,
        TargetPayloadFormat::new(Target::Ptx.aot_target_id(), 1).unwrap(),
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
        module_bytes,
    )
    .unwrap();
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope.attach_target_payload(payload).unwrap();
    envelope
}
