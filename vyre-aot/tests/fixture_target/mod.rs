use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_aot::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_foundation::ir::{BufferAccess, DataType, Program, ProgramGraph, ValueLifetime};
use vyre_megakernel::{
    compile_selected_modules, EmittedTargetModule, TargetModuleBundle, TargetModuleImage,
};
use vyre_megakernel::{TargetCompileError, TargetCompiler};

#[path = "../../../tests/support/artifact_fixtures.rs"]
mod artifact_fixtures;

use artifact_fixtures::{compile_graph, contract, graph_over};

pub(crate) const FIXTURE_TARGET_ID: vyre_aot::TargetId =
    vyre_aot::TargetId::expect_valid("fixture-target");

pub(crate) fn fixture_target() -> vyre_aot::TargetId {
    FIXTURE_TARGET_ID.clone()
}

fn unavailable_backend() -> Result<Box<dyn vyre_driver::VyreBackend>, vyre_driver::BackendError> {
    Err(vyre_driver::BackendError::new(
        "fixture target has no dispatch device. Fix: use it only for AOT package tests.",
    ))
}

fn no_operations() -> &'static HashSet<vyre_foundation::ir::OpId> {
    static OPERATIONS: LazyLock<HashSet<vyre_foundation::ir::OpId>> = LazyLock::new(HashSet::new);
    &OPERATIONS
}

struct FixtureTargetCompiler {
    format: TargetPayloadFormat,
    profile: TargetProfile,
}

impl TargetCompiler for FixtureTargetCompiler {
    fn format(&self) -> &TargetPayloadFormat {
        &self.format
    }

    fn profile(&self) -> &TargetProfile {
        &self.profile
    }

    fn compile(
        &self,
        artifact: &vyre_megakernel::Artifact,
    ) -> Result<TargetPayload, TargetCompileError> {
        compile_selected_modules(
            artifact,
            self.format.clone(),
            self.profile.clone(),
            |selected, _profile| {
                Ok(EmittedTargetModule {
                    entry_point: "main".to_string(),
                    resource_bindings: selected.canonical_bindings.clone(),
                    bytes: b"target-payload-fixture".to_vec(),
                })
            },
        )
    }
}

fn fixture_target_compiler() -> Result<Box<dyn TargetCompiler>, vyre_driver::BackendError> {
    let format = TargetPayloadFormat::new("fixture-target-format", 1).map_err(|error| {
        vyre_driver::BackendError::new(format!(
            "fixture target format is invalid: {error}. Fix: repair the fixture format."
        ))
    })?;
    let profile = TargetProfile::new("fixture-target-format", 1, [64, 1, 1], 64, 0, 0)
        .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
    Ok(Box::new(FixtureTargetCompiler { format, profile }))
}

inventory::submit! {
    vyre_driver::BackendRegistration {
        id: "fixture-target",
        target_id: FIXTURE_TARGET_ID,
        payload_format: Some("fixture-target-format"),
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: no_operations,
        semantic_operations: no_operations,
        target_compiler: Some(fixture_target_compiler),
        materializer: None,
    }
}

/// The launch geometry `neutral` recorded for `node`.
fn recorded_launch(
    neutral: &vyre_megakernel::Artifact,
    node: vyre_megakernel::ArtifactNodeId,
) -> &vyre_megakernel::GeometryRecord {
    neutral
        .geometry()
        .iter()
        .find(|record| record.node == node)
        .expect("the fixture artifact records geometry for every node it carries")
}

pub(crate) fn compiled_artifact() -> ArtifactEnvelope {
    let neutral = compile_graph(
        graph_over(
            "main",
            [64, 1, 1],
            &[
                (
                    "params",
                    contract(
                        DataType::U32,
                        256,
                        BufferAccess::ReadOnly,
                        ValueLifetime::Invocation,
                    ),
                ),
                (
                    "out",
                    contract(
                        DataType::U32,
                        64,
                        BufferAccess::WriteOnly,
                        ValueLifetime::Output,
                    ),
                ),
            ],
        ),
        0,
    );
    let node = neutral.nodes()[0].id;
    let launch = recorded_launch(&neutral, node);
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
    let program = Program::from_wire(&neutral.nodes()[0].program).unwrap();
    let descriptor = vyre_lower::lower_physical(&program)
        .unwrap()
        .into_descriptor();
    let program = program.to_wire().unwrap();
    let module_bytes = TargetModuleBundle::new(vec![TargetModuleImage {
        group: group.id,
        stage: group.stage,
        nodes: group.members.clone(),
        program,
        descriptor,
        entry_point: "main".into(),
        bytes: b"target-payload-fixture".to_vec(),
    }])
    .to_bytes()
    .unwrap();
    let payload = TargetPayload::new(
        &neutral,
        TargetPayloadFormat::new("fixture-target-format", 1).unwrap(),
        TargetProfile::new("fixture-target-format", 1, [64, 1, 1], 64, 0, 0).unwrap(),
        vec![TargetEntryPoint {
            name: "main".into(),
            node,
            workgroup_size: launch.workgroup_size,
            grid_size: launch.grid,
            dynamic_shared_bytes: launch.dynamic_shared_bytes,
            resource_bindings: vec![
                TargetResourceBinding {
                    resource: params,
                    group: 0,
                    slot: 0,
                    memory: TargetResourceMemory::Global,
                    access: TargetResourceAccess::ReadOnly,
                },
                TargetResourceBinding {
                    resource: out,
                    group: 0,
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

/// An artifact envelope over `program` carrying a synthetic target payload.
///
/// The neutral half comes from `compile_graph`, the one owner of fixture artifact
/// compilation, so this function only builds the target payload on top of it.
pub(crate) fn artifact_over(
    program: &Program,
    payload_format: &str,
    target_bytes: Vec<u8>,
) -> ArtifactEnvelope {
    let graph = ProgramGraph::from_program("main", program.clone())
        .expect("fixture Program must enter the canonical graph");
    let neutral = compile_graph(graph, 0);
    let node = neutral.nodes()[0].id;
    let launch = recorded_launch(&neutral, node);
    let entry = TargetEntryPoint {
        name: "main".to_string(),
        node,
        workgroup_size: launch.workgroup_size,
        grid_size: launch.grid,
        dynamic_shared_bytes: launch.dynamic_shared_bytes,
        resource_bindings: neutral
            .abi()
            .resources
            .iter()
            .map(|resource| TargetResourceBinding {
                resource: resource.value,
                group: 0,
                slot: resource.slot,
                memory: TargetResourceMemory::Global,
                access: match resource.access {
                    vyre_megakernel::AbiAccess::ReadOnly | vyre_megakernel::AbiAccess::Uniform => {
                        TargetResourceAccess::ReadOnly
                    }
                    vyre_megakernel::AbiAccess::WriteOnly => TargetResourceAccess::WriteOnly,
                    vyre_megakernel::AbiAccess::ReadWrite => TargetResourceAccess::ReadWrite,
                },
            })
            .collect(),
    };
    let payload = TargetPayload::new(
        &neutral,
        TargetPayloadFormat::new(payload_format, 1).unwrap(),
        TargetProfile::new(payload_format, 1, [1_024, 1_024, 64], 1_024, 65_536, 0).unwrap(),
        vec![entry],
        target_bytes,
    )
    .unwrap();
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope.attach_target_payload(payload).unwrap();
    envelope
}
