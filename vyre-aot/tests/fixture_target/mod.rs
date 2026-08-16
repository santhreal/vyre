use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_aot::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_foundation::ir::{BufferAccess, DataType, Program, ValueLifetime};
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
                    grid_size: [1, 1, 1],
                    dynamic_shared_bytes: 0,
                    workgroup_size: selected.descriptor.dispatch.workgroup_size,
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

pub(crate) fn compiled_artifact() -> ArtifactEnvelope {
    compiled_artifact_with_grid([1, 1, 1])
}

pub(crate) fn compiled_artifact_with_grid(grid_size: [u32; 3]) -> ArtifactEnvelope {
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
    let descriptor = vyre_lower::lower_verified(&program).unwrap().descriptor;
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
            workgroup_size: [64, 1, 1],
            grid_size,
            dynamic_shared_bytes: 0,
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
