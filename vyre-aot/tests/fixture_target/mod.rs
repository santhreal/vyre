use std::collections::HashSet;
use std::sync::LazyLock;

use vyre_driver::Device;
use vyre_aot::{
    ArtifactEnvelope, TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetProfile,
    TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};
use vyre_foundation::ir::{BufferAccess, DataType, Program, ProgramGraph, ValueLifetime};
use vyre_megakernel::{
    compile_selected_modules, EmittedTargetModule, TargetModuleBundle, TargetModuleImage,
};
use vyre_megakernel::{TargetCompileError, TargetCompiler};

use vyre_test_support::artifact_fixtures::{compile_graph, contract, graph_over};

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
struct FixtureMaterializer {
    device: vyre_driver::materialize::MaterializerDevice,
}

impl FixtureMaterializer {
    fn new() -> Result<Self, vyre_driver::BackendError> {
        let profile = TargetProfile::new("fixture-target-format", 1, [64, 1, 1], 64, 1_024, 0)
            .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
        let device = vyre_driver::materialize::MaterializerDevice::acquire(
            vyre_driver::materialize::DeviceSpec {
                backend: "fixture-target",
                device: "fixture-device".to_string(),
                format_extension: "fixture-target-format",
                format_version: 1,
                profile,
            },
        )?;
        Ok(Self { device })
    }
}

impl vyre_driver::ArtifactMaterializer for FixtureMaterializer {
    fn device(&self) -> &dyn vyre_driver::Device {
        &self.device
    }

    fn materialize(
        &self,
        artifact: &vyre_megakernel::Artifact,
        payload: &TargetPayload,
    ) -> Result<Box<dyn vyre_driver::ArtifactInstance>, vyre_driver::BackendError> {
        Ok(vyre_test_support::fixture_instance::FixtureInstance::neutral(
            artifact,
            payload,
            self.device.identity(),
        ))
    }

    fn allocate_resident(&self, _byte_len: usize) -> Result<vyre_driver::Resource, vyre_driver::BackendError> {
        Err(vyre_driver::BackendError::new("not implemented on fixture device"))
    }

    fn free_resident(&self, _resource: vyre_driver::Resource) -> Result<(), vyre_driver::BackendError> {
        Ok(())
    }

    fn upload_resident(&self, _resource: &vyre_driver::Resource, _bytes: &[u8]) -> Result<(), vyre_driver::BackendError> {
        Ok(())
    }

    fn upload_resident_at(
        &self,
        _resource: &vyre_driver::Resource,
        _offset_bytes: usize,
        _bytes: &[u8],
    ) -> Result<(), vyre_driver::BackendError> {
        Ok(())
    }
}

fn fixture_target_materializer() -> Result<Box<dyn vyre_driver::ArtifactMaterializer>, vyre_driver::BackendError> {
    Ok(Box::new(FixtureMaterializer::new()?))
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
        materializer: Some(fixture_target_materializer),
    }
}

/// Payload format owned by [`FIXTURE_TARGET_ID`].
pub(crate) const FIXTURE_FORMAT: &str = "fixture-target-format";

/// Payload format owned by [`LAUNCHER_FIXTURE_TARGET_ID`].
pub(crate) const LAUNCHER_FIXTURE_FORMAT: &str = "fixture-launcher-format";

/// A second fixture target that does own a launcher emitter.
///
/// [`FIXTURE_TARGET_ID`] must stay launcher-unregistered, because the bundle
/// contracts assert that packaging refuses a target with no linked emitter and
/// writes nothing first. `inventory::submit!` registers at link time even when
/// it is written inside a function body, so a launcher emitter submitted for
/// the shared fixture is visible to every test in the binary and turns those
/// two refusals into passes that never ran. The launcher tests get their own
/// target instead, and the two contracts stop competing for one registration.
pub(crate) const LAUNCHER_FIXTURE_TARGET_ID: vyre_aot::TargetId =
    vyre_aot::TargetId::expect_valid("fixture-launcher-target");

pub(crate) fn launcher_fixture_target() -> vyre_aot::TargetId {
    LAUNCHER_FIXTURE_TARGET_ID.clone()
}

fn launcher_fixture_compiler() -> Result<Box<dyn TargetCompiler>, vyre_driver::BackendError> {
    let format = TargetPayloadFormat::new(LAUNCHER_FIXTURE_FORMAT, 1).map_err(|error| {
        vyre_driver::BackendError::new(format!(
            "launcher fixture format is invalid: {error}. Fix: repair the fixture format."
        ))
    })?;
    let profile = TargetProfile::new(LAUNCHER_FIXTURE_FORMAT, 1, [64, 1, 1], 64, 0, 0)
        .map_err(|error| vyre_driver::BackendError::new(error.to_string()))?;
    Ok(Box::new(FixtureTargetCompiler { format, profile }))
}

fn emit_launcher_fixture(
    request: &vyre_driver::AotLauncherRequest<'_>,
) -> Result<vyre_driver::AotLauncherFiles, String> {
    let mut files = std::collections::BTreeMap::new();
    files.insert(
        std::path::PathBuf::from("src/main.rs"),
        format!(
            "// Generated launcher for {}\nfn main() {{ println!(\"ok\"); }}",
            request.crate_name
        ),
    );
    Ok(vyre_driver::AotLauncherFiles {
        dependencies: vec![],
        files,
    })
}

inventory::submit! {
    vyre_driver::BackendRegistration {
        id: "fixture-launcher-target",
        target_id: LAUNCHER_FIXTURE_TARGET_ID,
        payload_format: Some(LAUNCHER_FIXTURE_FORMAT),
        reference_oracle: false,
        factory: unavailable_backend,
        supported_ops: no_operations,
        semantic_operations: no_operations,
        target_compiler: Some(launcher_fixture_compiler),
        materializer: None,
    }
}

inventory::submit! {
    vyre_driver::AotLauncherEmitter {
        target: LAUNCHER_FIXTURE_TARGET_ID,
        emit: emit_launcher_fixture,
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
    compiled_artifact_with_format(FIXTURE_FORMAT)
}

/// An envelope whose payload format is owned by the launcher fixture target.
pub(crate) fn compiled_artifact_for_launcher() -> ArtifactEnvelope {
    compiled_artifact_with_format(LAUNCHER_FIXTURE_FORMAT)
}

fn compiled_artifact_with_format(format_name: &str) -> ArtifactEnvelope {
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
        numeric: vyre_megakernel::ModuleNumericRecord::default(),
        bytes: b"target-payload-fixture".to_vec(),
    }])
    .to_bytes()
    .unwrap();
    let payload = TargetPayload::new(
        &neutral,
        TargetPayloadFormat::new(format_name, 1).unwrap(),
        TargetProfile::new(format_name, 1, [64, 1, 1], 64, 0, 0).unwrap(),
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
