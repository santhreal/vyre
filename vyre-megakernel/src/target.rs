use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;
use vyre_foundation::{
    execution_plan::fusion::merge_programs_shared, ir::Program, schedule::SchedulePhase,
};
use vyre_lower::{KernelDescriptor, MemoryClass};

use crate::{
    Artifact, ArtifactAbi, ArtifactEnvelope, ArtifactNodeId, CompileError, FusionGroupId,
    FusionRecord, ResourceLifetime, TargetEntryPoint, TargetPayload, TargetPayloadFormat,
    TargetProfile, TargetResourceAccess, TargetResourceBinding, TargetResourceMemory,
};

/// One compiler-selected group decoded into verified semantic modules.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedModule {
    /// Stable selected group identity.
    pub group: FusionGroupId,
    /// Dependency stage selected by the whole-program planner.
    pub stage: u32,
    /// Typed graph node identities in deterministic emission order.
    pub nodes: Vec<ArtifactNodeId>,
    /// Canonical Programs corresponding one-for-one with `nodes`.
    pub programs: Vec<Program>,
}

/// One compiler-selected group after canonical semantic optimization and
/// verified representation lowering.
#[derive(Clone, Debug)]
pub struct SelectedLowering {
    /// Exact neutral artifact identity.
    pub artifact: crate::Digest,
    /// Stable selected group identity.
    pub group: FusionGroupId,
    /// Dependency stage selected by the whole-program planner.
    pub stage: u32,
    /// Typed graph node identities in deterministic emission order.
    pub nodes: Vec<ArtifactNodeId>,
    /// Exact backend-neutral selected schedule phase lowered into this kernel.
    pub schedule_phase: SchedulePhase,
    /// Verified backend-neutral physical kernel consumed by concrete emitters.
    physical: vyre_lower::PhysicalKernel,
    /// Canonical ABI slice for this selected group.
    pub abi: ArtifactAbi,
    /// Canonical descriptor-to-artifact resource association.
    pub canonical_bindings: Vec<TargetResourceBinding>,
    /// Authoritative logical invocation span before target grid projection.
    pub logical_element_count: u32,
    program: Program,
}

impl SelectedLowering {
    /// Borrow the verified physical descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &KernelDescriptor {
        self.physical.descriptor()
    }
}

/// Canonical target-module bundle schema carried inside one target payload.
///
/// Version 3 encodes an f32 literal by its IEEE-754 bits. Version 2 wrote the
/// number, which JSON cannot spell for a non-finite value: a bundle carrying an
/// infinity was written with `null` in its place and refused on decode. A stored
/// version 2 bundle is refused by version rather than reinterpreted, because the
/// two encodings read the same field differently.
pub const TARGET_MODULE_BUNDLE_SCHEMA_VERSION: u16 = 3;

/// One generated target module corresponding to one selected fusion group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetModuleImage {
    /// Stable selected fusion group.
    pub group: FusionGroupId,
    /// Dependency stage of this module.
    pub stage: u32,
    /// Exact selected node identities in deterministic order.
    pub nodes: Vec<ArtifactNodeId>,
    /// Canonical optimized Program wire consumed without semantic re-lowering.
    pub program: Vec<u8>,
    /// Verified physical descriptor consumed by materializers without re-lowering.
    pub descriptor: KernelDescriptor,
    /// Target entry-point name.
    pub entry_point: String,
    /// Immutable target-native module bytes.
    pub bytes: Vec<u8>,
}

impl TargetModuleImage {
    /// Resolve a Program buffer name to the exact target `(group, slot)`.
    ///
    /// Shared and scratch storage are not externally bound. Duplicate names are
    /// ambiguous and fail closed as `None`.
    #[must_use]
    pub fn binding_slot(&self, name: &str) -> Option<(u32, u32)> {
        let mut found = None;
        for slot in &self.descriptor.bindings.slots {
            if slot.name != name {
                continue;
            }
            let group = match slot.memory_class {
                MemoryClass::Shared | MemoryClass::Scratch => continue,
                MemoryClass::Uniform => 1,
                MemoryClass::Global | MemoryClass::Constant => 0,
            };
            if found.replace((group, slot.slot)).is_some() {
                return None;
            }
        }
        found
    }
}

/// Canonical ordered target modules for one neutral artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetModuleBundle {
    /// Bundle schema.
    pub schema_version: u16,
    /// Modules ordered by dependency stage and fusion-group identity.
    pub modules: Vec<TargetModuleImage>,
}

impl TargetModuleBundle {
    /// Construct and canonically order target modules.
    #[must_use]
    pub fn new(mut modules: Vec<TargetModuleImage>) -> Self {
        modules.sort_by_key(|module| (module.stage, module.group));
        Self {
            schema_version: TARGET_MODULE_BUNDLE_SCHEMA_VERSION,
            modules,
        }
    }

    /// Encode canonical target-module bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, TargetCompileError> {
        let body = serde_json::to_vec(self)
            .map_err(|error| TargetCompileError::ModuleBundle(error.to_string()))?;
        let digest = blake3::hash(&body);
        let mut bytes = Vec::with_capacity(32 + body.len());
        bytes.extend_from_slice(digest.as_bytes());
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }

    /// Decode and validate canonical target-module bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TargetCompileError> {
        let (expected, body) = bytes.split_at_checked(32).ok_or_else(|| {
            TargetCompileError::ModuleBundle("target module bundle is truncated".to_string())
        })?;
        let actual = blake3::hash(body);
        if actual.as_bytes() != expected {
            return Err(TargetCompileError::ModuleBundle(
                "target module bundle digest mismatch".to_string(),
            ));
        }
        let bundle: Self = serde_json::from_slice(body)
            .map_err(|error| TargetCompileError::ModuleBundle(error.to_string()))?;
        // Before any module content: the version says which encoding the fields
        // were written in, so a stale bundle is refused by version rather than
        // reported as a malformed descriptor it is not.
        if bundle.schema_version != TARGET_MODULE_BUNDLE_SCHEMA_VERSION {
            return Err(TargetCompileError::ModuleBundle(format!(
                "schema {} is unsupported; expected {}",
                bundle.schema_version, TARGET_MODULE_BUNDLE_SCHEMA_VERSION
            )));
        }
        for module in &bundle.modules {
            if module.nodes.is_empty() {
                return Err(TargetCompileError::ModuleBundle(format!(
                    "fusion group {} has no selected nodes",
                    module.group.0
                )));
            }
            Program::from_wire(&module.program).map_err(|error| {
                TargetCompileError::ModuleBundle(format!(
                    "fusion group {} selected Program is malformed: {error}",
                    module.group.0
                ))
            })?;
            vyre_lower::verify_descriptor(&module.descriptor).map_err(|error| {
                TargetCompileError::ModuleBundle(format!(
                    "fusion group {} descriptor is invalid: {error:?}",
                    module.group.0
                ))
            })?;
        }
        if bundle.modules.windows(2).any(|modules| {
            (modules[0].stage, modules[0].group) >= (modules[1].stage, modules[1].group)
        }) {
            return Err(TargetCompileError::ModuleBundle(
                "module bundle is not in canonical stage/group order".to_string(),
            ));
        }
        let canonical = bundle.to_bytes()?;
        if canonical != bytes {
            return Err(TargetCompileError::ModuleBundle(
                "module bundle is not in canonical stage/group order".to_string(),
            ));
        }
        Ok(bundle)
    }
}

/// Failure produced by a registered target compiler facet.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TargetCompileError {
    /// The neutral artifact could not be decoded into selected modules.
    #[error("target compiler rejected the neutral artifact: {0}")]
    InvalidArtifact(String),
    /// The target cannot represent one selected module or ABI contract.
    #[error("target capability rejected the selected plan: {0}")]
    Unsupported(String),
    /// Verified target lowering or emission failed.
    #[error("target emission failed: {0}")]
    Emission(String),
    /// Canonical target-module bundle encoding or decoding failed.
    #[error("target module bundle failed: {0}")]
    ModuleBundle(String),
    /// The emitted payload violated the canonical payload contract.
    #[error("target payload construction failed: {0}")]
    Payload(#[from] CompileError),
}

/// Pure compiler facet from a selected neutral artifact to immutable target bytes.
pub trait TargetCompiler: Send + Sync {
    /// Exact target payload format produced by this facet.
    fn format(&self) -> &TargetPayloadFormat;

    /// Immutable capability profile used by this pure compiler.
    fn profile(&self) -> &TargetProfile;

    /// Compile every selected module and project the canonical artifact ABI.
    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError>;
}
/// Compile and attach one target payload to its exact neutral artifact.
///
/// This is the only orchestration boundary from a pure target compiler facet to
/// an authenticated deployable envelope. It does not acquire a device or
/// materialize native handles.
pub fn attach_target(
    artifact: Artifact,
    compiler: &dyn TargetCompiler,
) -> Result<ArtifactEnvelope, TargetCompileError> {
    let payload = compiler.compile(&artifact)?;
    let mut envelope = ArtifactEnvelope::new(artifact);
    envelope.attach_target_payload(payload)?;
    Ok(envelope)
}

/// Decode compiler-selected modules from one authenticated neutral artifact.
///
/// Target compilers use the verified [`compile_selected_modules`] boundary
/// rather than reconstructing graph order or reading raw frontend Programs.
pub(crate) fn selected_modules(
    artifact: &Artifact,
) -> Result<Vec<SelectedModule>, TargetCompileError> {
    artifact
        .fusion()
        .iter()
        .map(|group| decode_group(artifact, group))
        .collect()
}

/// Form one generated semantic Program for a compiler-selected fusion group.
///
/// Programs in a graph composition use shared buffer names for connected
/// values. Shared fusion preserves those dataflow names, alpha-renames local
/// collisions, inserts required intra-kernel barriers, and rejects unsafe
/// geometry or aliasing.
fn fuse_selected_module(module: &SelectedModule) -> Result<Program, TargetCompileError> {
    merge_programs_shared(&module.programs).map_err(|error| {
        TargetCompileError::Unsupported(format!(
            "fusion group {} cannot form one target module: {error}",
            module.group.0
        ))
    })
}

/// Target-native bytes and the exact emitted entry metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedTargetModule {
    /// Entry point exported by the target-native module.
    pub entry_point: String,
    /// Exact target grid dimensions.
    pub grid_size: [u32; 3],
    /// Entry-local dynamic shared byte requirement.
    pub dynamic_shared_bytes: u32,
    /// Exact target workgroup dimensions.
    pub workgroup_size: [u32; 3],
    /// Exact target resource projection.
    pub resource_bindings: Vec<TargetResourceBinding>,
    /// Immutable target-native module bytes.
    pub bytes: Vec<u8>,
}

/// Compile all selected groups through the validated physical-kernel boundary
/// and package canonical target bytes.
pub fn compile_selected_modules(
    artifact: &Artifact,
    format: TargetPayloadFormat,
    profile: TargetProfile,
    mut emit: impl FnMut(
        &SelectedLowering,
        &TargetProfile,
    ) -> Result<EmittedTargetModule, TargetCompileError>,
) -> Result<TargetPayload, TargetCompileError> {
    let modules = selected_modules(artifact)?;
    let mut images = Vec::with_capacity(modules.len());
    let mut entries = Vec::with_capacity(modules.len());
    for module in modules {
        let program = fuse_selected_module(&module)?;
        let source_region = module.nodes.first().ok_or_else(|| {
            TargetCompileError::InvalidArtifact(format!(
                "fusion group {} has no source region for schedule lowering",
                module.group.0
            ))
        })?;
        let schedule = &artifact.selected_plan().schedule;
        let schedule_phase = schedule
            .phase_for_region(source_region.0)
            .cloned()
            .ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "fusion group {} has no selected schedule phase",
                    module.group.0
                ))
            })?;
        let lowered = vyre_lower::lower_scheduled(&program, schedule, schedule_phase.id).map_err(
            |error| {
                TargetCompileError::Emission(format!(
                    "verified physical lowering failed for fusion group {}: {error}",
                    module.group.0
                ))
            },
        )?;
        let bindings = selected_resource_bindings(artifact, &module, lowered.kernel.descriptor())?;
        let abi = selected_abi(artifact, &module);
        let logical_element_count =
            selected_logical_element_count(artifact, &module, &lowered.program);
        let selected = SelectedLowering {
            artifact: artifact.digest(),
            group: module.group,
            stage: module.stage,
            nodes: module.nodes,
            schedule_phase,
            physical: lowered.kernel,
            abi,
            canonical_bindings: bindings,
            logical_element_count,
            program: lowered.program,
        };
        let emitted = emit(&selected, &profile)?;
        let node = *selected.nodes.first().ok_or_else(|| {
            TargetCompileError::InvalidArtifact(format!(
                "fusion group {} has no member node",
                selected.group.0
            ))
        })?;
        let entry_point = emitted.entry_point;
        entries.push(TargetEntryPoint {
            name: entry_point.clone(),
            node,
            workgroup_size: emitted.workgroup_size,
            grid_size: emitted.grid_size,
            dynamic_shared_bytes: emitted.dynamic_shared_bytes,
            resource_bindings: emitted.resource_bindings,
        });
        let program = selected.program.to_wire().map_err(|error| {
            TargetCompileError::ModuleBundle(format!(
                "fusion group {} selected Program encoding failed: {error}",
                selected.group.0
            ))
        })?;
        images.push(TargetModuleImage {
            group: selected.group,
            stage: selected.stage,
            nodes: selected.nodes.clone(),
            program,
            descriptor: selected.descriptor().clone(),
            entry_point,
            bytes: emitted.bytes,
        });
    }
    let bytes = TargetModuleBundle::new(images).to_bytes()?;
    TargetPayload::new(artifact, format, profile, entries, bytes).map_err(Into::into)
}

/// Projects verified descriptor bindings onto the selected artifact resources.
fn selected_resource_bindings(
    artifact: &Artifact,
    module: &SelectedModule,
    descriptor: &KernelDescriptor,
) -> Result<Vec<TargetResourceBinding>, TargetCompileError> {
    // Named entry-ABI records own each node's directional value identity. The
    // artifact resource set supplies descriptor carriers that are intentionally
    // absent from one split node's graph boundary; descriptor positions never
    // participate in either lookup.
    let canonical_by_name = artifact
        .canonical_value_by_name()
        .map_err(|collision| TargetCompileError::InvalidArtifact(collision.to_string()))?;
    let constant_values = artifact
        .resources()
        .iter()
        .filter(|resource| resource.lifetime == ResourceLifetime::Constant)
        .map(|resource| resource.value)
        .collect::<HashSet<_>>();
    descriptor
        .bindings
        .slots
        .iter()
        .filter(|slot| {
            !matches!(
                slot.memory_class,
                MemoryClass::Shared | MemoryClass::Scratch
            ) && slot.name != vyre_lower::TRAP_SIDECAR_NAME
        })
        .map(|slot| {
            let mut first_input = None;
            let mut last_output = None;
            for node_id in &module.nodes {
                let Some(entry) = artifact
                    .abi()
                    .entries
                    .iter()
                    .find(|entry| entry.node == *node_id)
                else {
                    continue;
                };
                if first_input.is_none() {
                    first_input = entry
                        .input_bindings
                        .iter()
                        .find(|binding| binding.buffer == slot.name)
                        .map(|binding| binding.value);
                }
                if let Some(output) = entry
                    .output_bindings
                    .iter()
                    .find(|binding| binding.buffer == slot.name)
                    .map(|binding| binding.value)
                {
                    // A fused carrier publishes its final successor while
                    // retaining the first input as its launch predecessor.
                    last_output = Some(output);
                }
            }
            let resource = last_output
                .or(first_input)
                .or_else(|| canonical_by_name.get(slot.name.as_str()).copied())
                .ok_or_else(|| {
                    TargetCompileError::InvalidArtifact(format!(
                        "fusion group {} descriptor binding `{}` has no canonical artifact resource",
                        module.group.0, slot.name
                    ))
                })?;
            let inactive_access = artifact
                .abi()
                .resources
                .iter()
                .find(|binding| binding.value == resource)
                .and_then(|binding| match binding.access {
                    crate::AbiAccess::ReadOnly | crate::AbiAccess::Uniform => {
                        Some(TargetResourceAccess::ReadOnly)
                    }
                    crate::AbiAccess::WriteOnly | crate::AbiAccess::ReadWrite => artifact
                        .resources()
                        .iter()
                        .find(|candidate| candidate.value == resource)
                        .map(|record| {
                            if module.stage < record.first_stage {
                                TargetResourceAccess::WriteOnly
                            } else {
                                TargetResourceAccess::ReadWrite
                            }
                        }),
                })
                .unwrap_or_else(|| match slot.visibility {
                    vyre_lower::BindingVisibility::ReadOnly => TargetResourceAccess::ReadOnly,
                    vyre_lower::BindingVisibility::WriteOnly => TargetResourceAccess::WriteOnly,
                    vyre_lower::BindingVisibility::ReadWrite => TargetResourceAccess::ReadWrite,
                });
            Ok(TargetResourceBinding {
                resource,
                group: if matches!(slot.memory_class, MemoryClass::Uniform) {
                    1
                } else {
                    0
                },
                slot: slot.slot,
                memory: if matches!(
                    slot.memory_class,
                    MemoryClass::Constant | MemoryClass::Uniform
                ) || constant_values.contains(&resource)
                {
                    TargetResourceMemory::Constant
                } else {
                    TargetResourceMemory::Global
                },
                access: match (first_input.is_some(), last_output.is_some()) {
                    (true, true) => TargetResourceAccess::ReadWrite,
                    (true, false)
                        if slot.visibility == vyre_lower::BindingVisibility::ReadWrite =>
                    {
                        TargetResourceAccess::ReadWrite
                    }
                    (true, false) => TargetResourceAccess::ReadOnly,
                    (false, true) => TargetResourceAccess::WriteOnly,
                    (false, false) => inactive_access,
                },
            })
        })
        .collect()
}

fn selected_logical_element_count(
    artifact: &Artifact,
    module: &SelectedModule,
    program: &Program,
) -> u32 {
    let nodes = module.nodes.iter().copied().collect::<HashSet<_>>();
    let values = artifact
        .abi()
        .entries
        .iter()
        .filter(|entry| nodes.contains(&entry.node))
        .flat_map(|entry| entry.inputs.iter().chain(&entry.outputs))
        .copied()
        .collect::<HashSet<_>>();
    let full_span = program.stats().atomic_op_count > 0
        || vyre_foundation::program_caps::scan(program).subgroup_ops;
    let selected = artifact
        .resources()
        .iter()
        .filter(|resource| values.contains(&resource.value));
    let count = if full_span {
        selected.map(|resource| resource.element_count).max()
    } else {
        selected
            .filter(|resource| {
                artifact
                    .abi()
                    .resources
                    .iter()
                    .find(|abi| abi.value == resource.value)
                    .is_some_and(|abi| {
                        matches!(
                            abi.access,
                            crate::AbiAccess::WriteOnly | crate::AbiAccess::ReadWrite
                        )
                    })
            })
            .map(|resource| resource.element_count)
            .max()
            .or_else(|| {
                artifact
                    .resources()
                    .iter()
                    .filter(|resource| values.contains(&resource.value))
                    .map(|resource| resource.element_count)
                    .max()
            })
    }
    .unwrap_or(1)
    .max(1);
    u32::try_from(count).unwrap_or(u32::MAX)
}

fn selected_abi(artifact: &Artifact, module: &SelectedModule) -> ArtifactAbi {
    let nodes = module.nodes.iter().copied().collect::<HashSet<_>>();
    let entries = artifact
        .abi()
        .entries
        .iter()
        .filter(|entry| nodes.contains(&entry.node))
        .cloned()
        .collect::<Vec<_>>();
    let values = entries
        .iter()
        .flat_map(|entry| entry.inputs.iter().chain(&entry.outputs))
        .copied()
        .collect::<HashSet<_>>();
    ArtifactAbi {
        resources: artifact
            .abi()
            .resources
            .iter()
            .filter(|resource| values.contains(&resource.value))
            .cloned()
            .collect(),
        entries,
    }
}

/// Decode one selected group's programs at the geometry the search selected.
///
/// The workgroup a node program declares is an input to the compiler search, not
/// its result. The artifact records the selected geometry per node, so the
/// decoded programs are set to it before fusion and the emitted module declares
/// the shape every consumer launches.
fn decode_group(
    artifact: &Artifact,
    group: &FusionRecord,
) -> Result<SelectedModule, TargetCompileError> {
    let mut nodes = group.members.clone();
    nodes.sort();
    let programs = nodes
        .iter()
        .map(|node| {
            let record = artifact
                .nodes()
                .iter()
                .find(|record| record.id == *node)
                .ok_or_else(|| {
                    TargetCompileError::InvalidArtifact(format!(
                        "fusion group {} references missing node {}",
                        group.id.0, node.0
                    ))
                })?;
            let mut program = Program::from_wire(&record.program).map_err(|error| {
                TargetCompileError::InvalidArtifact(format!(
                    "node {} canonical Program failed to decode: {error}",
                    node.0
                ))
            })?;
            let geometry = artifact
                .geometry()
                .iter()
                .find(|geometry| geometry.node == *node)
                .ok_or_else(|| {
                    TargetCompileError::InvalidArtifact(format!(
                        "node {} has no selected launch geometry",
                        node.0
                    ))
                })?;
            if program.workgroup_size != geometry.workgroup_size {
                program.set_workgroup_size(geometry.workgroup_size);
            }
            Ok::<Program, TargetCompileError>(program)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SelectedModule {
        group: group.id,
        stage: group.stage,
        nodes,
        programs,
    })
}
