//! Target compilation, emission, and attachment boundaries.

use std::collections::BTreeMap;

use vyre_foundation::{
    execution_plan::fusion::{merge_programs_shared, rename_buffer},
    ir::{BufferAccess, BufferDecl, Program},
};
use vyre_lower::PhysicalSchedule;

use crate::target_bindings::{
    selected_abi, selected_logical_element_count, selected_resource_bindings,
};
use crate::{
    Artifact, ArtifactEnvelope, ArtifactNodeId, FusionRecord, GeometryRecord, TargetEntryPoint,
    TargetPayload, TargetPayloadFormat, TargetProfile, TargetResourceBinding,
};

use super::bundle::{
    ModuleNumericRecord, SelectedLowering, SelectedModule, TargetModuleBundle, TargetModuleImage,
};
use super::error::TargetCompileError;

/// Pure compiler facet from a selected neutral artifact to immutable target bytes.
pub trait TargetCompiler: Send + Sync {
    /// Exact target payload format produced by this facet.
    fn format(&self) -> &TargetPayloadFormat;

    /// Immutable capability profile used by this pure compiler.
    fn profile(&self) -> &TargetProfile;

    /// Compile every selected module and project the canonical artifact ABI.
    fn compile(&self, artifact: &Artifact) -> Result<TargetPayload, TargetCompileError>;
}

/// Compile one target payload and attach it for every device the artifact places
/// work on.
pub fn attach_target(
    artifact: Artifact,
    compiler: &dyn TargetCompiler,
) -> Result<ArtifactEnvelope, TargetCompileError> {
    let payload = compiler.compile(&artifact)?;
    let devices = artifact.topology().submission_devices();
    let mut envelope = ArtifactEnvelope::new(artifact);
    for device in devices {
        envelope.attach_target_payload(payload.for_device(device)?)?;
    }
    Ok(envelope)
}

/// Decode compiler-selected modules from one authenticated neutral artifact.
fn selected_modules(artifact: &Artifact) -> Result<Vec<SelectedModule>, TargetCompileError> {
    artifact
        .fusion()
        .iter()
        .map(|group| decode_group(artifact, group))
        .collect()
}

/// Form one generated semantic Program for a compiler-selected fusion group.
fn fuse_selected_module(
    artifact: &Artifact,
    module: &SelectedModule,
) -> Result<Program, TargetCompileError> {
    let unified = unify_intra_group_value_names(artifact, module)?;
    let programs = unified.as_deref().unwrap_or(&module.programs);
    let fused = merge_programs_shared(programs).map_err(|error| {
        TargetCompileError::Unsupported(format!(
            "fusion group {} cannot form one target module: {error}",
            module.group.0
        ))
    })?;
    size_module_owned_buffers(artifact, module, &fused)
}

/// True when the module, not a dispatch slot, has to state this buffer's size.
///
/// A runtime-sized storage declaration (`count = 0`) says a dispatch supplies
/// the bytes and their length. That holds for an input, and for an output the
/// dispatch layer rebinds with the host's capacity. It does not hold for a
/// buffer the module publishes as live-out or as its inlining result: the wire
/// format admits neither without a concrete positive element count, because no
/// host slot is left to size them.
fn needs_module_owned_size(buffer: &BufferDecl) -> bool {
    buffer.count() == 0
        && buffer.access() != BufferAccess::Workgroup
        && (buffer.is_pipeline_live_out() || buffer.is_output())
}

/// Value the group's entry ABI binds to `buffer`, output side first.
///
/// One name reaches both sides of a fused group: the producer writes the value
/// and, after name unification, the consumer reads it under the same name. The
/// output binding names the value the module computes into that storage, so it
/// wins, the same precedence descriptor projection resolves bindings under.
fn group_value_for_buffer(
    artifact: &Artifact,
    module: &SelectedModule,
    buffer: &str,
) -> Option<crate::ArtifactValueId> {
    let mut first_input = None;
    let mut last_output = None;
    for node in &module.nodes {
        let Some(entry) = artifact
            .abi()
            .entries
            .iter()
            .find(|entry| entry.node == *node)
        else {
            continue;
        };
        if first_input.is_none() {
            first_input = entry
                .input_bindings
                .iter()
                .find(|binding| binding.buffer.as_str() == buffer)
                .map(|binding| binding.value);
        }
        if let Some(output) = entry
            .output_bindings
            .iter()
            .find(|binding| binding.buffer.as_str() == buffer)
            .map(|binding| binding.value)
        {
            last_output = Some(output);
        }
    }
    last_output.or(first_input)
}

/// State the exact element count of every buffer the fused module owns.
///
/// An arm declares its storage runtime-sized because a dispatch sizes it.
/// Fusing a producer with its consumer removes that source for the value
/// between them: the merge unifies the two declarations into one `ReadWrite`
/// carrier the module writes before it reads and marks it live-out so launch
/// planning allocates the carrier instead of demanding host bytes for a value
/// the module computes itself. Nothing sizes that allocation afterwards, so a
/// carrier left at the arm's `count = 0` is an allocation with no length and
/// wire admission rejects it.
///
/// The artifact already resolved every value's extent against the validated
/// symbol bindings, so the count comes from the resource record for the value
/// the group's ABI binds to the buffer, converted through the buffer's own
/// element width so a declaration that reinterprets the value's element type
/// stays exact.
fn size_module_owned_buffers(
    artifact: &Artifact,
    module: &SelectedModule,
    fused: &Program,
) -> Result<Program, TargetCompileError> {
    if !fused.buffers().iter().any(needs_module_owned_size) {
        return Ok(fused.clone());
    }
    let mut buffers = Vec::with_capacity(fused.buffers().len());
    for buffer in fused.buffers() {
        let value = needs_module_owned_size(buffer)
            .then(|| group_value_for_buffer(artifact, module, buffer.name()))
            .flatten();
        let Some(value) = value else {
            buffers.push(buffer.clone());
            continue;
        };
        let byte_count = artifact
            .resources()
            .iter()
            .find(|record| record.value == value)
            .map(|record| record.byte_count)
            .ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "fusion group {} owns buffer `{}` for value {} that the artifact resource set omits",
                    module.group.0,
                    buffer.name(),
                    value.0
                ))
            })?;
        let element_bytes = buffer
            .element()
            .size_bytes()
            .filter(|width| *width != 0)
            .map(|width| width as u64)
            .ok_or_else(|| {
                TargetCompileError::Unsupported(format!(
                    "fusion group {} cannot size module-owned buffer `{}`: its element type has no fixed nonzero width. Fix: lower the carrier to a fixed-width GPU storage type before fusion.",
                    module.group.0,
                    buffer.name()
                ))
            })?;
        if byte_count % element_bytes != 0 {
            return Err(TargetCompileError::InvalidArtifact(format!(
                "fusion group {} owns buffer `{}` whose value {} spans {byte_count} bytes, not a whole number of {element_bytes}-byte elements",
                module.group.0,
                buffer.name(),
                value.0
            )));
        }
        let count = u32::try_from(byte_count / element_bytes).map_err(|_| {
            TargetCompileError::Unsupported(format!(
                "fusion group {} owns buffer `{}` with more elements than a buffer declaration can state. Fix: split the value across groups.",
                module.group.0,
                buffer.name()
            ))
        })?;
        let mut sized = buffer.clone();
        sized.count = count;
        buffers.push(sized);
    }
    Ok(fused.with_rewritten_buffers(buffers))
}

/// Give one buffer name to each value a group produces for its own members.
///
/// Fusion unifies buffers by name, so a value that never leaves the group is
/// one buffer only when the producing and consuming members spell it the same
/// way. They rarely do: each member's buffer names come from the Program its
/// caller built, so a producer writing `sum_out` and a consumer reading `s_in`
/// merged into a module with two buffers, no read-after-write barrier between
/// the arms, and a read-only declaration whose bytes the launch demanded from
/// the caller for a value it computes itself.
///
/// Members arrive sorted by node identity, which is a dependency order within
/// a group because a graph node is admitted only after the nodes it reads, so
/// one forward pass sees every producer before its consumers.
///
/// Returns `None` when every edge already agrees, which is every
/// single-member group, so the common case clones nothing.
fn unify_intra_group_value_names(
    artifact: &Artifact,
    module: &SelectedModule,
) -> Result<Option<Vec<Program>>, TargetCompileError> {
    let mut produced: BTreeMap<crate::ArtifactValueId, &str> = BTreeMap::new();
    let mut unified: Option<Vec<Program>> = None;
    for (arm, node) in module.nodes.iter().enumerate() {
        let Some(entry) = artifact
            .abi()
            .entries
            .iter()
            .find(|entry| entry.node == *node)
        else {
            continue;
        };
        for binding in &entry.input_bindings {
            let Some(&producer) = produced.get(&binding.value) else {
                continue;
            };
            if producer == binding.buffer.as_str() {
                continue;
            }
            let programs = unified.get_or_insert_with(|| module.programs.clone());
            let source = programs.get(arm).ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "fusion group {} lists node {} without a decoded Program",
                    module.group.0, node.0
                ))
            })?;
            let renamed = rename_buffer(source, &binding.buffer, producer).map_err(|error| {
                TargetCompileError::Unsupported(format!(
                    "fusion group {} cannot route value {} from `{producer}` to node {}: {error}",
                    module.group.0, binding.value.0, node.0
                ))
            })?;
            programs[arm] = renamed;
        }
        for binding in &entry.output_bindings {
            produced.insert(binding.value, binding.buffer.as_str());
        }
    }
    Ok(unified)
}

/// Target-native bytes and the exact emitted entry metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmittedTargetModule {
    /// Entry point exported by the target-native module.
    pub entry_point: String,
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
        let program = fuse_selected_module(artifact, &module)?;
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
            frontier_topology: artifact.selected_plan().frontier_topology,
            program: lowered.program,
        };
        let emitted = emit(&selected, &profile)?;
        let node = *selected.nodes.first().ok_or_else(|| {
            TargetCompileError::InvalidArtifact(format!(
                "fusion group {} has no member node",
                selected.group.0
            ))
        })?;
        let geometry = artifact
            .geometry()
            .iter()
            .find(|geometry| geometry.node == node)
            .ok_or_else(|| {
                TargetCompileError::InvalidArtifact(format!(
                    "node {} has no selected launch geometry",
                    node.0
                ))
            })?;
        let frozen = selected.physical.schedule().ok_or_else(|| {
            TargetCompileError::Emission(format!(
                "fusion group {} emitted without the frozen schedule facts. Fix: lower every emitted module through lower_scheduled.",
                selected.group.0
            ))
        })?;
        geometry_matches_frozen_schedule(geometry, frozen, node)?;
        let entry_point = emitted.entry_point;
        entries.push(TargetEntryPoint {
            name: entry_point.clone(),
            node,
            workgroup_size: geometry.workgroup_size,
            grid_size: geometry.grid,
            dynamic_shared_bytes: geometry.dynamic_shared_bytes,
            resource_bindings: emitted.resource_bindings,
        });
        let program = selected.program.to_wire().map_err(|error| {
            TargetCompileError::ModuleBundle(format!(
                "fusion group {} selected Program encoding failed: {error}",
                selected.group.0
            ))
        })?;
        let numeric = ModuleNumericRecord::of(&selected.program, &selected.schedule_phase);
        images.push(TargetModuleImage {
            group: selected.group,
            stage: selected.stage,
            nodes: selected.nodes.clone(),
            program,
            descriptor: selected.descriptor().clone(),
            entry_point,
            numeric,
            bytes: emitted.bytes,
        });
    }
    let bytes =
        TargetModuleBundle::with_topology(artifact.selected_plan().topology, images).to_bytes()?;
    TargetPayload::new(artifact, format, profile, entries, bytes).map_err(Into::into)
}

fn geometry_matches_frozen_schedule(
    geometry: &GeometryRecord,
    frozen: &PhysicalSchedule,
    node: ArtifactNodeId,
) -> Result<(), TargetCompileError> {
    let disagreement = |field: &str, recorded: String, projected: String| {
        TargetCompileError::InvalidArtifact(format!(
            "node {} records {field} {recorded} but was lowered under {projected}. Fix: project artifact geometry and physical lowering from the same selected phase.",
            node.0
        ))
    };
    if geometry.phase.0 != frozen.phase {
        return Err(disagreement(
            "schedule phase",
            geometry.phase.0.to_string(),
            frozen.phase.to_string(),
        ));
    }
    if geometry.logical_coverage != frozen.logical_coverage {
        return Err(disagreement(
            "logical coverage",
            format!("{:?}", geometry.logical_coverage),
            format!("{:?}", frozen.logical_coverage),
        ));
    }
    if geometry.workgroup_size != frozen.workgroup {
        return Err(disagreement(
            "workgroup",
            format!("{:?}", geometry.workgroup_size),
            format!("{:?}", frozen.workgroup),
        ));
    }
    if geometry.vector_width != frozen.vector_width {
        return Err(disagreement(
            "vector width",
            geometry.vector_width.to_string(),
            frozen.vector_width.to_string(),
        ));
    }
    if geometry.ring_slots != frozen.ring_slots || geometry.roles != frozen.roles {
        return Err(disagreement(
            "pipeline",
            format!(
                "{} slots across {} roles",
                geometry.ring_slots,
                geometry.roles.len()
            ),
            format!(
                "{} slots across {} roles",
                frozen.ring_slots,
                frozen.roles.len()
            ),
        ));
    }
    if geometry.barrier_phases.len() != frozen.barriers.len() {
        return Err(disagreement(
            "barrier boundaries",
            geometry.barrier_phases.len().to_string(),
            frozen.barriers.len().to_string(),
        ));
    }
    for (recorded, projected) in geometry.barrier_phases.iter().zip(&frozen.barriers) {
        if recorded.scope != projected.scope {
            return Err(disagreement(
                "barrier scope",
                format!("{:?}", recorded.scope),
                format!("{:?}", projected.scope),
            ));
        }
    }
    Ok(())
}

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
            let program = Program::from_wire(&record.program).map_err(|error| {
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
                return Err(TargetCompileError::InvalidArtifact(format!(
                    "node {} declares workgroup {:?} and the artifact selected {:?}",
                    node.0, program.workgroup_size, geometry.workgroup_size
                )));
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
