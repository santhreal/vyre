//! Target compilation, emission, and attachment boundaries.

use vyre_foundation::{
    execution_plan::fusion::merge_programs_shared,
    ir::Program,
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
fn selected_modules(
    artifact: &Artifact,
) -> Result<Vec<SelectedModule>, TargetCompileError> {
    artifact
        .fusion()
        .iter()
        .map(|group| decode_group(artifact, group))
        .collect()
}

/// Form one generated semantic Program for a compiler-selected fusion group.
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
