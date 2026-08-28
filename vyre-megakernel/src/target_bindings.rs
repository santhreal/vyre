//! Projection of verified descriptor bindings onto selected artifact resources.

use std::collections::HashSet;

use vyre_foundation::ir::Program;
use vyre_lower::{KernelDescriptor, MemoryClass};

use crate::allocation::AddressSpace;
use crate::target::{SelectedModule, TargetCompileError};
use crate::{
    Artifact, ArtifactAbi, ArtifactValueId, ResourceLifetime, TargetResourceAccess,
    TargetResourceBinding, TargetResourceMemory,
};

/// Projects verified descriptor bindings onto the selected artifact resources.
pub(crate) fn selected_resource_bindings(
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
            let memory = if matches!(
                slot.memory_class,
                MemoryClass::Constant | MemoryClass::Uniform
            ) || constant_values.contains(&resource)
            {
                TargetResourceMemory::Constant
            } else {
                TargetResourceMemory::Global
            };
            let access = match (first_input.is_some(), last_output.is_some()) {
                (true, true) => TargetResourceAccess::ReadWrite,
                (true, false) if slot.visibility == vyre_lower::BindingVisibility::ReadWrite => {
                    TargetResourceAccess::ReadWrite
                }
                (true, false) => TargetResourceAccess::ReadOnly,
                (false, true) => TargetResourceAccess::WriteOnly,
                (false, false) => inactive_access,
            };
            verify_placement(
                artifact,
                module,
                slot.name.as_str(),
                resource,
                first_input.is_some() || last_output.is_some(),
                access,
            )?;
            Ok(TargetResourceBinding {
                resource,
                group: if matches!(slot.memory_class, MemoryClass::Uniform) {
                    1
                } else {
                    0
                },
                slot: slot.slot,
                memory,
                access,
            })
        })
        .collect()
}

/// Verifies one projected binding against the selected allocation plan.
///
/// The plan states where every value lives, which stages hold it, and which
/// address space addresses it. Lowering binds storage, so it checks its own
/// bindings against that plan instead of restating it: a value the plan places
/// nowhere has no storage to bind, a group outside a placement's live range
/// binds bytes the plan has already reused, and a constant-space placement bound
/// writable writes storage the caller owns read-only. That the space agrees with
/// the value's lifetime is a plan invariant, checked where the plan is built.
fn verify_placement(
    artifact: &Artifact,
    module: &SelectedModule,
    slot: &str,
    resource: ArtifactValueId,
    accessed: bool,
    access: TargetResourceAccess,
) -> Result<(), TargetCompileError> {
    let invalid = |message: String| TargetCompileError::InvalidArtifact(message);
    let record = artifact
        .resources()
        .iter()
        .find(|candidate| candidate.value == resource);
    let Some((region, placement)) = artifact.allocation().placement(resource) else {
        if record.is_none_or(|record| record.byte_count == 0) {
            return Ok(());
        }
        return Err(invalid(format!(
            "fusion group {} binds `{slot}` to value {} the allocation plan places nowhere",
            module.group.0, resource.0
        )));
    };
    if accessed && (module.stage < placement.first_stage || module.stage > placement.last_stage) {
        return Err(invalid(format!(
            "fusion group {} at stage {} accesses `{slot}`, which the allocation plan holds over stages {}..={}",
            module.group.0, module.stage, placement.first_stage, placement.last_stage
        )));
    }
    if region.address_space == AddressSpace::Constant
        && !matches!(access, TargetResourceAccess::ReadOnly)
    {
        return Err(invalid(format!(
            "fusion group {} binds constant-space `{slot}` writable",
            module.group.0
        )));
    }
    Ok(())
}

pub(crate) fn selected_logical_element_count(
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
    let full_span = vyre_foundation::launch_covers_full_input_span(program);
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
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    vyre_foundation::admitted_logical_span(program, count)
}

pub(crate) fn selected_abi(artifact: &Artifact, module: &SelectedModule) -> ArtifactAbi {
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
