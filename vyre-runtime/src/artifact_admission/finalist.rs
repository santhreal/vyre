use std::collections::{BTreeMap, BTreeSet};

use vyre_driver::{
    ArtifactInstance, ArtifactMaterializer, BackendError, BindingSet, BoundResource,
};
use vyre_foundation::ir::GraphValueId;
use vyre_megakernel::measure::DeviceState;
use vyre_megakernel::{
    AbiAccess, Artifact, ArtifactValueId, EmittedResources, FinalistEvaluator, ResourceAbiRecord,
    ResourceRecord, TargetCompileError, TargetCompiler, TargetPayload,
};

use super::AdmittedArtifact;

pub(super) fn validate_instance(
    admitted: &AdmittedArtifact,
    materializer: &dyn ArtifactMaterializer,
    instance: &dyn ArtifactInstance,
) -> Result<(), BackendError> {
    if instance.artifact() != admitted.neutral().digest()
        || instance.payload() != admitted.target_payload().digest()
        || instance.device() != materializer.device().identity()
    {
        return Err(BackendError::InvalidProgram {
            fix: "Fix: materialized instance identities must exactly match the admitted artifact, target payload, and acquired device generation.".to_string(),
        });
    }
    Ok(())
}

/// Artifact ABI resources the caller supplies host bytes for, in slot order,
/// each paired with its canonical resource record.
///
/// One fact decides the set: whether an artifact entry produces the value. A
/// value no entry produces has no other source, so its contents at launch are
/// the caller's. A value some entry produces is device state, however many
/// entries also read it, and a retained value's successor is produced even
/// though its predecessor is bound by the caller.
///
/// The earlier form asked for the values in `entry.outputs` that were absent
/// from `entry.inputs`. That is the same set on every representable artifact,
/// because a node's newly minted outputs can never appear among the values it
/// binds as inputs, but it reads as though the arity depended on how the
/// compiler grouped the graph. Stating the rule directly removes the question.
///
/// Access then removes what nothing reads: a write-only slot's contents at
/// launch are unobservable, so binding bytes to it would ask the caller for a
/// buffer no kernel reads.
///
/// Measurement and caller submission select the same set: a measured launch that
/// bound a different set would not be timing the launch the caller performs.
///
/// # Errors
///
/// Returns an error when an ABI slot names a value the resource set does not
/// carry. Both describe one graph, so a gap is a malformed artifact, and
/// assuming a byte count for the missing value binds a buffer at the wrong size.
pub(super) fn host_input_resources(
    artifact: &Artifact,
) -> Result<Vec<(&ResourceAbiRecord, &ResourceRecord)>, BackendError> {
    let produced = entry_produced_values(artifact);
    let mut resources = Vec::new();
    for resource in &artifact.abi().resources {
        let record = artifact
            .resources()
            .iter()
            .find(|record| record.value == resource.value)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: artifact ABI slot {} names value {}, which the artifact resource set does not carry. Regenerate the artifact so its ABI and its resource set describe the same graph.",
                    resource.slot, resource.value.0
                ),
            })?;
        if !produced.contains(&resource.value) && kernel_reads_initial_bytes(resource.access) {
            resources.push((resource, record));
        }
    }
    resources.sort_unstable_by_key(|(resource, _)| resource.slot);
    Ok(resources)
}

/// Values some artifact entry produces.
fn entry_produced_values(artifact: &Artifact) -> BTreeSet<ArtifactValueId> {
    artifact
        .abi()
        .entries
        .iter()
        .flat_map(|entry| entry.outputs.iter().copied())
        .collect()
}

/// Whether the kernel reads what a slot holds at launch.
///
/// Exhaustive on purpose: a new access class must state whether its initial
/// contents are read before a caller can be asked for them, and a wildcard arm
/// would file it under whichever answer happened to be first.
fn kernel_reads_initial_bytes(access: AbiAccess) -> bool {
    match access {
        AbiAccess::ReadOnly | AbiAccess::Uniform | AbiAccess::ReadWrite => true,
        AbiAccess::WriteOnly => false,
    }
}

/// Compiler finalist evaluation on the acquired device.
///
/// The compiler decides which plans are finalists; this supplies the device half:
/// the registered target compiler, and one materialized launch per measurement
/// whose duration is the device timestamp the backend reports. Measured
/// compilation binds exact representative workload inputs for each host-input
/// resource, preventing traps from aborting compile-time device timing.
pub(super) struct DeviceFinalists<'a> {
    pub(super) compiler: &'a dyn TargetCompiler,
    pub(super) materializer: &'a dyn ArtifactMaterializer,
    pub(super) representative_inputs: &'a BTreeMap<GraphValueId, Vec<u8>>,
}

impl FinalistEvaluator for DeviceFinalists<'_> {
    fn target_compiler(&self) -> &dyn TargetCompiler {
        self.compiler
    }

    fn resources(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Vec<EmittedResources>, TargetCompileError> {
        self.materializer
            .materialize(artifact, payload)
            .and_then(|instance| instance.emitted_resources())
            .map_err(measurement_failure)
    }

    fn device_state(&self) -> DeviceState {
        self.materializer.device_state()
    }

    fn measure(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError> {
        let instance = self
            .materializer
            .materialize(artifact, payload)
            .map_err(measurement_failure)?;
        let mut bindings = BindingSet::new(artifact.digest());
        for (resource, record) in host_input_resources(artifact).map_err(measurement_failure)? {
            let byte_count = record.byte_count;
            let byte_count = usize::try_from(byte_count).map_err(|_| {
                TargetCompileError::Unsupported(format!(
                    "artifact value {} needs {byte_count} bytes, which exceeds host addressing",
                    resource.value.0
                ))
            })?;
            let bytes = self
                .representative_inputs
                .get(&GraphValueId(resource.value.0))
                .ok_or_else(|| {
                    TargetCompileError::Unsupported(format!(
                        "finalist measurement missing representative input for host-input resource `{}` (value {})",
                        record.name, resource.value.0
                    ))
                })?;
            if bytes.len() != byte_count {
                return Err(TargetCompileError::Unsupported(format!(
                    "finalist measurement representative input for resource `{}` (value {}) has {} bytes, but artifact requires {byte_count} bytes",
                    record.name, resource.value.0, bytes.len()
                )));
            }
            bindings.insert(resource.value, BoundResource::Host(bytes.clone()));
        }
        let completion = instance
            .submit(bindings)
            .and_then(|submission| submission.wait())
            .map_err(measurement_failure)?;
        completion.device_ns.ok_or_else(|| {
            TargetCompileError::Unsupported(
                "device reported no launch duration for a finalist measurement".to_string(),
            )
        })
    }
}

fn measurement_failure(error: BackendError) -> TargetCompileError {
    TargetCompileError::Unsupported(format!("finalist measurement failed on device: {error}"))
}
