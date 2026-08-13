//! Backend-neutral target-payload admission shared by every concrete driver.
//!
//! Materializing a target payload is two neutral checks bracketing one
//! backend-specific step: admit the payload against the artifact it claims to
//! implement, decode the dialect image, then project the artifact's resources
//! onto the instance. Only the middle step is target-specific.
//!
//! Copying the neutral halves per backend is what let them drift. Before this
//! module the same admission checks were written four times, and they had
//! stopped agreeing: two backends rejected a module whose entry point was not
//! `main` and two accepted it, and one spelled several shared failures with
//! different text than the other three. A payload rejected by one backend was
//! accepted by another for reasons nobody chose.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use vyre_foundation::ir::Program;
use vyre_megakernel::{
    Artifact, ArtifactValueId, FusionRecord, ResourceLifetime, TargetModuleBundle,
    TargetModuleImage, TargetPayload, TargetPayloadFormat, TargetProfile,
};

use crate::{BackendError, DispatchConfig};

/// Build the shared "recompile the payload" rejection.
#[must_use]
pub fn invalid_module(reason: &str) -> BackendError {
    BackendError::InvalidProgram {
        fix: format!("Fix: {reason}. Recompile the target payload from the neutral artifact."),
    }
}

/// Build the shared payload-decode failure for `backend`.
#[must_use]
pub fn compile_error(backend: &str, error: impl std::fmt::Display) -> BackendError {
    BackendError::KernelCompileFailed {
        backend: backend.to_string(),
        compiler_message: format!(
            "{error}. Fix: rebuild the target payload from the neutral artifact."
        ),
    }
}

/// What the acquired materializer accepts, as declared by its device.
#[derive(Clone, Copy, Debug)]
pub struct MaterializerTarget<'a> {
    /// Stable identity of the acquiring backend, used in rejection text.
    pub backend_id: &'a str,
    /// Payload format the materializer was acquired for.
    pub format: &'a TargetPayloadFormat,
    /// Device profile the materializer was acquired for.
    pub profile: &'a TargetProfile,
}

/// One target module whose identity matches the compiler-selected plan.
#[derive(Debug)]
pub struct AdmittedModule {
    /// The target-native module image, identity already verified.
    pub image: TargetModuleImage,
    /// Canonical Program decoded from the module wire.
    pub program: Arc<Program>,
    /// Dispatch configuration carried by the payload entry.
    pub config: DispatchConfig,
}

/// Admit a target payload against the artifact it claims to implement.
///
/// Every check here is a property of the neutral artifact and the payload
/// envelope, so it holds identically for every backend. The returned modules
/// are paired with their decoded Program and dispatch config, identity already
/// verified; the caller decodes `image.bytes` in its own dialect.
///
/// # Errors
///
/// Returns `BackendError::UnsupportedFeature` when the payload format is not
/// the one the materializer was acquired for, and `BackendError::InvalidProgram`
/// when the payload is not authenticated for this artifact, its profile
/// disagrees, or its module and entry counts do not match the compiler-selected
/// fusion plan.
pub fn admit(
    artifact: &Artifact,
    payload: &TargetPayload,
    target: MaterializerTarget<'_>,
) -> Result<Vec<AdmittedModule>, BackendError> {
    if payload.neutral_artifact() != artifact.digest() {
        return Err(invalid_module(
            "target payload is not authenticated for the supplied neutral artifact",
        ));
    }
    if payload.format() != target.format {
        return Err(BackendError::UnsupportedFeature {
            name: format!("target payload format `{}`", payload.format().identity()),
            backend: target.backend_id.to_string(),
        });
    }
    if payload.profile() != target.profile {
        return Err(invalid_module(
            "target payload profile does not match the acquired materializer profile",
        ));
    }

    let bundle = TargetModuleBundle::from_bytes(payload.bytes())
        .map_err(|error| compile_error(target.backend_id, error))?;
    let selected = artifact.fusion();
    if bundle.modules.len() != selected.len() {
        return Err(invalid_module(
            "target module count must equal the compiler-selected fusion-group count",
        ));
    }
    if payload.entries().len() != selected.len() {
        return Err(invalid_module(
            "target entry count must equal the compiler-selected fusion-group count",
        ));
    }

    let mut admitted = Vec::with_capacity(selected.len());
    for ((image, record), entry) in bundle
        .modules
        .into_iter()
        .zip(selected)
        .zip(payload.entries())
    {
        admit_module_identity(&image, record)?;
        if image.entry_point != "main" {
            return Err(invalid_module("target module entry point must be `main`"));
        }
        if entry.name != image.entry_point {
            return Err(invalid_module(
                "target entry metadata must name the emitted target entry point",
            ));
        }
        let program = Arc::new(Program::from_wire(&image.program).map_err(|error| {
            invalid_module(&format!("selected Program is malformed: {error}"))
        })?);
        let mut config = DispatchConfig::default();
        config.grid_override = Some(entry.grid_size);
        config.dispatch_grid = Some(entry.grid_size);
        admitted.push(AdmittedModule {
            image,
            program,
            config,
        });
    }
    Ok(admitted)
}

/// Reject a module whose identity disagrees with the neutral selected plan.
fn admit_module_identity(
    image: &TargetModuleImage,
    record: &FusionRecord,
) -> Result<(), BackendError> {
    if image.group != record.id || image.stage != record.stage || image.nodes != record.members {
        return Err(invalid_module(
            "target module group/stage/node identity must match the neutral selected plan",
        ));
    }
    Ok(())
}

/// Artifact resources sorted by lifetime, as an instance records them.
pub struct ResourceProjection {
    /// Every artifact resource by name.
    pub values: BTreeMap<String, ArtifactValueId>,
    /// Resources the artifact reports as outputs.
    pub outputs: BTreeSet<ArtifactValueId>,
    /// Resources the artifact retains across dispatches.
    pub retained: BTreeSet<ArtifactValueId>,
}

/// Project an artifact's resources onto the three sets every instance keeps.
///
/// One pass over the resource records; the per-backend copies walked them
/// three times to build the same three collections.
#[must_use]
pub fn project_resources(artifact: &Artifact) -> ResourceProjection {
    let mut projection = ResourceProjection {
        values: BTreeMap::new(),
        outputs: BTreeSet::new(),
        retained: BTreeSet::new(),
    };
    for resource in artifact.resources() {
        projection
            .values
            .insert(resource.name.clone(), resource.value);
        match resource.lifetime {
            ResourceLifetime::Output => {
                projection.outputs.insert(resource.value);
            }
            ResourceLifetime::Retained => {
                projection.retained.insert(resource.value);
            }
            _ => {}
        }
    }
    projection
}
