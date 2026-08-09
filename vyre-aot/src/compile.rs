//! Program to canonical neutral artifact plus attached target payload.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_foundation::ir::{
    inline_calls_with_resolver, BufferAccess, OpResolver, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileRequest, Digest, ExternalFacts, SearchBudget, TargetCompiler,
};

use crate::artifact::Target;

const MAX_NEUTRAL_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

/// Errors returned by [`compile`].
#[derive(Debug, Error)]
pub enum CompileError {
    /// The chosen target has no linked compiler facet.
    #[error(
        "vyre-aot: target {0:?} has no linked target compiler. Fix: link the concrete driver crate that registers this target."
    )]
    TargetNotEnabled(Target),

    /// Frontend call expansion failed.
    #[error("vyre-aot: frontend Program preparation failed: {0}")]
    ProgramPreparation(String),

    /// The Program cannot be represented accurately in the canonical graph.
    #[error("vyre-aot: artifact graph rejected Program: {0}")]
    ArtifactLayout(String),

    /// The selected target compiler rejected the canonical artifact.
    #[error("vyre-aot: target compiler rejected artifact: {0}")]
    TargetCompilation(String),

    /// Canonical artifact construction or payload association failed.
    #[error("vyre-aot: canonical artifact stage `{stage}` failed: {source}")]
    CanonicalArtifact {
        /// AOT stage that failed.
        stage: &'static str,
        /// Structured canonical artifact error, including its exact field path.
        #[source]
        source: vyre_megakernel::CompileError,
    },
}

/// Compile a `Program` through the canonical graph compiler and a registered target facet.
pub fn compile(program: &Program, target: Target) -> Result<ArtifactEnvelope, CompileError> {
    compile_with_resolver(program, target, None)
}

/// Compile with a caller-supplied resolver to inline `Expr::Call` nodes.
pub fn compile_with_resolver(
    program: &Program,
    target: Target,
    resolver: Option<OpResolver>,
) -> Result<ArtifactEnvelope, CompileError> {
    let inlined = match resolver {
        Some(resolver) => inline_calls_with_resolver(program, resolver)
            .map_err(|error| CompileError::ProgramPreparation(format!("{error:?}")))?,
        None => program.clone(),
    };
    let neutral = compile_neutral_artifact(&inlined)?;
    let compiler = registered_target_compiler(target)?;
    attach_target(neutral, compiler.as_ref())
}

/// Attach one target payload produced from the exact canonical neutral artifact.
pub fn attach_target(
    neutral: Artifact,
    compiler: &dyn TargetCompiler,
) -> Result<ArtifactEnvelope, CompileError> {
    let payload = compiler
        .compile(&neutral)
        .map_err(|error| CompileError::TargetCompilation(error.to_string()))?;
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope
        .attach_target_payload(payload)
        .map_err(|source| CompileError::CanonicalArtifact {
            stage: "payload-association",
            source,
        })?;
    Ok(envelope)
}

fn registered_target_compiler(target: Target) -> Result<Box<dyn TargetCompiler>, CompileError> {
    let identity = target.aot_target_id();
    for registration in vyre_driver::backend::registered_backends() {
        let Ok(compiler) = registration.target_compiler() else {
            continue;
        };
        if compiler.format().identity() == identity {
            return Ok(compiler);
        }
    }
    Err(CompileError::TargetNotEnabled(target))
}

fn compile_neutral_artifact(program: &Program) -> Result<Artifact, CompileError> {
    let mut graph = ProgramGraph::new();
    for buffer in program.buffers() {
        graph
            .add_external_value(
                buffer.name(),
                ValueContract {
                    dtype: buffer.element(),
                    shape: vec![ShapeDim::Known(u64::from(buffer.count()))],
                    access: buffer.access(),
                    lifetime: resource_lifetime(buffer.access()),
                },
            )
            .map_err(|error| {
                CompileError::ArtifactLayout(format!(
                    "resource `{}` cannot enter the canonical graph: {error}",
                    buffer.name()
                ))
            })?;
    }
    graph
        .add_node("main", program.clone(), Vec::new(), Vec::new())
        .map_err(|error| {
            CompileError::ArtifactLayout(format!(
                "Program cannot enter the canonical graph: {error}"
            ))
        })?;
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        MAX_NEUTRAL_ARTIFACT_BYTES,
    )
    .validate()
    .map_err(|source| CompileError::CanonicalArtifact {
        stage: "neutral-request",
        source,
    })?;
    vyre_megakernel::compile(&request).map_err(|source| CompileError::CanonicalArtifact {
        stage: "neutral-compile",
        source,
    })
}

fn resource_lifetime(access: BufferAccess) -> ValueLifetime {
    match access {
        BufferAccess::WriteOnly => ValueLifetime::Output,
        BufferAccess::ReadWrite => ValueLifetime::Retained,
        BufferAccess::ReadOnly | BufferAccess::Uniform | BufferAccess::Workgroup => {
            ValueLifetime::Invocation
        }
        _ => ValueLifetime::Invocation,
    }
}

#[cfg(test)]
pub(crate) fn artifact_fixture(program: &Program, target_bytes: Vec<u8>) -> ArtifactEnvelope {
    use vyre_megakernel::{
        TargetEntryPoint, TargetPayload, TargetPayloadFormat, TargetResourceAccess,
        TargetResourceBinding, TargetResourceMemory,
    };

    let neutral = compile_neutral_artifact(program).expect("test Program must compile neutrally");
    let entry = TargetEntryPoint {
        name: "main".to_string(),
        node: neutral.nodes()[0].id,
        grid_size: [1, 1, 1],
        dynamic_shared_bytes: 0,
        resource_bindings: neutral
            .abi()
            .resources
            .iter()
            .map(|resource| TargetResourceBinding {
                resource: resource.value,
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
        TargetPayloadFormat::new(Target::Ptx.aot_target_id(), 1).unwrap(),
        vec![entry],
        target_bytes,
    )
    .unwrap();
    let mut envelope = ArtifactEnvelope::new(neutral);
    envelope.attach_target_payload(payload).unwrap();
    envelope
}
