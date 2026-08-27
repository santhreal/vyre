//! Program to canonical neutral artifact plus attached target payload.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_foundation::transform::inline::inline_calls_with_resolver;
use vyre_foundation::transform::inline::OpResolver;
use vyre_megakernel::{
    Artifact, ArtifactEnvelope, CompileObjective, CompileRequest, DeviceFacts, Digest,
    ExternalFacts, ObjectiveMetric, SearchBudget, TargetCompiler,
};

use crate::artifact::{registration, TargetId};

const MAX_NEUTRAL_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;

/// Errors returned by [`compile`].
#[derive(Debug, Error)]
pub enum CompileError {
    /// The chosen target has no linked compiler facet.
    #[error(
        "vyre-aot: target `{0}` has no linked target compiler. Fix: link the concrete driver crate that registers this target."
    )]
    TargetNotEnabled(TargetId),

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
pub fn compile(program: &Program, target: TargetId) -> Result<ArtifactEnvelope, CompileError> {
    compile_with_resolver(program, target, None)
}

/// Compile with a caller-supplied resolver to inline `Expr::Call` nodes.
pub fn compile_with_resolver(
    program: &Program,
    target: TargetId,
    resolver: Option<OpResolver>,
) -> Result<ArtifactEnvelope, CompileError> {
    let inlined = match resolver {
        Some(resolver) => inline_calls_with_resolver(program, resolver)
            .map_err(|error| CompileError::ProgramPreparation(format!("{error:?}")))?,
        None => program.clone(),
    };
    let neutral = compile_neutral_artifact(&inlined)?;
    let compiler = registered_target_compiler(&target)?;
    vyre_megakernel::attach_target(neutral, compiler.as_ref())
        .map_err(|error| CompileError::TargetCompilation(error.to_string()))
}

fn registered_target_compiler(target: &TargetId) -> Result<Box<dyn TargetCompiler>, CompileError> {
    registration(target)
        .map_err(|_| CompileError::TargetNotEnabled(target.clone()))?
        .target_compiler()
        .map_err(|_| CompileError::TargetNotEnabled(target.clone()))
}

fn compile_neutral_artifact(program: &Program) -> Result<Artifact, CompileError> {
    let graph = ProgramGraph::from_program("main", program.clone()).map_err(|error| {
        CompileError::ArtifactLayout(format!("Program cannot enter the canonical graph: {error}"))
    })?;
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, MAX_NEUTRAL_ARTIFACT_BYTES),
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
