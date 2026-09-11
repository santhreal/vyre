//! End-to-end model compilation, artifact admission, and resource binding pipeline.
//!
//! Submits domain-neutral ProgramGraphs through the compiler seam, emits authenticated
//! target artifact envelopes, and binds physical resources into executable sessions.

use std::collections::BTreeMap;
use thiserror::Error;
use vyre::compiler::{
    compile, AbiAccess, Artifact, ArtifactEnvelope, ArtifactValueId, CompileError, CompileRequest,
    DeviceFacts, ExternalFacts, ResourceLifetime, ValidatedCompileRequest,
};
use vyre::ir::DataType;
use vyre::{
    ResourceIngestionError, ResourceManifest, ResourceManifestEntry, ResourceManifestSource,
    TypedResourceDataset,
};

use crate::config::ModelConfig;
use crate::manifest::CheckpointManifest;
use crate::translator::{ModelGraphBuilder, TranslationError};
use crate::workload::WorkloadEnvelope;

/// Error during model compilation or artifact session admission.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// Model graph translation error.
    #[error("Fix: model graph translation failed: {0}")]
    Translation(#[from] TranslationError),
    /// Compiler validation or schedule selection error.
    #[error("Fix: compiler failed to lower model graph: {0}")]
    Compile(#[from] CompileError),
    /// Artifact resource ingestion failure.
    #[error("Fix: resource ingestion failed: {0}")]
    ResourceIngestion(#[from] ResourceIngestionError),
}

/// Compiled production model artifact ready for runtime execution.
#[derive(Debug, Clone)]
pub struct CompiledModelArtifact {
    /// Canonical model configuration.
    pub config: ModelConfig,
    /// Workload envelope compiled for.
    pub workload: WorkloadEnvelope,
    /// Validated schedule-selected artifact envelope.
    pub envelope: ArtifactEnvelope,
    /// Parsed immutable artifact record.
    pub artifact: Artifact,
}

impl CompiledModelArtifact {
    /// Return the immutable artifact digest / identity hex string.
    #[must_use]
    pub fn digest(&self) -> String {
        self.artifact.digest().to_hex()
    }

    /// Return the declared node records count.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.artifact.nodes().len()
    }

    /// Return the declared executable ABI entry points count.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.artifact.abi().entries.len()
    }

    /// Return the required allocation byte count across all resources.
    #[must_use]
    pub fn required_resource_bytes(&self) -> u64 {
        self.artifact.resources().iter().map(|r| r.byte_count).sum()
    }
}

/// End-to-end model compilation pipeline coordinator.
pub struct ModelCompiler;

impl ModelCompiler {
    /// Compile a model architecture under a given workload envelope.
    ///
    /// This translates the architecture to a pure domain-neutral [`vyre_foundation::ir::ProgramGraph`],
    /// constructs an immutable [`CompileRequest`] containing only generic resource ABI,
    /// workload facts, and search budget, and runs whole-program compilation.
    pub fn compile_model(
        config: &ModelConfig,
        workload: &WorkloadEnvelope,
    ) -> Result<CompiledModelArtifact, PipelineError> {
        let builder = ModelGraphBuilder::new(config, workload);
        let graph = builder.build_graph()?;

        let manifest = CheckpointManifest::from_config(config);
        let mut constant_identities = BTreeMap::new();
        for val in graph.values() {
            if val.contract.lifetime == vyre::ir::ValueLifetime::Constant {
                let digest = if let Some(desc) = manifest.get(&val.name) {
                    desc.content_identity()
                } else {
                    let shape_dims: Vec<usize> = val
                        .contract
                        .shape
                        .iter()
                        .filter_map(|d| match d {
                            vyre::ir::ShapeDim::Known(k) => Some(*k as usize),
                            // A constant's content identity is derived from the
                            // statically known extents. The other forms resolve
                            // at run time and contribute no digest input.
                            vyre::ir::ShapeDim::Unresolved
                            | vyre::ir::ShapeDim::Symbol(_)
                            | vyre::ir::ShapeDim::Expr(_) => None,
                        })
                        .collect();
                    let desc = crate::manifest::TensorDescriptor::new(
                        &val.name,
                        shape_dims,
                        val.contract.dtype.clone(),
                    );
                    desc.content_identity()
                };
                constant_identities.insert(val.id, digest);
            }
        }

        let config_digest = config.configuration_digest();
        let mut external_facts = ExternalFacts::new(config_digest, BTreeMap::new())
            .with_expected_launch_batch(workload.expected_launch_count);
        external_facts.constant_identities = constant_identities;
        let device_facts = DeviceFacts::unknown();

        let request = CompileRequest::new(
            graph,
            external_facts,
            device_facts,
            workload.search_budget,
            workload.objective,
        );

        let validated: ValidatedCompileRequest = request.validate()?;
        let artifact = compile(&validated)?;
        let envelope = ArtifactEnvelope::new(artifact.clone());

        Ok(CompiledModelArtifact {
            config: config.clone(),
            workload: workload.clone(),
            envelope,
            artifact,
        })
    }

    /// Admit and validate a dataset manifest matching a compiled model artifact.
    ///
    /// Binds synthetic or loaded resource buffers and verifies schema conformance.
    pub fn admit_model(
        artifact: &CompiledModelArtifact,
        manifest: &CheckpointManifest,
    ) -> Result<TypedResourceDataset, PipelineError> {
        let mut entries = Vec::new();

        for (idx, resource) in artifact.artifact.resources().iter().enumerate() {
            let desc_match = manifest.get(&resource.name);
            let byte_count = desc_match
                .map(|d| d.byte_size as u64)
                .unwrap_or(resource.byte_count);

            let dtype = desc_match.map(|d| d.dtype.clone()).unwrap_or(DataType::U8);

            entries.push(ResourceManifestEntry {
                value: ArtifactValueId(idx as u32),
                name: Some(resource.name.clone()),
                dtype,
                element_count: byte_count,
                byte_count,
                lifetime: resource.lifetime,
                access: if resource.lifetime == ResourceLifetime::Constant {
                    AbiAccess::ReadOnly
                } else {
                    AbiAccess::ReadWrite
                },
                source: ResourceManifestSource::Zeroed { byte_count },
                identity: None,
            });
        }

        let runtime_manifest = ResourceManifest::new(Some(artifact.artifact.digest()), entries);
        let dataset = TypedResourceDataset::from_manifest(&runtime_manifest)?;
        Ok(dataset)
    }
}
