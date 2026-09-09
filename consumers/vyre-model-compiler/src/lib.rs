//! # vyre-model-compiler
//!
//! Independent downstream model-compiler package consuming neutral Vyre IR compositions.
//!
//! Translates published frontier model architecture families (LLaMA, Mistral, Qwen,
//! DeepSeek, Gemma, Multimodal Vision) and checkpoint manifests into domain-neutral
//! ProgramGraphs, compiles whole-application production target artifacts, and binds
//! physical resources without model-specific compiler branches.

pub mod config;
pub mod manifest;
pub mod multimodal;
pub mod pipeline;
pub mod tokenizer;
pub mod translator;
pub mod workload;

pub use config::{
    all_named_configs, ActivationKind, MlaConfig, ModelConfig, ModelFamily, MoeConfig,
    NamedModelConfig, NormKind,
};
pub use manifest::{CheckpointManifest, ManifestError, StateEdgeDescriptor, TensorDescriptor};
pub use multimodal::{
    ConnectorKind, MultimodalConnectorConfig, MultimodalError, VisionEncoderConfig,
    VisionEncoderKind,
};
pub use pipeline::{CompiledModelArtifact, ModelCompiler, PipelineError};
pub use tokenizer::{TokenizerError, TokenizerKind, TokenizerSchema};
pub use translator::{ModelGraphBuilder, TranslationError};
pub use workload::{ExecutionPhase, WorkloadEnvelope};
