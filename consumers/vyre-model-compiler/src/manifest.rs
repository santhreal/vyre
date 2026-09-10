//! Checkpoint manifests, tensor layouts, and parameter validation.
//!
//! Maps logical model parameters (e.g. `model.layers.0.self_attn.q_proj.weight`)
//! to physical checkpoint tensors and validates that all weights and state edges
//! match model architecture contracts.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use thiserror::Error;
use vyre::ir::DataType;
use vyre_safetensors::{SafetensorEntry, ShardedSafetensorIndex};

pub use vyre_safetensors::{
    ExpectedShardDigest, SafetensorDtype, SafetensorError, TransactionalCheckpoint,
};

use crate::config::ModelConfig;

/// Error during checkpoint manifest validation or tensor ingestion.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ManifestError {
    /// A required tensor is missing from the checkpoint manifest.
    #[error("Fix: missing required checkpoint tensor '{name}' for model '{model}'")]
    MissingTensor {
        /// Missing tensor name.
        name: String,
        /// Model identifier.
        model: String,
    },
    /// A tensor's shape does not match the architecture specification.
    #[error("Fix: shape mismatch for tensor '{name}': expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// Expected dimension array.
        expected: Vec<usize>,
        /// Actual dimension array from manifest.
        actual: Vec<usize>,
    },
    /// A tensor's data type does not match the architecture precision.
    #[error("Fix: dtype mismatch for tensor '{name}': expected {expected:?}, got {actual:?}")]
    DtypeMismatch {
        /// Tensor name.
        name: String,
        /// Expected data type.
        expected: DataType,
        /// Actual data type from manifest.
        actual: DataType,
    },
    /// A state edge (e.g. KV cache) declaration is invalid.
    #[error("Fix: invalid state edge '{name}': {reason}")]
    InvalidStateEdge {
        /// State edge name.
        name: String,
        /// Description of invalidity.
        reason: String,
    },
    /// A checkpoint tensor stores an element type the IR data contract does not define.
    #[error(
        "Fix: checkpoint tensor '{name}' stores element type {dtype:?}, which the IR data contract does not define; re-export the checkpoint with a supported element type"
    )]
    UnsupportedCheckpointDtype {
        /// Tensor name.
        name: String,
        /// Stored safetensors element type.
        dtype: SafetensorDtype,
    },
    /// A checkpoint tensor declares a dimension wider than the target address space.
    #[error(
        "Fix: checkpoint tensor '{name}' declares dimension {dimension}, which exceeds the address space of this target; ingest the checkpoint on a 64-bit target"
    )]
    ShapeOutOfRange {
        /// Tensor name.
        name: String,
        /// Declared dimension.
        dimension: u64,
    },
    /// A checkpoint tensor declares a payload wider than the target address space.
    #[error(
        "Fix: checkpoint tensor '{name}' declares {byte_size} payload bytes, which exceeds the address space of this target; ingest the checkpoint on a 64-bit target"
    )]
    ByteSizeOutOfRange {
        /// Tensor name.
        name: String,
        /// Declared payload byte count.
        byte_size: u64,
    },
    /// Reading or verifying the safetensors checkpoint failed.
    #[error("Fix: checkpoint ingestion failed: {0}")]
    Checkpoint(#[from] SafetensorError),
}

/// Metadata descriptor for a single tensor in a checkpoint manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorDescriptor {
    /// Canonical checkpoint tensor key.
    pub name: String,
    /// Tensor dimensions.
    pub shape: Vec<usize>,
    /// Tensor element data type.
    pub dtype: DataType,
    /// Byte size of the raw tensor data.
    pub byte_size: usize,
}

impl TensorDescriptor {
    /// Construct a descriptor with computed byte size.
    #[must_use]
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DataType) -> Self {
        let element_size = match dtype {
            DataType::F64 | DataType::U64 | DataType::I64 => 8,
            DataType::F32 | DataType::U32 | DataType::I32 => 4,
            DataType::F16 | DataType::BF16 | DataType::U16 | DataType::I16 => 2,
            DataType::U8 | DataType::I8 | DataType::Bool => 1,
            _ => 2,
        };
        let elements: usize = shape.iter().copied().product();
        Self {
            name: name.into(),
            shape,
            dtype,
            byte_size: elements * element_size,
        }
    }

    /// Return the deterministic content identity digest for this checkpoint tensor.
    #[must_use]
    pub fn content_identity(&self) -> vyre::compiler::Digest {
        let serialized = serde_json::to_vec(self).unwrap_or_default();
        vyre::compiler::Digest(vyre::hashing::domain_digest(
            b"vyre-model-checkpoint-constant\0",
            &serialized,
        ))
    }
}

/// Checkpoint manifest declaring the complete tensor set of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    /// Model name or checkpoint identifier.
    pub model_name: String,
    /// Collection of tensor descriptors keyed by tensor name.
    pub tensors: BTreeMap<String, TensorDescriptor>,
}

impl CheckpointManifest {
    /// Create an empty manifest for a model.
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            tensors: BTreeMap::new(),
        }
    }

    /// Add a tensor descriptor to the manifest.
    pub fn insert(&mut self, descriptor: TensorDescriptor) {
        self.tensors.insert(descriptor.name.clone(), descriptor);
    }

    /// Retrieve a tensor descriptor by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.get(name)
    }

    /// Open a sharded safetensors checkpoint, verify every shard against the
    /// trusted BLAKE3 digests, and build a manifest from the verified tensor set.
    ///
    /// The expected digest set must name every shard the index references,
    /// exactly once and no others. The returned [`TransactionalCheckpoint`]
    /// holds the file descriptors pinned during verification, so tensor reads
    /// through it observe the bytes that were verified.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError::Checkpoint`] when the index is unreadable or a
    /// shard digest does not match, and the ingestion variants below when a
    /// verified tensor cannot be described.
    pub fn ingest_verified_checkpoint<'a>(
        model_name: impl Into<String>,
        checkpoint_root: impl AsRef<Path>,
        index_path: impl AsRef<Path>,
        expected: impl IntoIterator<Item = ExpectedShardDigest<'a>>,
    ) -> Result<(Self, TransactionalCheckpoint), ManifestError> {
        let index = ShardedSafetensorIndex::open(checkpoint_root, index_path)?;
        let checkpoint = index.verify_transactional(expected)?;
        let manifest = Self::from_verified_checkpoint(model_name, &checkpoint)?;
        Ok((manifest, checkpoint))
    }

    /// Build a manifest from a checkpoint whose shard content is already verified.
    ///
    /// # Errors
    ///
    /// Returns [`ManifestError::UnsupportedCheckpointDtype`] for a stored
    /// element type the IR data contract does not define, and
    /// [`ManifestError::ShapeOutOfRange`] or [`ManifestError::ByteSizeOutOfRange`]
    /// for a tensor wider than the address space of the running target.
    pub fn from_verified_checkpoint(
        model_name: impl Into<String>,
        checkpoint: &TransactionalCheckpoint,
    ) -> Result<Self, ManifestError> {
        let mut manifest = Self::new(model_name);
        for (_, handle) in checkpoint.tensors() {
            manifest.insert(checkpoint_tensor_descriptor(handle.tensor())?);
        }
        Ok(manifest)
    }

    /// Return the deterministic content identity digest for this complete checkpoint manifest.
    #[must_use]
    pub fn manifest_digest(&self) -> vyre::compiler::Digest {
        let serialized = serde_json::to_vec(self).unwrap_or_default();
        vyre::compiler::Digest(vyre::hashing::domain_digest(
            b"vyre-checkpoint-manifest\0",
            &serialized,
        ))
    }

    /// Generate an ideal synthetic manifest matching a given [`ModelConfig`].
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        let mut manifest = Self::new(&config.name);
        let dtype = config.dtype.clone();

        // Embedding table: [vocab_size, hidden_dim] (if language model)
        if config.vocab_size > 0 {
            manifest.insert(TensorDescriptor::new(
                "model.embed_tokens.weight",
                vec![config.vocab_size as usize, config.hidden_dim as usize],
                dtype.clone(),
            ));
            manifest.insert(TensorDescriptor::new(
                "lm_head.weight",
                vec![config.vocab_size as usize, config.hidden_dim as usize],
                dtype.clone(),
            ));
        }

        // Final normalization scale: [hidden_dim]
        manifest.insert(TensorDescriptor::new(
            "model.norm.weight",
            vec![config.hidden_dim as usize],
            dtype.clone(),
        ));

        // Layers
        for layer_idx in 0..config.num_layers {
            let prefix = format!("model.layers.{layer_idx}");

            // Input / post-attention norm
            manifest.insert(TensorDescriptor::new(
                format!("{prefix}.input_layernorm.weight"),
                vec![config.hidden_dim as usize],
                dtype.clone(),
            ));
            manifest.insert(TensorDescriptor::new(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![config.hidden_dim as usize],
                dtype.clone(),
            ));

            // Attention projections
            if let Some(mla) = &config.mla {
                // DeepSeek MLA: compressed latent projections + RoPE
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.self_attn.w_uk"),
                    vec![
                        mla.kv_lora_rank as usize,
                        config.num_heads as usize * config.head_dim as usize,
                    ],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.self_attn.w_uv"),
                    vec![
                        mla.kv_lora_rank as usize,
                        config.num_heads as usize * config.head_dim as usize,
                    ],
                    dtype.clone(),
                ));
            } else {
                // Standard MHA / GQA
                let q_dim = config.num_heads as usize * config.head_dim as usize;
                let kv_dim = config.num_kv_heads as usize * config.head_dim as usize;
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![q_dim, config.hidden_dim as usize],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![kv_dim, config.hidden_dim as usize],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![kv_dim, config.hidden_dim as usize],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![config.hidden_dim as usize, q_dim],
                    dtype.clone(),
                ));
            }

            // MLP / MoE projections
            if let Some(moe) = &config.moe {
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.gate.weight"),
                    vec![moe.num_experts as usize, config.hidden_dim as usize],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.experts.gate_proj.weight"),
                    vec![
                        moe.num_experts as usize,
                        moe.expert_hidden_dim as usize,
                        config.hidden_dim as usize,
                    ],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.experts.up_proj.weight"),
                    vec![
                        moe.num_experts as usize,
                        moe.expert_hidden_dim as usize,
                        config.hidden_dim as usize,
                    ],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.experts.down_proj.weight"),
                    vec![
                        moe.num_experts as usize,
                        config.hidden_dim as usize,
                        moe.expert_hidden_dim as usize,
                    ],
                    dtype.clone(),
                ));
                if let Some(shared_dim) = moe.shared_expert_hidden_dim {
                    manifest.insert(TensorDescriptor::new(
                        format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                        vec![shared_dim as usize, config.hidden_dim as usize],
                        dtype.clone(),
                    ));
                    manifest.insert(TensorDescriptor::new(
                        format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                        vec![shared_dim as usize, config.hidden_dim as usize],
                        dtype.clone(),
                    ));
                    manifest.insert(TensorDescriptor::new(
                        format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                        vec![config.hidden_dim as usize, shared_dim as usize],
                        dtype.clone(),
                    ));
                }
            } else {
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.gate_proj.weight"),
                    vec![config.intermediate_dim as usize, config.hidden_dim as usize],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.up_proj.weight"),
                    vec![config.intermediate_dim as usize, config.hidden_dim as usize],
                    dtype.clone(),
                ));
                manifest.insert(TensorDescriptor::new(
                    format!("{prefix}.mlp.down_proj.weight"),
                    vec![config.hidden_dim as usize, config.intermediate_dim as usize],
                    dtype.clone(),
                ));
            }
        }

        manifest
    }

    /// Validate that all expected parameters for [`ModelConfig`] are present with exact shapes.
    pub fn validate_against_config(&self, config: &ModelConfig) -> Result<(), ManifestError> {
        let reference = Self::from_config(config);
        for (name, expected) in &reference.tensors {
            let actual = self
                .tensors
                .get(name)
                .ok_or_else(|| ManifestError::MissingTensor {
                    name: name.clone(),
                    model: config.name.clone(),
                })?;

            if actual.shape != expected.shape {
                return Err(ManifestError::ShapeMismatch {
                    name: name.clone(),
                    expected: expected.shape.clone(),
                    actual: actual.shape.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Map a safetensors element type onto the IR data type of identical width and
/// semantics.
///
/// Returns `None` for an element type the IR data contract does not define. The
/// match is exhaustive with no catch-all arm, so an element type added to
/// `vyre-safetensors` stops this crate from compiling instead of resolving to a
/// substitute type.
#[must_use]
pub const fn data_type_for_safetensor_dtype(dtype: SafetensorDtype) -> Option<DataType> {
    match dtype {
        SafetensorDtype::BOOL => Some(DataType::Bool),
        SafetensorDtype::U8 => Some(DataType::U8),
        SafetensorDtype::I8 => Some(DataType::I8),
        SafetensorDtype::U16 => Some(DataType::U16),
        SafetensorDtype::I16 => Some(DataType::I16),
        SafetensorDtype::U32 => Some(DataType::U32),
        SafetensorDtype::I32 => Some(DataType::I32),
        SafetensorDtype::U64 => Some(DataType::U64),
        SafetensorDtype::I64 => Some(DataType::I64),
        SafetensorDtype::F16 => Some(DataType::F16),
        SafetensorDtype::BF16 => Some(DataType::BF16),
        SafetensorDtype::F32 => Some(DataType::F32),
        SafetensorDtype::F64 => Some(DataType::F64),
        SafetensorDtype::F8E4M3 | SafetensorDtype::F8E5M2 => None,
    }
}

/// Describe one verified checkpoint tensor.
///
/// The byte size comes from the verified file range rather than from a
/// recomputed element width, so the descriptor states the payload extent the
/// digest covers.
fn checkpoint_tensor_descriptor(entry: &SafetensorEntry) -> Result<TensorDescriptor, ManifestError> {
    let dtype = data_type_for_safetensor_dtype(entry.dtype).ok_or_else(|| {
        ManifestError::UnsupportedCheckpointDtype {
            name: entry.name.clone(),
            dtype: entry.dtype,
        }
    })?;
    let mut shape = Vec::with_capacity(entry.shape.len());
    for &dimension in &entry.shape {
        shape.push(
            usize::try_from(dimension).map_err(|_| ManifestError::ShapeOutOfRange {
                name: entry.name.clone(),
                dimension,
            })?,
        );
    }
    let byte_size = entry.file_range.end - entry.file_range.start;
    Ok(TensorDescriptor {
        name: entry.name.clone(),
        shape,
        dtype,
        byte_size: usize::try_from(byte_size).map_err(|_| ManifestError::ByteSizeOutOfRange {
            name: entry.name.clone(),
            byte_size,
        })?,
    })
}

/// State edge descriptor for persistent Key/Value caches across autoregressive steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateEdgeDescriptor {
    /// Logical buffer name (e.g. `"kv_cache_layer_0"`).
    pub name: String,
    /// Layer index owning this state buffer.
    pub layer_index: u32,
    /// Dimensions `[batch, num_kv_heads, max_seq_len, head_dim]` or `[batch, max_seq_len, rank]`.
    pub shape: Vec<usize>,
    /// Element data type.
    pub dtype: DataType,
    /// Total allocation bytes.
    pub byte_size: usize,
}

impl StateEdgeDescriptor {
    /// Generate all state edge descriptors for a given model configuration and batch size.
    #[must_use]
    pub fn for_model(config: &ModelConfig, batch_size: usize) -> Vec<Self> {
        let mut edges = Vec::with_capacity(config.num_layers as usize * 2);
        let element_size = match config.dtype {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            _ => 2,
        };

        for layer_idx in 0..config.num_layers {
            if let Some(mla) = &config.mla {
                // Compressed latent state edge
                let shape = vec![
                    batch_size,
                    config.max_seq_len as usize,
                    mla.kv_lora_rank as usize,
                ];
                let byte_size = shape.iter().copied().product::<usize>() * element_size;
                edges.push(Self {
                    name: format!("kv_cache_layer_{layer_idx}"),
                    layer_index: layer_idx,
                    shape,
                    dtype: config.dtype.clone(),
                    byte_size,
                });
            } else {
                // Separate K and V state edges
                let shape = vec![
                    batch_size,
                    config.num_kv_heads as usize,
                    config.max_seq_len as usize,
                    config.head_dim as usize,
                ];
                let byte_size = shape.iter().copied().product::<usize>() * element_size;
                edges.push(Self {
                    name: format!("k_cache_layer_{layer_idx}"),
                    layer_index: layer_idx,
                    shape: shape.clone(),
                    dtype: config.dtype.clone(),
                    byte_size,
                });
                edges.push(Self {
                    name: format!("v_cache_layer_{layer_idx}"),
                    layer_index: layer_idx,
                    shape,
                    dtype: config.dtype.clone(),
                    byte_size,
                });
            }
        }

        edges
    }
}
