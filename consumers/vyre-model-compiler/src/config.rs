//! Published named configurations and model architecture families.
//!
//! This module owns all domain-specific model hyperparameters, checkpoint keys,
//! and architecture variants for frontier models.

use serde::{Deserialize, Serialize};
use vyre::ir::DataType;

/// Frontier model architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ModelFamily {
    /// LLaMA family dense models (LLaMA 2, LLaMA 3, LLaMA 3.1, LLaMA 3.3).
    Llama,
    /// Mistral dense and Mixtral MoE models.
    Mistral,
    /// Qwen family dense and MoE models (Qwen 2, Qwen 2.5, Qwen 3.5).
    Qwen,
    /// DeepSeek dense and MoE models with Multi-Head Latent Attention (V2, V3, V4 Flash).
    DeepSeek,
    /// Gemma family models (Gemma 1, Gemma 2).
    Gemma,
    /// Vision Transformer and multimodal backbones (CLIP, SigLIP).
    Vision,
}

/// Exhaustive slice of all canonical [`ModelFamily`] variants.
#[must_use]
pub const fn all_model_families() -> &'static [ModelFamily] {
    &[
        ModelFamily::Llama,
        ModelFamily::Mistral,
        ModelFamily::Qwen,
        ModelFamily::DeepSeek,
        ModelFamily::Gemma,
        ModelFamily::Vision,
    ]
}

impl ModelFamily {
    /// Exhaustive slice of all canonical [`ModelFamily`] variants.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        all_model_families()
    }

    /// Exhaustive ordinal index mapping every variant to its dense ordinal.
    ///
    /// Compile-time closure: adding a variant to [`ModelFamily`] without updating
    /// this exhaustive match triggers a compile error (E0004).
    #[must_use]
    pub const fn ordinal(self) -> usize {
        match self {
            Self::Llama => 0,
            Self::Mistral => 1,
            Self::Qwen => 2,
            Self::DeepSeek => 3,
            Self::Gemma => 4,
            Self::Vision => 5,
        }
    }
}

/// Normalization type used by the model architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormKind {
    /// Root Mean Square Layer Normalization.
    RmsNorm,
    /// Standard Layer Normalization with mean subtraction and scale.
    LayerNorm,
}

/// Activation function used in feed-forward networks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActivationKind {
    /// Swish-Gated Linear Unit (SwiGLU).
    SwiGlu,
    /// Gaussian Error Linear Unit (GELU).
    Gelu,
    /// Rectified Linear Unit (ReLU).
    Relu,
}

/// Mixture-of-Experts routing configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MoeConfig {
    /// Total routed experts in the layer.
    pub num_experts: u32,
    /// Top-K experts selected per token.
    pub top_k: u32,
    /// Hidden dimension of each routed expert MLP.
    pub expert_hidden_dim: u32,
    /// Hidden dimension of the shared expert MLP (if present).
    pub shared_expert_hidden_dim: Option<u32>,
}

/// Multi-Head Latent Attention (MLA) configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlaConfig {
    /// Compressed Key/Value latent rank.
    pub kv_lora_rank: u32,
    /// Decoupled RoPE dimension per head.
    pub qk_rope_head_dim: u32,
}

/// Complete structural configuration for a neural model architecture.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Canonical model identifier (e.g. `"deepseek-v4-flash"`, `"llama-3.1-8b"`).
    pub name: String,
    /// Model family classification.
    pub family: ModelFamily,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Model / hidden dimension (`d_model`).
    pub hidden_dim: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of query attention heads.
    pub num_heads: u32,
    /// Dimension per attention head.
    pub head_dim: u32,
    /// Number of Key/Value attention heads (for GQA / MQA).
    pub num_kv_heads: u32,
    /// Hidden dimension of the feed-forward / intermediate projection.
    pub intermediate_dim: u32,
    /// Epsilon for normalization layers.
    pub norm_eps: f32,
    /// Maximum sequence / context length.
    pub max_seq_len: u32,
    /// Activation function in FFN / MLP.
    pub activation: ActivationKind,
    /// Normalization kind.
    pub norm_kind: NormKind,
    /// Precision representation.
    pub dtype: DataType,
    /// Optional MoE routing configuration.
    pub moe: Option<MoeConfig>,
    /// Optional MLA compressed attention configuration.
    pub mla: Option<MlaConfig>,
}

impl ModelConfig {
    /// Compute total Key/Value cache bytes required per token.
    #[must_use]
    pub fn kv_cache_bytes_per_token(&self) -> usize {
        let element_bytes = match self.dtype {
            DataType::F16 | DataType::BF16 => 2,
            DataType::F32 => 4,
            DataType::U8 | DataType::I8 => 1,
            _ => 2,
        };
        if let Some(mla) = &self.mla {
            // MLA compresses KV into latent vector + decoupled RoPE key
            let latent_bytes = mla.kv_lora_rank as usize * element_bytes;
            let rope_bytes =
                (self.num_heads as usize * mla.qk_rope_head_dim as usize) * element_bytes;
            self.num_layers as usize * (latent_bytes + rope_bytes)
        } else {
            // Standard MHA / GQA: 2 * num_layers * num_kv_heads * head_dim * element_bytes
            2 * self.num_layers as usize
                * self.num_kv_heads as usize
                * self.head_dim as usize
                * element_bytes
        }
    }

    /// Compute approximate total weight parameters.
    #[must_use]
    pub fn total_parameter_count(&self) -> u64 {
        let embed = self.vocab_size as u64 * self.hidden_dim as u64;
        let lm_head = embed;
        let norm_per_layer = self.hidden_dim as u64 * 2; // pre-norm + post-norm
        let final_norm = self.hidden_dim as u64;

        // Attention weights per layer
        let q_proj = self.hidden_dim as u64 * (self.num_heads as u64 * self.head_dim as u64);
        let k_proj = self.hidden_dim as u64 * (self.num_kv_heads as u64 * self.head_dim as u64);
        let v_proj = self.hidden_dim as u64 * (self.num_kv_heads as u64 * self.head_dim as u64);
        let o_proj = (self.num_heads as u64 * self.head_dim as u64) * self.hidden_dim as u64;
        let attn_per_layer = q_proj + k_proj + v_proj + o_proj;

        // MLP weights per layer
        let mlp_per_layer = if let Some(moe) = &self.moe {
            let router = self.hidden_dim as u64 * moe.num_experts as u64;
            let expert_gate = self.hidden_dim as u64 * moe.expert_hidden_dim as u64;
            let expert_up = self.hidden_dim as u64 * moe.expert_hidden_dim as u64;
            let expert_down = moe.expert_hidden_dim as u64 * self.hidden_dim as u64;
            let routed_mlp = moe.num_experts as u64 * (expert_gate + expert_up + expert_down);
            let shared_mlp = if let Some(shared_dim) = moe.shared_expert_hidden_dim {
                3 * self.hidden_dim as u64 * shared_dim as u64
            } else {
                0
            };
            router + routed_mlp + shared_mlp
        } else {
            // Gate, Up, Down projections for SwiGLU / GeLU
            3 * self.hidden_dim as u64 * self.intermediate_dim as u64
        };

        let layer_total = attn_per_layer + mlp_per_layer + norm_per_layer;
        embed + (self.num_layers as u64 * layer_total) + final_norm + lm_head
    }

    /// Return the deterministic semantic configuration digest.
    #[must_use]
    pub fn configuration_digest(&self) -> vyre::compiler::Digest {
        let serialized = serde_json::to_vec(self).unwrap_or_default();
        vyre::compiler::Digest(vyre::hashing::domain_digest(
            b"vyre-model-configuration\0",
            &serialized,
        ))
    }
}

/// Enumeration of all published named model configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NamedModelConfig {
    /// DeepSeek V4 Flash (MLA + MoE).
    DeepSeekV4Flash,
    /// DeepSeek V3 (671B sparse MoE).
    DeepSeekV3,
    /// LLaMA 3.1 8B Instruct.
    Llama3_1_8B,
    /// LLaMA 3.1 70B Instruct.
    Llama3_1_70B,
    /// Mistral 7B v0.3.
    Mistral7B,
    /// Mixtral 8x7B MoE.
    Mixtral8x7B,
    /// Qwen 2.5 7B.
    Qwen2_5_7B,
    /// Qwen 2.5 72B.
    Qwen2_5_72B,
    /// Gemma 2 9B.
    Gemma2_9B,
    /// CLIP ViT-Large/14 vision encoder.
    ClipViTLarge14,
}

impl NamedModelConfig {
    /// Return the canonical identifier string for this named model configuration.
    #[must_use]
    pub const fn id(&self) -> &'static str {
        match self {
            Self::DeepSeekV4Flash => "deepseek-v4-flash",
            Self::DeepSeekV3 => "deepseek-v3",
            Self::Llama3_1_8B => "llama-3.1-8b",
            Self::Llama3_1_70B => "llama-3.1-70b",
            Self::Mistral7B => "mistral-7b",
            Self::Mixtral8x7B => "mixtral-8x7b",
            Self::Qwen2_5_7B => "qwen-2.5-7b",
            Self::Qwen2_5_72B => "qwen-2.5-72b",
            Self::Gemma2_9B => "gemma-2-9b",
            Self::ClipViTLarge14 => "clip-vit-large-14",
        }
    }

    /// Return the resolved [`ModelConfig`] for this named configuration.
    #[must_use]
    pub fn config(&self) -> ModelConfig {
        match self {
            Self::DeepSeekV4Flash => ModelConfig {
                name: "deepseek-v4-flash".to_string(),
                family: ModelFamily::DeepSeek,
                vocab_size: 129_280,
                hidden_dim: 7_168,
                num_layers: 61,
                num_heads: 128,
                head_dim: 128,
                num_kv_heads: 128,
                intermediate_dim: 18_432,
                norm_eps: 1e-6,
                max_seq_len: 4_096,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::F32,
                moe: Some(MoeConfig {
                    num_experts: 256,
                    top_k: 8,
                    expert_hidden_dim: 2_048,
                    shared_expert_hidden_dim: Some(18_432),
                }),
                mla: Some(MlaConfig {
                    kv_lora_rank: 512,
                    qk_rope_head_dim: 64,
                }),
            },
            Self::DeepSeekV3 => ModelConfig {
                name: "deepseek-v3".to_string(),
                family: ModelFamily::DeepSeek,
                vocab_size: 129_280,
                hidden_dim: 7_168,
                num_layers: 61,
                num_heads: 128,
                head_dim: 128,
                num_kv_heads: 128,
                intermediate_dim: 18_432,
                norm_eps: 1e-6,
                max_seq_len: 4_096,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::F32,
                moe: Some(MoeConfig {
                    num_experts: 256,
                    top_k: 8,
                    expert_hidden_dim: 2_048,
                    shared_expert_hidden_dim: Some(18_432),
                }),
                mla: Some(MlaConfig {
                    kv_lora_rank: 512,
                    qk_rope_head_dim: 64,
                }),
            },
            Self::Llama3_1_8B => ModelConfig {
                name: "llama-3.1-8b".to_string(),
                family: ModelFamily::Llama,
                vocab_size: 128_256,
                hidden_dim: 4_096,
                num_layers: 32,
                num_heads: 32,
                head_dim: 128,
                num_kv_heads: 8,
                intermediate_dim: 14_336,
                norm_eps: 1e-5,
                max_seq_len: 8_192,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: None,
                mla: None,
            },
            Self::Llama3_1_70B => ModelConfig {
                name: "llama-3.1-70b".to_string(),
                family: ModelFamily::Llama,
                vocab_size: 128_256,
                hidden_dim: 8_192,
                num_layers: 80,
                num_heads: 64,
                head_dim: 128,
                num_kv_heads: 8,
                intermediate_dim: 28_672,
                norm_eps: 1e-5,
                max_seq_len: 8_192,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: None,
                mla: None,
            },
            Self::Mistral7B => ModelConfig {
                name: "mistral-7b".to_string(),
                family: ModelFamily::Mistral,
                vocab_size: 32_768,
                hidden_dim: 4_096,
                num_layers: 32,
                num_heads: 32,
                head_dim: 128,
                num_kv_heads: 8,
                intermediate_dim: 14_336,
                norm_eps: 1e-5,
                max_seq_len: 4_096,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: None,
                mla: None,
            },
            Self::Mixtral8x7B => ModelConfig {
                name: "mixtral-8x7b".to_string(),
                family: ModelFamily::Mistral,
                vocab_size: 32_768,
                hidden_dim: 4_096,
                num_layers: 32,
                num_heads: 32,
                head_dim: 128,
                num_kv_heads: 8,
                intermediate_dim: 14_336,
                norm_eps: 1e-5,
                max_seq_len: 4_096,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: Some(MoeConfig {
                    num_experts: 8,
                    top_k: 2,
                    expert_hidden_dim: 14_336,
                    shared_expert_hidden_dim: None,
                }),
                mla: None,
            },
            Self::Qwen2_5_7B => ModelConfig {
                name: "qwen-2.5-7b".to_string(),
                family: ModelFamily::Qwen,
                vocab_size: 152_064,
                hidden_dim: 3_584,
                num_layers: 28,
                num_heads: 28,
                head_dim: 128,
                num_kv_heads: 4,
                intermediate_dim: 18_944,
                norm_eps: 1e-6,
                max_seq_len: 4_096,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: None,
                mla: None,
            },
            Self::Qwen2_5_72B => ModelConfig {
                name: "qwen-2.5-72b".to_string(),
                family: ModelFamily::Qwen,
                vocab_size: 152_064,
                hidden_dim: 8_192,
                num_layers: 80,
                num_heads: 64,
                head_dim: 128,
                num_kv_heads: 8,
                intermediate_dim: 29_568,
                norm_eps: 1e-6,
                max_seq_len: 4_096,
                activation: ActivationKind::SwiGlu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: None,
                mla: None,
            },
            Self::Gemma2_9B => ModelConfig {
                name: "gemma-2-9b".to_string(),
                family: ModelFamily::Gemma,
                vocab_size: 256_000,
                hidden_dim: 3_584,
                num_layers: 42,
                num_heads: 16,
                head_dim: 256,
                num_kv_heads: 8,
                intermediate_dim: 14_336,
                norm_eps: 1e-6,
                max_seq_len: 4_096,
                activation: ActivationKind::Gelu,
                norm_kind: NormKind::RmsNorm,
                dtype: DataType::BF16,
                moe: None,
                mla: None,
            },
            Self::ClipViTLarge14 => ModelConfig {
                name: "clip-vit-large-14".to_string(),
                family: ModelFamily::Vision,
                vocab_size: 0, // vision encoder has no token vocabulary
                hidden_dim: 1_024,
                num_layers: 24,
                num_heads: 16,
                head_dim: 64,
                num_kv_heads: 16,
                intermediate_dim: 4_096,
                norm_eps: 1e-5,
                max_seq_len: 257, // 256 patches + 1 CLS token
                activation: ActivationKind::Gelu,
                norm_kind: NormKind::LayerNorm,
                dtype: DataType::F32,
                moe: None,
                mla: None,
            },
        }
    }
}

/// Enumerate all declared named model configurations.
///
/// Acceptance requirement: test derives this set at run time and asserts an
/// end-to-end compile decision is recorded for every member.
#[must_use]
pub fn all_named_configs() -> Vec<NamedModelConfig> {
    vec![
        NamedModelConfig::DeepSeekV4Flash,
        NamedModelConfig::DeepSeekV3,
        NamedModelConfig::Llama3_1_8B,
        NamedModelConfig::Llama3_1_70B,
        NamedModelConfig::Mistral7B,
        NamedModelConfig::Mixtral8x7B,
        NamedModelConfig::Qwen2_5_7B,
        NamedModelConfig::Qwen2_5_72B,
        NamedModelConfig::Gemma2_9B,
        NamedModelConfig::ClipViTLarge14,
    ]
}
