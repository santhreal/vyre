//! Multimodal vision encoders and vision-language connector schemas.
//!
//! Owns patchification parameters, vision transformer specifications, and
//! cross-modal projection schemas (e.g. MLP projectors, Cross-Attention resamplers).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Architecture classification of vision encoder backbones.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisionEncoderKind {
    /// Vision Transformer with linear patch embedding (CLIP, SigLIP).
    ViT,
    /// Convolutional vision backbone.
    ConvNeXt,
}

/// Cross-modal projection connector type between vision and text spaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConnectorKind {
    /// Single linear projection layer.
    Linear,
    /// Two-layer MLP with activation (e.g. LLaVA 1.5, Qwen-VL).
    TwoLayerMlp,
    /// Perceiver Resampler with learned query tokens (Flamingo, IDEFICS).
    PerceiverResampler,
    /// Cross-Attention pooling connector.
    CrossAttention,
}

/// Error during multimodal schema resolution or image shape validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MultimodalError {
    /// Image dimensions not divisible by patch size.
    #[error("Fix: image size {image_size} not divisible by patch size {patch_size}")]
    IndivisiblePatchSize {
        /// Image spatial dimension.
        image_size: u32,
        /// Patch size.
        patch_size: u32,
    },
    /// Channel count mismatch.
    #[error("Fix: expected {expected} channels, got {actual}")]
    ChannelMismatch {
        /// Expected image channels (typically 3).
        expected: u32,
        /// Actual image channels.
        actual: u32,
    },
}

/// Structural configuration for a vision encoder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisionEncoderConfig {
    /// Backbone architecture.
    pub kind: VisionEncoderKind,
    /// Square input image resolution (e.g. 224, 336, 448).
    pub image_size: u32,
    /// Square patch size (e.g. 14, 16).
    pub patch_size: u32,
    /// Color channel count (usually 3 for RGB).
    pub num_channels: u32,
    /// Hidden dimension of the vision transformer.
    pub hidden_dim: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Feed-forward intermediate width.
    pub intermediate_dim: u32,
}

impl VisionEncoderConfig {
    /// Compute the number of visual patch tokens produced by patchification.
    ///
    /// `num_patches = (image_size / patch_size)^2 + (has_cls_token ? 1 : 0)`
    #[must_use]
    pub fn num_patches(&self, has_cls_token: bool) -> usize {
        let grid = (self.image_size / self.patch_size) as usize;
        let patches = grid * grid;
        if has_cls_token {
            patches + 1
        } else {
            patches
        }
    }

    /// CLIP ViT-Large/14 (224x224, patch 14, 1024 hidden dim, 24 layers).
    #[must_use]
    pub fn clip_vit_large_14() -> Self {
        Self {
            kind: VisionEncoderKind::ViT,
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_dim: 1_024,
            num_layers: 24,
            num_heads: 16,
            intermediate_dim: 4_096,
        }
    }

    /// SigLIP SO400M (384x384, patch 14, 1152 hidden dim, 27 layers).
    #[must_use]
    pub fn siglip_so400m() -> Self {
        Self {
            kind: VisionEncoderKind::ViT,
            image_size: 384,
            patch_size: 14,
            num_channels: 3,
            hidden_dim: 1_152,
            num_layers: 27,
            num_heads: 16,
            intermediate_dim: 4_304,
        }
    }
}

/// Configuration for the vision-to-language projection connector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MultimodalConnectorConfig {
    /// Connector architecture.
    pub kind: ConnectorKind,
    /// Input dimension from vision encoder.
    pub vision_hidden_dim: u32,
    /// Output dimension into language model residual stream.
    pub text_hidden_dim: u32,
    /// Optional fixed output token count (for resampler architectures).
    pub num_output_tokens: Option<u32>,
}

impl MultimodalConnectorConfig {
    /// Two-layer MLP projector (e.g. LLaVA 1.5 mapping CLIP 1024 to LLaMA 4096).
    #[must_use]
    pub fn llava_mlp(vision_dim: u32, text_dim: u32) -> Self {
        Self {
            kind: ConnectorKind::TwoLayerMlp,
            vision_hidden_dim: vision_dim,
            text_hidden_dim: text_dim,
            num_output_tokens: None,
        }
    }
}
