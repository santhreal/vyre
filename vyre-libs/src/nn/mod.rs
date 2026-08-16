//! Neural-net primitives  -  activation, linear, normalization, attention,
//! optimizer, quantization.
//!
//! Each function is a Category-A composition over foundation IR primitives and
//! lower-level `vyre-libs::math` functions.
//!
//! Organized into sub-dialects:
//! - `activation`  -  ReLU, LeakyReLU², LogitSoftcap, CrossEntropy, Embedding, SkipGate
//! - `linear`  -  affine linear layer
//! - `norm`  -  LayerNorm, RMSNorm, LayerwiseLNScale
//! - `attention`  -  softmax, scaled_dot_product_attention, QKGain, PartialRoPE
//! - `optim`  -  EMA, AdamW, Muon, Newton-Schulz, MuonEq-R
//! - `quant`  -  int6, int8 pack/unpack, byte_shuffle, GPTQ-SDClip
//! Consumers import operations from their category-owned modules.

#[cfg(feature = "nn-activation")]
pub mod activation;

#[cfg(feature = "nn-activation")]
pub mod backward;

#[cfg(feature = "nn-inference")]
pub mod conv;

#[cfg(feature = "nn-linear")]
pub mod linear;

#[cfg(feature = "nn-norm")]
pub mod norm;

#[cfg(feature = "nn-attention")]
pub mod attention;

#[cfg(feature = "nn-moe")]
pub mod moe;

#[cfg(any(feature = "nn-linear", feature = "nn-norm"))]
pub(crate) mod rms;

#[cfg(feature = "nn-norm")]
#[cfg(any(
    feature = "nn-inference",
    all(
        feature = "nn-activation",
        feature = "nn-linear",
        feature = "nn-norm",
        feature = "nn-attention",
        feature = "nn-moe"
    )
))]
pub mod inference_graph;

#[cfg(feature = "nn-inference")]
pub mod model;

#[cfg(feature = "nn-activation")]
pub mod optim;

#[cfg(feature = "nn-activation")]
pub mod quant;

/// Reusable attention score / normalization passes.
pub mod attention_passes;
/// Shared attention numeric-stability guards.
pub mod attention_stability;
/// Shared F32 numeric-stability guards.
pub mod f32_stability;
/// Reusable Quest-style KV paging passes.
pub mod quest_paging_passes;
