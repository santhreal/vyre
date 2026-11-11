//! Normalization sub-dialect: LayerNorm, RMSNorm, layerwise LN scale.
mod gated_rms_norm;
mod last_dim_l2_norm;
mod layer_norm;
pub(crate) mod layerwise_ln_scale;
mod rms_norm;
mod row_norm;

pub use gated_rms_norm::{
    gated_rms_norm, gated_rms_norm_with_weight_dtype, learned_rms_norm, GatedRmsNormError,
};
pub use last_dim_l2_norm::{last_dim_l2_norm, LastDimL2NormError};
pub use layer_norm::{layer_norm, LayerNorm};
pub use layerwise_ln_scale::layerwise_ln_scale;
pub use rms_norm::{rms_norm, rms_norm_reference};
