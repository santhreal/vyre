//! Mixture-of-Experts (MoE) sub-dialect.
pub(crate) mod gating;
pub(crate) mod softmax_top_k;
pub(crate) mod top_k;
mod topk_selection;

pub use gating::moe_gate;
pub use softmax_top_k::softmax_top_k;
pub use top_k::top_k;
