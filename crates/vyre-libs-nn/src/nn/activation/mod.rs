//! Activation sub-dialect + utility nn ops.
pub(crate) mod cross_entropy;
pub(crate) mod embedding;
pub(crate) mod gelu;
pub(crate) mod leaky_relu_sq;
pub(crate) mod logit_softcap;
pub(crate) mod mlp_4x_leaky_sq;
pub(crate) mod parallel_residual_block;
pub(crate) mod relu;
pub(crate) mod residual_add;
pub(crate) mod sigmoid_gate;
pub(crate) mod silu;
pub(crate) mod skip_gate;
pub(crate) mod swiglu;
pub(crate) mod unary;

pub use cross_entropy::{cross_entropy, try_cross_entropy};
pub use embedding::{embedding, embedding_typed};
pub use gelu::gelu;
pub use leaky_relu_sq::leaky_relu_sq;
pub use logit_softcap::logit_softcap;
pub use mlp_4x_leaky_sq::mlp_4x_leaky_sq;
pub use parallel_residual_block::parallel_residual_block;
pub use relu::relu;
pub use residual_add::{residual_add, residual_add_typed};
pub use sigmoid_gate::{sigmoid_gate, sigmoid_gate_typed};
pub use silu::silu;
pub use skip_gate::skip_gate;
pub use swiglu::{swiglu, swiglu_typed};
