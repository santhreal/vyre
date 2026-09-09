//! Cooperative tiled matrix multiplication.
//!
//! Category-A composition. Computes `out = a @ b` where `a` is `m × k`,
//! `b` is `k × n`, `out` is `m × n`. Each workgroup owns a rectangular
//! output tile and cooperatively stages A/B k-tiles through workgroup
//! memory before accumulating one output element per lane.
//!
//! The public surface is [`matmul_tiled`] and [`MatmulTiled`] for the no-bias
//! variant, and [`matmul_bias_tiled`] and [`MatmulBiasTiled`] for the
//! bias-fused variant. Tile geometry and cooperative body construction stay
//! inside this module boundary.

mod body;
mod mma_body;
pub(crate) mod mma_fragment;
mod ops;
pub(crate) mod program;
mod shape;
mod tensor_core_policy;
mod tile_coords;

pub use ops::{matmul_bias_tiled, matmul_tiled, MatmulBiasTiled, MatmulTiled};
pub use shape::MatrixShape;
pub use tensor_core_policy::{
    plan_matmul_kernel, F32MatmulMode, MatmulFallbackReason, MatmulKernelCapabilities,
    MatmulKernelPath, MatmulKernelPlan, TensorCoreTileShape,
};
