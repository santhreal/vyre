//! Fused `linear_4bit` constructor  -  unpack-on-demand 4-bit quantized linear.
//!
//! Instead of materializing an unpacked f32 weight buffer, this kernel loads
//! the packed u32 weight, extracts the correct nibble inside the inner `k`
//! loop, and accumulates directly. This eliminates the 8× memory expansion
//! of a separate unpack dispatch.
//!
//! [`unpack_on_demand`] owns the plain INT4 dot product. The grouped affine
//! path splits across [`affine_grouped`] and its weight-tile strategy in
//! [`affine_grouped_weight_reuse`], over the geometry in [`grouped_layout`].
//! [`quantized_spec`] validates first-class quantized metadata and
//! [`planner_evidence`] reports the cost of the fused path.

mod affine_grouped;
mod affine_grouped_weight_reuse;
mod grouped_layout;
mod op_registration;
mod planner_evidence;
mod quantized_spec;
mod unpack_on_demand;

pub use affine_grouped::{linear_4bit_affine_grouped, linear_4bit_affine_grouped_batched};
pub use planner_evidence::linear_4bit_affine_grouped_planner_evidence;
pub use quantized_spec::{
    linear_4bit_affine_grouped_batched_typed, linear_4bit_affine_grouped_typed,
};
pub use unpack_on_demand::linear_4bit;

use vyre_foundation::ir::DataType;

/// Maximum absolute output drift allowed for grouped INT4 planner evidence tests.
pub const LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE: f32 = 1.0e-4;

/// Planner evidence for fused grouped INT4 linear versus dequantized matmul.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedLinear4BitPlannerEvidence {
    /// Input feature dimension.
    pub in_dim: u32,
    /// Output feature dimension.
    pub out_dim: u32,
    /// Quantization group size.
    pub group_size: u32,
    /// Number of quantization groups.
    pub group_count: u32,
    /// Packed INT4 weight bytes.
    pub packed_weight_bytes: u64,
    /// Bytes that a materialized f32 dequantized weight matrix would require.
    pub dequantized_weight_bytes: u64,
    /// Scale plus zero-point sidecar bytes.
    pub sidecar_bytes: u64,
    /// Bias bytes.
    pub bias_bytes: u64,
    /// Output bytes.
    pub output_bytes: u64,
    /// Dequantized weight bytes avoided by the fused path.
    pub dequant_bytes_elided: u64,
    /// Equivalent matmul planner M dimension.
    pub matmul_m: u32,
    /// Equivalent matmul planner K dimension.
    pub matmul_k: u32,
    /// Equivalent matmul planner N dimension.
    pub matmul_n: u32,
    /// Equivalent matmul planner K tile.
    pub matmul_tile: u32,
    /// Selected shared matmul planner path.
    pub matmul_selected_path: &'static str,
    /// Candidate tensor-core path from the shared matmul planner, when any.
    pub matmul_candidate_path: Option<&'static str>,
    /// Shared matmul planner fallback reason, when the selected path is cooperative.
    pub matmul_fallback_reason: Option<&'static str>,
    /// Whether the shared matmul planner selected a tensor-core path.
    pub tensor_core_eligible: bool,
    /// Maximum absolute output drift accepted by evidence tests.
    pub output_drift_abs_tolerance: f32,
}

/// Typed metadata for fused grouped INT4 linear.
///
/// The actual packed weight buffer is still addressed as `u32` words because
/// the kernel extracts eight nibbles per word. This spec binds that physical
/// layout to the first-class `DataType::Quantized` contract so call sites do
/// not pass an untyped integer buffer and lose the scale/zero-point semantics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizedLinear4BitSpec {
    /// Input feature dimension.
    pub in_dim: u32,
    /// Output feature dimension.
    pub out_dim: u32,
    /// First-class quantized weight metadata.
    pub weight_type: DataType,
}
