//! Linear-layer sub-dialect: affine transforms built on `math::linalg`.
mod inner;

pub use inner::{
    batch_matmul, linear, linear_relu, linear_rows, linear_rows_no_bias,
    linear_rows_no_bias_out_in_typed, linear_rows_no_bias_typed, linear_silu, linear_tiled,
    linear_tiled_reference, rms_norm_linear, try_rms_norm_linear, Linear,
};
#[cfg(feature = "nn-linear-4bit")]
pub use inner::{
    linear_4bit, linear_4bit_affine_grouped, linear_4bit_affine_grouped_batched,
    linear_4bit_affine_grouped_batched_typed, linear_4bit_affine_grouped_planner_evidence,
    linear_4bit_affine_grouped_typed, QuantizedLinear4BitPlannerEvidence, QuantizedLinear4BitSpec,
    LINEAR_4BIT_AFFINE_GROUPED_OUTPUT_DRIFT_ABS_TOLERANCE,
};
