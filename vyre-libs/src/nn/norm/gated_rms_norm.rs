//! RMS normalization followed by learned scale and a SiLU gate.

use thiserror::Error;
use vyre_foundation::composition::wrap_anonymous_region;
use super::row_norm::row_sum_squares_body;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Program, UnOp};

const OP_ID: &str = "vyre-libs::nn::gated_rms_norm";
const LEARNED_OP_ID: &str = "vyre-libs::nn::learned_rms_norm";

/// Invalid gated RMSNorm construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GatedRmsNormError {
    /// A tensor dimension is zero.
    #[error(
        "gated RMSNorm requires nonzero rows and hidden size; got rows={rows}, hidden={hidden}"
    )]
    EmptyShape {
        /// Row count.
        rows: u32,
        /// Last-dimension width.
        hidden: u32,
    },
    /// Flattened element count exceeds u32 indexing.
    #[error("gated RMSNorm rows*hidden overflows u32; split the tensor")]
    ElementCountOverflow,
    /// Source dtype lacks the required floating conversion contract.
    #[error("gated RMSNorm supports F16, BF16, or F32 source tensors; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected dtype.
        dtype: DataType,
    },
}

/// Build gated RMSNorm over `rows` contiguous last-dimension vectors.
///
/// The operation order matches the model contract: float32 RMS normalization,
/// rounding back to `dtype`, learned scaling, float32 SiLU gating, then one
/// final conversion to `dtype`.
///
/// # Errors
///
/// Returns [`GatedRmsNormError`] for empty or overflowing shapes and for source
/// dtypes without F16, BF16, or F32 conversion semantics.
pub fn gated_rms_norm(
    input: &str,
    weight: &str,
    gate: &str,
    output: &str,
    rows: u32,
    hidden: u32,
    eps: f32,
    dtype: DataType,
) -> Result<Program, GatedRmsNormError> {
    rms_norm_impl(
        input,
        weight,
        Some(gate),
        output,
        rows,
        hidden,
        eps,
        dtype.clone(),
        dtype,
    )
}

/// Build gated RMSNorm with an independently typed learned scale.
///
/// This supports mixed-precision contracts where learned scales and recurrent
/// activations use different floating-point storage types.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_with_weight_dtype(
    input: &str,
    weight: &str,
    gate: &str,
    output: &str,
    rows: u32,
    hidden: u32,
    eps: f32,
    dtype: DataType,
    weight_dtype: DataType,
) -> Result<Program, GatedRmsNormError> {
    rms_norm_impl(
        input,
        weight,
        Some(gate),
        output,
        rows,
        hidden,
        eps,
        dtype,
        weight_dtype,
    )
}

/// Build learned RMSNorm over contiguous last-dimension rows without a gate.
///
/// F32 accumulation is followed by source-dtype normalization rounding,
/// learned scaling, and one final conversion to `dtype`.
pub fn learned_rms_norm(
    input: &str,
    weight: &str,
    output: &str,
    rows: u32,
    hidden: u32,
    eps: f32,
    dtype: DataType,
) -> Result<Program, GatedRmsNormError> {
    rms_norm_impl(
        input,
        weight,
        None,
        output,
        rows,
        hidden,
        eps,
        dtype.clone(),
        dtype,
    )
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_impl(
    input: &str,
    weight: &str,
    gate: Option<&str>,
    output: &str,
    rows: u32,
    hidden: u32,
    eps: f32,
    dtype: DataType,
    weight_dtype: DataType,
) -> Result<Program, GatedRmsNormError> {
    if rows == 0 || hidden == 0 {
        return Err(GatedRmsNormError::EmptyShape { rows, hidden });
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(GatedRmsNormError::UnsupportedDtype { dtype });
    }
    if !matches!(weight_dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(GatedRmsNormError::UnsupportedDtype {
            dtype: weight_dtype,
        });
    }
    let total = rows
        .checked_mul(hidden)
        .ok_or(GatedRmsNormError::ElementCountOverflow)?;
    let index = Expr::var("index");
    let row_start = Expr::mul(
        Expr::div(index.clone(), Expr::u32(hidden)),
        Expr::u32(hidden),
    );
    let source = Expr::cast(DataType::F32, Expr::load(input, index.clone()));
    let normalized = Expr::mul(
        source,
        Expr::UnOp {
            op: UnOp::InverseSqrt,
            operand: Box::new(Expr::add(
                Expr::div(Expr::var("sum_squares"), Expr::f32(hidden as f32)),
                Expr::f32(eps),
            )),
        },
    );
    let rounded_normalized = Expr::cast(dtype.clone(), normalized);
    let weighted = Expr::mul(
        Expr::cast(DataType::F32, rounded_normalized),
        Expr::cast(
            DataType::F32,
            Expr::load(weight, Expr::sub(index.clone(), row_start.clone())),
        ),
    );
    let result = gate.map_or(weighted.clone(), |gate| {
        let gate_f32 = Expr::cast(DataType::F32, Expr::load(gate, index.clone()));
        let silu_gate = Expr::div(
            gate_f32.clone(),
            Expr::add(
                Expr::f32(1.0),
                Expr::UnOp {
                    op: UnOp::Exp,
                    operand: Box::new(Expr::UnOp {
                        op: UnOp::Negate,
                        operand: Box::new(gate_f32),
                    }),
                },
            ),
        );
        Expr::mul(weighted, silu_gate)
    });
    let body = row_sum_squares_body(input, total, hidden, output, dtype.clone(), result);
    let mut buffers = vec![
        BufferDecl::storage(input, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(total),
        BufferDecl::storage(weight, 1, BufferAccess::ReadOnly, weight_dtype).with_count(hidden),
    ];
    if let Some(gate) = gate {
        buffers.push(
            BufferDecl::storage(gate, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(total),
        );
    }
    let output_slot = if gate.is_some() { 3 } else { 2 };
    buffers.push(BufferDecl::output(output, output_slot, dtype).with_count(total));
    Ok(Program::wrapped(
        buffers,
        [64, 1, 1],
        vec![wrap_anonymous_region(
            if gate.is_some() { OP_ID } else { LEARNED_OP_ID },
            body,
        )],
    ))
}
