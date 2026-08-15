//! Last-dimension L2 normalization with float32 accumulation.

use thiserror::Error;
use vyre_foundation::composition::wrap_anonymous_region;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program, UnOp};

const OP_ID: &str = "vyre-libs::nn::last_dim_l2_norm";

/// Invalid last-dimension L2 normalization construction.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LastDimL2NormError {
    /// A tensor dimension is zero.
    #[error("last-dimension L2 normalization requires nonzero rows and width; got rows={rows}, width={width}")]
    EmptyShape {
        /// Row count.
        rows: u32,
        /// Last-dimension width.
        width: u32,
    },
    /// Flattened element count exceeds u32 indexing.
    #[error("last-dimension L2 normalization rows*width overflows u32; split the tensor")]
    ElementCountOverflow,
    /// Source dtype lacks the required floating conversion contract.
    #[error("last-dimension L2 normalization supports F16, BF16, or F32 tensors; got {dtype:?}")]
    UnsupportedDtype {
        /// Rejected dtype.
        dtype: DataType,
    },
}

/// Build `output = input * rsqrt(sum(input², last_dim) + eps)`.
///
/// Every row accumulates in F32. The normalized result converts once to the
/// source dtype at the output boundary.
///
/// # Errors
///
/// Returns [`LastDimL2NormError`] for empty or overflowing shapes and for
/// source dtypes without F16, BF16, or F32 conversion semantics.
pub fn last_dim_l2_norm(
    input: &str,
    output: &str,
    rows: u32,
    width: u32,
    eps: f32,
    dtype: DataType,
) -> Result<Program, LastDimL2NormError> {
    if rows == 0 || width == 0 {
        return Err(LastDimL2NormError::EmptyShape { rows, width });
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(LastDimL2NormError::UnsupportedDtype { dtype });
    }
    let total = rows
        .checked_mul(width)
        .ok_or(LastDimL2NormError::ElementCountOverflow)?;
    let index = Expr::var("index");
    let row_start = Expr::mul(Expr::div(index.clone(), Expr::u32(width)), Expr::u32(width));
    let normalized = Expr::mul(
        Expr::cast(DataType::F32, Expr::load(input, index.clone())),
        Expr::UnOp {
            op: UnOp::InverseSqrt,
            operand: Box::new(Expr::add(Expr::var("sum_squares"), Expr::f32(eps))),
        },
    );
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(total)),
            vec![
                Node::let_bind("row_start", row_start),
                Node::let_bind("sum_squares", Expr::f32(0.0)),
                Node::loop_for(
                    "offset",
                    Expr::u32(0),
                    Expr::u32(width),
                    vec![
                        Node::let_bind(
                            "l2_value",
                            Expr::cast(
                                DataType::F32,
                                Expr::load(
                                    input,
                                    Expr::add(Expr::var("row_start"), Expr::var("offset")),
                                ),
                            ),
                        ),
                        Node::assign(
                            "sum_squares",
                            Expr::add(
                                Expr::var("sum_squares"),
                                Expr::mul(Expr::var("l2_value"), Expr::var("l2_value")),
                            ),
                        ),
                    ],
                ),
                Node::Store {
                    buffer: output.into(),
                    index,
                    value: Expr::cast(dtype.clone(), normalized),
                },
            ],
        ),
    ];
    Ok(Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(total),
            BufferDecl::output(output, 1, dtype).with_count(total),
        ],
        [64, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    ))
}
