//! Linear builder + the canonical `linear()` Cat-A constructor.

use vyre_foundation::composition::{tag_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::prelude::MatmulBias;
use crate::{
    builder::{check_tensors, BuildOptions},
    tensor_ref::{TensorRef, TensorRefError},
};

use super::tiled::{linear_tiled, LINEAR_TILED_MIN_WORK, LINEAR_TILED_TILE};

pub(super) const LINEAR_OP_ID: &str = "vyre-libs::nn::linear";
/// Typed Cat-A builder for [`linear`].
#[derive(Debug, Clone)]
pub struct Linear {
    x: TensorRef,
    w: TensorRef,
    b: TensorRef,
    out: TensorRef,
    options: BuildOptions,
}

impl Linear {
    /// Create a builder for `out[i] = sum_k x[k] * w[k, i] + b[i]`.
    #[must_use]
    pub fn new(x: TensorRef, w: TensorRef, b: TensorRef, out: TensorRef) -> Self {
        Self {
            x,
            w,
            b,
            out,
            options: BuildOptions::default(),
        }
    }

    /// Validate tensor metadata and materialize the linear Program.
    ///
    /// # Errors
    ///
    /// Returns [`TensorRefError`] when dtypes, names, shapes, or dimensions
    /// violate the linear-layer contract.
    pub fn build(self) -> Result<Program, TensorRefError> {
        check_tensors(
            LINEAR_OP_ID,
            &[
                (&self.x, DataType::U32),
                (&self.w, DataType::U32),
                (&self.b, DataType::U32),
                (&self.out, DataType::U32),
            ],
        )?;
        let x_shape = self.x.shape.as_ref();
        let w_shape = self.w.shape.as_ref();
        let b_shape = self.b.shape.as_ref();
        let out_shape = self.out.shape.as_ref();
        let expected_w = match x_shape {
            [in_dim] => match out_shape {
                [out_dim] => vec![*in_dim, *out_dim],
                _ => vec![],
            },
            _ => vec![],
        };
        if w_shape != expected_w.as_slice() {
            return Err(TensorRefError::ShapeMismatch {
                name: self.w.name_str().to_string(),
                found: self.w.shape.to_vec(),
                expected: expected_w,
                op: LINEAR_OP_ID,
            });
        }
        if b_shape != out_shape {
            return Err(TensorRefError::ShapeMismatch {
                name: self.b.name_str().to_string(),
                found: self.b.shape.to_vec(),
                expected: self.out.shape.to_vec(),
                op: LINEAR_OP_ID,
            });
        }
        let &[in_dim] = x_shape else {
            return Err(TensorRefError::ShapeMismatch {
                name: self.x.name_str().to_string(),
                found: self.x.shape.to_vec(),
                expected: vec![1],
                op: LINEAR_OP_ID,
            });
        };
        let &[out_dim] = out_shape else {
            return Err(TensorRefError::ShapeMismatch {
                name: self.out.name_str().to_string(),
                found: self.out.shape.to_vec(),
                expected: vec![1],
                op: LINEAR_OP_ID,
            });
        };
        if in_dim == 0 {
            return Err(TensorRefError::ShapeMismatch {
                name: self.x.name_str().to_string(),
                found: self.x.shape.to_vec(),
                expected: vec![1],
                op: LINEAR_OP_ID,
            });
        }
        if out_dim == 0 {
            return Err(TensorRefError::ShapeMismatch {
                name: self.out.name_str().to_string(),
                found: self.out.shape.to_vec(),
                expected: vec![1],
                op: LINEAR_OP_ID,
            });
        }
        build_linear_program(
            self.x.name_str(),
            self.w.name_str(),
            self.b.name_str(),
            self.out.name_str(),
            in_dim,
            out_dim,
            self.options,
        )
        .map_err(|_| TensorRefError::ElementCountOverflow {
            name: self.w.name_str().to_string(),
            shape: self.w.shape.to_vec(),
        })
    }
}

crate::builder::impl_cat_a_builder_options!(Linear);

/// Build a Program that computes `out[i] = sum_k x[k] * w[k, i] + b[i]`.
///
/// Shapes: `x: [in_dim]`, `w: [in_dim, out_dim]`, `b: [out_dim]`,
/// `out: [out_dim]`. Workgroup `[64, 1, 1]`  -  each invocation handles
/// one output index.
///
/// # Errors
/// Returns `Err` when `in_dim == 0` (FINDING-V7-TEST-010-LINEAR).
pub fn linear(
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    if in_dim
        .checked_mul(out_dim)
        .is_some_and(|work| work >= LINEAR_TILED_MIN_WORK)
    {
        return linear_tiled(x, w, b, out, in_dim, out_dim, LINEAR_TILED_TILE);
    }

    Linear::new(
        TensorRef::u32_1d(x, in_dim),
        TensorRef::u32_2d(w, in_dim, out_dim),
        TensorRef::u32_1d(b, out_dim),
        TensorRef::u32_1d(out, out_dim),
    )
    .build()
    .map_err(|error| format!("Fix: {LINEAR_OP_ID} build failed: {error}"))
}

/// Build a row-batched affine projection.
///
/// Shapes: `x: [rows, in_dim]`, `w: [in_dim, out_dim]`,
/// `b: [out_dim]`, and `out: [rows, out_dim]`.
///
/// # Errors
///
/// Returns `Err` for zero dimensions or any flattened element-count overflow.
pub fn linear_rows(
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    linear_rows_impl(
        x,
        w,
        Some(b),
        out,
        rows,
        in_dim,
        out_dim,
        DataType::F32,
        false,
    )
}

/// Build a row-batched bias-free projection.
///
/// Shapes: `x: [rows, in_dim]`, `w: [in_dim, out_dim]`, and
/// `out: [rows, out_dim]`.
///
/// # Errors
///
/// Returns `Err` for zero dimensions or any flattened element-count overflow.
pub fn linear_rows_no_bias(
    x: &str,
    w: &str,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
) -> Result<Program, String> {
    linear_rows_impl(x, w, None, out, rows, in_dim, out_dim, DataType::F32, false)
}

/// Build a typed row-batched bias-free projection with F32 accumulation.
///
/// # Errors
///
/// Returns `Err` for unsupported dtypes, zero dimensions, or flattened
/// element-count overflow.
#[allow(clippy::too_many_arguments)]
pub fn linear_rows_no_bias_typed(
    x: &str,
    w: &str,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    dtype: DataType,
) -> Result<Program, String> {
    linear_rows_impl(x, w, None, out, rows, in_dim, out_dim, dtype, false)
}

/// Build a typed bias-free projection from checkpoint-native `[out_dim, in_dim]` weights.
///
/// F32 accumulation is used for F16, BF16, and F32 source tensors.
///
/// # Errors
///
/// Returns `Err` for unsupported dtypes, zero dimensions, or flattened
/// element-count overflow.
#[allow(clippy::too_many_arguments)]
pub fn linear_rows_no_bias_out_in_typed(
    x: &str,
    w: &str,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    dtype: DataType,
) -> Result<Program, String> {
    linear_rows_impl(x, w, None, out, rows, in_dim, out_dim, dtype, true)
}

fn linear_rows_impl(
    x: &str,
    w: &str,
    bias: Option<&str>,
    out: &str,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    dtype: DataType,
    weight_out_in: bool,
) -> Result<Program, String> {
    if rows == 0 || in_dim == 0 || out_dim == 0 {
        return Err(
            "Fix: linear_rows requires nonzero rows, input dimension, and output dimension"
                .to_string(),
        );
    }
    if !matches!(dtype, DataType::F16 | DataType::BF16 | DataType::F32) {
        return Err(format!(
            "Fix: linear_rows supports F16, BF16, or F32 tensors; got {dtype:?}"
        ));
    }
    rows.checked_mul(in_dim).ok_or_else(|| {
        "Fix: linear_rows rows*in_dim overflows u32; split the row batch".to_string()
    })?;
    rows.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_rows rows*out_dim overflows u32; split the row batch".to_string()
    })?;
    in_dim.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear_rows in_dim*out_dim overflows u32; shard the projection".to_string()
    })?;
    let input_count = rows * in_dim;
    let output_count = rows * out_dim;
    let weight_count = in_dim * out_dim;
    let index = Expr::var("index");
    let row = Expr::div(index.clone(), Expr::u32(out_dim));
    let column = Expr::rem(index.clone(), Expr::u32(out_dim));
    let accumulator = bias.map_or_else(
        || Expr::f32(0.0),
        |name| Expr::cast(DataType::F32, Expr::load(name, Expr::var("column"))),
    );
    let weight_index = if weight_out_in {
        Expr::add(
            Expr::mul(Expr::var("column"), Expr::u32(in_dim)),
            Expr::var("inner"),
        )
    } else {
        Expr::add(
            Expr::mul(Expr::var("inner"), Expr::u32(out_dim)),
            Expr::var("column"),
        )
    };
    let body = vec![
        Node::let_bind("index", Expr::InvocationId { axis: 0 }),
        Node::if_then(
            Expr::lt(index.clone(), Expr::u32(output_count)),
            vec![
                Node::let_bind("row", row),
                Node::let_bind("column", column),
                Node::let_bind("accumulator", accumulator),
                Node::loop_for(
                    "inner",
                    Expr::u32(0),
                    Expr::u32(in_dim),
                    vec![Node::assign(
                        "accumulator",
                        Expr::add(
                            Expr::var("accumulator"),
                            Expr::mul(
                                Expr::cast(
                                    DataType::F32,
                                    Expr::load(
                                        x,
                                        Expr::add(
                                            Expr::mul(Expr::var("row"), Expr::u32(in_dim)),
                                            Expr::var("inner"),
                                        ),
                                    ),
                                ),
                                Expr::cast(DataType::F32, Expr::load(w, weight_index.clone())),
                            ),
                        ),
                    )],
                ),
                Node::Store {
                    buffer: out.into(),
                    index,
                    value: Expr::cast(dtype.clone(), Expr::var("accumulator")),
                },
            ],
        ),
    ];
    let mut buffers = vec![
        BufferDecl::storage(x, 0, BufferAccess::ReadOnly, dtype.clone()).with_count(input_count),
        BufferDecl::storage(w, 1, BufferAccess::ReadOnly, dtype.clone()).with_count(weight_count),
    ];
    if let Some(name) = bias {
        buffers.push(
            BufferDecl::storage(name, 2, BufferAccess::ReadOnly, dtype.clone()).with_count(out_dim),
        );
    }
    let output_slot = if bias.is_some() { 3 } else { 2 };
    buffers.push(BufferDecl::output(out, output_slot, dtype).with_count(output_count));
    Ok(Program::wrapped(
        buffers,
        [64, 1, 1],
        vec![wrap_anonymous_region("vyre-libs::nn::linear_rows", body)],
    ))
}

fn build_linear_program(
    x: &str,
    w: &str,
    b: &str,
    out: &str,
    in_dim: u32,
    out_dim: u32,
    options: BuildOptions,
) -> Result<Program, String> {
    in_dim.checked_mul(out_dim).ok_or_else(|| {
        "Fix: linear in_dim*out_dim overflows u32; reduce dimensions.".to_string()
    })?;
    let mut builder = MatmulBias::new(
        TensorRef::u32_2d(x, 1, in_dim),
        TensorRef::u32_2d(w, in_dim, out_dim),
        TensorRef::u32_1d(b, out_dim),
        TensorRef::u32_2d(out, 1, out_dim),
    );
    if let Some(workgroup_size) = options.workgroup_size {
        builder = builder.with_workgroup_size(workgroup_size);
    }
    let program = builder
        .build()
        .map_err(|error| format!("Fix: linear matmul_bias build failed: {error}"))?;
    Ok(tag_program(
        options.region_generator.unwrap_or(LINEAR_OP_ID),
        program,
    ))
}
