//! Shared fixed-point u32 matrix-contraction IR builder.
//!
//! Several primitive domains expose matrix multiplication under different
//! semantics: tensor-network contraction and categorical monoidal composition
//! are intentionally separate public ops, but their GPU kernel body is the same
//! 16.16 fixed-point row/column contraction. This module keeps that kernel in
//! one place while callers retain their own validation language and op ids.

use vyre_foundation::ir::{Expr, Program};

use crate::builder::gemm::ContractionComposer;
use crate::plumbing::operand::tensor_ref::TensorRef;

pub(crate) fn fixed_u32_matvec_program(
    op_id: &'static str,
    matrix: &str,
    vector: &str,
    out: &str,
    n: u32,
    matrix_cells: u32,
) -> Program {
    let matrix_ref = TensorRef::u32_2d(matrix, n, n);
    let vector_ref = TensorRef::u32_1d(vector, n);
    let out_ref = TensorRef::u32_1d(out, n);
    ContractionComposer::fixed_u32_matvec(op_id, matrix_ref, vector_ref, out_ref, n, matrix_cells)
        .with_region_generator(op_id)
        .build()
        .expect("Fix: fixed_u32_matvec_program failed to build contraction program")
}

pub(crate) struct FixedMatmulContext {
    pub op_id: &'static str,
    pub operation: &'static str,
    pub lhs_label: &'static str,
    pub rhs_label: &'static str,
    pub out_label: &'static str,
    pub dimensions: [&'static str; 3],
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fixed_u32_matmul(
    lhs: &str,
    rhs: &str,
    out: &str,
    rows: u32,
    shared: u32,
    cols: u32,
    context: &FixedMatmulContext,
) -> Result<Program, String> {
    if rows == 0 || shared == 0 || cols == 0 {
        let [rows_name, shared_name, cols_name] = context.dimensions;
        return Err(format!(
            "Fix: {} requires {rows_name}, {shared_name}, {cols_name} > 0, got {rows_name}={rows}, {shared_name}={shared}, {cols_name}={cols}.",
            context.operation
        ));
    }
    let lhs_cells = crate::plumbing::operand::shape::matrix_cells(context.lhs_label, rows, shared)?;
    let rhs_cells = crate::plumbing::operand::shape::matrix_cells(context.rhs_label, shared, cols)?;
    let out_cells = crate::plumbing::operand::shape::matrix_cells(context.out_label, rows, cols)?;
    Ok(fixed_u32_matmul_program(
        context.op_id,
        lhs,
        rhs,
        out,
        rows,
        shared,
        cols,
        lhs_cells,
        rhs_cells,
        out_cells,
    ))
}

/// Build a customizable u32 matrix contraction.
///
/// The caller supplies the scalar combine operation for each
/// `lhs[i, kk]`/`rhs[kk, j]` pair and the accumulator operation that folds it
/// into `acc`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn u32_matmul_program<C, A>(
    op_id: &'static str,
    lhs: &str,
    rhs: &str,
    out: &str,
    rows: u32,
    shared: u32,
    cols: u32,
    _lhs_cells: u32,
    _rhs_cells: u32,
    _out_cells: u32,
    identity: u32,
    combine: C,
    accumulate: A,
) -> Program
where
    C: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
    A: Fn(Expr, Expr) -> Expr + Send + Sync + 'static,
{
    let lhs_ref = TensorRef::u32_2d(lhs, rows, shared);
    let rhs_ref = TensorRef::u32_2d(rhs, shared, cols);
    let out_ref = TensorRef::u32_2d(out, rows, cols);
    ContractionComposer::custom_u32_2d(
        op_id, lhs_ref, rhs_ref, out_ref, rows, shared, cols, identity, combine, accumulate,
    )
    .with_region_generator(op_id)
    .build()
    .expect("Fix: u32_matmul_program failed to build contraction program")
}

/// Build `out[rows x cols] = lhs[rows x shared] * rhs[shared x cols]`.
///
/// Inputs and output use unsigned 16.16 fixed-point lanes packed as u32. The
/// caller owns all semantic naming and validation; this function only owns the
/// common kernel shape and buffer layout.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fixed_u32_matmul_program(
    op_id: &'static str,
    lhs: &str,
    rhs: &str,
    out: &str,
    _rows: u32,
    shared: u32,
    cols: u32,
    lhs_cells: u32,
    rhs_cells: u32,
    out_cells: u32,
) -> Program {
    u32_matmul_program(
        op_id,
        lhs,
        rhs,
        out,
        _rows,
        shared,
        cols,
        lhs_cells,
        rhs_cells,
        out_cells,
        0,
        crate::math::fixed::fixed_mul_16_16_expr,
        Expr::add,
    )
}
