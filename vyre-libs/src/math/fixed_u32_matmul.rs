//! Shared fixed-point u32 matrix-contraction IR builder.
//!
//! Several primitive domains expose matrix multiplication under different
//! semantics: tensor-network contraction and categorical monoidal composition
//! are intentionally separate public ops, but their GPU kernel body is the same
//! 16.16 fixed-point row/column contraction. This module keeps that kernel in
//! one place while callers retain their own validation language and op ids.

use vyre_foundation::composition::wrap_anonymous_region;

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

pub(crate) fn fixed_u32_matvec_program(
    op_id: &'static str,
    matrix: &str,
    vector: &str,
    out: &str,
    n: u32,
    matrix_cells: u32,
) -> Program {
    let row = Expr::InvocationId { axis: 0 };
    let body = vec![Node::if_then(
        Expr::lt(row.clone(), Expr::u32(n)),
        vec![
            Node::let_bind("acc", Expr::u32(0)),
            Node::let_bind("row_base", Expr::mul(row.clone(), Expr::u32(n))),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(n),
                vec![Node::assign(
                    "acc",
                    Expr::add(
                        Expr::var("acc"),
                        crate::math::fixed::fixed_mul_16_16_expr(
                            Expr::load(matrix, Expr::add(Expr::var("row_base"), Expr::var("j"))),
                            Expr::load(vector, Expr::var("j")),
                        ),
                    ),
                )],
            ),
            Node::store(out, row, Expr::var("acc")),
        ],
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage(matrix, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(matrix_cells),
            BufferDecl::storage(vector, 1, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(out, 2, BufferAccess::ReadWrite, DataType::U32).with_count(n),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
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
    _rows: u32,
    shared: u32,
    cols: u32,
    lhs_cells: u32,
    rhs_cells: u32,
    out_cells: u32,
    identity: u32,
    combine: C,
    accumulate: A,
) -> Program
where
    C: Fn(Expr, Expr) -> Expr,
    A: Fn(Expr, Expr) -> Expr,
{
    let t = Expr::InvocationId { axis: 0 };
    let i_expr = Expr::div(t.clone(), Expr::u32(cols));
    let j_expr = Expr::rem(t.clone(), Expr::u32(cols));
    let lhs_value = Expr::load(
        lhs,
        Expr::add(
            Expr::mul(Expr::var("i"), Expr::u32(shared)),
            Expr::var("kk"),
        ),
    );
    let rhs_value = Expr::load(
        rhs,
        Expr::add(Expr::mul(Expr::var("kk"), Expr::u32(cols)), Expr::var("j")),
    );
    let combined = combine(lhs_value, rhs_value);
    let folded = accumulate(Expr::var("acc"), combined);

    let body = vec![Node::if_then(
        Expr::lt(t.clone(), Expr::u32(out_cells)),
        vec![
            Node::let_bind("acc", Expr::u32(identity)),
            Node::let_bind("i", i_expr),
            Node::let_bind("j", j_expr),
            Node::loop_for(
                "kk",
                Expr::u32(0),
                Expr::u32(shared),
                vec![Node::assign("acc", folded)],
            ),
            Node::store(out, t, Expr::var("acc")),
        ],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(lhs, 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(lhs_cells),
            BufferDecl::storage(rhs, 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(rhs_cells),
            BufferDecl::storage(out, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(out_cells),
        ],
        [256, 1, 1],
        vec![wrap_anonymous_region(op_id, body)],
    )
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
