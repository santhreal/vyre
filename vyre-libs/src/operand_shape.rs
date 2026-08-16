//! Cell counts of a matrix operand, refused the same way by every op.
//!
//! These live outside the domain modules because two domains need them and
//! neither enables the other: `math-kernels` and `graph` are independent
//! features. While these sat in `math`, the seven graph call sites and the
//! fixed-point matmul builder only compiled because some other workspace member
//! happened to enable `math` too. A consumer enabling `graph` alone against the
//! published crate failed with E0433.

/// Cell count of a `rows x cols` operand, or the diagnostic the caller returns.
///
/// `context` identifies the shape in the message. Pass the op id when the op has
/// one shape, and qualify it with the operand name when it has several.
pub(crate) fn matrix_cells(context: &str, rows: u32, cols: u32) -> Result<u32, String> {
    rows.checked_mul(cols).ok_or_else(|| {
        format!(
            "Fix: {context} shape {rows}x{cols} overflows the u32 cell count. Shard or sparsify the operand before dispatch."
        )
    })
}

/// Cell count of the `n x n` operand `context` works over.
///
/// `n == 0` describes no matrix and is rejected before the multiplication.
pub(crate) fn square_matrix_cells(context: &str, n: u32) -> Result<u32, String> {
    if n == 0 {
        return Err(format!("Fix: {context} requires n > 0, got 0."));
    }
    matrix_cells(context, n, n)
}
