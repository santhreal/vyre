//! Read the diagonal of an `n x n` row-major f32 matrix into an `n`-element vector.
//!
//! A diagonalizing solver ends by publishing `diag(A)`, and so does any routine
//! that reads variances off a covariance matrix or singular values off a
//! rotated Gram matrix. The read-out is one operation, not a loop each of them
//! spells again.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::matrix_diagonal_extract";

/// Emit the loop that copies `matrix[i * n + i]` into `diagonal[i]`.
#[must_use]
pub fn matrix_diagonal_extract_body(matrix: &str, diagonal: &str, n: u32) -> Vec<Node> {
    vec![Node::loop_for(
        "mde_i",
        Expr::u32(0),
        Expr::u32(n),
        vec![Node::store(
            diagonal,
            Expr::var("mde_i"),
            Expr::load(
                matrix,
                Expr::add(
                    Expr::mul(Expr::var("mde_i"), Expr::u32(n)),
                    Expr::var("mde_i"),
                ),
            ),
        )],
    )]
}

/// Emit [`matrix_diagonal_extract_body`] as a child region of `parent_op_id`.
#[must_use]
pub fn matrix_diagonal_extract_region(
    parent_op_id: &str,
    matrix: &str,
    diagonal: &str,
    n: u32,
) -> Node {
    Node::Region {
        generator: Ident::from(OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(matrix_diagonal_extract_body(matrix, diagonal, n)),
    }
}

/// Build a standalone diagonal read-out Program.
#[must_use]
pub fn matrix_diagonal_extract(matrix: &str, diagonal: &str, n: u32) -> Program {
    if n == 0 {
        return crate::invalid_output_program(
            OP_ID,
            diagonal,
            DataType::F32,
            format!("Fix: matrix_diagonal_extract requires n > 0, got {n}."),
        );
    }
    let Some(cells) = n.checked_mul(n) else {
        return crate::invalid_output_program(
            OP_ID,
            diagonal,
            DataType::F32,
            format!("Fix: matrix_diagonal_extract n*n overflows matrix cell count for n={n}."),
        );
    };
    Program::wrapped(
        vec![
            BufferDecl::storage(matrix, 0, BufferAccess::ReadOnly, DataType::F32)
                .with_count(cells),
            BufferDecl::output(diagonal, 1, DataType::F32).with_count(n),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                matrix_diagonal_extract_body(matrix, diagonal, n),
            )]),
        }],
    )
}

// Canonical registration.
//
// WITNESS: the 3x3 matrix of 1..9 in row-major order. Every cell is distinct, so
// a transposed index, an off-by-one on the stride, or a read of the first row
// each produce a different vector.
//
// ORACLE: the diagonal of that matrix is [1, 5, 9] by inspection. Small integers
// are exact in f32 and the operation only copies, so no tolerance applies.
#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || matrix_diagonal_extract("m", "diag", 3),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![
                to_bytes(&[
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0, //
                    7.0, 8.0, 9.0,
                ]),
                to_bytes(&[0.0; 3]),
            ]]
        }),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![to_bytes(&[1.0, 5.0, 9.0])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_order_is_rejected() {
        let program = matrix_diagonal_extract("m", "diag", 0);
        assert!(
            program.entry().iter().any(|node| matches!(
                node,
                Node::Region { body, .. } if body.iter().any(|inner| matches!(inner, Node::Trap { .. }))
            )),
            "Fix: n = 0 must produce a trapping Program."
        );
    }

    #[test]
    fn extract_validates_as_a_program() {
        let program = matrix_diagonal_extract("m", "diag", 4);
        let errors = vyre_foundation::validate::validate(&program);
        assert!(
            errors.is_empty(),
            "Fix: the diagonal read-out must validate, got {:?}.",
            errors
        );
    }
}
