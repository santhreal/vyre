//! Read the diagonal of an `n x n` row-major f32 matrix into an `n`-element vector.
//!
//! A diagonalizing solver ends by publishing `diag(A)`, and so does any routine
//! that reads variances off a covariance matrix or singular values off a
//! rotated Gram matrix. The read-out is one operation, not a loop each of them
//! spells again.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::builder::cooperative::{for_each_index, LANES};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::matrix_diagonal_extract";

/// Emit the walk that copies `matrix[i * n + i]` into `diagonal[i]`.
///
/// The entries are independent, so `lanes` of them are copied at a time and the
/// walk needs `local` bound to the lane id. `lanes` is the width of the program
/// the nodes end up in: a program that declares fewer leaves entries unwritten.
#[must_use]
pub fn matrix_diagonal_extract_body(matrix: &str, diagonal: &str, n: u32, lanes: u32) -> Vec<Node> {
    let entry = Expr::var("mde_i");
    vec![for_each_index(
        n,
        lanes,
        "mde_i",
        vec![Node::store(
            diagonal,
            entry.clone(),
            Expr::load(
                matrix,
                Expr::add(Expr::mul(entry.clone(), Expr::u32(n)), entry),
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
    lanes: u32,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        matrix_diagonal_extract_body(matrix, diagonal, n, lanes),
    )
}

/// Build a standalone diagonal read-out Program.
#[must_use]
pub fn matrix_diagonal_extract(matrix: &str, diagonal: &str, n: u32) -> Program {
    let cells = match crate::plumbing::operand::shape::square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => return trap_program(OP_ID, Some((diagonal, DataType::F32)), message),
    };
    let mut body = vec![Node::let_bind("local", Expr::LocalId { axis: 0 })];
    body.extend(matrix_diagonal_extract_body(matrix, diagonal, n, LANES));
    Program::wrapped(
        vec![
            BufferDecl::storage(matrix, 0, BufferAccess::ReadOnly, DataType::F32).with_count(cells),
            BufferDecl::output(diagonal, 1, DataType::F32).with_count(n),
        ],
        [LANES, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
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
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || matrix_diagonal_extract("m", "diag", 3),
        Some(|| {
            let to_bytes = |vals: &[f32]| vyre_primitives::wire::pack_f32_slice(vals);
            vec![vec![
                to_bytes(&[
                    1.0, 2.0, 3.0, //
                    4.0, 5.0, 6.0, //
                    7.0, 8.0, 9.0,
                ]),
            ]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0xa0, 0x40, // 5.0
                0x00, 0x00, 0x10, 0x41, // 9.0
            ]]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::visit::any_descendant;

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

    #[test]
    fn diagonal_extract_dispatches_declared_lanes() {
        let program = matrix_diagonal_extract("m", "diag", 4);
        assert_eq!(
            program.workgroup_size(),
            [LANES, 1, 1],
            "Fix: matrix_diagonal_extract must dispatch LANES cooperative lanes."
        );
    }

    #[test]
    fn diagonal_extract_binds_local_id() {
        let program = matrix_diagonal_extract("m", "diag", 4);
        let has_local_binding = program.entry().iter().any(|node| {
            any_descendant(node, &mut |inner| match inner {
                Node::Let { name, value } => {
                    name == "local" && matches!(value, Expr::LocalId { axis: 0 })
                }
                _ => false,
            })
        });
        assert!(
            has_local_binding,
            "Fix: matrix_diagonal_extract must bind `local` to LocalId {{ axis: 0 }}."
        );
    }
}
