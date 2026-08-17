//! Fill an `n x n` row-major f32 matrix with the identity.
//!
//! Every rotation accumulator starts from the identity, so the seeding pass is
//! its own operation rather than three lines repeated inside each solver.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use crate::builder::cooperative::{for_each_index, LANES};
use crate::plumbing::operand::shape::square_matrix_cells;

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::matrix_identity_fill";

/// Emit the cell walk that writes `1.0` on the diagonal and `0.0` elsewhere.
///
/// The cells are independent, so `lanes` of them are written at a time and the
/// walk needs `local` bound to the lane id. `lanes` is the width of the program
/// the nodes end up in: every cell is offered to exactly one lane, so a program
/// that declares fewer lanes than it passes here leaves cells unwritten.
#[must_use]
pub fn matrix_identity_fill_body(matrix: &str, n: u32, lanes: u32) -> Vec<Node> {
    let cell = Expr::var("mif_cell");
    vec![for_each_index(
        n.saturating_mul(n),
        lanes,
        "mif_cell",
        vec![Node::store(
            matrix,
            cell.clone(),
            Expr::select(
                Expr::eq(
                    Expr::div(cell.clone(), Expr::u32(n)),
                    Expr::rem(cell, Expr::u32(n)),
                ),
                Expr::f32(1.0),
                Expr::f32(0.0),
            ),
        )],
    )]
}

/// Emit [`matrix_identity_fill_body`] as a child region of `parent_op_id`.
#[must_use]
pub fn matrix_identity_fill_region(parent_op_id: &str, matrix: &str, n: u32, lanes: u32) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        matrix_identity_fill_body(matrix, n, lanes),
    )
}

/// Build a standalone identity-fill Program over an `n x n` f32 matrix.
#[must_use]
pub fn matrix_identity_fill(matrix: &str, n: u32) -> Program {
    let cells = match square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => return trap_program(OP_ID, Some((matrix, DataType::F32)), message),
    };
    let mut body = vec![Node::let_bind("local", Expr::LocalId { axis: 0 })];
    body.extend(matrix_identity_fill_body(matrix, n, LANES));
    Program::wrapped(
        vec![BufferDecl::output(matrix, 0, DataType::F32).with_count(cells)],
        [LANES, 1, 1],
        vec![wrap_anonymous_region(OP_ID, body)],
    )
}

// Canonical registration.
//
// WITNESS: n = 3, so the fixture distinguishes row-major from column-major only
// if the fill is wrong on an off-diagonal cell, and it distinguishes the
// diagonal predicate from a fill of the first row or first column.
//
// ORACLE: the 3x3 identity. Every value is exact in f32.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || matrix_identity_fill("m", 3),
        Some(|| {
            vec![vec![]]
        }),
        Some(|| {
            vec![vec![vec![
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x80, 0x3f, // 1.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x00, 0x00, // 0.0
                0x00, 0x00, 0x80, 0x3f, // 1.0
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
        let program = matrix_identity_fill("m", 0);
        assert!(
            program.entry().iter().any(|node| matches!(
                node,
                Node::Region { body, .. } if body.iter().any(|inner| matches!(inner, Node::Trap { .. }))
            )),
            "Fix: n = 0 must produce a trapping Program."
        );
    }

    #[test]
    fn fill_validates_as_a_program() {
        let program = matrix_identity_fill("m", 4);
        let errors = vyre_foundation::validate::validate(&program);
        assert!(
            errors.is_empty(),
            "Fix: the identity fill must validate, got {:?}.",
            errors
        );
    }

    #[test]
    fn identity_fill_dispatches_declared_lanes() {
        let program = matrix_identity_fill("m", 4);
        assert_eq!(
            program.workgroup_size(),
            [LANES, 1, 1],
            "Fix: matrix_identity_fill must dispatch LANES cooperative lanes."
        );
    }

    #[test]
    fn identity_fill_binds_local_id() {
        let program = matrix_identity_fill("m", 4);
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
            "Fix: matrix_identity_fill must bind `local` to LocalId {{ axis: 0 }}."
        );
    }
}
