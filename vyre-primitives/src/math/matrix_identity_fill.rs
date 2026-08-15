//! Fill an `n x n` row-major f32 matrix with the identity.
//!
//! Every rotation accumulator starts from the identity, so the seeding pass is
//! its own operation rather than three lines repeated inside each solver.

use vyre_foundation::composition::{
    trap_program, wrap_anonymous_region, wrap_child_region,
};
use vyre_foundation::ir::model::expr::GeneratorRef;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

use crate::math::square_matrix_cells;

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::matrix_identity_fill";

/// Emit the nested loop that writes `1.0` on the diagonal and `0.0` elsewhere.
#[must_use]
pub fn matrix_identity_fill_body(matrix: &str, n: u32) -> Vec<Node> {
    vec![Node::loop_for(
        "mif_row",
        Expr::u32(0),
        Expr::u32(n),
        vec![Node::loop_for(
            "mif_col",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::store(
                matrix,
                Expr::add(
                    Expr::mul(Expr::var("mif_row"), Expr::u32(n)),
                    Expr::var("mif_col"),
                ),
                Expr::select(
                    Expr::eq(Expr::var("mif_row"), Expr::var("mif_col")),
                    Expr::f32(1.0),
                    Expr::f32(0.0),
                ),
            )],
        )],
    )]
}

/// Emit [`matrix_identity_fill_body`] as a child region of `parent_op_id`.
#[must_use]
pub fn matrix_identity_fill_region(parent_op_id: &str, matrix: &str, n: u32) -> Node {
    wrap_child_region(
        OP_ID,
        GeneratorRef {
            name: parent_op_id.to_string(),
        },
        matrix_identity_fill_body(matrix, n),
    )
}

/// Build a standalone identity-fill Program over an `n x n` f32 matrix.
#[must_use]
pub fn matrix_identity_fill(matrix: &str, n: u32) -> Program {
    let cells = match square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => return trap_program(OP_ID, Some((matrix, DataType::F32)), message),
    };
    Program::wrapped(
        vec![BufferDecl::output(matrix, 0, DataType::F32).with_count(cells)],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                matrix_identity_fill_body(matrix, n),
            )],
        )],
    )
}

// Canonical registration.
//
// WITNESS: n = 3, so the fixture distinguishes row-major from column-major only
// if the fill is wrong on an off-diagonal cell, and it distinguishes the
// diagonal predicate from a fill of the first row or first column.
//
// ORACLE: the 3x3 identity. Every value is exact in f32.
#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || matrix_identity_fill("m", 3),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![to_bytes(&[0.0; 9])]]
        }),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![to_bytes(&[
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                0.0, 0.0, 1.0,
            ])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
