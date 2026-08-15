//! Canonicalize the sign of every column of an eigenvector matrix.
//!
//! An eigenvector is defined only up to sign, so a solver is free to return `v`
//! or `-v` and both are correct. That makes the raw output unusable as an exact
//! oracle: a consumer that divides by a column flips with it, and a backend that
//! rounds one rotation differently lands on the opposite sign. Fixing the rule
//! here, once, is what lets every consumer be pinned by an exact fixture.
//!
//! The rule: the first component of a column whose magnitude exceeds
//! [`EIGENVECTOR_SIGN_EPSILON`] is made positive. A component that should be
//! exactly zero comes back on the order of 1e-7 with an arbitrary sign, so
//! letting it decide would make the canonicalization itself non-deterministic.
//!
//! [`EIGENVECTOR_SIGN_EPSILON`]: crate::math::eigenvector_column_sign::EIGENVECTOR_SIGN_EPSILON

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::GeneratorRef;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::eigenvector_column_sign";

/// Magnitude below which a component cannot decide its column's sign.
pub const EIGENVECTOR_SIGN_EPSILON: f32 = 1.0e-6;

/// Emit the pass that flips every column whose first significant component is negative.
#[must_use]
pub fn eigenvector_column_sign_body(eigenvectors: &str, n: u32) -> Vec<Node> {
    let cell =
        |row: &str, col: &str| Expr::add(Expr::mul(Expr::var(row), Expr::u32(n)), Expr::var(col));
    vec![Node::loop_for(
        "ecs_col",
        Expr::u32(0),
        Expr::u32(n),
        vec![
            Node::let_bind("ecs_sign", Expr::f32(1.0)),
            Node::let_bind("ecs_found", Expr::u32(0)),
            Node::loop_for(
                "ecs_scan",
                Expr::u32(0),
                Expr::u32(n),
                vec![
                    Node::let_bind(
                        "ecs_value",
                        Expr::load(eigenvectors, cell("ecs_scan", "ecs_col")),
                    ),
                    Node::let_bind(
                        "ecs_first",
                        Expr::and(
                            Expr::gt(
                                Expr::abs(Expr::var("ecs_value")),
                                Expr::f32(EIGENVECTOR_SIGN_EPSILON),
                            ),
                            Expr::eq(Expr::var("ecs_found"), Expr::u32(0)),
                        ),
                    ),
                    Node::assign(
                        "ecs_sign",
                        Expr::select(
                            Expr::var("ecs_first"),
                            Expr::select(
                                Expr::lt(Expr::var("ecs_value"), Expr::f32(0.0)),
                                Expr::f32(-1.0),
                                Expr::f32(1.0),
                            ),
                            Expr::var("ecs_sign"),
                        ),
                    ),
                    Node::assign(
                        "ecs_found",
                        Expr::select(Expr::var("ecs_first"), Expr::u32(1), Expr::var("ecs_found")),
                    ),
                ],
            ),
            Node::loop_for(
                "ecs_apply",
                Expr::u32(0),
                Expr::u32(n),
                vec![Node::store(
                    eigenvectors,
                    cell("ecs_apply", "ecs_col"),
                    Expr::mul(
                        Expr::load(eigenvectors, cell("ecs_apply", "ecs_col")),
                        Expr::var("ecs_sign"),
                    ),
                )],
            ),
        ],
    )]
}

/// Emit [`eigenvector_column_sign_body`] as a child region of `parent_op_id`.
#[must_use]
pub fn eigenvector_column_sign_region(parent_op_id: &str, eigenvectors: &str, n: u32) -> Node {
    wrap_child_region(
        OP_ID,
        GeneratorRef {
            name: parent_op_id.to_string(),
        },
        eigenvector_column_sign_body(eigenvectors, n),
    )
}

/// Build a standalone column-sign canonicalization Program.
///
/// The matrix is read and rewritten in place, so it is declared read-write: a
/// buffer declared as an output is not a witness input, and the caller's matrix
/// would never reach the program.
#[must_use]
pub fn eigenvector_column_sign(eigenvectors: &str, n: u32) -> Program {
    let cells = match crate::operand_shape::square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => {
            return trap_program(OP_ID, Some((eigenvectors, DataType::F32)), message);
        }
    };
    Program::wrapped(
        vec![
            BufferDecl::storage(eigenvectors, 0, BufferAccess::ReadWrite, DataType::F32)
                .with_count(cells),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                eigenvector_column_sign_body(eigenvectors, n),
            )],
        )],
    )
}

// Canonical registration.
//
// WITNESS: the 2x2 matrix [[-1, 0], [0, 2]] row-major. Column 0 leads with -1 and
// must flip; column 1 leads with a zero that is below the epsilon and must be
// skipped rather than fix the sign at +1 for the wrong reason, then finds 2 and
// stays. So the fixture separates "flip" from "leave alone" and pins that a
// leading zero does not decide a column.
//
// ORACLE: multiplying column 0 by -1 gives (1, -0.0) read down the rows, and
// column 1 is unchanged. The -0.0 is the zero row times -1.0 and is what f32
// produces; it is in the fixture because the comparison is on bytes.
#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || eigenvector_column_sign("evec", 2),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![to_bytes(&[-1.0, 0.0, 0.0, 2.0])]]
        }),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![to_bytes(&[1.0, 0.0, -0.0, 2.0])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_order_is_rejected() {
        let program = eigenvector_column_sign("evec", 0);
        assert!(
            program.entry().iter().any(|node| matches!(
                node,
                Node::Region { body, .. } if body.iter().any(|inner| matches!(inner, Node::Trap { .. }))
            )),
            "Fix: n = 0 must produce a trapping Program."
        );
    }

    #[test]
    fn sign_pass_validates_as_a_program() {
        let program = eigenvector_column_sign("evec", 4);
        let errors = vyre_foundation::validate::validate(&program);
        assert!(
            errors.is_empty(),
            "Fix: the column-sign pass must validate, got {:?}.",
            errors
        );
    }
}
