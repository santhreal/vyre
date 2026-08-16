//! Givens rotation of one strided element pair across a matrix axis.
//!
//! A Givens rotation always rewrites exactly two lines of a matrix, and the
//! arithmetic is the same whether those lines are two columns, two rows, or two
//! columns of an accumulated rotation matrix: read the pair, write
//! `(c*first - s*second, s*first + c*second)`. The three uses in
//! [`crate::math::jacobi_apply_rotation`] differ only in the base offset and the
//! stride between successive elements of a line, so they are one builder with
//! two address parameters rather than three copies of a five-node loop.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::givens_rotate_pair";

/// Address of element `step` of the line starting at `base`.
fn line_element(base: &Expr, stride: u32, step: &str) -> Expr {
    Expr::add(base.clone(), Expr::mul(Expr::var(step), Expr::u32(stride)))
}

/// Emit the loop that rotates the element pair `(first_base, second_base)` of
/// `matrix` through `count` steps of `stride`.
///
/// `loop_var` names the induction variable and prefixes the two temporaries, so
/// several rotations can sit in one scope without shadowing each other. Both
/// elements are read before either is written, which is what makes the rotation
/// in place correct.
#[must_use]
pub fn givens_rotate_pair(
    matrix: &str,
    loop_var: &str,
    count: u32,
    first_base: &Expr,
    second_base: &Expr,
    stride: u32,
    c: &Expr,
    s: &Expr,
) -> Node {
    let first = format!("{loop_var}_first");
    let second = format!("{loop_var}_second");
    Node::loop_for(
        loop_var,
        Expr::u32(0),
        Expr::u32(count),
        vec![
            Node::let_bind(
                &first,
                Expr::load(matrix, line_element(first_base, stride, loop_var)),
            ),
            Node::let_bind(
                &second,
                Expr::load(matrix, line_element(second_base, stride, loop_var)),
            ),
            Node::store(
                matrix,
                line_element(first_base, stride, loop_var),
                Expr::sub(
                    Expr::mul(c.clone(), Expr::var(&first)),
                    Expr::mul(s.clone(), Expr::var(&second)),
                ),
            ),
            Node::store(
                matrix,
                line_element(second_base, stride, loop_var),
                Expr::add(
                    Expr::mul(s.clone(), Expr::var(&first)),
                    Expr::mul(c.clone(), Expr::var(&second)),
                ),
            ),
        ],
    )
}

/// Emit [`givens_rotate_pair`] as a child region of `parent_op_id`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn givens_rotate_pair_region(
    parent_op_id: &str,
    matrix: &str,
    loop_var: &str,
    count: u32,
    first_base: &Expr,
    second_base: &Expr,
    stride: u32,
    c: &Expr,
    s: &Expr,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        vec![givens_rotate_pair(
            matrix,
            loop_var,
            count,
            first_base,
            second_base,
            stride,
            c,
            s,
        )],
    )
}

/// Build a standalone Program that rotates columns `first_col` and `second_col`
/// of an `n x n` row-major f32 matrix.
///
/// `coefficients` supplies `[c, s]`. Keeping them in a buffer rather than baking
/// them into the Program is what makes the rotation reusable across sweeps: the
/// angle changes every step, the program does not.
#[must_use]
pub fn givens_rotate_columns(
    matrix: &str,
    coefficients: &str,
    n: u32,
    first_col: u32,
    second_col: u32,
) -> Program {
    if n == 0 || first_col >= n || second_col >= n {
        return trap_program(OP_ID, Some((matrix, DataType::F32)), format!(
            "Fix: givens_rotate_columns needs n > 0 and both columns below n, got n={n}, first_col={first_col}, second_col={second_col}."
        ));
    }
    let cells = match crate::plumbing::operand::shape::square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => return trap_program(OP_ID, Some((matrix, DataType::F32)), message),
    };
    Program::wrapped(
        vec![
            BufferDecl::storage(matrix, 0, BufferAccess::ReadWrite, DataType::F32)
                .with_count(cells),
            BufferDecl::storage(coefficients, 1, BufferAccess::ReadOnly, DataType::F32)
                .with_count(2),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                vec![givens_rotate_pair(
                    matrix,
                    "giv_k",
                    n,
                    &Expr::u32(first_col),
                    &Expr::u32(second_col),
                    n,
                    &Expr::load(coefficients, Expr::u32(0)),
                    &Expr::load(coefficients, Expr::u32(1)),
                )],
            )],
        )],
    )
}

// Canonical registration.
//
// WITNESS: the 2x2 identity rotated by (c, s) = (0.6, 0.8). Both coefficients are
// exact in f32 and c^2 + s^2 = 1, so the result is the rotation matrix itself and
// every product below is exact: no rounding hides a sign or an index error.
//
// ORACLE: a Givens rotation of the identity is the rotation matrix by definition.
// Column 0 becomes (c, -s) and column 1 becomes (s, c), read down the rows, which
// is [0.6, 0.8, -0.8, 0.6] row-major. The read-before-write discipline is what the
// fixture is really pinning: writing the first column before reading the second
// would produce 0.6 and 0.36 in row 0.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || givens_rotate_columns("m", "coeff", 2, 0, 1),
        Some(|| {
            let to_bytes = |vals: &[f32]| vyre_primitives::wire::pack_f32_slice(vals);
            vec![vec![
                to_bytes(&[1.0, 0.0, 0.0, 1.0]),
                to_bytes(&[0.6, 0.8]),
            ]]
        }),
        Some(|| {
            let to_bytes = |vals: &[f32]| vyre_primitives::wire::pack_f32_slice(vals);
            vec![vec![to_bytes(&[0.6, 0.8, -0.8, 0.6])]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_reads_both_elements_before_writing_either() {
        let node = givens_rotate_pair(
            "m",
            "k",
            2,
            &Expr::u32(0),
            &Expr::u32(1),
            2,
            &Expr::f32(0.6),
            &Expr::f32(0.8),
        );
        let Node::Loop { body, .. } = node else {
            panic!("Fix: givens_rotate_pair must emit a Node::Loop, got {node:?}.");
        };
        assert!(
            matches!(body[0], Node::Let { .. }) && matches!(body[1], Node::Let { .. }),
            "Fix: both pair elements must be bound before the first store, got {body:?}."
        );
        assert!(
            matches!(body[2], Node::Store { .. }) && matches!(body[3], Node::Store { .. }),
            "Fix: the two stores must follow both loads, got {body:?}."
        );
    }

    #[test]
    fn distinct_loop_vars_do_not_collide() {
        let first = givens_rotate_pair(
            "m",
            "col",
            2,
            &Expr::u32(0),
            &Expr::u32(1),
            2,
            &Expr::f32(1.0),
            &Expr::f32(0.0),
        );
        let second = givens_rotate_pair(
            "m",
            "row",
            2,
            &Expr::u32(0),
            &Expr::u32(2),
            1,
            &Expr::f32(1.0),
            &Expr::f32(0.0),
        );
        let program = Program::wrapped(
            vec![BufferDecl::storage("m", 0, BufferAccess::ReadWrite, DataType::F32).with_count(4)],
            [1, 1, 1],
            vec![wrap_anonymous_region(OP_ID, vec![first, second])],
        );
        let errors = vyre_foundation::validate::validate(&program);
        assert!(
            errors.is_empty(),
            "Fix: two rotations in one scope must not shadow each other, got {:?}.",
            errors
        );
    }

    #[test]
    fn out_of_range_column_is_rejected() {
        let program = givens_rotate_columns("m", "coeff", 2, 0, 2);
        assert!(
            program.entry().iter().any(|node| matches!(
                node,
                Node::Region { body, .. } if body.iter().any(|inner| matches!(inner, Node::Trap { .. }))
            )),
            "Fix: a column at or above n must produce a trapping Program."
        );
    }
}
