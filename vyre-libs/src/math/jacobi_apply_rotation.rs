//! Apply one Jacobi rotation at pivot `(p, q)` to a symmetric matrix and its
//! accumulated rotation matrix.
//!
//! This is the whole numerical content of a Jacobi step: derive the rotation
//! that annihilates `A[p, q]`, rotate columns `p` and `q` of `A`, rotate rows
//! `p` and `q` of `A`, write the pivot entries as exact zeros, and accumulate
//! the same rotation into `V`. The three rotations are the same arithmetic over
//! different strides, so they are three calls to
//! [`crate::math::givens_rotate_pair`] rather than three copies.
//!
//! The pivot arrives as an expression so a caller running a sweep passes its
//! loop-carried pivot variables while the standalone Program passes constants.
//! The rotation is the operation; choosing the pivot is not part of it.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};

use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::math::givens_rotate_pair::givens_rotate_pair_region;

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::jacobi_apply_rotation";

/// `row * n + col` flat index for an `n`-column row-major matrix.
fn cell(row: &Expr, n: u32, col: &Expr) -> Expr {
    crate::builder::stencil::flat_index(row.clone(), n, col.clone())
}

/// Emit the rotation that annihilates `a[p, q]`, applied to `a` and accumulated
/// into `eigenvectors`.
///
/// The caller guarantees `p != q` and both below `n`; the rotation is undefined
/// for a diagonal pivot because `tau` divides by `a[p, q]`.
#[must_use]
pub fn jacobi_apply_rotation_body(
    a: &str,
    eigenvectors: &str,
    n: u32,
    p: &Expr,
    q: &Expr,
) -> Vec<Node> {
    let c = Expr::var("jar_c");
    let s = Expr::var("jar_s");
    vec![
        Node::let_bind("jar_app", Expr::load(a, cell(p, n, p))),
        Node::let_bind("jar_aqq", Expr::load(a, cell(q, n, q))),
        Node::let_bind("jar_apq", Expr::load(a, cell(p, n, q))),
        // tau = (aqq - app) / (2 * apq)
        Node::let_bind(
            "jar_tau",
            Expr::div(
                Expr::sub(Expr::var("jar_aqq"), Expr::var("jar_app")),
                Expr::mul(Expr::f32(2.0), Expr::var("jar_apq")),
            ),
        ),
        // t = sign(tau) / (|tau| + sqrt(1 + tau^2)). The sign must match Rust's
        // `f64::signum`, which returns +1 at +0.0: that is what makes the
        // app == aqq degenerate case (tau = +0) rotate by 45 degrees (t = 1)
        // instead of stalling. `UnOp::Sign` returns 0 at 0, so the sign is
        // spelled as an explicit `tau >= 0 ? 1 : -1`.
        Node::let_bind(
            "jar_t",
            Expr::div(
                Expr::select(
                    Expr::ge(Expr::var("jar_tau"), Expr::f32(0.0)),
                    Expr::f32(1.0),
                    Expr::f32(-1.0),
                ),
                Expr::add(
                    Expr::abs(Expr::var("jar_tau")),
                    Expr::sqrt(Expr::add(
                        Expr::f32(1.0),
                        Expr::mul(Expr::var("jar_tau"), Expr::var("jar_tau")),
                    )),
                ),
            ),
        ),
        // c = 1 / sqrt(1 + t^2); s = t * c
        Node::let_bind(
            "jar_c",
            Expr::inverse_sqrt(Expr::add(
                Expr::f32(1.0),
                Expr::mul(Expr::var("jar_t"), Expr::var("jar_t")),
            )),
        ),
        Node::let_bind("jar_s", Expr::mul(Expr::var("jar_t"), Expr::var("jar_c"))),
        // Columns p and q of A: element k of a column is k * n away from its head.
        givens_rotate_pair_region(OP_ID, a, "jar_col", n, p, q, n, &c, &s),
        // Rows p and q of A: element k of a row is one cell away from its head,
        // and the heads are p * n and q * n.
        givens_rotate_pair_region(
            OP_ID,
            a,
            "jar_row",
            n,
            &Expr::mul(p.clone(), Expr::u32(n)),
            &Expr::mul(q.clone(), Expr::u32(n)),
            1,
            &c,
            &s,
        ),
        // The rotation is chosen to annihilate the pivot, so write the exact
        // zeros rather than the residue the four-term products leave behind.
        Node::store(a, cell(p, n, q), Expr::f32(0.0)),
        Node::store(a, cell(q, n, p), Expr::f32(0.0)),
        // The same rotation on columns p and q of the accumulator.
        givens_rotate_pair_region(OP_ID, eigenvectors, "jar_vec", n, p, q, n, &c, &s),
    ]
}

/// Emit [`jacobi_apply_rotation_body`] as a child region of `parent_op_id`.
#[must_use]
pub fn jacobi_apply_rotation_region(
    parent_op_id: &str,
    a: &str,
    eigenvectors: &str,
    n: u32,
    p: &Expr,
    q: &Expr,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        jacobi_apply_rotation_body(a, eigenvectors, n, p, q),
    )
}

/// Build a standalone Program applying one Jacobi rotation at a fixed pivot.
#[must_use]
pub fn jacobi_apply_rotation(a: &str, eigenvectors: &str, n: u32, p: u32, q: u32) -> Program {
    if n == 0 || p >= n || q >= n || p == q {
        return trap_program(OP_ID, Some((a, DataType::F32)), format!(
            "Fix: jacobi_apply_rotation needs n > 0 and a distinct off-diagonal pivot below n, got n={n}, p={p}, q={q}."
        ));
    }
    let cells = match crate::plumbing::operand::shape::square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => return trap_program(OP_ID, Some((a, DataType::F32)), message),
    };
    Program::wrapped(
        vec![
            BufferDecl::storage(a, 0, BufferAccess::ReadWrite, DataType::F32).with_count(cells),
            BufferDecl::storage(eigenvectors, 1, BufferAccess::ReadWrite, DataType::F32)
                .with_count(cells),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::LogicalIndex { axis: 0 }, Expr::u32(0)),
                jacobi_apply_rotation_body(a, eigenvectors, n, &Expr::u32(p), &Expr::u32(q)),
            )],
        )],
    )
}

// Canonical registration.
//
// WITNESS: A = [[0, 1], [1, 0]] with V = I and pivot (0, 1). This is the
// degenerate case the rotation formula handles explicitly: app == aqq makes
// tau = +0, and only because the sign is spelled as `tau >= 0 ? 1 : -1` does
// t come out 1 (a 45 degree rotation) rather than 0 (no rotation at all). A
// witness with app != aqq would leave that branch unproven. The spectrum of
// this matrix is {-1, +1}, both simple.
//
// ORACLE: tau = 0, t = 1, c = 1 / sqrt(2), s = c. The rotation diagonalizes A
// to diag(-1, +1) and turns the identity into the rotation matrix itself. The
// expected bytes are the f32 evaluation of that closed form, computed
// independently of this Program: c rounds to 0.70710677, c * c rounds to
// 0.49999997 rather than 0.5, and the two four-term diagonal sums therefore
// land on 0.99999994 rather than 1.0. Both are exactly representable, so the
// fixture is exact and no tolerance is needed. The two off-diagonal cells are
// the forced zeros.
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || jacobi_apply_rotation("a", "evec", 2, 0, 1),
        Some(|| {
            let to_bytes = |vals: &[f32]| vyre_primitives::wire::pack_f32_slice(vals);
            vec![vec![
                to_bytes(&[0.0, 1.0, 1.0, 0.0]),
                to_bytes(&[1.0, 0.0, 0.0, 1.0]),
            ]]
        }),
        Some(|| {
            vec![vec![
                vec![
                    0xff, 0xff, 0x7f, 0xbf, // -0.99999994
                    0x00, 0x00, 0x00, 0x00, // 0.0
                    0x00, 0x00, 0x00, 0x00, // 0.0
                    0xff, 0xff, 0x7f, 0x3f, // 0.99999994
                ],
                vec![
                    0xf3, 0x04, 0x35, 0x3f, // 0.70710677
                    0xf3, 0x04, 0x35, 0x3f, // 0.70710677
                    0xf3, 0x04, 0x35, 0xbf, // -0.70710677
                    0xf3, 0x04, 0x35, 0x3f, // 0.70710677
                ],
            ]]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_pivot_is_rejected() {
        let program = jacobi_apply_rotation("a", "evec", 2, 1, 1);
        assert!(
            program.entry().iter().any(|node| matches!(
                node,
                Node::Region { body, .. } if body.iter().any(|inner| matches!(inner, Node::Trap { .. }))
            )),
            "Fix: a diagonal pivot divides by a[p, q] and must produce a trapping Program."
        );
    }

    #[test]
    fn rotation_validates_as_a_program() {
        let program = jacobi_apply_rotation("a", "evec", 4, 0, 2);
        let errors = vyre_foundation::validate::validate(&program);
        assert!(
            errors.is_empty(),
            "Fix: one Jacobi rotation must validate, got {:?}.",
            errors
        );
    }

    #[test]
    fn the_three_rotations_are_one_owner() {
        let body = jacobi_apply_rotation_body("a", "evec", 4, &Expr::u32(0), &Expr::u32(1));
        let regions = body
            .iter()
            .filter(|node| {
                matches!(node, Node::Region { generator, .. }
                    if generator.as_str() == crate::math::givens_rotate_pair::OP_ID)
            })
            .count();
        assert_eq!(
            regions, 3,
            "Fix: the column, row and accumulator rotations must all route through givens_rotate_pair."
        );
    }
}
