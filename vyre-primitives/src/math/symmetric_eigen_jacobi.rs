//! Symmetric eigendecomposition via the cyclic (max-pivot) Jacobi method (#P-PRIM-Jacobi).
//!
//! Diagonalizes a real symmetric `n x n` matrix `A` in f32: produces its eigenvalues (the diagonal
//! of the rotated matrix) and eigenvectors (the accumulated rotation matrix `V`, whose columns are
//! the eigenvectors). This is the numerical core of the tensor-train SVD (`tensor_train_decompose`)
//! (a truncated SVD of `A` is obtained from the eigendecomposition of the Gram matrix `AᵀA`).
//!
//! The algorithm is inherently sequential (each sweep picks the largest off-diagonal entry and
//! applies one Givens rotation that depends on the current matrix), so the kernel runs on a single
//! lane (`InvocationId == 0`), the canonical GPU serial-region idiom (cf. `sheaf_laplacian_eigenvalue`,
//! matroid). It mirrors the CPU reference [`crate::math::tensor_train_decompose`]'s
//! `symmetric_eigen_jacobi_into` step for step, so the two agree up to f32-vs-f64 rounding; the
//! kernel is verified by the basis/order-invariant eigenpair contract (`A·vᵢ ≈ λᵢ·vᵢ` and `VᵀV ≈ I`)
//! rather than element-wise, because near-degenerate eigenvalues admit different-but-valid
//! eigenvector bases.

use std::sync::Arc;
use vyre_foundation::algebra::composition::trap_program;
use vyre_foundation::ir::model::expr::{GeneratorRef, Ident};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::math::eigenvector_column_sign::eigenvector_column_sign_region;
use crate::math::jacobi_apply_rotation::jacobi_apply_rotation_region;
use crate::math::matrix_diagonal_extract::matrix_diagonal_extract_region;
use crate::math::matrix_identity_fill::matrix_identity_fill_region;

/// Op id.
pub const OP_ID: &str = "vyre-primitives::math::symmetric_eigen_jacobi";

/// Off-diagonal convergence threshold (f32). Sweeps stop rotating once the largest off-diagonal
/// magnitude falls below this; matches the role of the f64 reference's `1e-12` but scaled to f32's
/// usable precision.
const JACOBI_EPS: f32 = 1.0e-6;

/// `row * n + col` flat index for an `n`-column row-major matrix.
fn idx(row: Expr, n: u32, col: Expr) -> Expr {
    Expr::add(Expr::mul(row, Expr::u32(n)), col)
}

/// Number of Jacobi sweeps, matching the CPU reference `(16 * n² ).max(32)`.
#[must_use]
pub fn jacobi_sweeps(n: u32) -> u32 {
    (16u32.saturating_mul(n).saturating_mul(n)).max(32)
}

/// Build the serial Jacobi eigensolve body (already lane-guarded by the caller). `a` is the f32
/// symmetric matrix buffer (mutated in place to near-diagonal form; its diagonal becomes the
/// eigenvalues), `eigenvectors` receives the accumulated rotation matrix `V` (columns = eigenvectors),
/// `eigenvalues` receives `diag(A)` after convergence. All three are `n x n` / `n` f32 buffers.
///
/// Eigenvector columns come back sign-canonicalized by
/// [`crate::math::eigenvector_column_sign`]: the first component larger than
/// `EIGENVECTOR_SIGN_EPSILON` in magnitude is positive. An eigenvector is only defined up
/// to sign, so without that the same input can produce `v` or `-v` depending on rounding,
/// and no consumer of this body can be pinned by an exact fixture.
///
/// Emitted by exactly two callers: [`symmetric_eigen_jacobi`] (standalone Program) and
/// [`crate::math::tensor_train_decompose::tensor_train_decompose_step`] (via
/// [`jacobi_eigen_region`]), so the rotation policy lives in ONE place.
#[must_use]
pub fn jacobi_eigen_body(a: &str, eigenvectors: &str, eigenvalues: &str, n: u32) -> Vec<Node> {
    let sweeps = jacobi_sweeps(n);
    vec![
        matrix_identity_fill_region(OP_ID, eigenvectors, n),
        // Sweep loop: each iteration zeroes the largest off-diagonal entry via one
        // Givens rotation. The pivot search stays here because `jac_p`, `jac_q` and
        // `jac_maxod` are read by the rotation guard and a `Node::Region` scopes its
        // body, so searching inside a child region would not publish them.
        Node::loop_for(
            "jac_sweep",
            Expr::u32(0),
            Expr::u32(sweeps),
            vec![
                // Find (p, q) = argmax_{i<j} |A[i,j]| and maxod = that magnitude.
                Node::let_bind("jac_maxod", Expr::f32(0.0)),
                Node::let_bind("jac_p", Expr::u32(0)),
                Node::let_bind("jac_q", Expr::u32(0)),
                Node::loop_for(
                    "jac_si",
                    Expr::u32(0),
                    Expr::u32(n),
                    vec![Node::loop_for(
                        "jac_sj",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![Node::if_then(
                            Expr::lt(Expr::var("jac_si"), Expr::var("jac_sj")),
                            vec![
                                Node::let_bind(
                                    "jac_av",
                                    Expr::abs(Expr::load(
                                        a,
                                        idx(Expr::var("jac_si"), n, Expr::var("jac_sj")),
                                    )),
                                ),
                                Node::let_bind(
                                    "jac_isgt",
                                    Expr::gt(Expr::var("jac_av"), Expr::var("jac_maxod")),
                                ),
                                Node::assign(
                                    "jac_p",
                                    Expr::select(
                                        Expr::var("jac_isgt"),
                                        Expr::var("jac_si"),
                                        Expr::var("jac_p"),
                                    ),
                                ),
                                Node::assign(
                                    "jac_q",
                                    Expr::select(
                                        Expr::var("jac_isgt"),
                                        Expr::var("jac_sj"),
                                        Expr::var("jac_q"),
                                    ),
                                ),
                                Node::assign(
                                    "jac_maxod",
                                    Expr::select(
                                        Expr::var("jac_isgt"),
                                        Expr::var("jac_av"),
                                        Expr::var("jac_maxod"),
                                    ),
                                ),
                            ],
                        )],
                    )],
                ),
                // Rotate only when the largest off-diagonal exceeds the convergence
                // threshold.
                Node::if_then(
                    Expr::gt(Expr::var("jac_maxod"), Expr::f32(JACOBI_EPS)),
                    vec![jacobi_apply_rotation_region(
                        OP_ID,
                        a,
                        eigenvectors,
                        n,
                        &Expr::var("jac_p"),
                        &Expr::var("jac_q"),
                    )],
                ),
            ],
        ),
        eigenvector_column_sign_region(OP_ID, eigenvectors, n),
        matrix_diagonal_extract_region(OP_ID, a, eigenvalues, n),
    ]
}

/// Emit [`jacobi_eigen_body`] as a child region of `parent_op_id`.
///
/// The nodes are exactly the body; the `Node::Region` around them records the composition
/// edge: a body built by calling another operation's builder carries that operation's
/// generator and a `source_region` naming the caller. Splicing the body in bare (which is what `tensor_train_decompose_step` used to do)
/// leaves the IR indistinguishable from a hand-rolled eigensolve, so `print-composition`, the
/// region-inline debug trace, and the Gate 1 composed fraction all report the caller as a
/// monolith and no audit can tell that the two callers share one spelling.
#[must_use]
pub fn jacobi_eigen_region(
    parent_op_id: &str,
    a: &str,
    eigenvectors: &str,
    eigenvalues: &str,
    n: u32,
) -> Node {
    Node::Region {
        generator: Ident::from(OP_ID),
        source_region: Some(GeneratorRef {
            name: parent_op_id.to_string(),
        }),
        body: Arc::new(jacobi_eigen_body(a, eigenvectors, eigenvalues, n)),
    }
}

/// Build a standalone symmetric-eigendecomposition Program.
///
/// Inputs:
/// - `a`: `n x n` symmetric matrix (f32), OVERWRITTEN with the near-diagonal rotated matrix.
/// - `eigenvectors`: `n x n` output; column `k` is the eigenvector for eigenvalue `k`.
/// - `eigenvalues`: `n` output; `eigenvalues[k] = A_rotated[k,k]`.
#[must_use]
pub fn symmetric_eigen_jacobi(a: &str, eigenvectors: &str, eigenvalues: &str, n: u32) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((eigenvalues, DataType::F32)),
            format!("Fix: symmetric_eigen_jacobi requires n > 0, got {n}."),
        );
    }
    let Some(cells) = n.checked_mul(n) else {
        return trap_program(
            OP_ID,
            Some((eigenvalues, DataType::F32)),
            format!("Fix: symmetric_eigen_jacobi n*n overflows matrix cell count for n={n}."),
        );
    };

    let body = jacobi_eigen_body(a, eigenvectors, eigenvalues, n);
    Program::wrapped(
        vec![
            BufferDecl::storage(a, 0, BufferAccess::ReadWrite, DataType::F32).with_count(cells),
            BufferDecl::storage(eigenvectors, 1, BufferAccess::ReadWrite, DataType::F32)
                .with_count(cells),
            BufferDecl::storage(eigenvalues, 2, BufferAccess::ReadWrite, DataType::F32)
                .with_count(n),
        ],
        [1, 1, 1],
        vec![Node::Region {
            generator: Ident::from(OP_ID),
            source_region: None,
            body: Arc::new(vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                body,
            )]),
        }],
    )
}

// Canonical registration.
//
// WITNESS: A = [[6,2,0,0],[2,3,0,0],[0,0,12,1],[0,0,1,12]], two disjoint symmetric 2x2 blocks.
// Spectrum {7, 2, 11, 13}: every eigenvalue is simple and the smallest gap is 2 against a norm
// of 13, so the eigenbasis is unique up to column sign, and the sign pass below fixes that
// sign. Every row is strictly diagonally dominant.
//
// The block structure is not decoration, it is what makes the fixture pinnable at all. A
// rotation on (p, q) rewrites only rows and columns p and q, so with the two blocks disjoint
// the pivot (0,1) leaves A[2,3] untouched and both rotations write their pivot entries as
// exact zeros with no fill-in. The off-diagonal maximum therefore reaches exactly 0 after two
// rotations, in f32 and in f64 alike. A dense matrix converges instead of terminating: the f32
// body stops at JACOBI_EPS (1e-6) and the f64 oracle at 1e-12, leaving off-diagonal residues
// several decades apart in the `a` output buffer, which is unbounded in ULPs no matter how
// well separated the spectrum is. `a` is read-write and therefore part of expected_output, so
// that residue is not something a fixture can look away from.
//
// The witness still exercises the whole body: identity seeding of V, the i < j argmax (which
// picks (0,1) first because |2| > |1|), a rotation with app != aqq, a rotation with app == aqq
// (the tau = +0 case the t formula handles explicitly), the spectator rows a rotation must not
// touch, one column the sign pass flips and three it leaves alone, and the diagonal read-out.
//
// ORACLE: expected values come from the independent f64 CPU reference
// `math::tensor_train_decompose::symmetric_eigen_jacobi_into`, run on the same input bytes,
// then rounded to f32 and sign-canonicalized by the same rule this body applies. They are not
// captured from a run of the Program under test. Cross-checked analytically: [[6,2],[2,3]] has
// trace 9 and determinant 14, so its eigenvalues are (9 ± 5)/2 = 7 and 2 with eigenvectors
// (2,1)/√5 and (1,-2)/√5; [[12,1],[1,12]] has eigenvalues 12 ∓ 1 = 11 and 13 with eigenvectors
// (1,-1)/√2 and (1,1)/√2.
//
// The two -0.0 entries in the eigenvector fixture are the zero rows of column 1 after the sign
// pass multiplies that column by -1.0; f64 and f32 both produce them.
//
// TOLERANCE: 1 ULP. Each output element is produced by exactly one rotation, so the f32 body
// rounds c and s once each and then evaluates one four-term product per element across the
// column and row passes: a fixed handful of roundings, not an error that grows with sweep
// count. Measured against the f64 oracle, 22 of the 24 outputs are bit-identical and the two
// eigenvalues that are not (2.0 and 13.0, each a sum of two cancelling terms) land 1 ULP low.
// Nothing here justifies a wider window.
#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || symmetric_eigen_jacobi("a", "evec", "eval", 4),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            // One entry per declared buffer: a (4x4, read-write and overwritten), evec (4x4),
            // eval (4). The last two are zero-initialized, matching backend zero-allocation.
            vec![vec![
                to_bytes(&[
                    6.0, 2.0, 0.0, 0.0, //
                    2.0, 3.0, 0.0, 0.0, //
                    0.0, 0.0, 12.0, 1.0, //
                    0.0, 0.0, 1.0, 12.0,
                ]),
                to_bytes(&[0.0; 16]),
                to_bytes(&[0.0; 4]),
            ]]
        }),
        Some(|| {
            let to_bytes = |vals: &[f32]| crate::wire::pack_f32_slice(vals);
            vec![vec![
                // a, rotated to exact diagonal form.
                to_bytes(&[
                    7.0, 0.0, 0.0, 0.0, //
                    0.0, 2.0, 0.0, 0.0, //
                    0.0, 0.0, 11.0, 0.0, //
                    0.0, 0.0, 0.0, 13.0,
                ]),
                // evec, row-major; column k is the eigenvector for eval[k].
                to_bytes(&[
                    0.8944272, 0.4472136, 0.0, 0.0, //
                    0.4472136, -0.8944272, 0.0, 0.0, //
                    0.0, -0.0, 0.70710677, 0.70710677, //
                    0.0, -0.0, -0.70710677, 0.70710677,
                ]),
                to_bytes(&[7.0, 2.0, 11.0, 13.0]),
            ]]
        }),
    )
    .with_tolerance(vyre_foundation::operation::TolerancePolicy { f32_ulp: 1 })
}
