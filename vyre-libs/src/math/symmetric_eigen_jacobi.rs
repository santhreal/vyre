//! Symmetric eigendecomposition via the cyclic (max-pivot) Jacobi method (#P-PRIM-Jacobi).
//!
//! Diagonalizes a real symmetric `n x n` matrix `A` in f32: produces its eigenvalues (the diagonal
//! of the rotated matrix) and eigenvectors (the accumulated rotation matrix `V`, whose columns are
//! the eigenvectors). This is the numerical core of the tensor-train SVD (`tensor_train_decompose`)
//! (a truncated SVD of `A` is obtained from the eigendecomposition of the Gram matrix `AᵀA`).
//!
//! One sweep is sequential in the matrix: it picks the largest off-diagonal entry and applies one
//! Givens rotation that depends on the current matrix, so sweep `k + 1` cannot start before sweep
//! `k`'s rotation has landed. The pivot SEARCH inside a sweep is not sequential — it is an argmax
//! over the `n²` index pairs — so the kernel runs a workgroup of lanes: the search is a cooperative
//! reduction (`crate::builder::cooperative::Argmax`), the identity seeding, the sign pass and the
//! diagonal read-out each walk their own cells across the lanes, and only the rotation stays on one
//! lane, behind a barrier, because the next sweep reads what it wrote. The serial work per sweep
//! drops from `n²` iterations in one lane to `n² / lanes` plus a log-depth tree. It
//! mirrors the CPU reference [`crate::math::tensor_train_decompose`]'s `symmetric_eigen_jacobi_into`
//! step for step, so the two agree up to f32-vs-f64 rounding; the kernel is verified by the
//! basis/order-invariant eigenpair contract (`A·vᵢ ≈ λᵢ·vᵢ` and `VᵀV ≈ I`) rather than element-wise,
//! because near-degenerate eigenvalues admit different-but-valid eigenvector bases.

use vyre_foundation::composition::{trap_program, wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use crate::builder::cooperative::{Argmax, KeyKind, LANES};
use crate::math::eigenvector_column_sign::eigenvector_column_sign_region;
use crate::math::jacobi_apply_rotation::jacobi_apply_rotation_region;
use crate::math::matrix_diagonal_extract::matrix_diagonal_extract_region;
use crate::math::matrix_identity_fill::matrix_identity_fill_region;

/// Op id.
pub const OP_ID: &str = "vyre-libs::math::symmetric_eigen_jacobi";

/// Off-diagonal convergence threshold (f32). Sweeps stop rotating once the largest off-diagonal
/// magnitude falls below this; matches the role of the f64 reference's `1e-12` but scaled to f32's
/// usable precision.
const JACOBI_EPS: f32 = 1.0e-6;

/// Workgroup scratch the pivot key reduces through, one f32 entry per lane.
const JACOBI_PIVOT_KEY: &str = "jac_pivot_key";

/// Workgroup scratch the pivot index reduces through, one u32 entry per lane.
const JACOBI_PIVOT_INDEX: &str = "jac_pivot_index";

/// The workgroup shape a program splicing [`jacobi_eigen_body`] dispatches.
#[must_use]
pub fn jacobi_workgroup() -> [u32; 3] {
    [LANES, 1, 1]
}

/// The two workgroup scratch buffers [`jacobi_eigen_body`] reduces the pivot through.
///
/// Every program that splices the body declares these and dispatches
/// [`jacobi_workgroup`], so the two callers cannot disagree about a name or a width. A missing
/// declaration is not a wrong answer, it is a program the backend refuses to lower.
#[must_use]
pub fn jacobi_scratch_buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::workgroup(JACOBI_PIVOT_KEY, LANES, DataType::F32),
        BufferDecl::workgroup(JACOBI_PIVOT_INDEX, LANES, DataType::U32),
    ]
}

/// Number of Jacobi sweeps, matching the CPU reference `(16 * n² ).max(32)`.
#[must_use]
pub fn jacobi_sweeps(n: u32) -> u32 {
    (16u32.saturating_mul(n).saturating_mul(n)).max(32)
}

/// Build the Jacobi eigensolve body. `a` is the f32
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
///
/// The body binds `local` and reduces the pivot through the scratch of
/// [`jacobi_scratch_buffers`], so a program that splices it declares those buffers and runs
/// [`jacobi_workgroup`] lanes. One workgroup owns the whole sweep: the identity seeding, the
/// sign pass and the diagonal read-out spread across that workgroup's lanes and the rotation
/// runs on its lane 0. A second workgroup would re-seed `V` to the identity after the first
/// finished and then rotate nothing, since the rotation is lane-guarded, publishing an identity
/// eigenbasis over the answer. A dispatch is rounded up to whole workgroups, so the workgroup
/// guard is what keeps the extra ones out; it is uniform inside a workgroup, which is what lets
/// the barriers sit inside it.
#[must_use]
pub fn jacobi_eigen_body(a: &str, eigenvectors: &str, eigenvalues: &str, n: u32) -> Vec<Node> {
    let sweeps = jacobi_sweeps(n);
    let pivot = Argmax {
        op_id: OP_ID,
        count: n.saturating_mul(n),
        tile: LANES,
        key_scratch: JACOBI_PIVOT_KEY,
        key_kind: KeyKind::F32,
        index_scratch: JACOBI_PIVOT_INDEX,
        var: "jac_pair",
    };
    // The search key: `|A[i,j]|` on the strictly upper triangle, 0 elsewhere. The pair index is
    // row-major, so `pair / n` is the row and `pair % n` the column, and the diagonal and lower
    // triangle score 0, which loses to any entry the threshold would rotate on. A key of 0 also
    // makes an already-diagonal matrix pick pair 0 and rotate nothing.
    let key = |pair: Expr| {
        Expr::select(
            Expr::lt(
                Expr::div(pair.clone(), Expr::u32(n)),
                Expr::rem(pair.clone(), Expr::u32(n)),
            ),
            Expr::abs(Expr::load(a, pair)),
            Expr::f32(0.0),
        )
    };
    // One lane of one workgroup. A rotation rewrites two rows and two columns of `a` and of `V`,
    // and the next sweep's search reads what it wrote, so the phases that mutate shared state are
    // serial by the algorithm and only the search is not.
    let serial = |body: Vec<Node>| {
        Node::if_then(
            Expr::and(
                Expr::is_first_logical_tile(),
                Expr::eq(Expr::var("local"), Expr::u32(0)),
            ),
            body,
        )
    };

    // One sweep: cooperative argmax over the pair space, then the rotation the winner asks for.
    let mut sweep = pivot.nodes(key);
    sweep.extend([
        Node::let_bind("jac_pivot", Expr::load(JACOBI_PIVOT_INDEX, Expr::u32(0))),
        Node::let_bind("jac_p", Expr::div(Expr::var("jac_pivot"), Expr::u32(n))),
        Node::let_bind("jac_q", Expr::rem(Expr::var("jac_pivot"), Expr::u32(n))),
        Node::let_bind("jac_maxod", Expr::load(JACOBI_PIVOT_KEY, Expr::u32(0))),
        // Rotate only when the largest off-diagonal exceeds the convergence threshold.
        serial(vec![Node::if_then(
            Expr::gt(Expr::var("jac_maxod"), Expr::f32(JACOBI_EPS)),
            vec![jacobi_apply_rotation_region(
                OP_ID,
                a,
                eigenvectors,
                n,
                &Expr::var("jac_p"),
                &Expr::var("jac_q"),
            )],
        )]),
        // Publish the rotation to the lanes that search over it next sweep.
        Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
    ]);

    vec![Node::if_then(
        Expr::is_first_logical_tile(),
        vec![
            Node::let_bind("local", Expr::LogicalWithinTileId { axis: 0 }),
            // The accumulator seed, the sign canonicalization and the diagonal
            // read-out each touch cells no other lane touches, so they run across
            // the workgroup at the width this program declares. Only the rotation
            // is serial, and it is serial because the next sweep reads it.
            matrix_identity_fill_region(OP_ID, eigenvectors, n, LANES),
            Node::logical_barrier(vyre_foundation::ir::MemoryOrdering::SeqCst),
            Node::loop_for("jac_sweep", Expr::u32(0), Expr::u32(sweeps), sweep),
            eigenvector_column_sign_region(OP_ID, eigenvectors, n, LANES),
            matrix_diagonal_extract_region(OP_ID, a, eigenvalues, n, LANES),
        ],
    )]
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
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        jacobi_eigen_body(a, eigenvectors, eigenvalues, n),
    )
}

/// Build a standalone symmetric-eigendecomposition Program.
///
/// Inputs:
/// - `a`: `n x n` symmetric matrix (f32), OVERWRITTEN with the near-diagonal rotated matrix.
/// - `eigenvectors`: `n x n` output; column `k` is the eigenvector for eigenvalue `k`.
/// - `eigenvalues`: `n` output; `eigenvalues[k] = A_rotated[k,k]`.
#[must_use]
pub fn symmetric_eigen_jacobi(a: &str, eigenvectors: &str, eigenvalues: &str, n: u32) -> Program {
    let cells = match crate::plumbing::operand::shape::square_matrix_cells(OP_ID, n) {
        Ok(cells) => cells,
        Err(message) => return trap_program(OP_ID, Some((eigenvalues, DataType::F32)), message),
    };

    // The body owns the workgroup guard, so a wider dispatch cannot re-seed the
    // eigenbasis. Nothing is added here.
    let body = jacobi_eigen_body(a, eigenvectors, eigenvalues, n);
    let mut buffers = vec![
        BufferDecl::storage(a, 0, BufferAccess::ReadWrite, DataType::F32).with_count(cells),
        BufferDecl::storage(eigenvectors, 1, BufferAccess::ReadWrite, DataType::F32)
            .with_count(cells),
        BufferDecl::storage(eigenvalues, 2, BufferAccess::ReadWrite, DataType::F32).with_count(n),
    ];
    buffers.extend(jacobi_scratch_buffers());
    Program::wrapped(
        buffers,
        jacobi_workgroup(),
        vec![wrap_anonymous_region(OP_ID, body)],
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
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || symmetric_eigen_jacobi("a", "evec", "eval", 4),
        Some(|| {
            let to_bytes = |vals: &[f32]| vyre_primitives::wire::pack_f32_slice(vals);
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
            vec![vec![
                // a, rotated to exact diagonal form.
                vec![
                    0x00, 0x00, 0xe0, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x30, 0x41, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x41,
                ],
                // evec, row-major; column k is the eigenvector for eval[k].
                vec![
                    0x2e, 0xf9, 0x64, 0x3f, 0x2e, 0xf9, 0xe4, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x2e, 0xf9, 0xe4, 0x3e, 0x2e, 0xf9, 0x64, 0xbf, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xf3, 0x04, 0x35, 0x3f, 0xf3, 0x04, 0x35, 0x3f,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xf3, 0x04, 0x35, 0xbf, 0xf3, 0x04, 0x35, 0x3f,
                ],
                // eval: [7.0, 2.0, 11.0, 13.0]
                vec![
                    0x00, 0x00, 0xe0, 0x40, // 7.0
                    0x00, 0x00, 0x00, 0x40, // 2.0
                    0x00, 0x00, 0x30, 0x41, // 11.0
                    0x00, 0x00, 0x50, 0x41, // 13.0
                ],
            ]]
        }),
    )
    .with_numeric(vyre_foundation::numeric::NumericContract::ieee_f32(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_order_is_rejected() {
        let program = symmetric_eigen_jacobi("a", "evec", "eval", 0);
        crate::math::assert_trapping_region_on_zero(
            &program,
            "Fix: n = 0 must produce a trapping Program.",
        );
    }

    #[test]
    fn jacobi_validates_as_a_program() {
        let program = symmetric_eigen_jacobi("a", "evec", "eval", 4);
        let errors = vyre_foundation::validate::validate(&program);
        assert!(
            errors.is_empty(),
            "Fix: the Jacobi eigensolver must validate, got {:?}.",
            errors
        );
    }

    #[test]
    fn jacobi_dispatches_declared_lanes() {
        let program = symmetric_eigen_jacobi("a", "evec", "eval", 4);
        assert_eq!(
            program.workgroup_size(),
            [LANES, 1, 1],
            "Fix: symmetric_eigen_jacobi must dispatch [LANES, 1, 1] workgroup size."
        );
        assert_eq!(
            jacobi_workgroup(),
            [LANES, 1, 1],
            "Fix: jacobi_workgroup() must return [LANES, 1, 1]."
        );
    }
}
