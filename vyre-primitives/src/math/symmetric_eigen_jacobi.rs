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
use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

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
/// Reused by [`symmetric_eigen_jacobi`] and (later) the tensor-train SVD step so the rotation policy
/// lives in ONE place.
#[must_use]
pub fn jacobi_eigen_body(a: &str, eigenvectors: &str, eigenvalues: &str, n: u32) -> Vec<Node> {
    let sweeps = jacobi_sweeps(n);
    let mut nodes = Vec::new();

    // V = I
    nodes.push(Node::loop_for(
        "jac_vi",
        Expr::u32(0),
        Expr::u32(n),
        vec![Node::loop_for(
            "jac_vj",
            Expr::u32(0),
            Expr::u32(n),
            vec![Node::store(
                eigenvectors,
                idx(Expr::var("jac_vi"), n, Expr::var("jac_vj")),
                Expr::select(
                    Expr::eq(Expr::var("jac_vi"), Expr::var("jac_vj")),
                    Expr::f32(1.0),
                    Expr::f32(0.0),
                ),
            )],
        )],
    ));

    // Sweep loop: each iteration zeroes the largest off-diagonal entry via one Givens rotation.
    nodes.push(Node::loop_for(
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
            // Rotate only when the largest off-diagonal exceeds the convergence threshold.
            Node::if_then(
                Expr::gt(Expr::var("jac_maxod"), Expr::f32(JACOBI_EPS)),
                vec![
                    Node::let_bind(
                        "jac_app",
                        Expr::load(a, idx(Expr::var("jac_p"), n, Expr::var("jac_p"))),
                    ),
                    Node::let_bind(
                        "jac_aqq",
                        Expr::load(a, idx(Expr::var("jac_q"), n, Expr::var("jac_q"))),
                    ),
                    Node::let_bind(
                        "jac_apq",
                        Expr::load(a, idx(Expr::var("jac_p"), n, Expr::var("jac_q"))),
                    ),
                    // tau = (aqq - app) / (2 * apq)
                    Node::let_bind(
                        "jac_tau",
                        Expr::div(
                            Expr::sub(Expr::var("jac_aqq"), Expr::var("jac_app")),
                            Expr::mul(Expr::f32(2.0), Expr::var("jac_apq")),
                        ),
                    ),
                    // t = sign(tau) / (|tau| + sqrt(1 + tau^2)). NOTE: `sign` here must match the
                    // reference's Rust `f64::signum`, which returns +1 at +0.0, this is what makes
                    // the app==aqq degenerate case (tau=+0) rotate by 45° (t=1) instead of stalling.
                    // WGSL/`UnOp::Sign` returns 0 at 0, so we use an explicit `tau >= 0 ? 1 : -1`.
                    Node::let_bind(
                        "jac_t",
                        Expr::div(
                            Expr::select(
                                Expr::ge(Expr::var("jac_tau"), Expr::f32(0.0)),
                                Expr::f32(1.0),
                                Expr::f32(-1.0),
                            ),
                            Expr::add(
                                Expr::abs(Expr::var("jac_tau")),
                                Expr::sqrt(Expr::add(
                                    Expr::f32(1.0),
                                    Expr::mul(Expr::var("jac_tau"), Expr::var("jac_tau")),
                                )),
                            ),
                        ),
                    ),
                    // c = 1 / sqrt(1 + t^2); s = t * c
                    Node::let_bind(
                        "jac_c",
                        Expr::inverse_sqrt(Expr::add(
                            Expr::f32(1.0),
                            Expr::mul(Expr::var("jac_t"), Expr::var("jac_t")),
                        )),
                    ),
                    Node::let_bind("jac_s", Expr::mul(Expr::var("jac_t"), Expr::var("jac_c"))),
                    // Rotate columns p, q of A (over all rows k).
                    Node::loop_for(
                        "jac_ck",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![
                            Node::let_bind(
                                "jac_akp",
                                Expr::load(a, idx(Expr::var("jac_ck"), n, Expr::var("jac_p"))),
                            ),
                            Node::let_bind(
                                "jac_akq",
                                Expr::load(a, idx(Expr::var("jac_ck"), n, Expr::var("jac_q"))),
                            ),
                            Node::store(
                                a,
                                idx(Expr::var("jac_ck"), n, Expr::var("jac_p")),
                                Expr::sub(
                                    Expr::mul(Expr::var("jac_c"), Expr::var("jac_akp")),
                                    Expr::mul(Expr::var("jac_s"), Expr::var("jac_akq")),
                                ),
                            ),
                            Node::store(
                                a,
                                idx(Expr::var("jac_ck"), n, Expr::var("jac_q")),
                                Expr::add(
                                    Expr::mul(Expr::var("jac_s"), Expr::var("jac_akp")),
                                    Expr::mul(Expr::var("jac_c"), Expr::var("jac_akq")),
                                ),
                            ),
                        ],
                    ),
                    // Rotate rows p, q of A (over all columns k).
                    Node::loop_for(
                        "jac_rk",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![
                            Node::let_bind(
                                "jac_apk",
                                Expr::load(a, idx(Expr::var("jac_p"), n, Expr::var("jac_rk"))),
                            ),
                            Node::let_bind(
                                "jac_aqk",
                                Expr::load(a, idx(Expr::var("jac_q"), n, Expr::var("jac_rk"))),
                            ),
                            Node::store(
                                a,
                                idx(Expr::var("jac_p"), n, Expr::var("jac_rk")),
                                Expr::sub(
                                    Expr::mul(Expr::var("jac_c"), Expr::var("jac_apk")),
                                    Expr::mul(Expr::var("jac_s"), Expr::var("jac_aqk")),
                                ),
                            ),
                            Node::store(
                                a,
                                idx(Expr::var("jac_q"), n, Expr::var("jac_rk")),
                                Expr::add(
                                    Expr::mul(Expr::var("jac_s"), Expr::var("jac_apk")),
                                    Expr::mul(Expr::var("jac_c"), Expr::var("jac_aqk")),
                                ),
                            ),
                        ],
                    ),
                    // Force the pivot entries to exactly zero (matches the reference).
                    Node::store(
                        a,
                        idx(Expr::var("jac_p"), n, Expr::var("jac_q")),
                        Expr::f32(0.0),
                    ),
                    Node::store(
                        a,
                        idx(Expr::var("jac_q"), n, Expr::var("jac_p")),
                        Expr::f32(0.0),
                    ),
                    // Accumulate the rotation into V (columns p, q).
                    Node::loop_for(
                        "jac_vk",
                        Expr::u32(0),
                        Expr::u32(n),
                        vec![
                            Node::let_bind(
                                "jac_vkp",
                                Expr::load(
                                    eigenvectors,
                                    idx(Expr::var("jac_vk"), n, Expr::var("jac_p")),
                                ),
                            ),
                            Node::let_bind(
                                "jac_vkq",
                                Expr::load(
                                    eigenvectors,
                                    idx(Expr::var("jac_vk"), n, Expr::var("jac_q")),
                                ),
                            ),
                            Node::store(
                                eigenvectors,
                                idx(Expr::var("jac_vk"), n, Expr::var("jac_p")),
                                Expr::sub(
                                    Expr::mul(Expr::var("jac_c"), Expr::var("jac_vkp")),
                                    Expr::mul(Expr::var("jac_s"), Expr::var("jac_vkq")),
                                ),
                            ),
                            Node::store(
                                eigenvectors,
                                idx(Expr::var("jac_vk"), n, Expr::var("jac_q")),
                                Expr::add(
                                    Expr::mul(Expr::var("jac_s"), Expr::var("jac_vkp")),
                                    Expr::mul(Expr::var("jac_c"), Expr::var("jac_vkq")),
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    ));

    // eigenvalues = diag(A)
    nodes.push(Node::loop_for(
        "jac_ei",
        Expr::u32(0),
        Expr::u32(n),
        vec![Node::store(
            eigenvalues,
            Expr::var("jac_ei"),
            Expr::load(a, idx(Expr::var("jac_ei"), n, Expr::var("jac_ei"))),
        )],
    ));

    nodes
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
        return crate::invalid_output_program(
            OP_ID,
            eigenvalues,
            DataType::F32,
            format!("Fix: symmetric_eigen_jacobi requires n > 0, got {n}."),
        );
    }
    let Some(cells) = n.checked_mul(n) else {
        return crate::invalid_output_program(
            OP_ID,
            eigenvalues,
            DataType::F32,
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
