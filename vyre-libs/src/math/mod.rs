//! Linear algebra, scans, broadcasting, and atomic compositions.
//!
//! Every function here is a pure Category-A composition over foundation IR
//! primitives, except `atomic`, which requires backend support for
//! `Expr::Atomic`.
//!
//! Organized into sub-dialects so each concern has its own namespace:
//! - `linalg`  -  dot, matmul, matmul_tiled
//! - `scan`  -  scan_prefix_sum
//! - `broadcast`  -  broadcast
//! - `succinct`  -  rank/select bitvector metadata
//!

#[cfg(feature = "math-linalg")]
pub mod linalg;

#[cfg(feature = "math-scan")]
pub mod scan;

#[cfg(feature = "math-broadcast")]
pub mod broadcast;

/// Abstract algebraic structures for dataflow, security, and scheduling.
#[cfg(feature = "math-algebra")]
pub mod algebra;

/// Succinct bitvector rank metadata.
#[cfg(feature = "math-succinct")]
pub mod succinct;

/// Atomic read-modify-write compositions (add/and/or/xor/min/max/exchange/compare_exchange).
/// These are library compositions over the canonical `Expr::Atomic` IR variant.
#[cfg(feature = "math-dialect")]
pub(crate) mod atomic;
/// Average floor operation
#[cfg(feature = "math-dialect")]
pub mod avg_floor;
#[cfg(feature = "math-dialect")]
mod bit_count_ops;
#[cfg(feature = "math-dialect")]
mod bit_count_u32;
/// Clamp to [lo, hi] per lane.
#[cfg(feature = "math-dialect")]
pub(crate) mod clamp_u32;
/// 2D convolution  -  direct 3x3 kernel base case (ROADMAP H3).
#[cfg(feature = "math-dialect")]
pub mod conv;
/// Fast Fourier Transform  -  base-case 4-point complex FFT (ROADMAP H2).
#[cfg(feature = "math-dialect")]
pub mod fft;
/// Arithmetic mean reduction
#[cfg(feature = "math-dialect")]
pub(crate) mod reduce_mean;
/// Welford variance reduction
#[cfg(feature = "math-dialect")]
pub(crate) mod reduce_variance;
/// Element-wise square operation
#[cfg(feature = "math-dialect")]
pub(crate) mod square;
/// Block-FMA weighted-sum reduction (ROADMAP G7).
#[cfg(feature = "math-dialect")]
pub mod weighted_sum;
/// Welford sum-of-squares operation
#[cfg(feature = "math-dialect")]
pub(crate) mod welford;
/// Wrapping negation operation
#[cfg(feature = "math-dialect")]
pub mod wrapping_neg;

#[cfg(feature = "math-dialect")]
fn invalid_f32_reduction_program(
    op_id: &'static str,
    input: &str,
    output: &str,
    fix: &'static str,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(1),
            BufferDecl::output(output, 1, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![vyre_foundation::composition::wrap_anonymous_region(
            op_id,
            vec![Node::trap(Expr::u32(0), fix)],
        )],
    )
}

#[must_use]
pub(crate) fn trap_f32_output_program(
    op_id: &'static str,
    output: &str,
    error: String,
) -> vyre_foundation::ir::Program {
    vyre_foundation::composition::trap_program(
        op_id,
        Some((output, vyre_foundation::ir::DataType::F32)),
        error,
    )
}

#[cfg(test)]
pub(crate) fn wrap_unary_f32_scalar_program(
    op_id: &'static str,
    input: &str,
    output: &str,
    n: u32,
    body: Vec<vyre_foundation::ir::Node>,
) -> vyre_foundation::ir::Program {
    use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Program};
    Program::wrapped(
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::F32).with_count(n),
            BufferDecl::output(output, 1, DataType::F32).with_count(1),
        ],
        [1, 1, 1],
        vec![vyre_foundation::composition::wrap_anonymous_region(
            op_id, body,
        )],
    )
}

#[cfg(feature = "math-dialect")]
pub use atomic::{
    atomic_add_u32, atomic_and_u32, atomic_compare_exchange_u32, atomic_exchange_u32,
    atomic_lru_update_u32, atomic_max_u32, atomic_min_u32, atomic_or_u32, atomic_xor_u32,
};
#[cfg(feature = "math-dialect")]
pub use bit_count_ops::{lzcnt_u32, tzcnt_u32};
#[cfg(feature = "math-dialect")]
pub use clamp_u32::clamp_u32;
#[cfg(feature = "math-dialect")]
pub use reduce_mean::reduce_mean;
#[cfg(feature = "math-dialect")]
pub use reduce_mean::try_reduce_mean;
#[cfg(feature = "math-dialect")]
pub use reduce_variance::reduce_variance;
#[cfg(feature = "math-dialect")]
pub use reduce_variance::try_reduce_variance;
#[cfg(feature = "math-dialect")]
pub use square::square;
#[cfg(feature = "math-dialect")]
pub use welford::welford_sum_of_squares;

// ---------------------------------------------------------------------------
// Math kernels moved from vyre-primitives. Each module exposes one reusable
// GPU composition with a stable op id. Callers import the narrow module they
// need so region-chain audits can see which primitive owns the shared work.
//
// The `math-kernels` gate is the domain gate. A module that genuinely needs a
// sibling domain names that domain's feature as well, so enabling the math
// kernels alone never compiles the graph or fixpoint domain.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
pub(crate) fn wrap_fixpoint_program(
    op_id: &'static str,
    inner: &vyre_foundation::ir::Program,
    buffers: Vec<vyre_foundation::ir::BufferDecl>,
) -> vyre_foundation::ir::Program {
    vyre_foundation::ir::Program::wrapped(
        buffers,
        crate::fixpoint::persistent_fixpoint::PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
        vec![vyre_foundation::composition::wrap_anonymous_region(
            op_id,
            inner.entry().to_vec(),
        )],
    )
}

#[cfg(test)]
pub(crate) fn assert_trapping_region_on_zero(program: &vyre_foundation::ir::Program, msg: &str) {
    use vyre_foundation::ir::Node;
    assert!(
        program.entry().iter().any(|node| matches!(
            node,
            Node::Region { body, .. } if body.iter().any(|inner| matches!(inner, Node::Trap { .. }))
        )),
        "{msg}"
    );
}

#[cfg(test)]
pub(crate) fn assert_local_id_0_bound(program: &vyre_foundation::ir::Program, msg: &str) {
    use vyre_foundation::ir::{Expr, Node};
    use vyre_foundation::visit::any_descendant;
    let has_local_binding = program.entry().iter().any(|node| {
        any_descendant(node, &mut |inner| match inner {
            Node::Let { name, value } => {
                name == "local" && matches!(value, Expr::LocalId { axis: 0 })
            }
            _ => false,
        })
    });
    assert!(has_local_binding, "{msg}");
}

#[cfg(test)]
pub(crate) fn assert_slices_approx_eq(
    case: usize,
    actual: &[f64],
    expected: &[f64],
    approx_eq: impl Fn(f64, f64) -> bool,
) {
    assert_eq!(actual.len(), expected.len(), "case {case}: output length");
    for idx in 0..actual.len() {
        if expected[idx].is_nan() {
            assert!(actual[idx].is_nan(), "case {case} idx {idx}: expected NaN");
        } else {
            assert!(
                approx_eq(actual[idx], expected[idx]),
                "case {case} idx {idx}: expected {}, got {}",
                expected[idx],
                actual[idx]
            );
        }
    }
}

/// 1D separable convolution (domain-neutral: blur, signal processing, audio).
#[cfg(feature = "math-kernels")]
pub mod conv1d;
/// Shared dot-product partial accumulator.
#[cfg(feature = "math-kernels")]
pub mod dot_partial;
/// Value-set analysis interval arithmetic.
#[cfg(feature = "math-kernels")]
pub mod interval;
/// Classical RK4 next-state combiner for ODE integration.
#[cfg(feature = "math-kernels")]
pub mod ode_step;
/// Subgroup prefix-sum scan used by compaction, histograms, and reductions.
#[cfg(feature = "math-kernels")]
pub mod prefix_scan;
#[cfg(feature = "math-kernels")]
pub(crate) mod u32_binary_map;

/// Differential-privacy accountant  -  Gaussian-mechanism RDP step with
/// host-side `(epsilon, delta)` conversion.
#[cfg(feature = "math-kernels")]
pub mod dp_accountant;

/// Fractional-calculus kernel  -  Gruenwald-Letnikov weight generator that
/// feeds the existing `conv1d` primitive.
#[cfg(feature = "math-kernels")]
pub mod fractional;

/// Submodular greedy step  -  argmax-of-marginals primitive driving
/// (1 - 1/e)-approximation greedy maximization.
#[cfg(feature = "math-kernels")]
pub mod submodular_greedy;

/// Conformal prediction  -  finite-sample distribution-free uncertainty
/// intervals.
#[cfg(feature = "math-kernels")]
pub mod conformal;

/// Sinkhorn-Knopp scaling step for entropic optimal transport. Composes with
/// `semiring_gemm` for the matvec halves of the iteration.
#[cfg(feature = "math-kernels")]
pub mod sinkhorn;

/// Full iterative Sinkhorn balance primitive.
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
pub mod sinkhorn_iterate;

/// Differentiable algorithm primitives  -  softmax + temperature-scaled
/// argmax.
#[cfg(feature = "math-kernels")]
pub mod differentiable;

/// Quantized packing primitives for INT4 / packed low-bit tensors.
#[cfg(feature = "math-kernels")]
pub mod quantized;

/// Score-based generative one-step denoise combiner.
#[cfg(feature = "math-kernels")]
pub mod score_denoise;

/// KFAC block-diagonal inverse for natural gradient.
#[cfg(feature = "math-kernels")]
pub mod kfac_block_inverse;

/// Newton-Schulz inverse-square-root step (Shampoo / KFAC core kernel).
/// Matrix preconditioner without SVD.
#[cfg(feature = "math-kernels")]
pub mod preconditioner;

/// Natural-gradient block-apply  -  multiply gradient by precomputed
/// `M^{-1/2}` block.
#[cfg(feature = "math-kernels")]
pub mod natural_gradient;

/// Iterative hard thresholding for sparse signal recovery (#48).
#[cfg(feature = "math-kernels")]
pub mod sparse_recovery;

/// DP-SGD per-sample gradient clip (#42).
#[cfg(feature = "math-kernels")]
pub mod dp_clip;

/// Mori-Zwanzig Markovian projection step  -  closed-form coarse-graining of
/// dynamical systems (#58).
#[cfg(feature = "math-kernels")]
pub mod mori_zwanzig;

/// Information-geometry primitives  -  Bhattacharyya / Fisher-Rao / Amari
/// alpha-connection (#57).
#[cfg(feature = "math-kernels")]
pub mod info_geometry;

/// Fast Multipole Method primitives  -  P2M / M2L / L2P (#51).
#[cfg(feature = "math-kernels")]
pub mod fmm;

/// Algebraic Multigrid V-cycle Jacobi smoother step (#50).
#[cfg(feature = "math-kernels")]
pub mod multigrid;

/// Algebraic Multigrid V-cycle (#P-PRIM-3).
#[cfg(feature = "math-kernels")]
pub mod amg_v_cycle;

/// Sheaf Laplacian eigenvalue (#P-PRIM-9).
#[cfg(feature = "math-kernels")]
pub mod sheaf_laplacian_eigenvalue;

/// Canonical sign for every column of an eigenvector matrix, so an eigenbasis
/// that is only defined up to sign can be pinned by an exact fixture.
#[cfg(feature = "math-kernels")]
pub mod eigenvector_column_sign;

/// Givens rotation of one strided element pair, the shared arithmetic behind
/// every column, row and accumulator rotation.
#[cfg(feature = "math-kernels")]
pub mod givens_rotate_pair;

/// One Jacobi rotation at a given pivot, applied to a symmetric matrix and
/// accumulated into its rotation matrix.
#[cfg(feature = "math-kernels")]
pub mod jacobi_apply_rotation;

/// Diagonal read-out of a square row-major matrix.
#[cfg(feature = "math-kernels")]
pub mod matrix_diagonal_extract;

/// Identity seeding of a square row-major matrix.
#[cfg(feature = "math-kernels")]
pub mod matrix_identity_fill;

/// Symmetric eigendecomposition via cyclic (max-pivot) Jacobi rotations (f32,
/// serial single-lane). The numerical core of the tensor-train SVD; reusable
/// for any dense symmetric eigenproblem.
#[cfg(feature = "math-kernels")]
pub mod symmetric_eigen_jacobi;

/// Full Edmonds augmenting-path matroid intersection (#P-PRIM-10).
#[cfg(all(feature = "math-kernels", feature = "graph"))]
pub mod matroid_intersection_full;

/// Tensor-train decomposition via SVD-truncation per mode (#P-PRIM-12).
#[cfg(feature = "math-kernels")]
pub mod tensor_train_decompose;

/// Tensor-train one-step contraction (#6).
#[cfg(feature = "math-kernels")]
pub mod tensor_train;

/// Randomized SVD random-projection step (#3).
#[cfg(feature = "math-kernels")]
pub mod randomized_svd;

/// Sum-of-squares (Positivstellensatz) Gram-matrix construction (#14).
#[cfg(feature = "math-kernels")]
pub mod sos_certificate;

/// Quantum singular-value transform (classical) block-encoding + Chebyshev
/// apply (#34).
#[cfg(feature = "math-kernels")]
pub mod qsvt;

/// Pairwise tensor-network contraction (#35).
#[cfg(feature = "math-kernels")]
pub mod tensor_network;

/// RMT-based Marchenko-Pastur edge clip (#17).
#[cfg(feature = "math-kernels")]
pub mod spectral_shape;

/// p-adic Hensel-lift step (#54, research scaffold). Stable arithmetic for
/// ill-conditioned problems.
#[cfg(feature = "math-kernels")]
pub mod padic;

/// Multi-limb big-integer ripple-carry addition primitive (#P-PRIM-BIGINT).
/// Emits `(sum_partial, carry_partial)` per-limb for a downstream
/// parallel-prefix carry-fix wave.
#[cfg(feature = "math-kernels")]
pub mod bigint_add_carry;

/// Generic-semiring matrix multiply  -  spine of the LEGO substrate.
#[cfg(feature = "math-kernels")]
pub mod semiring_gemm;

/// Sparse-kernel selector evidence for library comparisons.
#[cfg(feature = "math-kernels")]
pub mod sparse_selector;

/// Bellman-Ford shortest path primitive over an edge list. Composes
/// `persistent_fixpoint`.
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
pub mod bellman_shortest_path;

#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
mod scallop_persistent;

/// Scallop-style probabilistic Datalog join (#39). Emits a lineage
/// semiring join inside a GPU-resident fixpoint kernel over `w`-word
/// lineage cells. User dialect: probabilistic Datalog.
/// Self-consumer: rule-provenance tracking
/// (`vyre_libs::encoding::scallop_provenance`).
#[cfg(all(feature = "math-kernels", feature = "fixpoint"))]
pub mod scallop_join;
/// Prefix-scan backed stream compaction over live-lane flags.
#[cfg(feature = "math-kernels")]
pub mod stream_compact;
/// SCC-local matrix fixpoint primitive for recursive graph components.
#[cfg(feature = "math-kernels")]
pub mod tensor_scc;

/// Signed 16.16 fixed-point arithmetic over `Expr`.
#[cfg(any(
    feature = "math-kernels",
    feature = "graph",
    feature = "geom",
    feature = "opt"
))]
pub(crate) mod fixed;

/// Fixed-point u32 matrix and matrix-vector program builders.
#[cfg(any(feature = "math-kernels", feature = "graph"))]
pub(crate) mod fixed_u32_matmul;
