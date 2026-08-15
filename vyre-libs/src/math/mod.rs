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

use vyre_foundation::composition::wrap_anonymous_region;

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
pub(crate) mod atomic;
/// Average floor operation
pub mod avg_floor;
mod bit_count_ops;
mod bit_count_u32;
/// Clamp to [lo, hi] per lane.
pub(crate) mod clamp_u32;
/// 2D convolution  -  direct 3x3 kernel base case (ROADMAP H3).
pub mod conv;
/// Fast Fourier Transform  -  base-case 4-point complex FFT (ROADMAP H2).
pub mod fft;
/// Arithmetic mean reduction
pub(crate) mod reduce_mean;
/// Welford variance reduction
pub(crate) mod reduce_variance;
/// Element-wise square operation
pub(crate) mod square;
/// Block-FMA weighted-sum reduction (ROADMAP G7).
pub mod weighted_sum;
/// Welford sum-of-squares operation
pub(crate) mod welford;
/// Wrapping negation operation
pub mod wrapping_neg;

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
        vec![wrap_anonymous_region(
            op_id,
            vec![Node::trap(Expr::u32(0), fix)],
        )],
    )
}

pub use atomic::{
    atomic_add_u32, atomic_and_u32, atomic_compare_exchange_u32, atomic_exchange_u32,
    atomic_lru_update_u32, atomic_max_u32, atomic_min_u32, atomic_or_u32, atomic_xor_u32,
};
pub use bit_count_ops::{lzcnt_u32, tzcnt_u32};
pub use clamp_u32::clamp_u32;
pub use reduce_mean::reduce_mean;
pub use reduce_mean::try_reduce_mean;
pub use reduce_variance::reduce_variance;
pub use reduce_variance::try_reduce_variance;
pub use square::square;
pub use welford::welford_sum_of_squares;
