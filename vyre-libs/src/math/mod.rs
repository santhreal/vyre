//! Linear algebra, scans, broadcasting, and atomic compositions.
//!
//! Every function here is a pure Category-A composition over
//! vyre-ops primitives, **except** `atomic` which is Category-B
//! (`Category::Intrinsic`) because it requires the backend to support
//! `Expr::Atomic` (F-IR-35).
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

/// Atomic read-modify-write compositions (add/and/or/xor/min/max/exchange/compare_exchange)
///  -  migrated from vyre-ops per the intrinsic-vs-library rule (Expr::Atomic is an
/// existing IR variant, so these are library compositions rather than intrinsics).
pub mod atomic;
/// Average floor operation
pub mod avg_floor;
mod bit_count_ops;
mod bit_count_u32;
/// Clamp to [lo, hi] per lane (migrated from vyre-ops per the intrinsic-vs-library rule).
pub mod clamp_u32;
/// 2D convolution  -  direct 3x3 kernel base case (ROADMAP H3).
pub mod conv;
/// Fast Fourier Transform  -  base-case 4-point complex FFT (ROADMAP H2).
pub mod fft;
/// Arithmetic mean reduction
pub mod reduce_mean;
/// Welford variance reduction
pub mod reduce_variance;
/// Element-wise square operation
pub mod square;
/// Block-FMA weighted-sum reduction (ROADMAP G7).
pub mod weighted_sum;
/// Welford sum-of-squares operation
pub mod welford;
/// Wrapping negation operation
pub mod wrapping_neg;

pub(crate) mod elementwise;

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
        vec![crate::region::wrap_anonymous(
            op_id,
            vec![Node::trap(Expr::u32(0), fix)],
        )],
    )
}

pub use atomic::{
    atomic_add_u32, atomic_and_u32, atomic_compare_exchange_u32, atomic_exchange_u32,
    atomic_max_u32, atomic_min_u32, atomic_or_u32, atomic_xor_u32,
};
pub use bit_count_ops::lzcnt_u32::lzcnt_u32;
pub use bit_count_ops::tzcnt_u32::tzcnt_u32;
pub use bit_count_ops::{lzcnt_u32, tzcnt_u32};
pub use clamp_u32::clamp_u32;
pub use reduce_mean::reduce_mean;
pub use reduce_variance::reduce_variance;
pub use square::square;
pub use welford::welford_sum_of_squares;
