//! Shared dense Chebyshev-expansion CPU oracle.
//!
//! Two public primitives evaluate the same polynomial. `graph::chebyshev_filter`
//! applies `Σ_k c_k T_k(L̂) x` to a graph signal in f32, and `math::qsvt`
//! applies `Σ_k c_k T_k(A) v` to a vector in f64 as the classical half of a
//! quantum-singular-value-transform block encoding. Those are different ops with
//! different validation language, but the recurrence underneath is one:
//!
//! ```text
//!   T_0(M) v = v
//!   T_1(M) v = M v
//!   T_{k+1}(M) v = 2 M T_k(M) v - T_{k-1}(M) v
//! ```
//!
//! Both had their own copy of the accumulate-and-rotate step, so a change to
//! how a term is folded into the output, or to which buffer the rotation leaves
//! the newest iterate in, had to be made twice. This module owns the kernel;
//! callers keep their own op ids, error text, and buffer reservation.
//!
//! # Missing cells read as zero
//!
//! [`dense_mat_vec_into`] reads a matrix cell past the end of `matrix` as the
//! scalar zero rather than panicking, and [`chebyshev_expansion_into`] does the
//! same for `vector` and `coefficients`. That is not leniency this module chose:
//! `graph::chebyshev_filter`'s CPU oracle pins it, so a caller that passes a
//! Laplacian shorter than `n * n` gets the filter of the zero-padded operator.
//! `math::qsvt` rejects those lengths before it calls in, so the fallback is
//! unreachable from that side. Tighten it here and the pinned contract on the
//! other side breaks; validate in the caller instead, as qsvt does.

use std::ops::{AddAssign, Mul, Sub};

/// Scalar an expansion is evaluated over.
///
/// Satisfied by `f32` and `f64`. `Default` supplies the additive identity and
/// `From<u8>` the literal 2 the recurrence multiplies by, so no numeric trait of
/// this crate's own has to exist for two call sites.
pub(crate) trait ExpansionScalar:
    Copy + Default + From<u8> + Mul<Output = Self> + Sub<Output = Self> + AddAssign
{
}

impl<T> ExpansionScalar for T where
    T: Copy + Default + From<u8> + Mul<Output = T> + Sub<Output = T> + AddAssign
{
}

/// Read `slice[index]`, or the scalar zero when the slice is shorter.
fn at<T: ExpansionScalar>(slice: &[T], index: usize) -> T {
    slice.get(index).copied().unwrap_or_default()
}

/// `out[i] = Σ_j matrix[i * n + j] · vector[j]` for every `i < n`.
///
/// `out` must be at least `n` long. Writes, never accumulates, so the caller
/// does not have to zero it first.
pub(crate) fn dense_mat_vec_into<T: ExpansionScalar>(
    matrix: &[T],
    vector: &[T],
    n: usize,
    out: &mut [T],
) {
    for i in 0..n {
        let mut sum = T::default();
        for j in 0..n {
            sum += at(matrix, i * n + j) * at(vector, j);
        }
        out[i] = sum;
    }
}

/// Evaluate `out = Σ_{k=0..=degree} coefficients[k] · T_k(matrix) · vector`.
///
/// `matrix` is `n × n` row-major. `out`, `t_prev` and `t_curr` must each be at
/// least `n` long, and `t_next` must be too when `degree >= 2`; below that the
/// recurrence never runs and `t_next` stays as the caller left it, which is what
/// lets a caller skip sizing it. None of the four has to be zeroed.
///
/// The ping-pong takes the three iterate buffers as `Vec` so the rotation is
/// [`std::mem::swap`] on the caller's own storage: no iterate is ever copied,
/// and on return `t_curr` holds `T_degree(matrix) · vector` and `t_prev` holds
/// `T_{degree-1}(matrix) · vector`, so a caller can continue the recurrence or
/// read the last two iterates back.
pub(crate) fn chebyshev_expansion_into<T: ExpansionScalar>(
    matrix: &[T],
    vector: &[T],
    coefficients: &[T],
    n: usize,
    degree: usize,
    out: &mut [T],
    t_prev: &mut Vec<T>,
    t_curr: &mut Vec<T>,
    t_next: &mut Vec<T>,
) {
    let two = T::from(2_u8);
    let c0 = at(coefficients, 0);
    for i in 0..n {
        out[i] = c0 * at(vector, i);
    }
    if degree == 0 {
        return;
    }

    // T_0 = vector, T_1 = matrix · vector.
    for i in 0..n {
        t_prev[i] = at(vector, i);
    }
    dense_mat_vec_into(matrix, vector, n, t_curr);
    let c1 = at(coefficients, 1);
    for i in 0..n {
        out[i] += c1 * t_curr[i];
    }

    for k in 2..=degree {
        dense_mat_vec_into(matrix, t_curr, n, t_next);
        let c_k = at(coefficients, k);
        for i in 0..n {
            t_next[i] = two * t_next[i] - t_prev[i];
            out[i] += c_k * t_next[i];
        }
        std::mem::swap(t_prev, t_curr);
        std::mem::swap(t_curr, t_next);
    }
}
