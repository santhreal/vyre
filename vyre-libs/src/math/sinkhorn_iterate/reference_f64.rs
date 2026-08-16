//! Canonical f64 Sinkhorn-Knopp reference with tolerance-based convergence.

// ===== P-PRIM-11: Full iterative-balance Sinkhorn (f64) ===========
//
// The fixed-point u32 cpu_ref above is the GPU-targeted reference;
// the math operates on quantized fractions. This block ships an
// f64 reference that performs the canonical Sinkhorn-Knopp iterative
// matrix-balancing algorithm with tolerance-based convergence  -
// the operation many user dialects ask for when they say "balanced
// transport plan."

/// Tolerance-based Sinkhorn-Knopp iterative balancing in f64.
///
/// Inputs:
/// - `k`: kernel matrix `m × n`, row-major. Strictly positive entries.
/// - `a`: target row marginal, length m. Strictly positive entries.
/// - `b`: target column marginal, length n. Strictly positive entries.
/// - `tolerance`: stop when `||u_new - u_old||_∞ < tolerance`.
/// - `max_iterations`: hard cap.
///
/// Returns `(u, v, iterations)` such that `diag(u) · k · diag(v)`
/// has row sums approximately `a` and column sums approximately `b`,
/// up to the supplied tolerance.
///
/// Pre/post conditions:
/// * Caller guarantees `sum(a) == sum(b)` (mass-conservation;
///   Sinkhorn-Knopp converges only on balanced marginals).
/// * Returns the iteration that stopped  -  < `max_iterations` means
///   tolerance reached, == `max_iterations` means cap hit.
///
/// # Panics
///
/// Panics on length mismatch.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_iterate_f64(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> (Vec<f64>, Vec<f64>, u32) {
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut u_old = Vec::new();
    let iters = sinkhorn_iterate_f64_into(
        k,
        a,
        b,
        tolerance,
        max_iterations,
        &mut u,
        &mut v,
        &mut u_old,
    );
    (u, v, iters)
}

/// Fallible tolerance-based Sinkhorn-Knopp iterative balancing in f64.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_sinkhorn_iterate_f64(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> Result<(Vec<f64>, Vec<f64>, u32), String> {
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut u_old = Vec::new();
    let iters = try_sinkhorn_iterate_f64_into(
        k,
        a,
        b,
        tolerance,
        max_iterations,
        &mut u,
        &mut v,
        &mut u_old,
    )?;
    Ok((u, v, iters))
}

/// Tolerance-based Sinkhorn-Knopp iterative balancing in f64 using
/// caller-owned buffers.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_iterate_f64_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    u_old: &mut Vec<f64>,
) -> u32 {
    match try_sinkhorn_iterate_f64_into(k, a, b, tolerance, max_iterations, u, v, u_old) {
        Ok(iters) => iters,
        // Clearing the buffers and returning 0 iterations on failure makes a
        // GPU-vs-CPU parity assertion pass on empty==empty, silently masking a
        // divergence (Law 10 / Law 6). Fail loud; callers use the try_ variant.
        Err(error) => panic!("vyre-primitives Sinkhorn iterate CPU reference failed: {error}"),
    }
}

/// Fallible tolerance-based Sinkhorn-Knopp iterative balancing in f64 using
/// caller-owned buffers.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_sinkhorn_iterate_f64_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    u_old: &mut Vec<f64>,
) -> Result<u32, String> {
    let m = a.len();
    let n = b.len();
    if k.len() != m * n || tolerance <= 0.0 || !tolerance.is_finite() {
        return Err(format!(
            "sinkhorn_iterate_f64 requires k.len()==a.len()*b.len() and finite positive tolerance, got k={}, m={m}, n={n}, tolerance={tolerance}.",
            k.len()
        ));
    }
    reserve_f64_vec(u, m, "u output")?;
    reserve_f64_vec(v, n, "v output")?;
    reserve_f64_vec(u_old, m, "u convergence scratch")?;

    u.clear();
    v.clear();
    u_old.clear();
    u.resize(m, 1.0_f64);
    v.resize(n, 1.0_f64);

    for iter in 0..max_iterations {
        u_old.clear();
        u_old.extend_from_slice(u);

        // u <- a / (k · v)
        for i in 0..m {
            let mut sum = 0.0_f64;
            for j in 0..n {
                sum += k[i * n + j] * v[j];
            }
            // Guard against division by zero  -  sinkhorn requires k > 0,
            // but defensive callers benefit from a non-NaN result.
            u[i] = if sum == 0.0 { 0.0 } else { a[i] / sum };
        }

        // v <- b / (kᵀ · u)
        for j in 0..n {
            let mut sum = 0.0_f64;
            for i in 0..m {
                sum += k[i * n + j] * u[i];
            }
            v[j] = if sum == 0.0 { 0.0 } else { b[j] / sum };
        }

        // Convergence check on u (Sinkhorn-Knopp stops when one
        // marginal is stable; the other follows by construction).
        let max_delta = u
            .iter()
            .zip(u_old.iter())
            .map(|(new, old)| (new - old).abs())
            .fold(0.0_f64, f64::max);
        if max_delta < tolerance {
            return Ok(iter + 1);
        }
    }
    Ok(max_iterations)
}

crate::scratch::define_reserve_capacity!(
    reserve_f64_vec,
    f64,
    "Sinkhorn iterate f64 CPU oracle"
);

#[cfg(any(test, feature = "cpu-parity"))]
fn max_residual(target: &[f64], sum_at: impl Fn(usize) -> f64) -> f64 {
    target
        .iter()
        .enumerate()
        .map(|(index, expected)| (sum_at(index) - expected).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Clone, Copy)]
enum ResidualAxis {
    Row,
    Column,
}

#[cfg(any(test, feature = "cpu-parity"))]
fn sinkhorn_residual(k: &[f64], u: &[f64], v: &[f64], target: &[f64], axis: ResidualAxis) -> f64 {
    let m = u.len();
    let n = v.len();
    assert_eq!(k.len(), m * n);
    match axis {
        ResidualAxis::Row => {
            assert_eq!(target.len(), m);
            max_residual(target, |i| (0..n).map(|j| u[i] * k[i * n + j] * v[j]).sum())
        }
        ResidualAxis::Column => {
            assert_eq!(target.len(), n);
            max_residual(target, |j| (0..m).map(|i| u[i] * k[i * n + j] * v[j]).sum())
        }
    }
}

/// Compute the row-sum residual `||row_sum(diag(u) · k · diag(v)) - a||_∞`.
/// Useful for testing convergence of [`sinkhorn_iterate_f64`].
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_row_residual(k: &[f64], u: &[f64], v: &[f64], a: &[f64]) -> f64 {
    sinkhorn_residual(k, u, v, a, ResidualAxis::Row)
}

/// Compute the column-sum residual `||col_sum(diag(u) · k · diag(v)) - b||_∞`.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_col_residual(k: &[f64], u: &[f64], v: &[f64], b: &[f64]) -> f64 {
    sinkhorn_residual(k, u, v, b, ResidualAxis::Column)
}
