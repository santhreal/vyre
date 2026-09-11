//! Sequential mathematical witnesses for sheaf diffusion, spectral gap, and sheaf clustering.

/// Apply one diagonal sheaf-diffusion step in scalar arithmetic.
#[must_use]
pub fn sheaf_diffusion_step_witness(
    stalks: &[f64],
    restriction_diagonal: &[f64],
    damping: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    sheaf_diffusion_step_witness_into(stalks, restriction_diagonal, damping, &mut out);
    out
}

/// Apply one diagonal sheaf-diffusion step into caller-owned storage.
pub fn sheaf_diffusion_step_witness_into(
    stalks: &[f64],
    restriction_diagonal: &[f64],
    damping: f64,
    out: &mut Vec<f64>,
) {
    out.clear();
    out.extend(
        stalks
            .iter()
            .zip(restriction_diagonal)
            .map(|(&stalk, &restriction)| stalk - damping * restriction * stalk),
    );
}

/// Iterate diagonal sheaf diffusion into caller-owned ping-pong storage.
pub fn sheaf_diffusion_equilibrium_witness_into(
    initial_stalks: &[f64],
    restriction_diagonal: &[f64],
    damping: f64,
    tolerance: f64,
    max_iterations: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> u32 {
    out.clear();
    out.extend_from_slice(initial_stalks);
    for iteration in 0..max_iterations {
        sheaf_diffusion_step_witness_into(out, restriction_diagonal, damping, scratch);
        let max_change = scratch
            .iter()
            .zip(out.iter())
            .map(|(next, current)| (next - current).abs())
            .fold(0.0_f64, f64::max);
        std::mem::swap(out, scratch);
        if max_change < tolerance {
            return iteration + 1;
        }
    }
    max_iterations
}

/// Mark stalks whose diffusion displacement exceeds the declared threshold.
pub fn sheaf_fusion_incompatible_witness_into(
    initial_stalks: &[f64],
    diffused_stalks: &[f64],
    divergence_threshold: f64,
    out: &mut Vec<u32>,
) {
    out.clear();
    out.extend(
        initial_stalks
            .iter()
            .zip(diffused_stalks)
            .map(|(&initial, &diffused)| {
                u32::from((initial - diffused).abs() > divergence_threshold)
            }),
    );
}

/// Allocate incompatibility flags for one sheaf diffusion result.
#[must_use]
pub fn sheaf_fusion_incompatible_witness(
    initial_stalks: &[f64],
    diffused_stalks: &[f64],
    divergence_threshold: f64,
) -> Vec<u32> {
    let mut out = Vec::new();
    sheaf_fusion_incompatible_witness_into(
        initial_stalks,
        diffused_stalks,
        divergence_threshold,
        &mut out,
    );
    out
}

/// Compute the dominant eigenvalue and eigenvector of a diagonal sheaf Laplacian into caller storage.
///
/// Returns the dominant eigenvalue `max_i r[i]`. The eigenvector `v_out` is resized to `n` and
/// set to zero except `v_out[max_idx] = 1.0`, where `max_idx` is the index of the first maximum element.
/// If `restriction_diag` is empty, returns `0.0` and clears `v_out`.
pub fn sheaf_dominant_spectrum_witness_into(
    restriction_diag: &[f64],
    _iterations: u32,
    v_out: &mut Vec<f64>,
) -> f64 {
    v_out.clear();
    let n = restriction_diag.len();
    if n == 0 {
        return 0.0;
    }
    if v_out.capacity() < n {
        v_out.reserve(n.saturating_sub(v_out.len()));
    }
    v_out.resize(n, 0.0);
    let mut max_val = 0.0_f64;
    let mut max_idx = 0;
    for (i, &r) in restriction_diag.iter().enumerate() {
        if r > max_val {
            max_val = r;
            max_idx = i;
        }
    }
    v_out[max_idx] = 1.0;
    max_val
}

/// Compute the dominant eigenvalue and eigenvector of a diagonal sheaf Laplacian.
#[must_use]
pub fn sheaf_dominant_spectrum_witness(
    restriction_diag: &[f64],
    iterations: u32,
) -> (f64, Vec<f64>) {
    let mut v = Vec::with_capacity(restriction_diag.len());
    let lambda = sheaf_dominant_spectrum_witness_into(restriction_diag, iterations, &mut v);
    (lambda, v)
}

/// Compute the spectral gap signal in `[0, 1]` into caller eigenvector scratch.
pub fn sheaf_spectral_gap_witness_into(
    restriction_diag: &[f64],
    iterations: u32,
    v_scratch: &mut Vec<f64>,
) -> f64 {
    let lambda = sheaf_dominant_spectrum_witness_into(restriction_diag, iterations, v_scratch);
    let max_diag = restriction_diag.iter().cloned().fold(0.0_f64, f64::max);
    if max_diag <= 1e-20 {
        0.0
    } else {
        (lambda / max_diag).clamp(0.0, 1.0)
    }
}

/// Compute the spectral gap signal in `[0, 1]` derived from the dominant eigenvalue.
#[must_use]
pub fn sheaf_spectral_gap_witness(restriction_diag: &[f64], iterations: u32) -> f64 {
    let mut scratch = Vec::with_capacity(restriction_diag.len());
    sheaf_spectral_gap_witness_into(restriction_diag, iterations, &mut scratch)
}

/// Derive a suggested cluster count from the principal eigenvector sign pattern.
///
/// Items whose eigenvector entry has the same sign belong in the same cluster;
/// flips between consecutive items suggest cluster boundaries.
/// Returns the count of distinct sign runs (>= 1 for non-empty eigenvector, 0 for empty).
#[must_use]
pub fn sheaf_suggested_cluster_count_witness(eigenvector: &[f64]) -> u32 {
    if eigenvector.is_empty() {
        return 0;
    }
    let mut count: u32 = 1;
    let mut last_sign = eigenvector[0].signum();
    for &x in eigenvector.iter().skip(1) {
        let sign = x.signum();
        if sign != 0.0 && sign != last_sign && last_sign != 0.0 {
            count = count.saturating_add(1);
            last_sign = sign;
        } else if last_sign == 0.0 && sign != 0.0 {
            last_sign = sign;
        }
    }
    count
}
