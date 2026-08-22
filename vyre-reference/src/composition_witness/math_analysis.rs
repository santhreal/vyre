//! Sequential fractional derivatives, Newton-Schulz iteration, differential privacy, and norm witnesses.

/// Fallible sequential Grünwald-Letnikov kernel generator into caller-owned storage.
pub fn try_grunwald_letnikov_kernel_witness_into(
    alpha: f64,
    n: u32,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    out.clear();
    let n = n as usize;
    if n == 0 || !alpha.is_finite() {
        return Ok(());
    }
    out.resize(n, 0.0);
    out[0] = 1.0;
    for k in 1..n {
        out[k] = (1.0 - (alpha + 1.0) / (k as f64)) * out[k - 1];
    }
    Ok(())
}

/// Fallible sequential Grünwald-Letnikov kernel generator.
pub fn try_grunwald_letnikov_kernel_witness(alpha: f64, n: u32) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    try_grunwald_letnikov_kernel_witness_into(alpha, n, &mut out)?;
    Ok(out)
}

/// Sequential Grünwald-Letnikov kernel generator.
///
/// # Panics
///
/// Panics if generating the Grünwald-Letnikov kernel fails.
#[must_use]
pub fn grunwald_letnikov_kernel_witness(alpha: f64, n: u32) -> Vec<f64> {
    try_grunwald_letnikov_kernel_witness(alpha, n)
        .unwrap_or_else(|error| panic!("Grünwald-Letnikov kernel witness failed: {error}"))
}

/// Fallible sequential fractional derivative convolution witness into caller-owned storage.
pub fn try_fractional_derivative_witness_into(
    f: &[f64],
    alpha: f64,
    step: f64,
    kernel: &mut Vec<f64>,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    out.clear();
    kernel.clear();
    if step <= 0.0 || !step.is_finite() || !alpha.is_finite() {
        return Ok(());
    }
    let n = f.len();
    if n == 0 {
        return Ok(());
    }
    let n_u32 = u32::try_from(n).map_err(|_| format!("signal length {n} exceeds u32"))?;
    try_grunwald_letnikov_kernel_witness_into(alpha, n_u32, kernel)?;
    if kernel.len() != n {
        return Ok(());
    }
    let scale = step.powf(-alpha);
    out.reserve(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..=i {
            acc += kernel[k] * f[i - k];
        }
        out.push(acc * scale);
    }
    Ok(())
}

/// Fallible sequential fractional derivative convolution witness.
pub fn try_fractional_derivative_witness(
    f: &[f64],
    alpha: f64,
    step: f64,
) -> Result<Vec<f64>, String> {
    let mut kernel = Vec::new();
    let mut out = Vec::new();
    try_fractional_derivative_witness_into(f, alpha, step, &mut kernel, &mut out)?;
    Ok(out)
}

/// Sequential fractional derivative convolution witness.
///
/// # Panics
///
/// Panics if signal length exceeds `u32::MAX` or if kernel convolution fails.
#[must_use]
pub fn fractional_derivative_witness(f: &[f64], alpha: f64, step: f64) -> Vec<f64> {
    try_fractional_derivative_witness(f, alpha, step)
        .unwrap_or_else(|error| panic!("Fractional derivative witness failed: {error}"))
}

/// Fallible conversion of a Grünwald-Letnikov kernel into caller-owned 16.16 fixed point.
pub fn try_kernel_to_fixed_16_16_witness_into(
    kernel: &[f64],
    step: f64,
    alpha: f64,
    out: &mut Vec<u32>,
) -> Result<(), String> {
    if out.capacity() < kernel.len() {
        out.reserve(kernel.len().saturating_sub(out.len()));
    }
    out.clear();
    if step <= 0.0 || !step.is_finite() || !alpha.is_finite() {
        return Ok(());
    }
    let scale = step.powf(-alpha);
    for &w in kernel {
        let scaled = w * scale * 65536.0;
        out.push(scaled.round() as i64 as u32);
    }
    Ok(())
}

/// Convert a Grünwald-Letnikov kernel into 16.16 fixed point in caller-owned storage.
///
/// # Panics
///
/// Panics if converting the kernel to 16.16 fixed point fails.
pub fn kernel_to_fixed_16_16_witness_into(
    kernel: &[f64],
    step: f64,
    alpha: f64,
    out: &mut Vec<u32>,
) {
    try_kernel_to_fixed_16_16_witness_into(kernel, step, alpha, out)
        .unwrap_or_else(|error| panic!("kernel_to_fixed_16_16 witness failed: {error}"));
}

/// Convert a Grünwald-Letnikov kernel into 16.16 fixed point.
#[must_use]
pub fn kernel_to_fixed_16_16_witness(kernel: &[f64], step: f64, alpha: f64) -> Vec<u32> {
    let mut out = Vec::new();
    kernel_to_fixed_16_16_witness_into(kernel, step, alpha, &mut out);
    out
}

/// Sequential dense matrix multiplication witness into caller-owned storage: `C = A * B`.
pub fn dense_matrix_multiply_witness_into(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    c: &mut Vec<f64>,
) {
    c.clear();
    c.resize(m * n, 0.0);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a.get(i * k + p).copied().unwrap_or(0.0)
                    * b.get(p * n + j).copied().unwrap_or(0.0);
            }
            c[i * n + j] = sum;
        }
    }
}

/// Sequential dense matrix multiplication witness: `C = A * B`.
#[must_use]
pub fn dense_matrix_multiply_witness(
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f64> {
    let mut out = Vec::new();
    dense_matrix_multiply_witness_into(a, b, m, k, n, &mut out);
    out
}

/// Scratch space for sequential Newton-Schulz matrix square root inversion witness.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NewtonSchulzScratchWitness {
    /// Intermediate matrix Y.
    pub y: Vec<f64>,
    /// Intermediate matrix Z.
    pub z: Vec<f64>,
    /// Matrix product Z * Y.
    pub zy: Vec<f64>,
    /// Intermediate matrix 3*I - Z*Y.
    pub three_i_minus_zy: Vec<f64>,
}

impl NewtonSchulzScratchWitness {
    /// Create fresh scratch space.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Sequential Newton-Schulz Y-step witness into caller-owned storage.
pub fn newton_schulz_y_step_witness_into(y_curr: &[f64], yzy: &[f64], out: &mut Vec<f64>) {
    let len = y_curr.len().min(yzy.len());
    out.clear();
    out.reserve(len);
    for i in 0..len {
        out.push(0.5 * (3.0 * y_curr[i] - yzy[i]));
    }
}

/// Sequential Newton-Schulz Y-step witness: `Y_{k+1} = 0.5 * (3 * Y_k - YZY)`.
#[must_use]
pub fn newton_schulz_y_step_witness(y_curr: &[f64], yzy: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    newton_schulz_y_step_witness_into(y_curr, yzy, &mut out);
    out
}

/// Sequential Newton-Schulz inverse square root witness into caller-owned storage.
pub fn newton_schulz_inverse_sqrt_witness_into(
    m: &[f64],
    n: usize,
    iters: u32,
    out: &mut Vec<f64>,
    scratch: &mut NewtonSchulzScratchWitness,
) {
    let cells = n * n;
    let mut norm = 0.0;
    for i in 0..n {
        norm += m.get(i * n + i).copied().unwrap_or(0.0);
    }
    if norm <= 0.0 {
        norm = 1.0;
    }

    scratch.y.clear();
    scratch.y.resize(cells, 0.0);
    for (index, value) in scratch.y.iter_mut().enumerate() {
        *value = m.get(index).copied().unwrap_or(0.0) / norm;
    }

    scratch.z.clear();
    scratch.z.resize(cells, 0.0);
    for i in 0..n {
        scratch.z[i * n + i] = 1.0;
    }

    for _ in 0..iters {
        dense_matrix_multiply_witness_into(&scratch.z, &scratch.y, n, n, n, &mut scratch.zy);
        scratch.three_i_minus_zy.clear();
        scratch.three_i_minus_zy.resize(cells, 0.0);
        for i in 0..n {
            for j in 0..n {
                let id_val = if i == j { 3.0 } else { 0.0 };
                scratch.three_i_minus_zy[i * n + j] = id_val - scratch.zy[i * n + j];
            }
        }
        dense_matrix_multiply_witness_into(
            &scratch.y,
            &scratch.three_i_minus_zy,
            n,
            n,
            n,
            &mut scratch.zy,
        );
        scratch.y.copy_from_slice(&scratch.zy);
        for value in &mut scratch.y {
            *value *= 0.5;
        }
        dense_matrix_multiply_witness_into(
            &scratch.three_i_minus_zy,
            &scratch.z,
            n,
            n,
            n,
            &mut scratch.zy,
        );
        scratch.z.copy_from_slice(&scratch.zy);
        for value in &mut scratch.z {
            *value *= 0.5;
        }
    }
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    out.extend(scratch.z.iter().map(|value| value / norm.sqrt()));
}

/// Sequential Newton-Schulz inverse square root witness for symmetric positive definite matrix `m`.
#[must_use]
pub fn newton_schulz_inverse_sqrt_witness(m: &[f64], n: usize, iters: u32) -> Vec<f64> {
    let mut out = Vec::new();
    let mut scratch = NewtonSchulzScratchWitness::new();
    newton_schulz_inverse_sqrt_witness_into(m, n, iters, &mut out, &mut scratch);
    out
}

/// Sequential Runge-Kutta 4th order step witness into caller-owned storage.
pub fn rk4_step_witness_into(
    y_prev: &[f64],
    k1: &[f64],
    k2: &[f64],
    k3: &[f64],
    k4: &[f64],
    h: f64,
    out: &mut Vec<f64>,
) {
    let n = y_prev.len();
    out.clear();
    out.reserve(n);
    for i in 0..n {
        let y = y_prev[i];
        let k1_v = k1.get(i).copied().unwrap_or(0.0);
        let k2_v = k2.get(i).copied().unwrap_or(0.0);
        let k3_v = k3.get(i).copied().unwrap_or(0.0);
        let k4_v = k4.get(i).copied().unwrap_or(0.0);
        let next = y + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v);
        out.push(next);
    }
}

/// Sequential Runge-Kutta 4th order step witness.
#[must_use]
pub fn rk4_step_witness(
    y_prev: &[f64],
    k1: &[f64],
    k2: &[f64],
    k3: &[f64],
    k4: &[f64],
    h: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    rk4_step_witness_into(y_prev, k1, k2, k3, k4, h, &mut out);
    out
}

/// Sequential score-based diffusion denoising step witness into caller-owned storage.
pub fn score_denoise_step_witness_into(
    x: &[f64],
    score: &[f64],
    noise: &[f64],
    alpha: f64,
    beta: f64,
    sigma: f64,
    out: &mut Vec<f64>,
) {
    let n = x.len();
    out.clear();
    out.reserve(n);
    for i in 0..n {
        let x_v = x[i];
        let s_v = score.get(i).copied().unwrap_or(0.0);
        let z_v = noise.get(i).copied().unwrap_or(0.0);
        let denoised = alpha * x_v + beta * s_v + sigma * z_v;
        out.push(denoised);
    }
}

/// Sequential score-based diffusion denoising step witness.
#[must_use]
pub fn score_denoise_step_witness(
    x: &[f64],
    score: &[f64],
    noise: &[f64],
    alpha: f64,
    beta: f64,
    sigma: f64,
) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let (xv, sv, zv) = (
                x[i],
                score.get(i).copied().unwrap_or(0.0),
                noise.get(i).copied().unwrap_or(0.0),
            );
            alpha * xv + beta * sv + sigma * zv
        })
        .collect()
}

/// Sequential greedy tensor network contraction order witness.
#[must_use]
pub fn greedy_tensor_contract_order_witness(dims: &[u32]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..dims.len()).collect();
    order.sort_by(|&left, &right| dims[right].cmp(&dims[left]).then_with(|| left.cmp(&right)));
    order
}

/// Sequential Bhattacharyya coefficient witness between probability vectors `p` and `q`.
#[must_use]
pub fn bhattacharyya_coefficient_witness(p: &[f64], q: &[f64]) -> f64 {
    p.iter().zip(q).map(|(&a, &b)| (a * b).sqrt()).sum()
}

/// Sequential Fisher-Rao distance witness between probability vectors `p` and `q`.
#[must_use]
pub fn fisher_rao_distance_witness(p: &[f64], q: &[f64]) -> f64 {
    let bc = bhattacharyya_coefficient_witness(p, q);
    2.0 * bc.clamp(0.0, 1.0).acos()
}

/// Sequential Amari alpha-geodesic step witness into caller-owned storage.
pub fn amari_alpha_step_witness_into(p: &[f64], q: &[f64], alpha: f64, t: f64, out: &mut Vec<f64>) {
    let t = t.clamp(0.0, 1.0);
    let s = 1.0 - t;
    let n = p.len().min(q.len());
    out.clear();
    out.reserve(n);
    p.iter().zip(q.iter()).for_each(|(&pi, &qi)| {
        if (alpha - 1.0).abs() < 1e-12 {
            out.push(pi.powf(t) * qi.powf(s));
        } else if (alpha + 1.0).abs() < 1e-12 {
            out.push(t * pi + s * qi);
        } else if alpha.abs() < 1e-12 {
            let sp = pi.max(0.0).sqrt();
            let sq = qi.max(0.0).sqrt();
            let blended = t * sp + s * sq;
            out.push(blended * blended);
        } else {
            let beta = (1.0 - alpha) / 2.0;
            let blended = t * pi.max(0.0).powf(beta) + s * qi.max(0.0).powf(beta);
            out.push(blended.powf(1.0 / beta));
        }
    });
}

/// Sequential Amari alpha-geodesic step witness.
#[must_use]
pub fn amari_alpha_step_witness(p: &[f64], q: &[f64], alpha: f64, t: f64) -> Vec<f64> {
    let mut out = Vec::new();
    amari_alpha_step_witness_into(p, q, alpha, t, &mut out);
    out
}

/// Sequential Hensel lifting step witness: `x_{k+1} = x - f(x) * (f'(x))^{-1}`.
#[must_use]
pub fn hensel_lift_step_witness(x: f64, f_x: f64, inv_f_prime: f64) -> f64 {
    x - f_x * inv_f_prime
}

/// Sequential positive semi-definite diagonal positivity witness.
#[must_use]
pub fn is_psd_matrix_witness(matrix: &[f64], n: u32) -> bool {
    let n = n as usize;
    for i in 0..n {
        if matrix.get(i * n + i).copied().unwrap_or(0.0) < 0.0 {
            return false;
        }
    }
    true
}

/// Sequential Modified Gram-Schmidt orthogonalization witness into caller-owned storage.
pub fn modified_gram_schmidt_witness_into(y: &[f64], m: u32, l: u32, q: &mut Vec<f64>) {
    let (m, l) = (m as usize, l as usize);
    let mut cols: Vec<Vec<f64>> = (0..l)
        .map(|j| {
            (0..m)
                .map(|i| y.get(i * l + j).copied().unwrap_or(0.0))
                .collect()
        })
        .collect();
    for j in 0..l {
        let norm = cols[j].iter().map(|&v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            cols[j].iter_mut().for_each(|v| *v /= norm);
        }
        for k in (j + 1)..l {
            let dot: f64 = (0..m).map(|i| cols[j][i] * cols[k][i]).sum();
            for i in 0..m {
                cols[k][i] -= dot * cols[j][i];
            }
        }
    }
    q.clear();
    q.reserve(m * l);
    for i in 0..m {
        for j in 0..l {
            q.push(cols[j][i]);
        }
    }
}

/// Sequential Modified Gram-Schmidt orthogonalization witness for `m x l` column matrix `y`.
#[must_use]
pub fn modified_gram_schmidt_witness(y: &[f64], m: u32, l: u32) -> Vec<f64> {
    let mut q = Vec::new();
    modified_gram_schmidt_witness_into(y, m, l, &mut q);
    q
}

/// Fallible sequential Gaussian Rényi Differential Privacy step witness into caller-owned storage.
pub fn try_gaussian_rdp_step_witness_into(
    alpha: &[f64],
    sigma_squared: &[f64],
    out: &mut Vec<f64>,
) -> Result<(), String> {
    let n = alpha.len().min(sigma_squared.len());
    if out.capacity() < n {
        out.reserve(n.saturating_sub(out.len()));
    }
    out.clear();
    for i in 0..n {
        let a = alpha[i];
        let s2 = sigma_squared[i];
        out.push(a / (2.0 * s2));
    }
    Ok(())
}

/// Sequential Gaussian Rényi Differential Privacy step witness into caller-owned storage.
pub fn gaussian_rdp_step_witness_into(alpha: &[f64], sigma_squared: &[f64], out: &mut Vec<f64>) {
    let _ = try_gaussian_rdp_step_witness_into(alpha, sigma_squared, out);
}

/// Sequential Gaussian Rényi Differential Privacy step witness.
#[must_use]
pub fn gaussian_rdp_step_witness(alpha: &[f64], sigma_squared: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    gaussian_rdp_step_witness_into(alpha, sigma_squared, &mut out);
    out
}

/// Convert RDP(α) to (ε, δ)-DP via Mironov's standard inequality.
#[must_use]
pub fn rdp_to_dp_witness(rdp: f64, alpha: f64, delta: f64) -> f64 {
    if alpha <= 1.0 || !(delta > 0.0 && delta < 1.0) {
        return f64::INFINITY;
    }
    rdp + (1.0 / delta).ln() / (alpha - 1.0)
}

/// Convert RDP to epsilon for private telemetry accounting witness.
#[must_use]
pub fn privacy_epsilon_from_rdp_witness(rdp: f64, alpha: f64, delta: f64) -> f64 {
    rdp_to_dp_witness(rdp, alpha, delta)
}

/// Sequential RMS-normalized linear projection layer witness.
#[must_use]
pub fn rms_norm_linear_witness(
    input: &[f32],
    normalized: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_dim: u32,
    in_dim: u32,
    n: u32,
    eps: f32,
) -> Vec<f32> {
    assert_eq!(
        normalized.len(),
        n as usize,
        "rms_norm_linear_witness must receive exactly n normalized values: got {} vs {}",
        normalized.len(),
        n
    );
    let inv_scale =
        1.0_f32 / ((normalized.iter().map(|&v| v * v).sum::<f32>() / (n as f32)) + eps).sqrt();
    let (out_dim, in_dim) = (out_dim as usize, in_dim as usize);
    (0..out_dim)
        .map(|j| {
            let b = bias.get(j).copied().unwrap_or(0.0);
            let mut dot = 0.0_f32;
            for k in 0..in_dim {
                let in_val = input.get(k).copied().unwrap_or(0.0);
                let norm_val = in_val * inv_scale;
                let w_val = weights.get(k * out_dim + j).copied().unwrap_or(0.0);
                dot += norm_val * w_val;
            }
            dot + b
        })
        .collect()
}

/// Sequential zero-padded 3x3 im2col patch reshape witness writing into caller storage.
pub fn im2col_3x3_witness_into(input: &[f32], h: usize, w: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(h * w * 9, 0.0);
    for y in 0..h {
        for x in 0..w {
            let flat = y * w + x;
            for ky in 0..3usize {
                for kx in 0..3usize {
                    let ny = (y as i32) + (ky as i32) - 1;
                    let nx = (x as i32) + (kx as i32) - 1;
                    let value = if ny < 0 || ny >= h as i32 || nx < 0 || nx >= w as i32 {
                        0.0
                    } else {
                        input[(ny as usize) * w + (nx as usize)]
                    };
                    out[flat * 9 + ky * 3 + kx] = value;
                }
            }
        }
    }
}

/// Sequential zero-padded 3x3 im2col patch reshape witness.
#[must_use]
pub fn im2col_3x3_witness(input: &[f32], h: usize, w: usize) -> Vec<f32> {
    let mut out = Vec::new();
    im2col_3x3_witness_into(input, h, w, &mut out);
    out
}
