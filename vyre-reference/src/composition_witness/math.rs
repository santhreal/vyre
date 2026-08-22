//! Sequential mathematical witnesses for semiring linear algebra, polynomial filters, inversion, and hypervectors.

use vyre_spec::Semiring;

/// Sequential mathematical witness for a generalized semiring matrix multiplication: `C = A ⊗ B`.
///
/// Shape: `A` is `m × k`, `B` is `k × n`, output `C` is `m × n`.
///
/// # Panics
/// Panics if input slice dimensions do not match `m * k` and `k * n`.
/// Sequential mathematical witness for a generalized semiring matrix multiplication into caller-owned storage: `C = A ⊗ B`.
pub fn semiring_gemm_witness_into(
    a: &[u32],
    b: &[u32],
    m: usize,
    n: usize,
    k: usize,
    semiring: Semiring,
    c: &mut Vec<u32>,
) {
    assert_eq!(
        a.len(),
        m * k,
        "A dimension mismatch in semiring GEMM witness"
    );
    assert_eq!(
        b.len(),
        k * n,
        "B dimension mismatch in semiring GEMM witness"
    );

    let zero = semiring.identity();
    c.clear();
    c.resize(m * n, zero);

    for i in 0..m {
        for j in 0..n {
            let mut acc = zero;
            for p in 0..k {
                let a_val = a[i * k + p];
                let b_val = b[p * n + j];

                let term = match semiring {
                    Semiring::Real | Semiring::MaxTimes => a_val.wrapping_mul(b_val),
                    Semiring::MinPlus => {
                        if a_val == u32::MAX || b_val == u32::MAX {
                            u32::MAX
                        } else {
                            a_val.saturating_add(b_val)
                        }
                    }
                    Semiring::MaxPlus => a_val.saturating_add(b_val),
                    Semiring::BoolOr | Semiring::Gf2 => a_val & b_val,
                    Semiring::BoolAnd => a_val | b_val,
                    Semiring::Lineage => {
                        if a_val == 0 || b_val == 0 {
                            0
                        } else {
                            a_val | b_val
                        }
                    }
                };

                acc = match semiring {
                    Semiring::Real => acc.wrapping_add(term),
                    Semiring::MinPlus => acc.min(term),
                    Semiring::MaxPlus | Semiring::MaxTimes => acc.max(term),
                    Semiring::BoolOr | Semiring::Lineage => acc | term,
                    Semiring::BoolAnd => acc & term,
                    Semiring::Gf2 => acc ^ term,
                };
            }
            c[i * n + j] = acc;
        }
    }
}

/// Sequential mathematical witness for a generalized semiring matrix multiplication: `C = A ⊗ B`.
#[must_use]
pub fn semiring_gemm_witness(
    a: &[u32],
    b: &[u32],
    m: usize,
    n: usize,
    k: usize,
    semiring: Semiring,
) -> Vec<u32> {
    let mut c = Vec::new();
    semiring_gemm_witness_into(a, b, m, n, k, semiring, &mut c);
    c
}

/// Sequential mathematical witness for u32 matrix multiplication with optional bias into caller-owned storage.
pub fn matmul_u32_witness_into(
    a: &[u32],
    b: &[u32],
    bias: Option<&[u32]>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut Vec<u32>,
) {
    out.clear();
    out.reserve(m * n);
    for row in 0..m {
        for col in 0..n {
            let mut acc = bias.map_or(0, |values| values[col]);
            for kk in 0..k {
                let av = a[row * k + kk];
                let bv = b[kk * n + col];
                acc = acc.wrapping_add(av.wrapping_mul(bv));
            }
            out.push(acc);
        }
    }
}

/// Sequential mathematical witness for u32 matrix multiplication with optional bias.
#[must_use]
pub fn matmul_u32_witness(
    a: &[u32],
    b: &[u32],
    bias: Option<&[u32]>,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<u32> {
    let mut out = Vec::new();
    matmul_u32_witness_into(a, b, bias, m, k, n, &mut out);
    out
}

/// Sequential split-carry addition over little-endian `u32` limbs into caller storage.
pub fn bigint_add_carry_witness_into(
    left: &[u32],
    right: &[u32],
    sums: &mut Vec<u32>,
    carries: &mut Vec<u32>,
) -> Result<(), String> {
    if left.len() != right.len() {
        return Err(format!(
            "bigint limb-count mismatch: left={}, right={}",
            left.len(),
            right.len()
        ));
    }
    if sums.capacity() < left.len() {
        sums.reserve(left.len().saturating_sub(sums.len()));
    }
    if carries.capacity() < left.len() {
        carries.reserve(left.len().saturating_sub(carries.len()));
    }
    sums.clear();
    carries.clear();
    for (&left_val, &right_val) in left.iter().zip(right) {
        let (sum, carry) = left_val.overflowing_add(right_val);
        sums.push(sum);
        carries.push(u32::from(carry));
    }
    Ok(())
}

/// Sequential split-carry addition over little-endian `u32` limbs.
pub fn bigint_add_carry_witness(
    left: &[u32],
    right: &[u32],
) -> Result<(Vec<u32>, Vec<u32>), String> {
    let mut sums = Vec::new();
    let mut carries = Vec::new();
    bigint_add_carry_witness_into(left, right, &mut sums, &mut carries)?;
    Ok((sums, carries))
}

/// Sequentially resolve split carry limbs into the full little-endian sum writing into caller storage.
pub fn resolve_bigint_carry_chain_witness_into(
    partial_sums: &[u32],
    partial_carries: &[u32],
    output: &mut Vec<u32>,
) -> Result<u32, String> {
    if partial_sums.len() != partial_carries.len() {
        return Err(format!(
            "split-carry limb-count mismatch: sums={}, carries={}",
            partial_sums.len(),
            partial_carries.len()
        ));
    }
    if output.capacity() < partial_sums.len() {
        output.reserve(partial_sums.len().saturating_sub(output.len()));
    }
    output.clear();
    let mut carry_in = 0_u32;
    for (&sum, &carry) in partial_sums.iter().zip(partial_carries) {
        let (resolved, overflow) = sum.overflowing_add(carry_in);
        output.push(resolved);
        carry_in = carry | u32::from(overflow);
    }
    Ok(carry_in)
}

/// Sequentially resolve split carry limbs into the full little-endian sum.
pub fn resolve_bigint_carry_chain_witness(
    partial_sums: &[u32],
    partial_carries: &[u32],
) -> Result<(Vec<u32>, u32), String> {
    let mut output = Vec::new();
    let carry_in =
        resolve_bigint_carry_chain_witness_into(partial_sums, partial_carries, &mut output)?;
    Ok((output, carry_in))
}

/// Sequential Chebyshev matrix-polynomial filter witness writing into caller-provided storage.
#[allow(clippy::too_many_arguments)]
pub fn chebyshev_filter_witness_into(
    laplacian: &[f32],
    signal: &[f32],
    coefficients: &[f32],
    n: u32,
    k_steps: u32,
    out: &mut Vec<f32>,
    t_prev: &mut Vec<f32>,
    t_curr: &mut Vec<f32>,
    t_next: &mut Vec<f32>,
) {
    let n_usize = n as usize;
    if out.capacity() < n_usize {
        out.reserve(n_usize.saturating_sub(out.len()));
    }
    if t_prev.capacity() < n_usize {
        t_prev.reserve(n_usize.saturating_sub(t_prev.len()));
    }
    if t_curr.capacity() < n_usize {
        t_curr.reserve(n_usize.saturating_sub(t_curr.len()));
    }
    if t_next.capacity() < n_usize {
        t_next.reserve(n_usize.saturating_sub(t_next.len()));
    }

    out.clear();
    t_prev.clear();
    t_curr.clear();
    t_next.clear();

    let c0 = coefficients.first().copied().unwrap_or(0.0);
    for idx in 0..n_usize {
        out.push(c0 * signal.get(idx).copied().unwrap_or(0.0));
    }
    for idx in 0..n_usize {
        t_prev.push(signal.get(idx).copied().unwrap_or(0.0));
    }

    if k_steps == 0 {
        t_curr.extend_from_slice(t_prev);
        t_next.extend_from_slice(t_prev);
        return;
    }

    t_curr.resize(n_usize, 0.0);
    for i in 0..n_usize {
        let mut sum = 0.0f32;
        for j in 0..n_usize {
            sum += laplacian.get(i * n_usize + j).copied().unwrap_or(0.0) * t_prev[j];
        }
        t_curr[i] = sum;
    }

    let c1 = coefficients.get(1).copied().unwrap_or(0.0);
    for idx in 0..n_usize {
        out[idx] += c1 * t_curr[idx];
    }

    if k_steps < 2 {
        t_next.extend_from_slice(t_curr);
        return;
    }

    t_next.resize(n_usize, 0.0);
    for k in 2..=k_steps as usize {
        for i in 0..n_usize {
            let mut sum = 0.0f32;
            for j in 0..n_usize {
                sum += laplacian.get(i * n_usize + j).copied().unwrap_or(0.0) * t_curr[j];
            }
            t_next[i] = 2.0 * sum - t_prev[i];
        }
        let ck = coefficients.get(k).copied().unwrap_or(0.0);
        for idx in 0..n_usize {
            out[idx] += ck * t_next[idx];
        }
        t_prev.copy_from_slice(t_curr);
        t_curr.copy_from_slice(t_next);
    }
}

/// Fallible sequential Chebyshev matrix-polynomial filter witness writing into caller-provided storage.
#[allow(clippy::too_many_arguments)]
pub fn try_chebyshev_filter_witness_into(
    laplacian: &[f32],
    signal: &[f32],
    coefficients: &[f32],
    n: u32,
    k_steps: u32,
    out: &mut Vec<f32>,
    t_prev: &mut Vec<f32>,
    t_curr: &mut Vec<f32>,
    t_next: &mut Vec<f32>,
) -> Result<(), String> {
    let n_usize = n as usize;
    n_usize.checked_mul(n_usize).ok_or_else(|| {
        format!("chebyshev_filter_witness n={n} overflows dense Laplacian indexing.")
    })?;
    chebyshev_filter_witness_into(
        laplacian,
        signal,
        coefficients,
        n,
        k_steps,
        out,
        t_prev,
        t_curr,
        t_next,
    );
    Ok(())
}

/// Fallible sequential Chebyshev matrix-polynomial filter witness.
pub fn try_chebyshev_filter_witness(
    laplacian: &[f32],
    signal: &[f32],
    coefficients: &[f32],
    n: u32,
    k_steps: u32,
) -> Result<Vec<f32>, String> {
    let mut out = Vec::new();
    try_chebyshev_filter_witness_into(
        laplacian,
        signal,
        coefficients,
        n,
        k_steps,
        &mut out,
        &mut Vec::new(),
        &mut Vec::new(),
        &mut Vec::new(),
    )?;
    Ok(out)
}

/// Sequential Chebyshev matrix-polynomial filter witness.
#[must_use]
pub fn chebyshev_filter_witness(
    laplacian: &[f32],
    signal: &[f32],
    coefficients: &[f32],
    n: u32,
    k_steps: u32,
) -> Vec<f32> {
    try_chebyshev_filter_witness(laplacian, signal, coefficients, n, k_steps).unwrap_or_default()
}

/// Sequential per-block Gauss-Jordan inverse witness without pivoting using caller-provided scratch buffers.
pub fn kfac_block_inverse_witness_into(
    blocks: &[f32],
    num_blocks: u32,
    n: u32,
    out: &mut Vec<f32>,
    mat: &mut Vec<f32>,
    inv: &mut Vec<f32>,
) {
    let n = n as usize;
    let block_cells = n * n;
    let total_cells = (num_blocks as usize) * block_cells;
    assert_eq!(blocks.len(), total_cells);

    if out.capacity() < total_cells {
        out.reserve(total_cells.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(total_cells, 0.0);

    if mat.capacity() < block_cells {
        mat.reserve(block_cells.saturating_sub(mat.len()));
    }
    mat.clear();
    mat.resize(block_cells, 0.0);

    if inv.capacity() < block_cells {
        inv.reserve(block_cells.saturating_sub(inv.len()));
    }
    inv.clear();
    inv.resize(block_cells, 0.0);

    for b in 0..num_blocks as usize {
        let block_offset = b * block_cells;
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                mat[idx] = blocks[block_offset + idx];
                inv[idx] = if i == j { 1.0 } else { 0.0 };
            }
        }
        // Gauss-Jordan
        for i in 0..n {
            let pivot = mat[i * n + i];
            for j in 0..n {
                mat[i * n + j] /= pivot;
                inv[i * n + j] /= pivot;
            }
            for k in 0..n {
                if k != i {
                    let factor = mat[k * n + i];
                    for j in 0..n {
                        mat[k * n + j] -= factor * mat[i * n + j];
                        inv[k * n + j] -= factor * inv[i * n + j];
                    }
                }
            }
        }
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                out[block_offset + idx] = inv[idx];
            }
        }
    }
}

/// Sequential per-block Gauss-Jordan inverse witness without pivoting.
#[must_use]
pub fn kfac_block_inverse_witness(blocks: &[f32], num_blocks: u32, n: u32) -> Vec<f32> {
    let mut out = Vec::new();
    let mut mat = Vec::new();
    let mut inv = Vec::new();
    kfac_block_inverse_witness_into(blocks, num_blocks, n, &mut out, &mut mat, &mut inv);
    out
}
/// Sequential split-conformal threshold witness.
#[must_use]
pub fn conformal_threshold_witness(scores: &[u32], alpha: f64) -> u32 {
    if scores.is_empty() || !(0.0 < alpha && alpha < 1.0) {
        return 0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_unstable();
    let rank = ((1.0 - alpha) * (sorted.len() as f64 + 1.0)).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

/// Sequential conformal rank witness: `k = ⌈(1 - α)(n + 1)⌉` clamped to `[1, n]`.
#[must_use]
pub fn conformal_rank_witness(n: u32, alpha: f64) -> u32 {
    if n == 0 || !(alpha > 0.0 && alpha < 1.0) {
        return 0;
    }
    let raw = (1.0 - alpha) * (n as f64 + 1.0);
    let rank = raw.ceil() as u32;
    rank.clamp(1, n)
}

/// Sequential prediction interval witness: `[y - q_hat, y + q_hat]` with saturation.
#[must_use]
pub fn predict_interval_witness(y: u32, q_hat: u32) -> (u32, u32) {
    let lo = y.saturating_sub(q_hat);
    let hi = y.saturating_add(q_hat);
    (lo, hi)
}

/// Select the lowest-index unpicked candidate having maximum gain.
#[must_use]
pub fn argmax_of_marginals_witness(gains: &[u32], picked: &[u32]) -> (u32, u32) {
    let mut winner = u32::MAX;
    let mut winner_gain = 0_u32;
    for (index, (&gain, &is_picked)) in gains.iter().zip(picked).enumerate() {
        if is_picked == 0 && (winner == u32::MAX || gain > winner_gain) {
            winner = index as u32;
            winner_gain = gain;
        }
    }
    (winner, winner_gain)
}

/// Sequential greedy cache-retention selector writing into caller storage:
/// selects up to `k` items by repeatedly taking the argmax of marginal gains.
pub fn select_retention_set_witness_into(gains: &mut [u32], n: u32, k: u32, picked: &mut Vec<u32>) {
    let count = (n as usize).min(gains.len());
    picked.clear();
    picked.resize(count, 0);
    for _ in 0..k.min(count as u32) {
        let (winner, _) = argmax_of_marginals_witness(gains, picked);
        if winner == u32::MAX || winner as usize >= count {
            break;
        }
        picked[winner as usize] = 1;
        gains[winner as usize] = 0;
    }
}

/// Sequential greedy cache-retention selector: selects up to `k` items by repeatedly taking the argmax of marginal gains.
#[must_use]
pub fn select_retention_set_witness(gains: &mut [u32], n: u32, k: u32) -> Vec<u32> {
    let mut picked = Vec::with_capacity(n as usize);
    select_retention_set_witness_into(gains, n, k, &mut picked);
    picked
}

/// Dense matrix-vector natural-gradient witness.
#[must_use]
pub fn natural_gradient_block_apply_witness(matrix: &[f64], gradient: &[f64], n: u32) -> Vec<f64> {
    let mut output = Vec::new();
    natural_gradient_block_apply_witness_into(matrix, gradient, n, &mut output);
    output
}

/// Dense matrix-vector natural-gradient witness into caller-owned storage.
pub fn natural_gradient_block_apply_witness_into(
    matrix: &[f64],
    gradient: &[f64],
    n: u32,
    output: &mut Vec<f64>,
) {
    let n = n as usize;
    output.clear();
    output.resize(n, 0.0);
    for (row, value) in output.iter_mut().enumerate() {
        *value = (0..n)
            .map(|column| {
                matrix.get(row * n + column).copied().unwrap_or(0.0)
                    * gradient.get(column).copied().unwrap_or(0.0)
            })
            .sum();
    }
}

/// Fallible dense matrix-vector natural-gradient witness into caller-owned storage.
pub fn try_natural_gradient_block_apply_witness_into(
    matrix: &[f64],
    gradient: &[f64],
    n: u32,
    output: &mut Vec<f64>,
) -> Result<(), String> {
    let n_us = n as usize;
    let required_matrix = n_us.checked_mul(n_us).ok_or("matrix cells overflow")?;
    if matrix.len() < required_matrix {
        return Err(format!(
            "matrix too short: expected {required_matrix}, got {}",
            matrix.len()
        ));
    }
    if gradient.len() < n_us {
        return Err(format!(
            "gradient too short: expected {n_us}, got {}",
            gradient.len()
        ));
    }
    if output.capacity() < n_us {
        output.reserve(n_us.saturating_sub(output.len()));
    }
    output.clear();
    output.resize(n_us, 0.0);
    for (row, value) in output.iter_mut().enumerate() {
        *value = (0..n_us)
            .map(|column| matrix[row * n_us + column] * gradient[column])
            .sum();
    }
    Ok(())
}

/// Fallible natural-gradient autotuner step into caller-owned storage.
pub fn try_natural_gradient_autotune_step_witness_into(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
    learning_rate: f64,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    try_natural_gradient_block_apply_witness_into(m_inv_sqrt, grad, n, out)?;
    for value in out.iter_mut() {
        *value *= -learning_rate;
    }
    Ok(())
}

/// Natural-gradient autotuner step into caller-owned storage.
///
/// # Panics
///
/// Panics if matrix dimension `n` overflows `usize` or if slice lengths do not match `n`.
pub fn natural_gradient_autotune_step_witness_into(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
    learning_rate: f64,
    out: &mut Vec<f64>,
) {
    try_natural_gradient_autotune_step_witness_into(m_inv_sqrt, grad, n, learning_rate, out)
        .expect(
            "Fix: provide m_inv_sqrt of length n*n and grad of length n with n*n fitting in usize",
        );
}

/// Natural-gradient autotuner step.
#[must_use]
pub fn natural_gradient_autotune_step_witness(
    m_inv_sqrt: &[f64],
    grad: &[f64],
    n: u32,
    learning_rate: f64,
) -> Vec<f64> {
    let mut out = Vec::new();
    natural_gradient_autotune_step_witness_into(m_inv_sqrt, grad, n, learning_rate, &mut out);
    out
}

/// Fallible identity matrix into caller-owned storage.
pub fn try_identity_matrix_witness_into(n: u32, out: &mut Vec<f64>) -> Result<(), String> {
    let n_us = n as usize;
    let cells = n_us.checked_mul(n_us).ok_or("matrix dimension overflow")?;
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(cells, 0.0);
    for i in 0..n_us {
        out[i * n_us + i] = 1.0;
    }
    Ok(())
}

/// Identity matrix into caller-owned storage.
///
/// # Panics
///
/// Panics if matrix dimension `n * n` overflows `usize`.
pub fn identity_matrix_witness_into(n: u32, out: &mut Vec<f64>) {
    try_identity_matrix_witness_into(n, out)
        .expect("Fix: choose matrix dimension n such that n * n does not overflow usize");
}

/// Identity matrix of size n x n.
#[must_use]
pub fn identity_matrix_witness(n: u32) -> Vec<f64> {
    let mut out = Vec::new();
    identity_matrix_witness_into(n, &mut out);
    out
}

/// Clip each sample gradient by its supplied L2 norm.
/// Clip each sample gradient by its supplied L2 norm into caller storage.
pub fn dp_clip_per_sample_witness_into(
    gradients: &[f64],
    norms: &[f64],
    clip_norm: f64,
    batch: u32,
    dimensions: u32,
    out: &mut Vec<f64>,
) {
    let batch = batch as usize;
    let dimensions = dimensions as usize;
    let total = batch.saturating_mul(dimensions);
    if out.capacity() < total {
        out.reserve(total.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(total, 0.0_f64);
    for sample in 0..batch {
        let norm = norms.get(sample).copied().unwrap_or(0.0);
        let scale = if norm > clip_norm && norm != 0.0 {
            clip_norm / norm
        } else {
            1.0
        };
        for dimension in 0..dimensions {
            let index = sample * dimensions + dimension;
            out[index] = gradients.get(index).copied().unwrap_or(0.0) * scale;
        }
    }
}

/// Clip each sample gradient by its supplied L2 norm.
#[must_use]
pub fn dp_clip_per_sample_witness(
    gradients: &[f64],
    norms: &[f64],
    clip_norm: f64,
    batch: u32,
    dimensions: u32,
) -> Vec<f64> {
    let mut out = Vec::new();
    dp_clip_per_sample_witness_into(gradients, norms, clip_norm, batch, dimensions, &mut out);
    out
}
/// Compact payloads whose parallel flags are nonzero into caller-owned storage.
pub fn stream_compact_witness_into(payloads: &[u32], flags: &[u32], out: &mut Vec<u32>) -> u32 {
    out.clear();
    let count = payloads.len().min(flags.len());
    out.reserve(count);
    for (&payload, &flag) in payloads.iter().zip(flags) {
        if flag != 0 {
            out.push(payload);
        }
    }
    out.len() as u32
}

/// Compact payloads whose parallel flags are nonzero.
#[must_use]
pub fn stream_compact_witness(payloads: &[u32], flags: &[u32]) -> (Vec<u32>, u32) {
    let mut out = Vec::new();
    let live_count = stream_compact_witness_into(payloads, flags, &mut out);
    (out, live_count)
}

/// Monotone lineage-semiring matrix fixpoint witness into caller storage.
///
/// # Panics
///
/// Panics if `n * n * words_per_cell` overflows `usize` or if `state` or `join_rules`
/// lengths do not match the expected word count.
pub fn scallop_join_fixpoint_witness_into(
    state: &[u32],
    join_rules: &[u32],
    n: u32,
    words_per_cell: u32,
    max_iterations: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> u32 {
    let words = n
        .checked_mul(n)
        .and_then(|cells| cells.checked_mul(words_per_cell))
        .and_then(|count| usize::try_from(count).ok())
        .expect(
            "Fix: keep n * n * words_per_cell within usize bounds for lineage matrix allocation",
        );
    assert_eq!(state.len(), words, "complete n*n*w state matrix");
    assert_eq!(join_rules.len(), words, "complete n*n*w join-rule matrix");
    let width = words_per_cell as usize;
    let n = n as usize;
    if current.capacity() < words {
        current.reserve(words.saturating_sub(current.len()));
    }
    current.clear();
    current.extend_from_slice(state);

    if next.capacity() < words {
        next.reserve(words.saturating_sub(next.len()));
    }
    next.clear();
    next.resize(words, 0_u32);

    for iteration in 0..max_iterations {
        next.fill(0);
        for row in 0..n {
            for column in 0..n {
                let destination = (row * n + column) * width;
                for middle in 0..n {
                    let left = (row * n + middle) * width;
                    let right = (middle * n + column) * width;
                    let left_present = current[left..left + width].iter().any(|&word| word != 0);
                    let right_present = join_rules[right..right + width]
                        .iter()
                        .any(|&word| word != 0);
                    if left_present && right_present {
                        for word in 0..width {
                            next[destination + word] |=
                                current[left + word] | join_rules[right + word];
                        }
                    }
                }
            }
        }
        let mut changed = false;
        for (value, derived) in current.iter_mut().zip(next.iter()) {
            let accumulated = *value | *derived;
            changed |= accumulated != *value;
            *value = accumulated;
        }
        if !changed {
            return iteration;
        }
    }
    max_iterations
}

/// Monotone lineage-semiring matrix fixpoint witness.
#[must_use]
pub fn scallop_join_fixpoint_witness(
    state: &[u32],
    join_rules: &[u32],
    n: u32,
    words_per_cell: u32,
    max_iterations: u32,
) -> (Vec<u32>, u32) {
    let mut current = Vec::new();
    let mut next = Vec::new();
    let iters = scallop_join_fixpoint_witness_into(
        state,
        join_rules,
        n,
        words_per_cell,
        max_iterations,
        &mut current,
        &mut next,
    );
    (current, iters)
}

/// Saturating Bellman-Ford edge-list fixpoint witness.
#[must_use]
pub fn bellman_shortest_path_witness(
    sources: &[u32],
    destinations: &[u32],
    weights: &[u32],
    initial: &[u32],
    node_count: u32,
    max_iterations: u32,
) -> (Vec<u32>, u32) {
    let node_count = node_count as usize;
    let mut current = vec![u32::MAX; node_count];
    for (destination, &value) in current.iter_mut().zip(initial) {
        *destination = value;
    }
    let mut next = current.clone();
    let edge_count = sources.len().min(destinations.len()).min(weights.len());
    for iteration in 0..max_iterations {
        for edge in 0..edge_count {
            let source = sources[edge] as usize;
            let destination = destinations[edge] as usize;
            if source >= node_count || destination >= node_count || current[source] == u32::MAX {
                continue;
            }
            next[destination] =
                next[destination].min(current[source].saturating_add(weights[edge]));
        }
        if next == current {
            return (current, iteration);
        }
        current.copy_from_slice(&next);
    }
    (current, max_iterations)
}

/// Group-masked row-bitset transitive closure witness.
#[must_use]
pub fn tensor_scc_witness(
    matrix_rows: &[u32],
    seed: u32,
    group_mask: u32,
    iteration_limit: u32,
) -> u32 {
    let mut active = seed & group_mask;
    for _ in 0..iteration_limit {
        let mut expanded = active;
        for (node, &row) in matrix_rows.iter().take(32).enumerate() {
            if active & (1_u32 << node) != 0 {
                expanded |= row & group_mask;
            }
        }
        if expanded == active {
            break;
        }
        active = expanded;
    }
    active
}

pub use super::math_amg::*;
pub use super::math_analysis::*;
pub use super::math_physics::*;
pub use super::math_quant::*;
pub use super::math_schedule::*;
pub use super::math_tensor::*;
