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

/// One tensor-train matrix-vector contraction step writing into caller storage.
pub fn try_tensor_train_contract_step_witness_into(
    accumulator: &[f64],
    core: &[f64],
    previous_rank: u32,
    next_rank: u32,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    if previous_rank == 0 || next_rank == 0 {
        return Err(format!(
            "tt_contract_step CPU oracle requires non-zero ranks, got r_prev={previous_rank}, r_next={next_rank}."
        ));
    }
    let previous_rank = usize::try_from(previous_rank)
        .map_err(|_| format!("TT step r_prev={previous_rank} does not fit usize."))?;
    let next_rank = usize::try_from(next_rank)
        .map_err(|_| format!("TT step r_next={next_rank} does not fit usize."))?;
    let _cells = previous_rank
        .checked_mul(next_rank)
        .ok_or_else(|| "TT step core-slice shape overflows usize.".to_string())?;
    if out.capacity() < next_rank {
        out.reserve(next_rank.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(next_rank, 0.0);
    for b in 0..next_rank {
        let mut acc = 0.0;
        for a in 0..previous_rank {
            let lhs = accumulator.get(a).copied().unwrap_or(0.0);
            let rhs = core.get(a * next_rank + b).copied().unwrap_or(0.0);
            acc += lhs * rhs;
        }
        out[b] = acc;
    }
    Ok(())
}

/// Infallible tensor-train contraction into caller storage.
///
/// # Panics
///
/// Panics if ranks are zero, do not fit `usize`, or if `previous_rank * next_rank` overflows `usize`.
pub fn tensor_train_contract_step_witness_into(
    accumulator: &[f64],
    core: &[f64],
    previous_rank: u32,
    next_rank: u32,
    out: &mut Vec<f64>,
) {
    try_tensor_train_contract_step_witness_into(accumulator, core, previous_rank, next_rank, out)
        .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - tt_contract_step_cpu_into failed: invalid TT contraction shape");
}

/// Fallible tensor-train contraction witness.
pub fn try_tensor_train_contract_step_witness(
    accumulator: &[f64],
    core: &[f64],
    previous_rank: u32,
    next_rank: u32,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    try_tensor_train_contract_step_witness_into(
        accumulator,
        core,
        previous_rank,
        next_rank,
        &mut out,
    )?;
    Ok(out)
}

/// Infallible tensor-train contraction witness.
#[must_use]
pub fn tensor_train_contract_step_witness(
    accumulator: &[f64],
    core: &[f64],
    previous_rank: u32,
    next_rank: u32,
) -> Vec<f64> {
    let mut out = Vec::new();
    tensor_train_contract_step_witness_into(accumulator, core, previous_rank, next_rank, &mut out);
    out
}

/// Fallible full-chain contraction using caller-owned accumulators.
pub fn try_tensor_train_full_chain_witness_into(
    cores: &[Vec<f64>],
    ranks: &[u32],
    mode_dims: &[u32],
    indices: &[u32],
    acc: &mut Vec<f64>,
    next: &mut Vec<f64>,
) -> Result<f64, String> {
    let n = cores.len();
    if n == 0 || ranks.first().copied().unwrap_or(0) != 1 || ranks.get(n).copied().unwrap_or(0) != 1
    {
        return Ok(0.0);
    }

    if acc.capacity() < 1 {
        acc.reserve(1_usize.saturating_sub(acc.len()));
    }
    acc.clear();
    acc.push(1.0);
    for k in 0..n {
        let r_p = ranks.get(k).copied().unwrap_or(0);
        let r_n = ranks.get(k + 1).copied().unwrap_or(0);
        let nk = mode_dims.get(k).copied().unwrap_or(0);
        let i = indices.get(k).copied().unwrap_or(0);
        if r_p == 0 || r_n == 0 || nk == 0 || i >= nk {
            return Ok(0.0);
        }

        let r_n_usize = usize::try_from(r_n)
            .map_err(|_| format!("TT chain rank r_next={r_n} does not fit usize."))?;
        if next.capacity() < r_n_usize {
            next.reserve(r_n_usize.saturating_sub(next.len()));
        }
        next.clear();
        next.resize(r_n_usize, 0.0);
        for b in 0..r_n_usize {
            let mut value = 0.0;
            for a in 0..r_p as usize {
                let lhs = acc.get(a).copied().unwrap_or(0.0);
                let idx = ((a as u32 * nk + i) * r_n + b as u32) as usize;
                let rhs = cores[k].get(idx).copied().unwrap_or(0.0);
                value += lhs * rhs;
            }
            next[b] = value;
        }
        std::mem::swap(acc, next);
    }
    Ok(acc.first().copied().unwrap_or(0.0))
}

/// Infallible full-chain contraction into caller-owned accumulators.
///
/// # Panics
///
/// Panics if rank arrays, mode dimensions, or core tensors are inconsistent or invalid.
pub fn tensor_train_full_chain_witness_into(
    cores: &[Vec<f64>],
    ranks: &[u32],
    mode_dims: &[u32],
    indices: &[u32],
    acc: &mut Vec<f64>,
    next: &mut Vec<f64>,
) -> f64 {
    try_tensor_train_full_chain_witness_into(cores, ranks, mode_dims, indices, acc, next)
        .expect("Fix: scratch allocation must succeed for declared sizes; shrink test fixture or return Err - tt_full_chain_cpu_with_scratch failed: scratch allocation failed")
}

/// Fallible full-chain contraction witness.
pub fn try_tensor_train_full_chain_witness(
    cores: &[Vec<f64>],
    ranks: &[u32],
    mode_dims: &[u32],
    indices: &[u32],
) -> Result<f64, String> {
    let mut acc = Vec::with_capacity(1);
    let mut next = Vec::with_capacity(1);
    try_tensor_train_full_chain_witness_into(cores, ranks, mode_dims, indices, &mut acc, &mut next)
}

/// Sequential full Tensor-Train tensor contraction chain witness.
#[must_use]
pub fn tensor_train_full_chain_witness(
    cores: &[Vec<f64>],
    ranks: &[u32],
    mode_dims: &[u32],
    indices: &[u32],
) -> f64 {
    let mut acc = Vec::with_capacity(1);
    let mut next = Vec::with_capacity(1);
    tensor_train_full_chain_witness_into(cores, ranks, mode_dims, indices, &mut acc, &mut next)
}

/// Fallible Tensor-Train chain fusion pressure calculation using caller-owned scratch accumulators.
pub fn try_tensor_train_fusion_pressure_witness_with_scratch(
    shared_buffer_ranks: &[u32],
    acc: &mut Vec<f64>,
    next: &mut Vec<f64>,
) -> Result<f64, String> {
    if shared_buffer_ranks.is_empty() {
        return Ok(0.0);
    }
    if acc.capacity() < 1 {
        acc.reserve(1_usize.saturating_sub(acc.len()));
    }
    acc.clear();
    acc.push(1.0);

    for &r_next in shared_buffer_ranks {
        let r_prev = acc.len() as u32;
        if r_next == 0 {
            continue;
        }
        let cells = (r_prev as usize)
            .checked_mul(r_next as usize)
            .ok_or("TT core size overflow")?;
        let core_slice = vec![1.0; cells];
        try_tensor_train_contract_step_witness_into(acc, &core_slice, r_prev, r_next, next)?;
        std::mem::swap(acc, next);
    }

    let r_last = acc.len() as u32;
    let core_last = vec![1.0; r_last as usize];
    try_tensor_train_contract_step_witness_into(acc, &core_last, r_last, 1, next)?;
    Ok(next.first().copied().unwrap_or(0.0))
}

/// Tensor-Train chain fusion pressure calculation using caller-owned scratch accumulators.
///
/// # Panics
///
/// Panics if intermediate tensor train core shapes overflow `usize`.
pub fn tensor_train_fusion_pressure_witness_with_scratch(
    shared_buffer_ranks: &[u32],
    acc: &mut Vec<f64>,
    next: &mut Vec<f64>,
) -> f64 {
    try_tensor_train_fusion_pressure_witness_with_scratch(shared_buffer_ranks, acc, next)
        .expect("Fix: ensure adjacent tensor train rank products fit within usize bounds")
}

/// Tensor-Train chain fusion pressure calculation.
#[must_use]
pub fn tensor_train_fusion_pressure_witness(shared_buffer_ranks: &[u32]) -> f64 {
    let mut acc = Vec::new();
    let mut next = Vec::new();
    tensor_train_fusion_pressure_witness_with_scratch(shared_buffer_ranks, &mut acc, &mut next)
}

/// Decide whether to fuse a chain based on its TT fusion pressure.
///
/// A chain should be fused if its total intermediate volume (pressure)
/// is below the threshold relative to the number of regions.
#[must_use]
pub fn should_fuse_chain_witness(shared_buffer_ranks: &[u32], threshold_per_link: f64) -> bool {
    if shared_buffer_ranks.is_empty() {
        return false;
    }
    let n = shared_buffer_ranks.len() as f64;
    let log_sum = shared_buffer_ranks
        .iter()
        .copied()
        .filter(|&rank| rank != 0)
        .map(|rank| (rank as f64).ln())
        .sum::<f64>();
    let avg_log_rank = log_sum / n;
    avg_log_rank <= threshold_per_link.ln()
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

/// Compute the residual `b - A * x` into caller storage.
pub fn amg_residual_witness_into(
    matrix: &[f64],
    rhs: &[f64],
    solution: &[f64],
    n: u32,
    out: &mut Vec<f64>,
) {
    let n = n as usize;
    if out.capacity() < n {
        out.reserve(n.saturating_sub(out.len()));
    }
    out.clear();
    for row in 0..n {
        let mut ax = 0.0;
        for col in 0..n {
            ax += matrix[row * n + col] * solution[col];
        }
        out.push(rhs[row] - ax);
    }
}

/// Caller-owned scratch storage for one algebraic-multigrid (AMG) V-cycle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AmgVcycleScratchWitness {
    /// Fine-level iterate scratch.
    pub fine: Vec<f64>,
    /// Fine-level residual scratch (`r = b - A*x`).
    pub residual: Vec<f64>,
    /// Coarse-level right-hand side scratch (`r_c = R*r`).
    pub coarse_rhs: Vec<f64>,
    /// Coarse-level iterate scratch.
    pub coarse: Vec<f64>,
    /// Coarse-level next iterate scratch.
    pub coarse_next: Vec<f64>,
}

impl AmgVcycleScratchWitness {
    /// Create empty AMG V-cycle scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-reserve capacity for `fine_count` fine nodes and `coarse_count` coarse nodes.
    pub fn reserve(&mut self, fine_count: usize, coarse_count: usize) {
        if self.fine.capacity() < fine_count {
            self.fine
                .reserve(fine_count.saturating_sub(self.fine.len()));
        }
        if self.residual.capacity() < fine_count {
            self.residual
                .reserve(fine_count.saturating_sub(self.residual.len()));
        }
        if self.coarse_rhs.capacity() < coarse_count {
            self.coarse_rhs
                .reserve(coarse_count.saturating_sub(self.coarse_rhs.len()));
        }
        if self.coarse.capacity() < coarse_count {
            self.coarse
                .reserve(coarse_count.saturating_sub(self.coarse.len()));
        }
        if self.coarse_next.capacity() < coarse_count {
            self.coarse_next
                .reserve(coarse_count.saturating_sub(self.coarse_next.len()));
        }
    }
}

/// Caller-owned scratch storage for algebraic-multigrid (AMG) iterative solver to tolerance.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AmgSolveScratchWitness {
    /// Work vectors for one AMG V-cycle.
    pub v_cycle: AmgVcycleScratchWitness,
    /// Secondary solution scratch for tolerance solve iterations.
    pub next_iterate: Vec<f64>,
}

impl AmgSolveScratchWitness {
    /// Create empty AMG solver scratch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Pre-reserve capacity for `fine_count` fine nodes and `coarse_count` coarse nodes.
    pub fn reserve(&mut self, fine_count: usize, coarse_count: usize) {
        self.v_cycle.reserve(fine_count, coarse_count);
        if self.next_iterate.capacity() < fine_count {
            self.next_iterate
                .reserve(fine_count.saturating_sub(self.next_iterate.len()));
        }
    }
}
/// Fallible two-level algebraic-multigrid V-cycle writing into caller storage using reusable scratch.
#[allow(clippy::too_many_arguments)]
pub fn try_amg_v_cycle_witness_with_scratch_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    scratch: &mut AmgVcycleScratchWitness,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    let nf = fine_count as usize;
    let nc = coarse_count as usize;
    if fine_matrix.len() < nf * nf {
        return Err(format!(
            "buffer `a` is too short: expected {}, got {}",
            nf * nf,
            fine_matrix.len()
        ));
    }
    if fine_rhs.len() < nf {
        return Err(format!(
            "buffer `b` is too short: expected {}, got {}",
            nf,
            fine_rhs.len()
        ));
    }
    if initial.len() < nf {
        return Err(format!(
            "buffer `x` is too short: expected {}, got {}",
            nf,
            initial.len()
        ));
    }
    if restriction.len() < nc * nf {
        return Err(format!(
            "buffer `r_mat` is too short: expected {}, got {}",
            nc * nf,
            restriction.len()
        ));
    }
    if prolongation.len() < nf * nc {
        return Err(format!(
            "buffer `p_mat` is too short: expected {}, got {}",
            nf * nc,
            prolongation.len()
        ));
    }
    if coarse_matrix.len() < nc * nc {
        return Err(format!(
            "buffer `a_c` is too short: expected {}, got {}",
            nc * nc,
            coarse_matrix.len()
        ));
    }

    if out.capacity() < nf {
        out.reserve(nf.saturating_sub(out.len()));
    }
    out.clear();

    if nf == 0 {
        return Ok(());
    }

    let jacobi = |matrix: &[f64], rhs: &[f64], current: &[f64], n: usize, dest: &mut Vec<f64>| {
        dest.clear();
        if dest.capacity() < n {
            dest.reserve(n.saturating_sub(dest.len()));
        }
        for row in 0..n {
            let off_diagonal = (0..n)
                .filter(|&column| column != row)
                .map(|column| matrix[row * n + column] * current[column])
                .sum::<f64>();
            let target = (rhs[row] - off_diagonal) / matrix[row * n + row];
            dest.push(current[row] + omega * (target - current[row]));
        }
    };

    jacobi(fine_matrix, fine_rhs, initial, nf, &mut scratch.fine);

    scratch.residual.clear();
    if scratch.residual.capacity() < nf {
        scratch
            .residual
            .reserve(nf.saturating_sub(scratch.residual.len()));
    }
    for row in 0..nf {
        let ax_row = (0..nf)
            .map(|col| fine_matrix[row * nf + col] * scratch.fine[col])
            .sum::<f64>();
        scratch.residual.push(fine_rhs[row] - ax_row);
    }

    scratch.coarse_rhs.clear();
    if scratch.coarse_rhs.capacity() < nc {
        scratch
            .coarse_rhs
            .reserve(nc.saturating_sub(scratch.coarse_rhs.len()));
    }
    for row in 0..nc {
        let val = (0..nf)
            .map(|col| restriction[row * nf + col] * scratch.residual[col])
            .sum::<f64>();
        scratch.coarse_rhs.push(val);
    }

    scratch.coarse.clear();
    scratch.coarse.resize(nc, 0.0_f64);
    for _ in 0..4 {
        jacobi(
            coarse_matrix,
            &scratch.coarse_rhs,
            &scratch.coarse,
            nc,
            &mut scratch.coarse_next,
        );
        std::mem::swap(&mut scratch.coarse, &mut scratch.coarse_next);
    }

    for row in 0..nf {
        scratch.fine[row] += (0..nc)
            .map(|col| prolongation[row * nc + col] * scratch.coarse[col])
            .sum::<f64>();
    }

    jacobi(fine_matrix, fine_rhs, &scratch.fine, nf, out);
    Ok(())
}

/// Two-level algebraic-multigrid V-cycle writing into caller storage with reusable scratch.
///
/// # Panics
///
/// Panics if matrix or vector buffer shapes do not match `fine_count` or `coarse_count`.
#[allow(clippy::too_many_arguments)]
pub fn amg_v_cycle_witness_with_scratch_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    scratch: &mut AmgVcycleScratchWitness,
    out: &mut Vec<f64>,
) {
    try_amg_v_cycle_witness_with_scratch_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        scratch,
        out,
    )
    .expect(
        "Fix: provide fine and coarse matrix and transfer operator buffers matching dimensions",
    );
}

/// Fallible two-level algebraic-multigrid V-cycle writing into caller storage.
#[allow(clippy::too_many_arguments)]
pub fn try_amg_v_cycle_witness_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    let mut scratch = AmgVcycleScratchWitness::new();
    try_amg_v_cycle_witness_with_scratch_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        &mut scratch,
        out,
    )
}

/// Two-level algebraic-multigrid V-cycle writing into caller storage.
///
/// # Panics
///
/// Panics if matrix or vector buffer shapes do not match `fine_count` or `coarse_count`.
#[allow(clippy::too_many_arguments)]
pub fn amg_v_cycle_witness_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    out: &mut Vec<f64>,
) {
    try_amg_v_cycle_witness_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        out,
    )
    .expect(
        "Fix: provide fine and coarse matrix and transfer operator buffers matching dimensions",
    );
}

/// One two-level algebraic-multigrid V-cycle in scalar floating-point arithmetic.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn amg_v_cycle_witness(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(fine_count as usize);
    amg_v_cycle_witness_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        &mut out,
    );
    out
}

/// Fallible iterative AMG V-cycle solver to tolerance into caller-owned storage using reusable scratch.
#[allow(clippy::too_many_arguments)]
pub fn try_amg_solve_to_tolerance_witness_with_scratch_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    tolerance: f64,
    max_cycles: u32,
    scratch: &mut AmgSolveScratchWitness,
    out: &mut Vec<f64>,
) -> Result<u32, String> {
    let nf = fine_count as usize;
    let nc = coarse_count as usize;
    if fine_matrix.len() < nf * nf {
        return Err(format!(
            "buffer `a` is too short: expected {}, got {}",
            nf * nf,
            fine_matrix.len()
        ));
    }
    if fine_rhs.len() < nf {
        return Err(format!(
            "buffer `b` is too short: expected {}, got {}",
            nf,
            fine_rhs.len()
        ));
    }
    if initial.len() < nf {
        return Err(format!(
            "buffer `x` is too short: expected {}, got {}",
            nf,
            initial.len()
        ));
    }
    if restriction.len() < nc * nf {
        return Err(format!(
            "buffer `r_mat` is too short: expected {}, got {}",
            nc * nf,
            restriction.len()
        ));
    }
    if prolongation.len() < nf * nc {
        return Err(format!(
            "buffer `p_mat` is too short: expected {}, got {}",
            nf * nc,
            prolongation.len()
        ));
    }
    if coarse_matrix.len() < nc * nc {
        return Err(format!(
            "buffer `a_c` is too short: expected {}, got {}",
            nc * nc,
            coarse_matrix.len()
        ));
    }
    if tolerance <= 0.0 || !tolerance.is_finite() {
        return Err(format!(
            "tolerance must be finite positive, got {tolerance}"
        ));
    }

    if nf == 0 {
        out.clear();
        scratch.next_iterate.clear();
        return Ok(0);
    }

    if out.capacity() < nf {
        out.reserve(nf.saturating_sub(out.len()));
    }
    if scratch.next_iterate.capacity() < nf {
        scratch
            .next_iterate
            .reserve(nf.saturating_sub(scratch.next_iterate.len()));
    }

    out.clear();
    out.extend_from_slice(&initial[..nf]);
    scratch.next_iterate.clear();

    for cycle in 0..max_cycles {
        try_amg_v_cycle_witness_with_scratch_into(
            fine_matrix,
            fine_rhs,
            out,
            restriction,
            prolongation,
            coarse_matrix,
            omega,
            fine_count,
            coarse_count,
            &mut scratch.v_cycle,
            &mut scratch.next_iterate,
        )?;
        out.clear();
        out.extend_from_slice(&scratch.next_iterate);

        let mut max_resid: f64 = 0.0;
        for i in 0..nf {
            let row_dot: f64 = (0..nf).map(|j| fine_matrix[i * nf + j] * out[j]).sum();
            let r = (row_dot - fine_rhs[i]).abs();
            if r > max_resid {
                max_resid = r;
            }
        }
        if max_resid < tolerance {
            return Ok(cycle + 1);
        }
    }
    Ok(max_cycles)
}

/// Iterative AMG V-cycle solver to tolerance writing into caller-owned storage using reusable scratch.
///
/// # Panics
///
/// Panics if matrix or vector buffer shapes are invalid or if `tolerance` is non-positive or non-finite.
#[allow(clippy::too_many_arguments)]
pub fn amg_solve_to_tolerance_witness_with_scratch_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    tolerance: f64,
    max_cycles: u32,
    scratch: &mut AmgSolveScratchWitness,
    out: &mut Vec<f64>,
) -> u32 {
    try_amg_solve_to_tolerance_witness_with_scratch_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        tolerance,
        max_cycles,
        scratch,
        out,
    )
    .expect("Fix: supply matching fine/coarse AMG operator buffers and a finite positive tolerance")
}

/// Fallible iterative AMG V-cycle solver to tolerance into caller-owned storage.
///
/// Input validation completes before any caller-owned vector is changed.
/// Returns the number of cycles executed (<= max_cycles).
#[allow(clippy::too_many_arguments)]
pub fn try_amg_solve_to_tolerance_witness_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    tolerance: f64,
    max_cycles: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> Result<u32, String> {
    let mut solve_scratch = AmgSolveScratchWitness::new();
    let res = try_amg_solve_to_tolerance_witness_with_scratch_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        tolerance,
        max_cycles,
        &mut solve_scratch,
        out,
    )?;
    scratch.clear();
    scratch.extend_from_slice(&solve_scratch.next_iterate);
    Ok(res)
}

/// Iterative AMG V-cycle solver to tolerance writing into caller-owned storage.
///
/// # Panics
///
/// Panics if matrix or vector buffer shapes are invalid or if `tolerance` is non-positive or non-finite.
#[allow(clippy::too_many_arguments)]
pub fn amg_solve_to_tolerance_witness_into(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    tolerance: f64,
    max_cycles: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> u32 {
    try_amg_solve_to_tolerance_witness_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        tolerance,
        max_cycles,
        out,
        scratch,
    )
    .expect("Fix: supply matching fine/coarse AMG operator buffers and a finite positive tolerance")
}

/// Iterative AMG V-cycle solver to tolerance returning solution vector and cycle count.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn amg_solve_to_tolerance_witness(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    omega: f64,
    fine_count: u32,
    coarse_count: u32,
    tolerance: f64,
    max_cycles: u32,
) -> (Vec<f64>, u32) {
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    let cycles = amg_solve_to_tolerance_witness_into(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        omega,
        fine_count,
        coarse_count,
        tolerance,
        max_cycles,
        &mut out,
        &mut scratch,
    );
    (out, cycles)
}

/// Fallible iterative Jacobi solver to tolerance writing into caller-owned storage.
pub fn try_jacobi_solve_to_tolerance_witness_into(
    matrix: &[f64],
    rhs: &[f64],
    initial: &[f64],
    omega: f64,
    n: u32,
    tolerance: f64,
    max_iters: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> Result<u32, String> {
    let n_us = n as usize;
    if matrix.len() < n_us * n_us {
        return Err(format!(
            "matrix too short: expected {}, got {}",
            n_us * n_us,
            matrix.len()
        ));
    }
    if rhs.len() < n_us {
        return Err(format!("rhs too short: expected {n_us}, got {}", rhs.len()));
    }
    if initial.len() < n_us {
        return Err(format!(
            "initial vector too short: expected {n_us}, got {}",
            initial.len()
        ));
    }
    if tolerance <= 0.0 || !tolerance.is_finite() {
        return Err(format!(
            "tolerance must be finite positive, got {tolerance}"
        ));
    }

    if n == 0 {
        out.clear();
        scratch.clear();
        return Ok(0);
    }

    if out.capacity() < n_us {
        out.reserve(n_us.saturating_sub(out.len()));
    }
    if scratch.capacity() < n_us {
        scratch.reserve(n_us.saturating_sub(scratch.len()));
    }

    out.clear();
    out.extend_from_slice(&initial[..n_us]);
    scratch.clear();

    for iter in 0..max_iters {
        scratch.clear();
        scratch.reserve(n_us);
        for row in 0..n_us {
            let off_diag: f64 = (0..n_us)
                .filter(|&col| col != row)
                .map(|col| matrix[row * n_us + col] * out[col])
                .sum();
            let diag = matrix[row * n_us + row];
            let target = (rhs[row] - off_diag) / diag;
            scratch.push(out[row] + omega * (target - out[row]));
        }
        out.copy_from_slice(scratch);

        let mut max_resid: f64 = 0.0;
        for i in 0..n_us {
            let row_dot: f64 = (0..n_us).map(|j| matrix[i * n_us + j] * out[j]).sum();
            let r = (row_dot - rhs[i]).abs();
            if r > max_resid {
                max_resid = r;
            }
        }
        if max_resid < tolerance {
            return Ok(iter + 1);
        }
    }
    Ok(max_iters)
}

/// Iterative Jacobi solver to tolerance writing into caller-owned storage.
///
/// # Panics
///
/// Panics if matrix or vector buffer shapes are invalid or if `tolerance` is non-positive or non-finite.
pub fn jacobi_solve_to_tolerance_witness_into(
    matrix: &[f64],
    rhs: &[f64],
    initial: &[f64],
    omega: f64,
    n: u32,
    tolerance: f64,
    max_iters: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> u32 {
    try_jacobi_solve_to_tolerance_witness_into(
        matrix, rhs, initial, omega, n, tolerance, max_iters, out, scratch,
    )
    .expect("Fix: provide a square matrix of size n*n, vectors of size n, and a finite positive tolerance")
}

/// Iterative Jacobi solver to tolerance returning solution vector and iteration count.
#[must_use]
pub fn jacobi_solve_to_tolerance_witness(
    matrix: &[f64],
    rhs: &[f64],
    initial: &[f64],
    omega: f64,
    n: u32,
    tolerance: f64,
    max_iters: u32,
) -> (Vec<f64>, u32) {
    let mut out = Vec::new();
    let mut scratch = Vec::new();
    let iters = jacobi_solve_to_tolerance_witness_into(
        matrix,
        rhs,
        initial,
        omega,
        n,
        tolerance,
        max_iters,
        &mut out,
        &mut scratch,
    );
    (out, iters)
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

/// Clamp spectral values to the Marchenko-Pastur upper edge into caller storage.
pub fn mp_edge_clip_witness_into(values: &[f64], upper_edge: f64, out: &mut Vec<f64>) {
    if out.capacity() < values.len() {
        out.reserve(values.len().saturating_sub(out.len()));
    }
    out.clear();
    out.extend(values.iter().map(|&value| value.min(upper_edge)));
}

/// Clamp spectral values to the Marchenko-Pastur upper edge.
#[must_use]
pub fn mp_edge_clip_witness(values: &[f64], upper_edge: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    mp_edge_clip_witness_into(values, upper_edge, &mut out);
    out
}
/// Pack signed four-bit lanes into caller-owned storage, eight lanes per little-endian `u32` word.
pub fn pack_i4x8_witness_into(lanes: &[i32], out: &mut Vec<u32>) {
    let word_count = lanes.len().div_ceil(8);
    out.clear();
    out.resize(word_count, 0_u32);
    for (index, &lane) in lanes.iter().enumerate() {
        let nibble = (lane.clamp(-8, 7) as i8 as u8) & 0xF;
        out[index / 8] |= u32::from(nibble) << ((index % 8) * 4);
    }
}

/// Pack signed four-bit lanes, eight lanes per little-endian `u32` word.
#[must_use]
pub fn pack_i4x8_witness(lanes: &[i32]) -> Vec<u32> {
    let mut output = Vec::new();
    pack_i4x8_witness_into(lanes, &mut output);
    output
}

/// Unpack signed four-bit lanes from little-endian `u32` words into caller-owned storage.
pub fn unpack_i4x8_witness_into(words: &[u32], lane_count: u32, out: &mut Vec<i32>) {
    let count = lane_count as usize;
    out.clear();
    out.reserve(count);
    for index in 0..count {
        let nibble = words.get(index / 8).copied().unwrap_or(0) >> ((index % 8) * 4) & 0xF;
        out.push((nibble as i32) << 28 >> 28);
    }
}

/// Unpack signed four-bit lanes from little-endian `u32` words.
#[must_use]
pub fn unpack_i4x8_witness(words: &[u32], lane_count: u32) -> Vec<i32> {
    let mut out = Vec::new();
    unpack_i4x8_witness_into(words, lane_count, &mut out);
    out
}
/// Sequential dot product over packed signed four-bit lanes.
#[must_use]
pub fn i4x8_dot_i32_witness(lhs: &[u32], rhs: &[u32], lane_count: u32) -> i32 {
    unpack_i4x8_witness(lhs, lane_count)
        .into_iter()
        .zip(unpack_i4x8_witness(rhs, lane_count))
        .fold(0_i32, |sum, (left, right)| {
            sum.wrapping_add(left.wrapping_mul(right))
        })
}

/// Sequential scaled dot product over packed signed four-bit lanes.
#[must_use]
pub fn i4x8_dot_f32_scaled_witness(
    lhs: &[u32],
    rhs: &[u32],
    lhs_scale: f32,
    rhs_scale: f32,
    lane_count: u32,
) -> f32 {
    unpack_i4x8_witness(lhs, lane_count)
        .into_iter()
        .zip(unpack_i4x8_witness(rhs, lane_count))
        .fold(0.0_f32, |sum, (left, right)| {
            sum + left as f32 * right as f32
        })
        * lhs_scale
        * rhs_scale
}

/// Sequential row-scaled matrix-vector product over packed INT4 weights.
#[must_use]
pub fn i4x8_matvec_f32_scaled_witness(
    weights: &[u32],
    vector: &[f32],
    row_scales: &[f32],
    row_count: u32,
    lane_count: u32,
) -> Vec<f32> {
    let words_per_row = lane_count.div_ceil(8) as usize;
    (0..row_count as usize)
        .map(|row| {
            let row_words = weights
                .get(row * words_per_row..(row + 1) * words_per_row)
                .unwrap_or_default();
            let lanes = unpack_i4x8_witness(row_words, lane_count);
            let sum = lanes
                .into_iter()
                .zip(vector.iter().copied().chain(std::iter::repeat(0.0)))
                .take(lane_count as usize)
                .fold(0.0_f32, |sum, (weight, value)| sum + weight as f32 * value);
            sum * row_scales.get(row).copied().unwrap_or(0.0)
        })
        .collect()
}

/// Sequential batched row-scaled matrix-vector product over packed INT4 weights.
#[must_use]
pub fn i4x8_batched_matvec_f32_scaled_witness(
    weights: &[u32],
    vectors: &[f32],
    row_scales: &[f32],
    batch_count: u32,
    row_count: u32,
    lane_count: u32,
) -> Vec<f32> {
    let mut output = Vec::with_capacity((batch_count * row_count) as usize);
    for batch in 0..batch_count as usize {
        let start = batch * lane_count as usize;
        let end = start + lane_count as usize;
        output.extend(i4x8_matvec_f32_scaled_witness(
            weights,
            vectors.get(start..end).unwrap_or_default(),
            row_scales,
            row_count,
            lane_count,
        ));
    }
    output
}

/// Sequential scaled batched matrix multiplication over packed INT4 rows.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn i4x8_batched_matmul_f32_scaled_witness(
    weights: &[u32],
    activations: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
    batch_count: u32,
    row_count: u32,
    lane_count: u32,
) -> Vec<f32> {
    let words_per_row = lane_count.div_ceil(8) as usize;
    let mut output = Vec::with_capacity((batch_count * row_count) as usize);
    for batch in 0..batch_count as usize {
        let activation = activations
            .get(batch * words_per_row..(batch + 1) * words_per_row)
            .unwrap_or_default();
        for row in 0..row_count as usize {
            let weight = weights
                .get(row * words_per_row..(row + 1) * words_per_row)
                .unwrap_or_default();
            output.push(
                i4x8_dot_i32_witness(weight, activation, lane_count) as f32
                    * row_scales.get(row).copied().unwrap_or(0.0)
                    * batch_scales.get(batch).copied().unwrap_or(0.0),
            );
        }
    }
    output
}

/// Select the highest scaled matrix-product row for each batch.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn i4x8_batched_matmul_top1_f32_scaled_witness(
    weights: &[u32],
    activations: &[u32],
    row_scales: &[f32],
    batch_scales: &[f32],
    batch_count: u32,
    row_count: u32,
    lane_count: u32,
) -> (Vec<f32>, Vec<u32>) {
    let logits = i4x8_batched_matmul_f32_scaled_witness(
        weights,
        activations,
        row_scales,
        batch_scales,
        batch_count,
        row_count,
        lane_count,
    );
    let mut scores = Vec::with_capacity(batch_count as usize);
    let mut indices = Vec::with_capacity(batch_count as usize);
    for batch in 0..batch_count as usize {
        let row = logits
            .get(batch * row_count as usize..(batch + 1) * row_count as usize)
            .unwrap_or_default();
        let mut best_score = f32::MIN;
        let mut best_index = 0;
        for (index, &score) in row.iter().enumerate() {
            if score > best_score {
                best_score = score;
                best_index = index as u32;
            }
        }
        scores.push(best_score);
        indices.push(best_index);
    }
    (scores, indices)
}

/// Sequential wrapping-integer Sinkhorn scaling iteration.
///
/// # Panics
///
/// Panics if matrix dimensions `m * n` overflow `usize` or if input buffer shapes do not match `m` and `n`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_iterate_witness(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_init: &[u32],
    v_init: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
) -> (Vec<u32>, Vec<u32>, u32) {
    try_sinkhorn_iterate_witness(k, k_t, a, b, u_init, v_init, m, n, max_iterations)
        .unwrap_or_else(|error| panic!("Sinkhorn witness failed: {error}"))
}

/// Fallible sequential wrapping-integer Sinkhorn scaling iteration into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn try_sinkhorn_iterate_witness_into(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_init: &[u32],
    v_init: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
    u_out: &mut Vec<u32>,
    v_out: &mut Vec<u32>,
    u_old: &mut Vec<u32>,
) -> Result<u32, String> {
    let (m, n) = (m as usize, n as usize);
    if m == 0 || n == 0 {
        return Err("Sinkhorn dimensions must be non-zero".to_owned());
    }
    let required = [
        ("k", k.len(), m.saturating_mul(n)),
        ("k_t", k_t.len(), m.saturating_mul(n)),
        ("a", a.len(), m),
        ("b", b.len(), n),
        ("u_init", u_init.len(), m),
        ("v_init", v_init.len(), n),
    ];
    if let Some((name, got, need)) = required.into_iter().find(|(_, got, need)| got < need) {
        return Err(format!(
            "buffer `{name}` is too short: got {got}, need {need}"
        ));
    }
    u_out.clear();
    u_out.extend_from_slice(&u_init[..m]);
    v_out.clear();
    v_out.extend_from_slice(&v_init[..n]);
    u_old.clear();
    u_old.extend_from_slice(&u_init[..m]);

    let mut iterations = 0;
    for iteration in 0..max_iterations {
        u_old.copy_from_slice(u_out);
        let step_u32 = |mat: &[u32],
                        in_v: &[u32],
                        tgt: &[u32],
                        out_v: &mut [u32],
                        rows: usize,
                        cols: usize| {
            for r in 0..rows {
                let sum = (0..cols).fold(0_u32, |acc, c| {
                    acc.wrapping_add(mat[r * cols + c].wrapping_mul(in_v[c]))
                });
                out_v[r] = tgt[r] / sum.max(1);
            }
        };
        step_u32(k, v_out, a, u_out, m, n);
        step_u32(k_t, u_out, b, v_out, n, m);
        if u_out == u_old {
            return Ok(iteration);
        }
        iterations = iteration + 1;
    }
    Ok(iterations)
}

/// Fallible sequential wrapping-integer Sinkhorn scaling iteration.
#[allow(clippy::too_many_arguments)]
pub fn try_sinkhorn_iterate_witness(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_init: &[u32],
    v_init: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
) -> Result<(Vec<u32>, Vec<u32>, u32), String> {
    let (mut u, mut v, mut u_old) = (Vec::new(), Vec::new(), Vec::new());
    let iters = try_sinkhorn_iterate_witness_into(
        k,
        k_t,
        a,
        b,
        u_init,
        v_init,
        m,
        n,
        max_iterations,
        &mut u,
        &mut v,
        &mut u_old,
    )?;
    Ok((u, v, iters))
}

/// Sequential floating-point Sinkhorn clustering witness.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_clustering_witness(
    region_features: &[f32],
    cluster_centroids: &[f32],
    region_weights: &[f32],
    cluster_capacities: &[f32],
    m: u32,
    n: u32,
    d: u32,
    iterations: u32,
    epsilon: f32,
) -> Vec<u32> {
    let (m, n, d) = (m as usize, n as usize, d as usize);
    let mut kernel = vec![0.0_f32; m * n];
    for region in 0..m {
        for cluster in 0..n {
            let cost = (0..d).fold(0.0_f32, |sum, dimension| {
                let difference = region_features[region * d + dimension]
                    - cluster_centroids[cluster * d + dimension];
                sum + difference * difference
            });
            kernel[region * n + cluster] = (-cost / epsilon).exp();
        }
    }

    let mut u = vec![1.0_f32; m];
    let mut v = vec![1.0_f32; n];
    for _ in 0..iterations {
        let step_f32 = |in_v: &[f32],
                        weights: &[f32],
                        out_u: &mut [f32],
                        rows: usize,
                        cols: usize,
                        trans: bool| {
            for r in 0..rows {
                let sum = (0..cols).fold(0.0_f32, |sum, c| {
                    let idx = if trans { c * rows + r } else { r * cols + c };
                    sum + kernel[idx] * in_v[c]
                });
                out_u[r] = weights[r] / sum.max(1.0e-10);
            }
        };
        step_f32(&v, region_weights, &mut u, m, n, false);
        step_f32(&u, cluster_capacities, &mut v, n, m, true);
    }

    (0..m)
        .map(|region| {
            let mut best_cluster = 0;
            let mut best_score = -1.0_f32;
            for cluster in 0..n {
                let score = kernel[region * n + cluster] * v[cluster];
                if score > best_score {
                    best_score = score;
                    best_cluster = cluster as u32;
                }
            }
            best_cluster
        })
        .collect()
}

/// Sequential edge-clamped one-dimensional wrapping convolution writing into caller storage.
pub fn conv1d_witness_into(input: &[u32], weights: &[u32], stride: u32, out: &mut Vec<u32>) {
    out.clear();
    if input.is_empty() {
        return;
    }
    if out.capacity() < input.len() {
        out.reserve(input.len().saturating_sub(out.len()));
    }
    let radius = weights.len() / 2;
    let stride = stride as usize;
    out.extend((0..input.len()).map(|index| {
        weights
            .iter()
            .enumerate()
            .fold(0_u32, |sum, (kernel, &weight)| {
                let source = if kernel >= radius {
                    index
                        .saturating_add((kernel - radius).saturating_mul(stride))
                        .min(input.len() - 1)
                } else {
                    index.saturating_sub((radius - kernel).saturating_mul(stride))
                };
                sum.wrapping_add(input[source].wrapping_mul(weight))
            })
    }));
}

/// Sequential edge-clamped one-dimensional wrapping convolution.
#[must_use]
pub fn conv1d_witness(input: &[u32], weights: &[u32], stride: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    conv1d_witness_into(input, weights, stride, &mut out);
    out
}

/// Gather polynomial coefficients into a row-major Gram matrix into caller storage.
pub fn sos_gram_construct_witness_into(
    monomial_pairs: &[u32],
    polynomial_coefficients: &[u32],
    matrix_size: u32,
    out: &mut Vec<u32>,
) {
    let cells = matrix_size.saturating_mul(matrix_size) as usize;
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    out.extend((0..cells).map(|cell| {
        monomial_pairs
            .get(cell)
            .and_then(|&index| polynomial_coefficients.get(index as usize))
            .copied()
            .unwrap_or(0)
    }));
}

/// Gather polynomial coefficients into a row-major Gram matrix.
#[must_use]
pub fn sos_gram_construct_witness(
    monomial_pairs: &[u32],
    polynomial_coefficients: &[u32],
    matrix_size: u32,
) -> Vec<u32> {
    let cells = matrix_size.saturating_mul(matrix_size) as usize;
    let mut out = Vec::with_capacity(cells);
    sos_gram_construct_witness_into(
        monomial_pairs,
        polynomial_coefficients,
        matrix_size,
        &mut out,
    );
    out
}
/// Construct a row-major identity linear map.
#[must_use]
pub fn identity_arrow_witness(size: u32) -> Vec<f64> {
    let mut output = vec![0.0; (size * size) as usize];
    for index in 0..size as usize {
        output[index * size as usize + index] = 1.0;
    }
    output
}

/// Compose row-major linear maps `A -> B` and `B -> C`.
#[must_use]
pub fn compose_ir_arrows_witness(
    first: &[f64],
    second: &[f64],
    a: u32,
    b: u32,
    c: u32,
) -> Vec<f64> {
    let (a, b, c) = (a as usize, b as usize, c as usize);
    let mut output = vec![0.0; a * c];
    for row in 0..a {
        for column in 0..c {
            output[row * c + column] = (0..b)
                .map(|middle| first[row * b + middle] * second[middle * c + column])
                .sum();
        }
    }
    output
}

/// Compare both parenthesizations of three compatible linear maps.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn composition_associates_witness(
    first: &[f64],
    second: &[f64],
    third: &[f64],
    a: u32,
    b: u32,
    c: u32,
    d: u32,
) -> bool {
    let left = compose_ir_arrows_witness(
        &compose_ir_arrows_witness(first, second, a, b, c),
        third,
        a,
        c,
        d,
    );
    let right = compose_ir_arrows_witness(
        first,
        &compose_ir_arrows_witness(second, third, b, c, d),
        a,
        b,
        d,
    );
    left.iter()
        .zip(right)
        .all(|(lhs, rhs)| (lhs - rhs).abs() <= 1e-9 * (1.0 + lhs.abs() + rhs.abs()))
}

/// Evaluate a topologically ordered floating-point sum-product circuit into caller-owned storage.
///
/// Kinds `0`, `1`, and `2` denote leaf, weighted-sum, and product nodes.
pub fn sum_product_evaluate_witness_into(
    kinds: &[u32],
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    weights: &[f64],
    leaf_values: &[f64],
    topological_order: &[u32],
    output: &mut Vec<f64>,
) {
    let n = kinds.len();
    if output.capacity() < n {
        output.reserve(n.saturating_sub(output.len()));
    }
    output.clear();
    output.resize(n, 0.0);
    for &node in topological_order {
        let node = node as usize;
        let start = child_offsets[node] as usize;
        let end = start + child_counts[node] as usize;
        output[node] = match kinds[node] {
            0 => leaf_values[node],
            1 => children[start..end]
                .iter()
                .zip(&weights[start..end])
                .map(|(&child, &weight)| output[child as usize] * weight)
                .sum(),
            2 => children[start..end]
                .iter()
                .map(|&child| output[child as usize])
                .product(),
            _ => 0.0,
        };
    }
}

/// Evaluate a topologically ordered floating-point sum-product circuit.
///
/// Kinds `0`, `1`, and `2` denote leaf, weighted-sum, and product nodes.
#[must_use]
pub fn sum_product_evaluate_witness(
    kinds: &[u32],
    child_offsets: &[u32],
    child_counts: &[u32],
    children: &[u32],
    weights: &[f64],
    leaf_values: &[f64],
    topological_order: &[u32],
) -> Vec<f64> {
    let mut output = Vec::new();
    sum_product_evaluate_witness_into(
        kinds,
        child_offsets,
        child_counts,
        children,
        weights,
        leaf_values,
        topological_order,
        &mut output,
    );
    output
}

/// Sequential numerically stable softmax into caller storage.
pub fn softmax_witness_into(input: &[f64], out: &mut Vec<f64>) {
    if input.is_empty() {
        out.clear();
        return;
    }
    if out.capacity() < input.len() {
        out.reserve(input.len().saturating_sub(out.len()));
    }
    out.clear();
    let max = input.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    out.extend(input.iter().map(|value| (value - max).exp()));
    let sum: f64 = out.iter().sum();
    for value in out.iter_mut() {
        *value /= sum;
    }
}

/// Sequential numerically stable softmax.
#[must_use]
pub fn softmax_witness(input: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(input.len());
    softmax_witness_into(input, &mut out);
    out
}
/// Sequential temperature-scaled soft argmax into caller storage.
pub fn differentiable_argmax_witness_into(
    input: &[f64],
    temperature: f64,
    scaled: &mut Vec<f64>,
    out: &mut Vec<f64>,
) {
    if temperature <= 0.0 || !temperature.is_finite() {
        scaled.clear();
        out.clear();
        return;
    }
    if scaled.capacity() < input.len() {
        scaled.reserve(input.len().saturating_sub(scaled.len()));
    }
    scaled.clear();
    scaled.extend(input.iter().map(|value| value / temperature));
    softmax_witness_into(scaled, out);
}

/// Sequential temperature-scaled soft argmax.
#[must_use]
pub fn differentiable_argmax_witness(input: &[f64], temperature: f64) -> Vec<f64> {
    let mut scaled = Vec::new();
    let mut out = Vec::new();
    differentiable_argmax_witness_into(input, temperature, &mut scaled, &mut out);
    out
}

/// Fallible sequential argmin with total_cmp tie breaking.
pub fn try_argmin_cost_witness(costs: &[f64]) -> Result<usize, String> {
    if costs.is_empty() {
        return Err("costs must not be empty".to_string());
    }
    let mut best = 0usize;
    let mut best_cost = costs[0];
    for (i, &cost) in costs.iter().enumerate().skip(1) {
        if cost.total_cmp(&best_cost).is_lt() {
            best = i;
            best_cost = cost;
        }
    }
    Ok(best)
}

/// Sequential argmin with total_cmp tie breaking.
///
/// # Panics
///
/// Panics if `costs` is empty.
#[must_use]
pub fn argmin_cost_witness(costs: &[f64]) -> usize {
    try_argmin_cost_witness(costs).expect("Fix: pick_best_config requires at least one candidate.")
}

/// Fallible differentiable autotune config score gradient into caller-owned storage.
pub fn try_differentiable_autotune_gradient_witness_into(
    costs: &[f64],
    temperature: f64,
    neg_costs: &mut Vec<f64>,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err("temperature must be positive".to_string());
    }
    if neg_costs.capacity() < costs.len() {
        neg_costs.reserve(costs.len().saturating_sub(neg_costs.len()));
    }
    neg_costs.clear();
    neg_costs.extend(costs.iter().map(|&c| -c / temperature));
    softmax_witness_into(neg_costs, out);
    for value in out.iter_mut() {
        *value = -*value;
    }
    Ok(())
}

/// Differentiable autotune config score gradient into caller-owned storage.
///
/// # Panics
///
/// Panics if `temperature` is non-positive or non-finite.
pub fn differentiable_autotune_gradient_witness_into(
    costs: &[f64],
    temperature: f64,
    neg_costs: &mut Vec<f64>,
    out: &mut Vec<f64>,
) {
    try_differentiable_autotune_gradient_witness_into(costs, temperature, neg_costs, out)
        .expect("differentiable_autotune_gradient_witness_into failed: invalid temperature");
}

/// Differentiable autotune config score gradient.
#[must_use]
pub fn differentiable_autotune_gradient_witness(costs: &[f64], temperature: f64) -> Vec<f64> {
    let mut neg_costs = Vec::new();
    let mut out = Vec::new();
    differentiable_autotune_gradient_witness_into(costs, temperature, &mut neg_costs, &mut out);
    out
}

/// Fallible differentiable autotune configuration pick probabilities into caller-owned storage.
pub fn try_differentiable_autotune_pick_config_witness_into(
    costs: &[f64],
    temperature: f64,
    neg_costs: &mut Vec<f64>,
    scaled: &mut Vec<f64>,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return Err("temperature must be positive".to_string());
    }
    if neg_costs.capacity() < costs.len() {
        neg_costs.reserve(costs.len().saturating_sub(neg_costs.len()));
    }
    neg_costs.clear();
    neg_costs.extend(costs.iter().map(|&c| -c));
    differentiable_argmax_witness_into(neg_costs, temperature, scaled, out);
    Ok(())
}

/// Differentiable autotune configuration pick probabilities into caller-owned storage.
///
/// # Panics
///
/// Panics if `temperature` is non-positive or non-finite.
pub fn differentiable_autotune_pick_config_witness_into(
    costs: &[f64],
    temperature: f64,
    neg_costs: &mut Vec<f64>,
    scaled: &mut Vec<f64>,
    out: &mut Vec<f64>,
) {
    try_differentiable_autotune_pick_config_witness_into(
        costs,
        temperature,
        neg_costs,
        scaled,
        out,
    )
    .expect("Fix: supply a finite positive temperature parameter for differentiable autotune configuration selection");
}

/// Differentiable autotune configuration pick probabilities.
#[must_use]
pub fn differentiable_autotune_pick_config_witness(costs: &[f64], temperature: f64) -> Vec<f64> {
    let mut neg_costs = Vec::new();
    let mut scaled = Vec::new();
    let mut out = Vec::new();
    differentiable_autotune_pick_config_witness_into(
        costs,
        temperature,
        &mut neg_costs,
        &mut scaled,
        &mut out,
    );
    out
}

/// Sequential alternating-sign message aggregation over triangle edges.
#[must_use]
pub fn simplicial_triangle_message_witness(
    edge_features: &[f64],
    triangle_edges: &[u32],
    edge_count: u32,
    triangle_count: u32,
    dimensions: u32,
) -> Vec<f64> {
    let (edge_count, triangle_count, dimensions) = (
        edge_count as usize,
        triangle_count as usize,
        dimensions as usize,
    );
    let mut output = vec![0.0; triangle_count * dimensions];
    for triangle in 0..triangle_count {
        let Some((&edge_jk, rest)) = triangle_edges
            .get(triangle * 3)
            .zip(triangle_edges.get(triangle * 3 + 1..triangle * 3 + 3))
        else {
            continue;
        };
        let (edge_jk, edge_ik, edge_ij) = (edge_jk as usize, rest[0] as usize, rest[1] as usize);
        if edge_jk >= edge_count || edge_ik >= edge_count || edge_ij >= edge_count {
            continue;
        }
        for dimension in 0..dimensions {
            let Some((&jk, (&ik, &ij))) = edge_features.get(edge_jk * dimensions + dimension).zip(
                edge_features
                    .get(edge_ik * dimensions + dimension)
                    .zip(edge_features.get(edge_ij * dimensions + dimension)),
            ) else {
                continue;
            };
            output[triangle * dimensions + dimension] = jk - ik + ij;
        }
    }
    output
}

/// Sequential Vietoris-Rips upper-triangular edge mask at one scale.
#[must_use]
pub fn vietoris_rips_edge_filter_witness(
    distances: &[f64],
    epsilon: f64,
    point_count: u32,
) -> Vec<u32> {
    let points = point_count as usize;
    let mut output = vec![0_u32; points * points];
    for row in 0..points {
        for column in (row + 1)..points {
            let index = row * points + column;
            if distances.get(index).copied().unwrap_or(f64::INFINITY) <= epsilon {
                output[index] = 1;
            }
        }
    }
    output
}

/// Extract ordered upper-triangular edges from a Vietoris-Rips mask.
#[must_use]
pub fn vietoris_rips_edges_witness(mask: &[u32], point_count: u32) -> Vec<(u32, u32)> {
    let points = point_count as usize;
    let mut output = Vec::new();
    for row in 0..points {
        for column in (row + 1)..points {
            if mask
                .get(row * points + column)
                .is_some_and(|&value| value != 0)
            {
                output.push((row as u32, column as u32));
            }
        }
    }
    output
}

/// Sequential conservative merge of paired unsigned intervals.
#[must_use]
pub fn interval_merge_witness(
    mins_a: &[u32],
    maxs_a: &[u32],
    mins_b: &[u32],
    maxs_b: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    let length = mins_a
        .len()
        .min(maxs_a.len())
        .min(mins_b.len())
        .min(maxs_b.len());
    let mins = (0..length)
        .map(|index| mins_a[index].min(mins_b[index]))
        .collect();
    let maxs = (0..length)
        .map(|index| maxs_a[index].max(maxs_b[index]))
        .collect();
    (mins, maxs)
}

/// Sequential hard threshold retaining the `k` largest finite magnitudes into caller-owned storage.
pub fn iht_top_k_witness_into(
    values: &[f64],
    k: usize,
    out: &mut Vec<f64>,
    order_scratch: &mut Vec<usize>,
) -> f64 {
    let n = values.len();
    if out.capacity() < n {
        out.reserve(n.saturating_sub(out.len()));
    }
    if k >= n {
        out.clear();
        out.extend_from_slice(values);
        order_scratch.clear();
        return 0.0;
    }
    if k == 0 {
        out.clear();
        out.resize(n, 0.0);
        order_scratch.clear();
        return f64::INFINITY;
    }
    let score = |value: f64| {
        let magnitude = value.abs();
        if magnitude.is_nan() {
            f64::NEG_INFINITY
        } else {
            magnitude
        }
    };
    if order_scratch.capacity() < n {
        order_scratch.reserve(n.saturating_sub(order_scratch.len()));
    }
    order_scratch.clear();
    order_scratch.extend(0..n);
    order_scratch.sort_by(|&left, &right| score(values[right]).total_cmp(&score(values[left])));
    let threshold = values[order_scratch[k - 1]].abs();
    out.clear();
    out.resize(n, 0.0);
    for &index in &order_scratch[..k] {
        out[index] = values[index];
    }
    order_scratch.clear();
    threshold
}

/// Sequential hard threshold retaining the `k` largest finite magnitudes.
#[must_use]
pub fn iht_top_k_witness(values: &[f64], k: usize) -> (Vec<f64>, f64) {
    let mut out = Vec::with_capacity(values.len());
    let mut order_scratch = Vec::with_capacity(values.len());
    let threshold = iht_top_k_witness_into(values, k, &mut out, &mut order_scratch);
    (out, threshold)
}

/// Sequential FMM particle-to-multipole zeroth-moment aggregation.
///
/// # Panics
///
/// Panics if `charges` and `cell_assignment` have different lengths or if cell IDs exceed host bounds.
#[must_use]
pub fn p2m_zeroth_moment_witness(charges: &[f64], cell_assignment: &[u32]) -> Vec<f64> {
    let mut moments = Vec::new();
    try_p2m_zeroth_moment_witness_into(charges, cell_assignment, &mut moments)
        .expect("P2M witness inputs must have matching lengths and representable cell ids");
    moments
}

/// Fallible sequential P2M aggregation into caller-owned storage.
///
/// Validation and reservation complete before `moments` is mutated.
pub fn try_p2m_zeroth_moment_witness_into(
    charges: &[f64],
    cell_assignment: &[u32],
    moments: &mut Vec<f64>,
) -> Result<(), String> {
    if charges.len() != cell_assignment.len() {
        return Err(format!(
            "charge count {} does not match cell assignment count {}",
            charges.len(),
            cell_assignment.len()
        ));
    }
    let cell_count = cell_assignment
        .iter()
        .copied()
        .max()
        .map_or(Ok(0usize), |cell| {
            usize::try_from(cell)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| format!("cell id {cell} is not representable"))
        })?;
    moments
        .try_reserve(cell_count.saturating_sub(moments.len()))
        .map_err(|error| format!("failed to reserve {cell_count} P2M moments: {error}"))?;

    moments.clear();
    moments.resize(cell_count, 0.0);
    for (&charge, &cell) in charges.iter().zip(cell_assignment) {
        moments[cell as usize] += charge;
    }
    Ok(())
}

/// Fallible sequential P2M aggregation with historical truncation of mismatched inputs into caller-owned storage.
pub fn try_p2m_zeroth_moment_truncating_witness_into(
    charges: &[f64],
    cell_assignment: &[u32],
    moments: &mut Vec<f64>,
) -> Result<(), String> {
    if charges.is_empty() {
        moments.clear();
        return Ok(());
    }
    let cell_count = cell_assignment
        .iter()
        .copied()
        .max()
        .map_or(Ok(1usize), |cell| {
            usize::try_from(cell)
                .ok()
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| format!("cell id {cell} is not representable"))
        })?;
    moments
        .try_reserve(cell_count.saturating_sub(moments.len()))
        .map_err(|error| format!("failed to reserve {cell_count} P2M moments: {error}"))?;

    moments.clear();
    moments.resize(cell_count, 0.0);
    for (&charge, &cell) in charges.iter().zip(cell_assignment) {
        moments[cell as usize] += charge;
    }
    Ok(())
}

/// Sequential P2M aggregation with historical truncation of mismatched inputs into caller-owned storage.
///
/// # Panics
///
/// Panics if cell IDs exceed host bounds or if memory allocation fails.
pub fn p2m_zeroth_moment_truncating_witness_into(
    charges: &[f64],
    cell_assignment: &[u32],
    moments: &mut Vec<f64>,
) {
    try_p2m_zeroth_moment_truncating_witness_into(charges, cell_assignment, moments)
        .expect("P2M truncating witness failed");
}

/// Sequential P2M aggregation with historical truncation of mismatched inputs.
#[must_use]
pub fn p2m_zeroth_moment_truncating_witness(charges: &[f64], cell_assignment: &[u32]) -> Vec<f64> {
    let mut moments = Vec::new();
    p2m_zeroth_moment_truncating_witness_into(charges, cell_assignment, &mut moments);
    moments
}

/// Sequential FMM multipole-to-local zeroth-order translation.
#[must_use]
pub fn m2l_zeroth_translate_witness(source_moment: f64, distance: f64) -> f64 {
    source_moment / distance.max(1.0e-12)
}

/// Sequential all-cell M2L zeroth-order translation.
///
/// # Panics
///
/// Panics if `cell_distances` is not a square matrix matching `cell_moments.len()` or if allocation fails.
#[must_use]
pub fn m2l_zeroth_all_witness(cell_moments: &[f64], cell_distances: &[f64]) -> Vec<f64> {
    let mut local = Vec::new();
    try_m2l_zeroth_all_witness_into(cell_moments, cell_distances, &mut local)
        .expect("M2L witness distance matrix must be square");
    local
}

/// Fallible sequential all-cell M2L translation into caller-owned storage.
///
/// Validation and reservation complete before `local` is mutated.
pub fn try_m2l_zeroth_all_witness_into(
    cell_moments: &[f64],
    cell_distances: &[f64],
    local: &mut Vec<f64>,
) -> Result<(), String> {
    let cell_count = cell_moments.len();
    let expected_distances = cell_count
        .checked_mul(cell_count)
        .ok_or_else(|| format!("cell count {cell_count} overflows square distance shape"))?;
    if cell_distances.len() != expected_distances {
        return Err(format!(
            "distance count {} does not match {cell_count}x{cell_count} matrix",
            cell_distances.len()
        ));
    }
    local
        .try_reserve(cell_count.saturating_sub(local.len()))
        .map_err(|error| format!("failed to reserve {cell_count} M2L locals: {error}"))?;

    local.clear();
    local.resize(cell_count, 0.0);
    for target in 0..cell_count {
        for source in 0..cell_count {
            if target != source {
                let distance = cell_distances[target * cell_count + source];
                local[target] += m2l_zeroth_translate_witness(cell_moments[source], distance);
            }
        }
    }
    Ok(())
}

/// Sequential FMM local-to-particle zeroth-order evaluation.
#[must_use]
pub const fn l2p_zeroth_eval_witness(local_moment: f64) -> f64 {
    local_moment
}

/// Sequential all-region L2P zeroth-order evaluation.
///
/// # Panics
///
/// Panics if `cell_assignment` length does not match `region_count`, if any assignment references
/// an out-of-bounds cell, or if allocation fails.
#[must_use]
pub fn l2p_zeroth_all_witness(
    cell_local: &[f64],
    cell_assignment: &[u32],
    region_count: u32,
) -> Vec<f64> {
    let mut output = Vec::new();
    try_l2p_zeroth_all_witness_into(cell_local, cell_assignment, region_count, &mut output)
        .expect("Fix: provide cell assignments matching region_count and indexing valid cells in cell_local");
    output
}

/// Fallible sequential all-region L2P evaluation into caller-owned storage.
///
/// Validation and reservation complete before `output` is mutated.
pub fn try_l2p_zeroth_all_witness_into(
    cell_local: &[f64],
    cell_assignment: &[u32],
    region_count: u32,
    output: &mut Vec<f64>,
) -> Result<(), String> {
    let region_count = region_count as usize;
    if cell_assignment.len() != region_count {
        return Err(format!(
            "cell assignment count {} does not match region count {region_count}",
            cell_assignment.len()
        ));
    }
    if let Some((region, &cell)) = cell_assignment
        .iter()
        .enumerate()
        .find(|(_, cell)| **cell as usize >= cell_local.len())
    {
        return Err(format!(
            "region {region} references cell {cell}, but only {} cells exist",
            cell_local.len()
        ));
    }
    output
        .try_reserve(region_count.saturating_sub(output.len()))
        .map_err(|error| format!("failed to reserve {region_count} L2P outputs: {error}"))?;

    output.clear();
    output.extend(
        cell_assignment
            .iter()
            .map(|&cell| l2p_zeroth_eval_witness(cell_local[cell as usize])),
    );
    Ok(())
}

/// Sequential dense Mori-Zwanzig projection with zero-padded short inputs into caller-owned storage.
pub fn mori_zwanzig_project_witness_into(
    projector: &[f64],
    forcing: &[f64],
    dimension: u32,
    out: &mut Vec<f64>,
) {
    let dimension = dimension as usize;
    out.clear();
    out.reserve(dimension);
    for row in 0..dimension {
        let mut sum = 0.0;
        for column in 0..dimension {
            sum += projector
                .get(row * dimension + column)
                .copied()
                .unwrap_or(0.0)
                * forcing.get(column).copied().unwrap_or(0.0);
        }
        out.push(sum);
    }
}

/// Sequential dense Mori-Zwanzig projection with zero-padded short inputs.
#[must_use]
pub fn mori_zwanzig_project_witness(
    projector: &[f64],
    forcing: &[f64],
    dimension: u32,
) -> Vec<f64> {
    let mut out = Vec::new();
    mori_zwanzig_project_witness_into(projector, forcing, dimension, &mut out);
    out
}

/// Fallible cluster-projection matrix construction into caller-owned storage.
pub fn try_cluster_projection_matrix_witness_into(
    assignments: &[u32],
    n: u32,
    k: u32,
    cluster_sizes: &mut Vec<u32>,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    if n == 0 {
        return Err("n must be positive".to_string());
    }
    if k == 0 {
        return Err("k must be positive".to_string());
    }
    let n_us = n as usize;
    let k_us = k as usize;
    if assignments.len() != n_us {
        return Err(format!(
            "assignments length mismatch: expected {n_us}, got {}",
            assignments.len()
        ));
    }
    for &c in assignments {
        if (c as usize) >= k_us {
            return Err(format!("Fix: assignment {c} exceeds cluster count {k}."));
        }
    }

    if cluster_sizes.capacity() < k_us {
        cluster_sizes.reserve(k_us.saturating_sub(cluster_sizes.len()));
    }
    cluster_sizes.clear();
    cluster_sizes.resize(k_us, 0);
    for &c in assignments {
        cluster_sizes[c as usize] += 1;
    }

    let cells = n_us.checked_mul(n_us).ok_or("matrix dimension overflow")?;
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    out.resize(cells, 0.0);
    for i in 0..n_us {
        let ci = assignments[i] as usize;
        let size = cluster_sizes[ci] as f64;
        if size == 0.0 {
            continue;
        }
        let inv = 1.0 / size;
        for j in 0..n_us {
            if assignments[j] as usize == ci {
                out[i * n_us + j] = inv;
            }
        }
    }
    Ok(())
}

/// Cluster-projection matrix construction into caller-owned storage.
///
/// # Panics
///
/// Panics if `assignments` length does not match `n`, if `n * n` overflows `usize`,
/// or if cluster assignments are invalid.
pub fn cluster_projection_matrix_witness_into(
    assignments: &[u32],
    n: u32,
    k: u32,
    cluster_sizes: &mut Vec<u32>,
    out: &mut Vec<f64>,
) {
    try_cluster_projection_matrix_witness_into(assignments, n, k, cluster_sizes, out)
        .expect("cluster_projection_matrix_witness_into failed");
}

/// Cluster-projection matrix construction.
#[must_use]
pub fn cluster_projection_matrix_witness(assignments: &[u32], n: u32, k: u32) -> Vec<f64> {
    let mut cluster_sizes = Vec::new();
    let mut out = Vec::new();
    cluster_projection_matrix_witness_into(assignments, n, k, &mut cluster_sizes, &mut out);
    out
}

/// Fallible Mori-Zwanzig coarsening via clustering into caller-owned storage.
pub fn try_mori_zwanzig_coarsen_via_clustering_witness_into(
    state: &[f64],
    assignments: &[u32],
    n: u32,
    k: u32,
    cluster_sizes: &mut Vec<u32>,
    projection: &mut Vec<f64>,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    try_cluster_projection_matrix_witness_into(assignments, n, k, cluster_sizes, projection)?;
    mori_zwanzig_project_witness_into(projection, state, n, out);
    Ok(())
}

/// Mori-Zwanzig coarsening via clustering into caller-owned storage.
///
/// # Panics
///
/// Panics if `assignments` or `state` lengths do not match `n`, if `n * n` overflows `usize`,
/// or if cluster assignments are invalid.
pub fn mori_zwanzig_coarsen_via_clustering_witness_into(
    state: &[f64],
    assignments: &[u32],
    n: u32,
    k: u32,
    cluster_sizes: &mut Vec<u32>,
    projection: &mut Vec<f64>,
    out: &mut Vec<f64>,
) {
    try_mori_zwanzig_coarsen_via_clustering_witness_into(
        state,
        assignments,
        n,
        k,
        cluster_sizes,
        projection,
        out,
    )
    .expect("mori_zwanzig_coarsen_via_clustering_witness_into failed");
}

/// Mori-Zwanzig coarsening via clustering.
#[must_use]
pub fn mori_zwanzig_coarsen_via_clustering_witness(
    state: &[f64],
    assignments: &[u32],
    n: u32,
    k: u32,
) -> Vec<f64> {
    let mut cluster_sizes = Vec::new();
    let mut projection = Vec::new();
    let mut out = Vec::new();
    mori_zwanzig_coarsen_via_clustering_witness_into(
        state,
        assignments,
        n,
        k,
        &mut cluster_sizes,
        &mut projection,
        &mut out,
    );
    out
}

/// Sequential Frobenius block encoding with zero-padded short matrices writing into caller storage.
///
/// # Panics
///
/// Panics if `dimension * dimension` overflows `usize`.
pub fn qsvt_block_encode_witness_into(matrix: &[f64], dimension: u32, out: &mut Vec<f64>) -> f64 {
    let cells = (dimension as usize)
        .checked_mul(dimension as usize)
        .expect("Fix: choose block encoding dimension such that dimension * dimension does not overflow usize");
    if out.capacity() < cells {
        out.reserve(cells.saturating_sub(out.len()));
    }
    out.clear();
    let norm = matrix.iter().map(|value| value * value).sum::<f64>().sqrt();
    let safe_norm = norm.max(1e-30);
    out.extend((0..cells).map(|index| matrix.get(index).copied().unwrap_or(0.0) / safe_norm));
    norm
}

/// Sequential Frobenius block encoding with zero-padded short matrices.
///
/// # Panics
///
/// Panics if `dimension * dimension` overflows `usize`.
#[must_use]
pub fn qsvt_block_encode_witness(matrix: &[f64], dimension: u32) -> (Vec<f64>, f64) {
    let cells = (dimension as usize)
        .checked_mul(dimension as usize)
        .expect("Fix: choose block encoding dimension such that dimension * dimension does not overflow usize");
    let mut scaled = Vec::with_capacity(cells);
    let norm = qsvt_block_encode_witness_into(matrix, dimension, &mut scaled);
    (scaled, norm)
}

fn qsvt_matvec_into(matrix: &[f64], input: &[f64], dimension: usize, out: &mut [f64]) {
    out.fill(0.0);
    for row in 0..dimension {
        for column in 0..dimension {
            out[row] += matrix[row * dimension + column] * input[column];
        }
    }
}

/// Sequential Chebyshev matrix-function expansion using caller-owned recurrence storage.
///
/// # Errors
///
/// Returns a diagnostic before mutating any caller storage when coefficients
/// are empty or matrix/vector storage is shorter than the declared dimension.
#[allow(clippy::too_many_arguments)]
pub fn qsvt_apply_witness_with_scratch_into(
    matrix: &[f64],
    vector: &[f64],
    coefficients: &[f64],
    dimension: u32,
    out: &mut Vec<f64>,
    previous: &mut Vec<f64>,
    current: &mut Vec<f64>,
    next: &mut Vec<f64>,
) -> Result<(), String> {
    let dimension = dimension as usize;
    let cells = dimension
        .checked_mul(dimension)
        .ok_or_else(|| "QSVT matrix dimensions overflow usize".to_string())?;
    if coefficients.is_empty() {
        return Err("QSVT expansion requires at least one coefficient".to_string());
    }
    if matrix.len() < cells {
        return Err(format!(
            "QSVT scaled matrix length {} is shorter than {cells}",
            matrix.len()
        ));
    }
    if vector.len() < dimension {
        return Err(format!(
            "QSVT vector length {} is shorter than {dimension}",
            vector.len()
        ));
    }
    for buffer in [&mut *out, &mut *previous, &mut *current, &mut *next] {
        if buffer.capacity() < dimension {
            buffer.reserve(dimension.saturating_sub(buffer.len()));
        }
    }
    out.clear();
    previous.clear();
    current.clear();
    next.clear();
    out.extend(
        vector[..dimension]
            .iter()
            .map(|value| coefficients[0] * value),
    );
    if coefficients.len() == 1 {
        return Ok(());
    }
    previous.extend_from_slice(&vector[..dimension]);
    current.resize(dimension, 0.0);
    qsvt_matvec_into(matrix, previous, dimension, current);
    for (value, term) in out.iter_mut().zip(current.iter()) {
        *value += coefficients[1] * term;
    }
    for &coefficient in &coefficients[2..] {
        next.resize(dimension, 0.0);
        qsvt_matvec_into(matrix, current, dimension, next);
        for index in 0..dimension {
            next[index] = 2.0 * next[index] - previous[index];
            out[index] += coefficient * next[index];
        }
        std::mem::swap(previous, current);
        std::mem::swap(current, next);
    }
    Ok(())
}

/// Sequential Chebyshev matrix-function expansion applied to a vector writing into caller storage.
pub fn qsvt_apply_witness_into(
    matrix: &[f64],
    vector: &[f64],
    coefficients: &[f64],
    dimension: u32,
    out: &mut Vec<f64>,
) -> Result<(), String> {
    let mut previous = Vec::new();
    let mut current = Vec::new();
    let mut next = Vec::new();
    qsvt_apply_witness_with_scratch_into(
        matrix,
        vector,
        coefficients,
        dimension,
        out,
        &mut previous,
        &mut current,
        &mut next,
    )
}

/// Sequential Chebyshev matrix-function expansion applied to a vector.
#[must_use]
pub fn qsvt_apply_witness(
    matrix: &[f64],
    vector: &[f64],
    coefficients: &[f64],
    dimension: u32,
) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    qsvt_apply_witness_into(matrix, vector, coefficients, dimension, &mut out)?;
    Ok(out)
}

/// Write negative-truncation Chebyshev coefficients into caller-owned storage.
pub fn negative_truncator_coeffs_witness_into(k_steps: u32, out: &mut Vec<f64>) {
    let pi = std::f64::consts::PI;
    let all = [
        -1.0 / pi,
        -0.5,
        -2.0 / (3.0 * pi),
        0.0,
        2.0 / (15.0 * pi),
        0.0,
        -2.0 / (35.0 * pi),
        0.0,
    ];
    let count = (k_steps as usize).min(all.len());
    if out.capacity() < count {
        out.reserve(count.saturating_sub(out.len()));
    }
    out.clear();
    out.extend(all.iter().take(k_steps as usize).copied());
}

/// Compute negative-truncation Chebyshev coefficients of length `k_steps`.
#[must_use]
pub fn negative_truncator_coeffs_witness(k_steps: u32) -> Vec<f64> {
    let mut out = Vec::new();
    negative_truncator_coeffs_witness_into(k_steps, &mut out);
    out
}

/// Derive fusion-affinity scores into caller-owned storage.
pub fn fusion_affinity_witness_into(transport_residual: &[f64], out: &mut Vec<f64>) {
    if out.capacity() < transport_residual.len() {
        out.reserve(transport_residual.len().saturating_sub(out.len()));
    }
    out.clear();
    out.extend(transport_residual.iter().map(|&v| -v.abs()));
}

/// Derive fusion-affinity scores from transport residual.
#[must_use]
pub fn fusion_affinity_witness(transport_residual: &[f64]) -> Vec<f64> {
    let mut out = Vec::new();
    fusion_affinity_witness_into(transport_residual, &mut out);
    out
}
/// Sequential Euler predictor for one homotopy continuation step.
#[must_use]
pub fn homotopy_euler_predictor_witness(state: &[f64], velocity: &[f64], step: f64) -> Vec<f64> {
    state
        .iter()
        .zip(velocity)
        .map(|(&state, &velocity)| state + step * velocity)
        .collect()
}

/// Sequential linear homotopy between compatible vectors.
#[must_use]
pub fn linear_homotopy_witness(start: &[f64], end: &[f64], parameter: f64) -> Vec<f64> {
    start
        .iter()
        .zip(end)
        .map(|(&start, &end)| (1.0 - parameter) * start + parameter * end)
        .collect()
}
/// One neutral scale-aware telemetry sample for reference megakernel scheduling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MegakernelScaleSampleWitness {
    /// Observed candidate dispatch cost in nanoseconds.
    pub dispatch_cost_ns: f64,
    /// Observed active-frontier density in `[0, 1]`.
    pub frontier_density: f64,
    /// Observed final readback byte volume.
    pub readback_bytes: u64,
}

/// Pure mathematical calculation of launch dominance factor.
#[must_use]
pub fn launch_dominance_witness(launch_overhead_ns: f64, candidate_cost_ns: f64) -> f64 {
    let denom = launch_overhead_ns + candidate_cost_ns;
    if denom == 0.0 {
        0.0
    } else {
        (launch_overhead_ns / denom).clamp(0.0, 1.0)
    }
}

/// Pure mathematical calculation of scale-aware fusion pressure.
#[must_use]
pub fn scale_aware_pressure_witness(
    cost_pressure: f64,
    readback_pressure: f64,
    launch_pressure: f64,
    frontier_pressure: f64,
) -> f64 {
    let density_adjusted_cost = cost_pressure * (0.65 + 0.35 * frontier_pressure);
    (0.55 * density_adjusted_cost
        + 0.25 * readback_pressure
        + 0.15 * launch_pressure
        + 0.05 * frontier_pressure)
        .clamp(0.0, 1.0)
}

/// Sequential homotopy continuation schedule witness over raw cost slices into caller-owned storage.
pub fn schedule_via_homotopy_witness_into(
    costs: &[f64],
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) {
    let n = costs.len();
    out.clear();
    out.resize(n, 0.0);
    if n == 0 || n_steps == 0 {
        return;
    }
    let max_cost = costs
        .iter()
        .copied()
        .fold(0.0f64, |max, cost| max.max(cost));
    if max_cost == 0.0 {
        return;
    }
    let step_size = dt.clamp(0.0, 1.0);
    let inv_max_cost = 1.0 / max_cost;
    for step in 0..n_steps {
        let alpha = f64::from(step + 1) / f64::from(n_steps);
        for (value, &cost) in out.iter_mut().zip(costs) {
            let cost_pressure = cost * inv_max_cost;
            let target = alpha * cost_pressure;
            *value += step_size * (target - *value);
        }
    }
    for value in out.iter_mut() {
        *value = value.clamp(0.0, 1.0);
    }
}

/// Sequential homotopy continuation schedule witness over raw cost slices.
#[must_use]
pub fn schedule_via_homotopy_witness(costs: &[f64], n_steps: u32, dt: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(costs.len());
    schedule_via_homotopy_witness_into(costs, n_steps, dt, &mut out);
    out
}

/// Sequential scale-aware homotopy schedule witness over neutral slices into caller-owned storage.
pub fn schedule_via_scale_aware_telemetry_witness_into(
    costs: &[f64],
    frontier_density: &[f64],
    readback_bytes: &[u64],
    launch_overhead_ns: f64,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) {
    let n = costs.len();
    out.clear();
    out.resize(n, 0.0);
    if n == 0 || n_steps == 0 {
        return;
    }
    let max_cost = costs
        .iter()
        .copied()
        .fold(0.0f64, |max, cost| max.max(cost));
    let max_readback = readback_bytes.iter().copied().max().unwrap_or(0);
    if max_cost == 0.0 && max_readback == 0 && launch_overhead_ns == 0.0 {
        return;
    }
    let step_size = dt.clamp(0.0, 1.0);
    let inv_max_cost = if max_cost == 0.0 { 0.0 } else { 1.0 / max_cost };
    let inv_max_readback = if max_readback == 0 {
        0.0
    } else {
        1.0 / max_readback as f64
    };
    for step in 0..n_steps {
        let alpha = f64::from(step + 1) / f64::from(n_steps);
        for i in 0..n {
            let cost_pressure = costs[i] * inv_max_cost;
            let readback_pressure = readback_bytes[i] as f64 * inv_max_readback;
            let launch_pressure = launch_dominance_witness(launch_overhead_ns, costs[i]);
            let frontier_pressure = frontier_density[i];
            let target = alpha
                * scale_aware_pressure_witness(
                    cost_pressure,
                    readback_pressure,
                    launch_pressure,
                    frontier_pressure,
                );
            out[i] += step_size * (target - out[i]);
        }
    }
    for value in out.iter_mut() {
        *value = value.clamp(0.0, 1.0);
    }
}

/// Sequential scale-aware homotopy schedule witness over neutral samples into caller-owned storage.
pub fn schedule_via_scale_aware_samples_witness_into(
    samples: &[MegakernelScaleSampleWitness],
    launch_overhead_ns: f64,
    n_steps: u32,
    dt: f64,
    out: &mut Vec<f64>,
) {
    let n = samples.len();
    out.clear();
    out.resize(n, 0.0);
    if n == 0 || n_steps == 0 {
        return;
    }
    let max_cost = samples
        .iter()
        .fold(0.0f64, |max, s| max.max(s.dispatch_cost_ns));
    let max_readback = samples.iter().map(|s| s.readback_bytes).max().unwrap_or(0);
    if max_cost == 0.0 && max_readback == 0 && launch_overhead_ns == 0.0 {
        return;
    }
    let step_size = dt.clamp(0.0, 1.0);
    let inv_max_cost = if max_cost == 0.0 { 0.0 } else { 1.0 / max_cost };
    let inv_max_readback = if max_readback == 0 {
        0.0
    } else {
        1.0 / max_readback as f64
    };
    for step in 0..n_steps {
        let alpha = f64::from(step + 1) / f64::from(n_steps);
        for (value, sample) in out.iter_mut().zip(samples) {
            let cost = sample.dispatch_cost_ns;
            let cost_pressure = cost * inv_max_cost;
            let readback_pressure = sample.readback_bytes as f64 * inv_max_readback;
            let launch_pressure = launch_dominance_witness(launch_overhead_ns, cost);
            let frontier_pressure = sample.frontier_density;
            let target = alpha
                * scale_aware_pressure_witness(
                    cost_pressure,
                    readback_pressure,
                    launch_pressure,
                    frontier_pressure,
                );
            *value += step_size * (target - *value);
        }
    }
    for value in out.iter_mut() {
        *value = value.clamp(0.0, 1.0);
    }
}

/// Sequential scale-aware homotopy schedule witness over neutral samples.
#[must_use]
pub fn schedule_via_scale_aware_samples_witness(
    samples: &[MegakernelScaleSampleWitness],
    launch_overhead_ns: f64,
    n_steps: u32,
    dt: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(samples.len());
    schedule_via_scale_aware_samples_witness_into(
        samples,
        launch_overhead_ns,
        n_steps,
        dt,
        &mut out,
    );
    out
}

/// Fallible sequential floating-point Sinkhorn scaling iteration into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn try_sinkhorn_iterate_f64_witness_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
    u_out: &mut Vec<f64>,
    v_out: &mut Vec<f64>,
    u_old: &mut Vec<f64>,
) -> Result<u32, String> {
    let (m, n) = (a.len(), b.len());
    if k.len() != m * n || !(tolerance > 0.0 && tolerance.is_finite()) {
        return Err(format!(
            "invalid sinkhorn parameters: k.len={}, m={m}, n={n}, tol={tolerance}",
            k.len()
        ));
    }
    u_out.clear();
    u_out.resize(m, 1.0_f64);
    v_out.clear();
    v_out.resize(n, 1.0_f64);
    u_old.clear();
    u_old.resize(m, 0.0_f64);

    for iter in 0..max_iterations {
        u_old.copy_from_slice(u_out);

        let step_f64 = |in_v: &[f64],
                        tgt: &[f64],
                        out_v: &mut [f64],
                        rows: usize,
                        cols: usize,
                        trans: bool| {
            for r in 0..rows {
                let mut sum = 0.0_f64;
                for c in 0..cols {
                    let idx = if trans { c * rows + r } else { r * cols + c };
                    sum += k[idx] * in_v[c];
                }
                out_v[r] = if sum == 0.0 { 0.0 } else { tgt[r] / sum };
            }
        };
        step_f64(v_out, a, u_out, m, n, false);
        step_f64(u_out, b, v_out, n, m, true);

        let max_delta = u_out
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

/// Fallible sequential floating-point Sinkhorn scaling iteration.
pub fn try_sinkhorn_iterate_f64_witness(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> Result<(Vec<f64>, Vec<f64>, u32), String> {
    let (mut u, mut v, mut u_old) = (Vec::new(), Vec::new(), Vec::new());
    let iters = try_sinkhorn_iterate_f64_witness_into(
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

/// Sequential floating-point Sinkhorn scaling iteration.
///
/// # Panics
///
/// Panics if input vector lengths do not match matrix dimensions or if tolerance is non-positive or non-finite.
#[must_use]
pub fn sinkhorn_iterate_f64_witness(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> (Vec<f64>, Vec<f64>, u32) {
    try_sinkhorn_iterate_f64_witness(k, a, b, tolerance, max_iterations)
        .unwrap_or_else(|error| panic!("Sinkhorn f64 witness failed: {error}"))
}

fn sinkhorn_residual(k: &[f64], u: &[f64], v: &[f64], target: &[f64], trans: bool) -> f64 {
    let (m, n) = (u.len(), v.len());
    assert_eq!(k.len(), m * n);
    let (outer, inner) = if trans { (n, m) } else { (m, n) };
    assert_eq!(target.len(), outer);
    target
        .iter()
        .enumerate()
        .map(|(o, &exp)| {
            let act: f64 = (0..inner)
                .map(|i| {
                    let (r, c) = if trans { (i, o) } else { (o, i) };
                    u[r] * k[r * n + c] * v[c]
                })
                .sum();
            (act - exp).abs()
        })
        .fold(0.0, f64::max)
}

/// Sequential floating-point Sinkhorn row residual calculation.
#[must_use]
pub fn sinkhorn_row_residual_witness(k: &[f64], u: &[f64], v: &[f64], a: &[f64]) -> f64 {
    sinkhorn_residual(k, u, v, a, false)
}

/// Sequential floating-point Sinkhorn column residual calculation.
#[must_use]
pub fn sinkhorn_col_residual_witness(k: &[f64], u: &[f64], v: &[f64], b: &[f64]) -> f64 {
    sinkhorn_residual(k, u, v, b, true)
}

/// One sequential step of floating-point Sinkhorn normalization into caller-owned storage.
pub fn sinkhorn_iter_f64_step_witness_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    v_in: &[f64],
    m: u32,
    n: u32,
    u_out: &mut Vec<f64>,
    v_out: &mut Vec<f64>,
) {
    let (m, n) = (m as usize, n as usize);
    u_out.clear();
    u_out.resize(m, 0.0_f64);
    v_out.clear();
    v_out.resize(n, 0.0_f64);
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..n {
            sum += k.get(i * n + j).copied().unwrap_or(0.0) * v_in.get(j).copied().unwrap_or(0.0);
        }
        u_out[i] = if sum == 0.0 {
            0.0
        } else {
            a.get(i).copied().unwrap_or(0.0) / sum
        };
    }
    for j in 0..n {
        let mut sum = 0.0;
        for i in 0..m {
            sum += k.get(i * n + j).copied().unwrap_or(0.0) * u_out[i];
        }
        v_out[j] = if sum == 0.0 {
            0.0
        } else {
            b.get(j).copied().unwrap_or(0.0) / sum
        };
    }
}

/// One sequential step of floating-point Sinkhorn normalization.
#[must_use]
pub fn sinkhorn_iter_f64_step_witness(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    _u_in: &[f64],
    v_in: &[f64],
    m: u32,
    n: u32,
) -> (Vec<f64>, Vec<f64>) {
    let mut u_out = Vec::new();
    let mut v_out = Vec::new();
    sinkhorn_iter_f64_step_witness_into(k, a, b, v_in, m, n, &mut u_out, &mut v_out);
    (u_out, v_out)
}

/// Fallible sequential one-step Sinkhorn iteration updating caller-owned `u` and `v` in place with scratch buffers.
#[allow(clippy::too_many_arguments)]
pub fn try_sinkhorn_iter_f64_in_place_witness_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    u: &mut [f64],
    v: &mut [f64],
    m: u32,
    n: u32,
    kv: &mut Vec<f64>,
    ktu: &mut Vec<f64>,
) -> Result<(), String> {
    let m = usize::try_from(m)
        .map_err(|_| format!("sinkhorn_iter witness m={m} does not fit usize."))?;
    let n = usize::try_from(n)
        .map_err(|_| format!("sinkhorn_iter witness n={n} does not fit usize."))?;
    m.checked_mul(n)
        .ok_or_else(|| format!("sinkhorn_iter witness K shape overflows: m={m}, n={n}."))?;

    let step = |k_slice: &[f64],
                in_v: &[f64],
                target_a: &[f64],
                out_u: &mut [f64],
                rows: usize,
                cols: usize,
                trans: bool,
                scr: &mut Vec<f64>| {
        if scr.capacity() < rows {
            scr.reserve(rows.saturating_sub(scr.len()));
        }
        scr.clear();
        scr.resize(rows, 0.0);
        for r in 0..rows {
            for c in 0..cols {
                let idx = if trans { c * rows + r } else { r * cols + c };
                scr[r] +=
                    k_slice.get(idx).copied().unwrap_or(0.0) * in_v.get(c).copied().unwrap_or(0.0);
            }
            if let Some(slot) = out_u.get_mut(r) {
                *slot = target_a.get(r).copied().unwrap_or(0.0) / scr[r].max(1e-30);
            }
        }
    };
    step(k, v, a, u, m, n, false, kv);
    step(k, u, b, v, n, m, true, ktu);
    Ok(())
}

/// Sequential one-step Sinkhorn iteration updating caller-owned `u` and `v` in place with scratch buffers.
///
/// # Panics
///
/// Panics if buffer shapes do not match `m` and `n` or if `m * n` overflows `usize`.
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_iter_f64_in_place_witness_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    u: &mut [f64],
    v: &mut [f64],
    m: u32,
    n: u32,
    kv: &mut Vec<f64>,
    ktu: &mut Vec<f64>,
) {
    try_sinkhorn_iter_f64_in_place_witness_into(k, a, b, u, v, m, n, kv, ktu).expect(
        "Fix: ensure m and n fit usize with m*n within bounds for Sinkhorn kernel matrix iteration",
    );
}

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
            let dot: f32 = (0..in_dim)
                .map(|k| {
                    input.get(k).copied().unwrap_or(0.0)
                        * weights.get(k * out_dim + j).copied().unwrap_or(0.0)
                })
                .sum();
            b + dot * inv_scale
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
