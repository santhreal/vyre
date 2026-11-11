//! Sequential Tensor-Train contraction and fusion witnesses.

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
