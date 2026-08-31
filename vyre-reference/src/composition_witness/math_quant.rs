//! Sequential quantized matrix operations, 1D convolution, and autotuning witnesses.

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
        .expect("Fix: supply a finite positive temperature parameter for differentiable autotune score gradient calculation");
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
