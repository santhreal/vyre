//! Sequential simplicial topology, Fast Multipole Method (FMM), Mori-Zwanzig, and QSVT witnesses.

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
        .expect("Fix: provide charges and cell assignments with matching lengths and representable cell ids");
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
        .expect("Fix: provide representable cell ids and sufficient memory capacity for P2M moment aggregation");
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
        .expect("Fix: provide a square cell_distances matrix matching cell_moments.len() * cell_moments.len()");
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
        .expect("Fix: provide positive n and k, assignments of length n with values less than k, and n * n within usize bounds");
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
    .expect("Fix: provide positive n and k, assignments of length n with values less than k, and n * n within usize bounds");
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
