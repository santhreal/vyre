//! Sequential homotopy scheduling, scale-aware telemetry, and Sinkhorn f64 witnesses.

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

    let mut kv = Vec::new();
    let mut ktu = Vec::new();
    for iter in 0..max_iterations {
        u_old.copy_from_slice(u_out);
        try_sinkhorn_iter_f64_in_place_witness_into(
            k, a, b, u_out, v_out, m as u32, n as u32, &mut kv, &mut ktu,
        )?;

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
    let (m_us, n_us) = (m as usize, n as usize);
    u_out.clear();
    u_out.resize(m_us, 0.0);
    v_out.clear();
    v_out.extend_from_slice(v_in);
    if v_out.len() < n_us {
        v_out.resize(n_us, 0.0);
    }
    let mut kv = Vec::new();
    let mut ktu = Vec::new();
    let _ =
        try_sinkhorn_iter_f64_in_place_witness_into(k, a, b, u_out, v_out, m, n, &mut kv, &mut ktu);
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
