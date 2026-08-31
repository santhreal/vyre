//! Sequential Sinkhorn f64 scaling and residual witnesses.

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
