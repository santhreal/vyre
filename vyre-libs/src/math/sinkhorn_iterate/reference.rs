//! Fixed-point u32 CPU reference for iterative Sinkhorn balance.

/// CPU reference for iterative Sinkhorn.
#[cfg(any(test, feature = "cpu-parity"))]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn cpu_ref(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_curr: &[u32],
    v: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
) -> (Vec<u32>, Vec<u32>, u32) {
    let mut u = Vec::new();
    let mut v_mut = Vec::new();
    let mut u_old = Vec::new();
    let iters = try_cpu_ref_into(
        k,
        k_t,
        a,
        b,
        u_curr,
        v,
        m,
        n,
        max_iterations,
        &mut u,
        &mut v_mut,
        &mut u_old,
    )
    .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - sinkhorn_iterate cpu_ref failed: invalid fixed-point Sinkhorn buffers");
    (u, v_mut, iters)
}

/// Fallible CPU reference for iterative Sinkhorn.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_cpu_ref(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_curr: &[u32],
    v: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
) -> Result<(Vec<u32>, Vec<u32>, u32), String> {
    let mut u = Vec::new();
    let mut v_mut = Vec::new();
    let mut u_old = Vec::new();
    let iters = try_cpu_ref_into(
        k,
        k_t,
        a,
        b,
        u_curr,
        v,
        m,
        n,
        max_iterations,
        &mut u,
        &mut v_mut,
        &mut u_old,
    )?;
    Ok((u, v_mut, iters))
}

/// CPU reference for iterative Sinkhorn using caller-owned buffers.
///
/// `u_out` and `v_out` receive the final states. `u_old` is retained
/// as convergence scratch to avoid cloning `u` every iteration.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn cpu_ref_into(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_curr: &[u32],
    v: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
    u_out: &mut Vec<u32>,
    v_out: &mut Vec<u32>,
    u_old: &mut Vec<u32>,
) -> u32 {
    try_cpu_ref_into(
        k,
        k_t,
        a,
        b,
        u_curr,
        v,
        m,
        n,
        max_iterations,
        u_out,
        v_out,
        u_old,
    )
    .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - sinkhorn_iterate cpu_ref_into failed: invalid fixed-point Sinkhorn buffers")
}

/// Fallible CPU reference for iterative Sinkhorn using caller-owned buffers.
#[cfg(any(test, feature = "cpu-parity"))]
#[allow(clippy::too_many_arguments)]
pub fn try_cpu_ref_into(
    k: &[u32],
    k_t: &[u32],
    a: &[u32],
    b: &[u32],
    u_curr: &[u32],
    v: &[u32],
    m: u32,
    n: u32,
    max_iterations: u32,
    u_out: &mut Vec<u32>,
    v_out: &mut Vec<u32>,
    u_old: &mut Vec<u32>,
) -> Result<u32, String> {
    let (m_usize, n_usize, matrix_cells) = checked_fixed_sinkhorn_shape(m, n)?;
    require_fixed_len("k", k.len(), matrix_cells)?;
    require_fixed_len("k_t", k_t.len(), matrix_cells)?;
    require_fixed_len("a", a.len(), m_usize)?;
    require_fixed_len("b", b.len(), n_usize)?;
    require_fixed_len("u_curr", u_curr.len(), m_usize)?;
    require_fixed_len("v", v.len(), n_usize)?;
    reserve_u32_vec(u_out, m_usize, "u output")?;
    reserve_u32_vec(v_out, n_usize, "v output")?;
    reserve_u32_vec(u_old, m_usize, "u convergence scratch")?;

    u_out.clear();
    u_out.extend_from_slice(&u_curr[..m_usize]);
    v_out.clear();
    v_out.extend_from_slice(&v[..n_usize]);

    let mut iters = 0;
    for iter in 0..max_iterations {
        u_old.clear();
        u_old.extend_from_slice(u_out);

        // 1 & 2. Kv & u
        for i in 0..m_usize {
            let mut sum = 0u32;
            for j in 0..n_usize {
                sum = sum.wrapping_add(k[i * n_usize + j].wrapping_mul(v_out[j]));
            }
            let divisor = if sum == 0 { 1 } else { sum };
            u_out[i] = a[i] / divisor;
        }

        // 3 & 4. Ktu & v
        for j in 0..n_usize {
            let mut sum = 0u32;
            for i in 0..m_usize {
                sum = sum.wrapping_add(k_t[j * m_usize + i].wrapping_mul(u_out[i]));
            }
            let divisor = if sum == 0 { 1 } else { sum };
            v_out[j] = b[j] / divisor;
        }

        if u_out == u_old {
            return Ok(iter);
        }
        iters = iter + 1;
    }
    Ok(iters)
}

#[cfg(any(test, feature = "cpu-parity"))]
fn checked_fixed_sinkhorn_shape(m: u32, n: u32) -> Result<(usize, usize, usize), String> {
    if m == 0 || n == 0 {
        return Err(format!(
            "sinkhorn_iterate CPU oracle requires non-zero dimensions, got m={m}, n={n}."
        ));
    }
    let m_usize =
        usize::try_from(m).map_err(|_| format!("sinkhorn_iterate m={m} does not fit usize."))?;
    let n_usize =
        usize::try_from(n).map_err(|_| format!("sinkhorn_iterate n={n} does not fit usize."))?;
    let matrix_cells = m_usize.checked_mul(n_usize).ok_or_else(|| {
        format!("sinkhorn_iterate CPU oracle matrix cells overflow: m={m}, n={n}.")
    })?;
    Ok((m_usize, n_usize, matrix_cells))
}

#[cfg(any(test, feature = "cpu-parity"))]
fn require_fixed_len(name: &str, got: usize, need: usize) -> Result<(), String> {
    if got < need {
        Err(format!(
            "sinkhorn_iterate CPU oracle buffer `{name}` is too short: got {got}, need {need}."
        ))
    } else {
        Ok(())
    }
}

crate::scratch::define_reserve_capacity!(
    reserve_u32_vec,
    u32,
    "Sinkhorn iterate CPU oracle"
);
