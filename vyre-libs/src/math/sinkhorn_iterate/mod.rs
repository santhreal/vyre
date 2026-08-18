//! Full iterative Sinkhorn balance.
//!
//! Alternates row-normalize and column-normalize until converged.
//! Composes `sinkhorn_scale` + `semiring_gemm` + a persistent fixpoint
//! harness.

mod program;

#[cfg(test)]
mod f64_tests;
#[cfg(test)]
#[path = "../../../tests/internal/math/sinkhorn_iterate/mod.rs"]
mod tests;

pub use program::sinkhorn_iterate;

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_ref(
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cpu_ref(
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_ref_into(
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cpu_ref_into(
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

    if m_usize > u_out.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            u_out,
            m_usize - u_out.len(),
            "Sinkhorn iterate CPU oracle",
            "u output",
        )?;
    }
    if n_usize > v_out.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            v_out,
            n_usize - v_out.len(),
            "Sinkhorn iterate CPU oracle",
            "v output",
        )?;
    }
    if m_usize > u_old.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            u_old,
            m_usize - u_old.len(),
            "Sinkhorn iterate CPU oracle",
            "u convergence scratch",
        )?;
    }

    vyre_reference::composition_witness::try_sinkhorn_iterate_witness_into(
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
}

#[cfg(test)]
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

#[cfg(test)]
fn require_fixed_len(name: &str, got: usize, need: usize) -> Result<(), String> {
    if got < need {
        Err(format!(
            "sinkhorn_iterate CPU oracle buffer `{name}` is too short: got {got}, need {need}."
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn sinkhorn_iterate_f64(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> (Vec<f64>, Vec<f64>, u32) {
    match try_sinkhorn_iterate_f64(k, a, b, tolerance, max_iterations) {
        Ok(result) => result,
        Err(error) => panic!("Sinkhorn iterate CPU reference failed: {error}"),
    }
}

#[cfg(test)]
pub(crate) fn try_sinkhorn_iterate_f64(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> Result<(Vec<f64>, Vec<f64>, u32), String> {
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut u_old = Vec::new();
    let iters = try_sinkhorn_iterate_f64_into(
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sinkhorn_iterate_f64_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    u_old: &mut Vec<f64>,
) -> u32 {
    match try_sinkhorn_iterate_f64_into(k, a, b, tolerance, max_iterations, u, v, u_old) {
        Ok(iters) => iters,
        Err(error) => panic!("Sinkhorn iterate CPU reference failed: {error}"),
    }
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_sinkhorn_iterate_f64_into(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
    u: &mut Vec<f64>,
    v: &mut Vec<f64>,
    u_old: &mut Vec<f64>,
) -> Result<u32, String> {
    let m = a.len();
    let n = b.len();
    if k.len() != m * n || tolerance <= 0.0 || !tolerance.is_finite() {
        return Err(format!(
            "sinkhorn_iterate_f64 requires k.len()==a.len()*b.len() and finite positive tolerance, got k={}, m={m}, n={n}, tolerance={tolerance}.",
            k.len()
        ));
    }

    if m > u.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            u,
            m - u.len(),
            "Sinkhorn iterate f64 CPU oracle",
            "u output",
        )?;
    }
    if n > v.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            v,
            n - v.len(),
            "Sinkhorn iterate f64 CPU oracle",
            "v output",
        )?;
    }
    if m > u_old.capacity() {
        crate::plumbing::host::scratch::reserve_items(
            u_old,
            m - u_old.len(),
            "Sinkhorn iterate f64 CPU oracle",
            "u convergence scratch",
        )?;
    }

    vyre_reference::composition_witness::try_sinkhorn_iterate_f64_witness_into(
        k,
        a,
        b,
        tolerance,
        max_iterations,
        u,
        v,
        u_old,
    )
}

#[cfg(test)]
pub(crate) fn sinkhorn_row_residual(k: &[f64], u: &[f64], v: &[f64], a: &[f64]) -> f64 {
    vyre_reference::composition_witness::sinkhorn_row_residual_witness(k, u, v, a)
}

#[cfg(test)]
pub(crate) fn sinkhorn_col_residual(k: &[f64], u: &[f64], v: &[f64], b: &[f64]) -> f64 {
    vyre_reference::composition_witness::sinkhorn_col_residual_witness(k, u, v, b)
}
/// Stable registry id for the iterative Sinkhorn primitive.
pub const OP_ID: &str = "vyre-libs::math::sinkhorn_iterate";

/// The ten buffer bindings one iterative-Sinkhorn program declares.
///
/// Every one of them is a `&str`, so a positional call of the ten accepted any
/// permutation of them, and one caller took that offer: the crate's own IR
/// parity test passed the names in BINDING order rather than parameter order,
/// so the emitted program named its kernel matrix `u_curr`, its `u` ping-pong
/// half `k_t`, and its convergence flag `kv`. Nothing noticed, because that
/// test feeds `reference_eval` by binding index, where a name is only a label.
/// A consumer that binds by name reads the scaling vector as the kernel.
/// Naming each binding at the construction site is what makes a transposition a
/// diff instead of a silent argument swap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SinkhornBuffers<'a> {
    /// `m x n` kernel matrix.
    pub k: &'a str,
    /// `n x m` transposed kernel matrix.
    pub k_t: &'a str,
    /// `m` target marginals.
    pub a: &'a str,
    /// `n` target marginals.
    pub b: &'a str,
    /// `m` elements, current half of the `u` ping-pong.
    pub u_curr: &'a str,
    /// `m` elements, next half of the `u` ping-pong.
    pub u_next: &'a str,
    /// `n` elements, current state for `v`.
    pub v: &'a str,
    /// `m` elements of `K v` scratch.
    pub kv: &'a str,
    /// `n` elements of `K_T u` scratch.
    pub ktu: &'a str,
    /// Convergence flag. One element on the single-workgroup form,
    /// `max_iterations` elements on the grid form; see the form note on
    /// [`sinkhorn_iterate`].
    pub changed: &'a str,
}

impl SinkhornBuffers<'static> {
    /// The canonical binding names for a Sinkhorn program.
    ///
    /// A caller that has no naming of its own gets one here instead of
    /// inventing ten strings, and every program built from it declares the
    /// same bindings in the same order, which is what makes two such programs
    /// comparable byte for byte.
    pub const CANONICAL: Self = Self {
        k: "k",
        k_t: "kt",
        a: "a",
        b: "b",
        u_curr: "uc",
        u_next: "un",
        v: "v",
        kv: "kv",
        ktu: "ktu",
        changed: "c",
    };
}

/// The problem extents and iteration cap one iterative-Sinkhorn program is
/// built for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SinkhornExtents {
    /// Row count of the kernel matrix, and the length of `a`, `u_curr`,
    /// `u_next` and `kv`.
    pub m: u32,
    /// Column count of the kernel matrix, and the length of `b`, `v` and `ktu`.
    pub n: u32,
    /// Hard cap on iterations.
    pub max_iterations: u32,
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        OP_ID,
        || {
            sinkhorn_iterate(
                SinkhornBuffers::CANONICAL,
                SinkhornExtents {
                    m: 2,
                    n: 2,
                    max_iterations: 5,
                },
            )
        },
        Some(|| {
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[65536, 65536]), // u_curr
                to_bytes(&[0, 0]), // u_next
                to_bytes(&[0]), // changed
                to_bytes(&[65536, 65536, 65536, 65536]), // k
                to_bytes(&[65536, 65536, 65536, 65536]), // k_t
                to_bytes(&[32768, 32768]), // a
                to_bytes(&[32768, 32768]), // b
                to_bytes(&[65536, 65536]), // v
                to_bytes(&[0, 0]), // kv
                to_bytes(&[0, 0]), // ktu
            ]]
        }),
        Some(|| {
            vec![vec![
                // u_curr: [32768, 32768]
                vec![0x00, 0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00],
                // u_next: [32768, 32768]
                vec![0x00, 0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00],
                // changed: [0]
                vec![0x00, 0x00, 0x00, 0x00],
                // v: [32768, 32768]
                vec![0x00, 0x80, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00],
                // kv: [0, 0]
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                // ktu: [0, 0]
                vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ]]
        }),
    )
}
