//! Full iterative Sinkhorn balance.
//!
//! Alternates row-normalize and column-normalize until converged.
//! Composes `sinkhorn_scale` + `semiring_gemm` + a persistent fixpoint
//! harness.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Node, Program};

use crate::fixpoint::persistent_fixpoint::{
    persistent_fixpoint, persistent_fixpoint_grid, PERSISTENT_FIXPOINT_WORKGROUP_SIZE,
};
use crate::math::semiring_gemm::{semiring_gemm, Semiring};
use crate::math::sinkhorn::sinkhorn_scale;

/// Stable registry id for the iterative Sinkhorn primitive.
pub const OP_ID: &str = "vyre-primitives::math::sinkhorn_iterate";

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

/// Sinkhorn full iteration.
///
/// Runs Sinkhorn matrix-scaling iterations to convergence over the bindings
/// [`SinkhornBuffers`] names, at the extents [`SinkhornExtents`] fixes.
///
/// # Convergence-flag form
///
/// The launch spans `m * n` lanes, not `m`:
/// `dispatch_element_count_for_program`
/// (`vyre-driver/src/program_walks/dispatch_params.rs:19`) sizes an
/// atomic-carrying program's launch from its WIDEST declared buffer, this
/// program carries the harness's `atomic_or`, and the widest buffers are the
/// `m * n` kernel matrices `k` and `k_t`. So the cell count, not the scaling
/// vector length, selects the harness:
///
/// - `m * n <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`: one workgroup covers
///   the launch, so [`persistent_fixpoint`] runs with its single shared
///   `changed[0]` word. That word is cleared by a plain store fenced only by
///   a workgroup-scope barrier; with one group the fence is incidentally
///   grid-wide, so the clear cannot race the `atomic_or` that sets the flag.
/// - `m * n` above that width: [`persistent_fixpoint_grid`], which never
///   clears the flag, gives each iteration its own `changed` word, and
///   separates waves with `MemoryOrdering::GridSync`. The single-word form is
///   limited to one workgroup precisely because its clear and its set are
///   unordered across groups: group 0's clear can erase another group's set,
///   that group then reads 0 and returns early with an unbalanced scaling
///   vector, and the flag the host reads afterwards reports a convergence no
///   group agreed to.
///
/// A `17 x 17` problem is already 289 cells, so this threshold is crossed at
/// modest sizes with both extents far under one workgroup width.
#[must_use]
pub fn sinkhorn_iterate(buffers: SinkhornBuffers<'_>, extents: SinkhornExtents) -> Program {
    let SinkhornExtents {
        m,
        n,
        max_iterations,
    } = extents;
    if m == 0 {
        return crate::invalid_output_program(
            OP_ID,
            buffers.u_curr,
            DataType::U32,
            "Fix: sinkhorn_iterate requires m > 0, got 0.".to_string(),
        );
    }
    if n == 0 {
        return crate::invalid_output_program(
            OP_ID,
            buffers.u_curr,
            DataType::U32,
            "Fix: sinkhorn_iterate requires n > 0, got 0.".to_string(),
        );
    }
    let Some(matrix_cells) = m.checked_mul(n) else {
        return crate::invalid_output_program(
            OP_ID,
            buffers.u_curr,
            DataType::U32,
            format!("Fix: sinkhorn_iterate m*n overflows u32: {m}*{n}."),
        );
    };

    let transfer_body = sinkhorn_transfer_body(buffers, extents);

    // `m` alone does NOT decide the harness: see the form note above. `k` and
    // `k_t` are `m * n` long, which dominates `m` and `n` for any non-zero
    // extents, so the launch spans `matrix_cells` lanes and a modest matrix
    // makes the dispatch multi-workgroup while both extents still fit one group.
    let needs_grid_sync = matrix_cells > PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0];

    let inner = if needs_grid_sync {
        persistent_fixpoint_grid(
            transfer_body,
            buffers.u_curr,
            buffers.u_next,
            buffers.changed,
            m,
            max_iterations,
        )
    } else {
        persistent_fixpoint(
            transfer_body,
            buffers.u_curr,
            buffers.u_next,
            buffers.changed,
            m,
            max_iterations,
        )
    };

    // Mirrors the count the chosen harness declares for `changed`: one
    // never-cleared word per iteration for the grid form, which indexes
    // `changed[iteration]`, and one shared word for the single-workgroup form.
    let changed_words = if needs_grid_sync {
        max_iterations.max(1)
    } else {
        1
    };

    sinkhorn_wrap(&inner, buffers, extents, matrix_cells, changed_words)
}

/// One full Sinkhorn sweep: `Kv`, then `u`, then `Ktu`, then `v`.
///
/// Every composed pass is PARTITIONED by global invocation id, not chunked: the
/// gemms gate on `t < out_cells` (`m` for `Kv`, `n` for `Ktu`) and the scales on
/// `t < count`, and each lane sums over the shared dimension internally. So the
/// widest lane gate here is `max(m, n)`, which is what decides whether groups
/// above 0 own any state. It is NOT `m * n`: nothing walks the kernel matrices
/// one cell per lane, so a launch made multi-workgroup only by the size of `k`
/// and `k_t` leaves every gate inside group 0.
///
/// Takes the whole binding record even though the sweep never reads `u_curr` or
/// `changed`: those two belong to the convergence harness, and forwarding the
/// record is what keeps the sweep from restating a second copy of the list.
fn sinkhorn_transfer_body(buffers: SinkhornBuffers<'_>, extents: SinkhornExtents) -> Vec<Node> {
    let SinkhornBuffers {
        k,
        k_t,
        a,
        b,
        u_next,
        v,
        kv,
        ktu,
        ..
    } = buffers;
    let SinkhornExtents { m, n, .. } = extents;
    let extract_body = |p: Program| -> Vec<Node> {
        let mut body = Vec::new();
        for node in p.entry() {
            if let Node::Region {
                body: region_body, ..
            } = node
            {
                body.extend(region_body.iter().cloned());
            }
        }
        body
    };
    let seq_cst = || Node::Barrier {
        ordering: vyre_foundation::MemoryOrdering::SeqCst,
    };

    let mut transfer_body = Vec::new();

    // 1. Kv = K * v (m x n * n x 1 -> m x 1)
    transfer_body.extend(extract_body(semiring_gemm(
        k,
        v,
        kv,
        m,
        1,
        n,
        Semiring::Real,
    )));
    transfer_body.push(seq_cst());

    // 2. u_next = a ./ Kv
    transfer_body.extend(extract_body(sinkhorn_scale(a, kv, u_next, m)));
    transfer_body.push(seq_cst());

    // 3. Ktu = K_T * u_next (n x m * m x 1 -> n x 1)
    transfer_body.extend(extract_body(semiring_gemm(
        k_t,
        u_next,
        ktu,
        n,
        1,
        m,
        Semiring::Real,
    )));
    transfer_body.push(seq_cst());

    // 4. v = b ./ Ktu
    transfer_body.extend(extract_body(sinkhorn_scale(b, ktu, v, n)));
    transfer_body.push(seq_cst());

    transfer_body
}

/// Wrap a convergence harness in the Sinkhorn Region and buffer declarations.
///
/// Single owner of those ten declarations, so the two routed forms and the
/// single-word form the divergence test builds cannot drift apart in binding
/// order, counts, or access modes.
fn sinkhorn_wrap(
    inner: &Program,
    buffers: SinkhornBuffers<'_>,
    extents: SinkhornExtents,
    matrix_cells: u32,
    changed_words: u32,
) -> Program {
    let SinkhornBuffers {
        k,
        k_t,
        a,
        b,
        u_curr,
        u_next,
        v,
        kv,
        ktu,
        changed,
    } = buffers;
    let SinkhornExtents { m, n, .. } = extents;
    super::wrap_fixpoint_program(
        OP_ID,
        inner,
        vec![
            BufferDecl::storage(u_curr, 0, BufferAccess::ReadWrite, DataType::U32).with_count(m),
            BufferDecl::storage(u_next, 1, BufferAccess::ReadWrite, DataType::U32).with_count(m),
            BufferDecl::storage(changed, 2, BufferAccess::ReadWrite, DataType::U32)
                .with_count(changed_words),
            BufferDecl::storage(k, 3, BufferAccess::ReadOnly, DataType::U32)
                .with_count(matrix_cells),
            BufferDecl::storage(k_t, 4, BufferAccess::ReadOnly, DataType::U32)
                .with_count(matrix_cells),
            BufferDecl::storage(a, 5, BufferAccess::ReadOnly, DataType::U32).with_count(m),
            BufferDecl::storage(b, 6, BufferAccess::ReadOnly, DataType::U32).with_count(n),
            BufferDecl::storage(v, 7, BufferAccess::ReadWrite, DataType::U32).with_count(n),
            BufferDecl::storage(kv, 8, BufferAccess::ReadWrite, DataType::U32).with_count(m),
            BufferDecl::storage(ktu, 9, BufferAccess::ReadWrite, DataType::U32).with_count(n),
        ],
    )
}

/// The pre-routing program: the Sinkhorn transfer body on the single-word
/// convergence harness at ANY size, which is exactly what [`sinkhorn_iterate`]
/// emitted before the dispatch-span routing landed.
///
/// Exists only so the divergence test can OBSERVE what the racing shared flag
/// produces above one workgroup. Production code must never take this path above
/// one workgroup width.
#[cfg(test)]
fn sinkhorn_single_word_harness(buffers: SinkhornBuffers<'_>, extents: SinkhornExtents) -> Program {
    let matrix_cells = extents
        .m
        .checked_mul(extents.n)
        .expect("Fix: the divergence fixture must use non-overflowing extents.");
    let transfer_body = sinkhorn_transfer_body(buffers, extents);
    let inner = persistent_fixpoint(
        transfer_body,
        buffers.u_curr,
        buffers.u_next,
        buffers.changed,
        extents.m,
        extents.max_iterations,
    );
    sinkhorn_wrap(&inner, buffers, extents, matrix_cells, 1)
}

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

crate::graph::scratch::define_reserve_graph_capacity!(
    reserve_u32_vec,
    u32,
    "Sinkhorn iterate CPU oracle"
);

#[cfg(feature = "inventory-registry")]
inventory::submit! {
    vyre_foundation::operation::OperationRegistration::primitive(
        OP_ID,
        || {
            sinkhorn_iterate(
                SinkhornBuffers {
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
                },
                SinkhornExtents {
                    m: 2,
                    n: 2,
                    max_iterations: 5,
                },
            )
        },
        Some(|| {
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
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
            let to_bytes = |w: &[u32]| crate::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[32768, 32768]), // u_curr
                to_bytes(&[32768, 32768]), // u_next
                to_bytes(&[0]),            // changed
                to_bytes(&[32768, 32768]), // v
                to_bytes(&[0, 0]),         // kv
                to_bytes(&[0, 0]),         // ktu
            ]]
        }),
    )
}

#[cfg(test)]
mod tests;

// ===== P-PRIM-11: Full iterative-balance Sinkhorn (f64) ===========
//
// The fixed-point u32 cpu_ref above is the GPU-targeted reference;
// the math operates on quantized fractions. This block ships an
// f64 reference that performs the canonical Sinkhorn-Knopp iterative
// matrix-balancing algorithm with tolerance-based convergence  -
// the operation many user dialects ask for when they say "balanced
// transport plan."

/// Tolerance-based Sinkhorn-Knopp iterative balancing in f64.
///
/// Inputs:
/// - `k`: kernel matrix `m × n`, row-major. Strictly positive entries.
/// - `a`: target row marginal, length m. Strictly positive entries.
/// - `b`: target column marginal, length n. Strictly positive entries.
/// - `tolerance`: stop when `||u_new - u_old||_∞ < tolerance`.
/// - `max_iterations`: hard cap.
///
/// Returns `(u, v, iterations)` such that `diag(u) · k · diag(v)`
/// has row sums approximately `a` and column sums approximately `b`,
/// up to the supplied tolerance.
///
/// Pre/post conditions:
/// * Caller guarantees `sum(a) == sum(b)` (mass-conservation;
///   Sinkhorn-Knopp converges only on balanced marginals).
/// * Returns the iteration that stopped  -  < `max_iterations` means
///   tolerance reached, == `max_iterations` means cap hit.
///
/// # Panics
///
/// Panics on length mismatch.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_iterate_f64(
    k: &[f64],
    a: &[f64],
    b: &[f64],
    tolerance: f64,
    max_iterations: u32,
) -> (Vec<f64>, Vec<f64>, u32) {
    let mut u = Vec::new();
    let mut v = Vec::new();
    let mut u_old = Vec::new();
    let iters = sinkhorn_iterate_f64_into(
        k,
        a,
        b,
        tolerance,
        max_iterations,
        &mut u,
        &mut v,
        &mut u_old,
    );
    (u, v, iters)
}

/// Fallible tolerance-based Sinkhorn-Knopp iterative balancing in f64.
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_sinkhorn_iterate_f64(
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

/// Tolerance-based Sinkhorn-Knopp iterative balancing in f64 using
/// caller-owned buffers.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_iterate_f64_into(
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
        // Clearing the buffers and returning 0 iterations on failure makes a
        // GPU-vs-CPU parity assertion pass on empty==empty, silently masking a
        // divergence (Law 10 / Law 6). Fail loud; callers use the try_ variant.
        Err(error) => panic!("vyre-primitives Sinkhorn iterate CPU reference failed: {error}"),
    }
}

/// Fallible tolerance-based Sinkhorn-Knopp iterative balancing in f64 using
/// caller-owned buffers.
#[allow(clippy::too_many_arguments)]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn try_sinkhorn_iterate_f64_into(
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
    reserve_f64_vec(u, m, "u output")?;
    reserve_f64_vec(v, n, "v output")?;
    reserve_f64_vec(u_old, m, "u convergence scratch")?;

    u.clear();
    v.clear();
    u_old.clear();
    u.resize(m, 1.0_f64);
    v.resize(n, 1.0_f64);

    for iter in 0..max_iterations {
        u_old.clear();
        u_old.extend_from_slice(u);

        // u <- a / (k · v)
        for i in 0..m {
            let mut sum = 0.0_f64;
            for j in 0..n {
                sum += k[i * n + j] * v[j];
            }
            // Guard against division by zero  -  sinkhorn requires k > 0,
            // but defensive callers benefit from a non-NaN result.
            u[i] = if sum == 0.0 { 0.0 } else { a[i] / sum };
        }

        // v <- b / (kᵀ · u)
        for j in 0..n {
            let mut sum = 0.0_f64;
            for i in 0..m {
                sum += k[i * n + j] * u[i];
            }
            v[j] = if sum == 0.0 { 0.0 } else { b[j] / sum };
        }

        // Convergence check on u (Sinkhorn-Knopp stops when one
        // marginal is stable; the other follows by construction).
        let max_delta = u
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

crate::graph::scratch::define_reserve_graph_capacity!(
    reserve_f64_vec,
    f64,
    "Sinkhorn iterate f64 CPU oracle"
);

#[cfg(any(test, feature = "cpu-parity"))]
fn max_residual(target: &[f64], sum_at: impl Fn(usize) -> f64) -> f64 {
    target
        .iter()
        .enumerate()
        .map(|(index, expected)| (sum_at(index) - expected).abs())
        .fold(0.0_f64, f64::max)
}

#[cfg(any(test, feature = "cpu-parity"))]
#[derive(Clone, Copy)]
enum ResidualAxis {
    Row,
    Column,
}

#[cfg(any(test, feature = "cpu-parity"))]
fn sinkhorn_residual(k: &[f64], u: &[f64], v: &[f64], target: &[f64], axis: ResidualAxis) -> f64 {
    let m = u.len();
    let n = v.len();
    assert_eq!(k.len(), m * n);
    match axis {
        ResidualAxis::Row => {
            assert_eq!(target.len(), m);
            max_residual(target, |i| (0..n).map(|j| u[i] * k[i * n + j] * v[j]).sum())
        }
        ResidualAxis::Column => {
            assert_eq!(target.len(), n);
            max_residual(target, |j| (0..m).map(|i| u[i] * k[i * n + j] * v[j]).sum())
        }
    }
}

/// Compute the row-sum residual `||row_sum(diag(u) · k · diag(v)) - a||_∞`.
/// Useful for testing convergence of [`sinkhorn_iterate_f64`].
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_row_residual(k: &[f64], u: &[f64], v: &[f64], a: &[f64]) -> f64 {
    sinkhorn_residual(k, u, v, a, ResidualAxis::Row)
}

/// Compute the column-sum residual `||col_sum(diag(u) · k · diag(v)) - b||_∞`.
#[must_use]
#[cfg(any(test, feature = "cpu-parity"))]
pub fn sinkhorn_col_residual(k: &[f64], u: &[f64], v: &[f64], b: &[f64]) -> f64 {
    sinkhorn_residual(k, u, v, b, ResidualAxis::Column)
}

#[cfg(test)]
mod f64_tests;
