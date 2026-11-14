//! Sequential Algebraic Multigrid (AMG) and Jacobi solver witnesses.

fn reserve_at_least(buf: &mut Vec<f64>, target: usize) {
    if buf.capacity() < target {
        buf.reserve(target.saturating_sub(buf.len()));
    }
}

fn check_slice_len(slice: &[f64], expected: usize, label: &str) -> Result<(), String> {
    if slice.len() < expected {
        return Err(format!("{label}: expected {expected}, got {}", slice.len()));
    }
    Ok(())
}

fn validate_amg_buffers(
    fine_matrix: &[f64],
    fine_rhs: &[f64],
    initial: &[f64],
    restriction: &[f64],
    prolongation: &[f64],
    coarse_matrix: &[f64],
    nf: usize,
    nc: usize,
) -> Result<(), String> {
    check_slice_len(fine_matrix, nf * nf, "buffer `a` is too short")?;
    check_slice_len(fine_rhs, nf, "buffer `b` is too short")?;
    check_slice_len(initial, nf, "buffer `x` is too short")?;
    check_slice_len(restriction, nc * nf, "buffer `r_mat` is too short")?;
    check_slice_len(prolongation, nf * nc, "buffer `p_mat` is too short")?;
    check_slice_len(coarse_matrix, nc * nc, "buffer `a_c` is too short")
}

fn jacobi_sweep(
    matrix: &[f64],
    rhs: &[f64],
    current: &[f64],
    omega: f64,
    n: usize,
    dest: &mut Vec<f64>,
) {
    dest.clear();
    reserve_at_least(dest, n);
    for row in 0..n {
        let off_diagonal: f64 = (0..n)
            .filter(|&col| col != row)
            .map(|col| matrix[row * n + col] * current[col])
            .sum();
        let target = (rhs[row] - off_diagonal) / matrix[row * n + row];
        dest.push(current[row] + omega * (target - current[row]));
    }
}

fn max_abs_residual(matrix: &[f64], rhs: &[f64], x: &[f64], n: usize) -> f64 {
    let mut max_resid = 0.0_f64;
    for i in 0..n {
        let row_dot: f64 = (0..n).map(|j| matrix[i * n + j] * x[j]).sum();
        max_resid = max_resid.max((row_dot - rhs[i]).abs());
    }
    max_resid
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
    reserve_at_least(out, n);
    out.clear();
    for row in 0..n {
        let ax: f64 = (0..n)
            .map(|col| matrix[row * n + col] * solution[col])
            .sum();
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
        reserve_at_least(&mut self.fine, fine_count);
        reserve_at_least(&mut self.residual, fine_count);
        reserve_at_least(&mut self.coarse_rhs, coarse_count);
        reserve_at_least(&mut self.coarse, coarse_count);
        reserve_at_least(&mut self.coarse_next, coarse_count);
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
        reserve_at_least(&mut self.next_iterate, fine_count);
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
    validate_amg_buffers(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        nf,
        nc,
    )?;

    reserve_at_least(out, nf);
    out.clear();
    if nf == 0 {
        return Ok(());
    }

    jacobi_sweep(fine_matrix, fine_rhs, initial, omega, nf, &mut scratch.fine);
    amg_residual_witness_into(
        fine_matrix,
        fine_rhs,
        &scratch.fine,
        fine_count,
        &mut scratch.residual,
    );

    scratch.coarse_rhs.clear();
    reserve_at_least(&mut scratch.coarse_rhs, nc);
    for row in 0..nc {
        let val: f64 = (0..nf)
            .map(|col| restriction[row * nf + col] * scratch.residual[col])
            .sum();
        scratch.coarse_rhs.push(val);
    }

    scratch.coarse.clear();
    scratch.coarse.resize(nc, 0.0_f64);
    for _ in 0..4 {
        jacobi_sweep(
            coarse_matrix,
            &scratch.coarse_rhs,
            &scratch.coarse,
            omega,
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

    jacobi_sweep(fine_matrix, fine_rhs, &scratch.fine, omega, nf, out);
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
    validate_amg_buffers(
        fine_matrix,
        fine_rhs,
        initial,
        restriction,
        prolongation,
        coarse_matrix,
        nf,
        nc,
    )?;
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

    reserve_at_least(out, nf);
    reserve_at_least(&mut scratch.next_iterate, nf);
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
        if max_abs_residual(fine_matrix, fine_rhs, out, nf) < tolerance {
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
    check_slice_len(matrix, n_us * n_us, "matrix too short")?;
    check_slice_len(rhs, n_us, "rhs too short")?;
    check_slice_len(initial, n_us, "initial vector too short")?;
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

    reserve_at_least(out, n_us);
    reserve_at_least(scratch, n_us);
    out.clear();
    out.extend_from_slice(&initial[..n_us]);
    scratch.clear();

    for iter in 0..max_iters {
        jacobi_sweep(matrix, rhs, out, omega, n_us, scratch);
        out.clear();
        out.extend_from_slice(scratch);
        if max_abs_residual(matrix, rhs, out, n_us) < tolerance {
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
