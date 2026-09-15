//! Sequential Algebraic Multigrid (AMG) and Jacobi solver witnesses.

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
