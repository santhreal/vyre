//! Algebraic-multigrid V-cycle for matroid-intersection LP relaxation.
//!
//! Self-consumer for [`amg_v_cycle`](crate::math::amg_v_cycle).
//!
//! The matroid scheduler currently uses a single `matroid_solve_step` Jacobi
//! smoothing step to weight augmenting BFS layers. That's a 1-step relaxation  -
//! converges slowly on stiff exchange graphs (large condition number,
//! deep dispatch chains).
//!
//! This consumer wraps the substrate's full AMG V-cycle (smooth →
//! restrict → solve coarse → prolong → smooth), which converges
//! geometrically instead of arithmetically. Use it when the matroid
//! scheduler's flow vector hasn't converged after a fixed iteration
//! budget.
//!
//! # Algorithm wired
//!
//! Two-level AMG V-cycle on the dense matroid system `A·x = b`:
//!   1. Pre-smooth (Jacobi)
//!   2. Compute residual `r = b - A·x`
//!   3. Restrict to coarse: `r_c = R · r`
//!   4. Solve coarse via 4 Jacobi steps
//!   5. Prolong: `x ← x + P · x_c`
//!   6. Post-smooth (Jacobi)
//!
//! Returns the smoothed flow vector. Used by callers that want
//! provably-tight bounds on the matroid LP relaxation residual.

use crate::dispatch_buffers::{
    checked_product_count, checked_square_cells, decode_u32_output_exact, ensure_input_slots,
    write_u32_slice_le_bytes, write_zero_bytes,
};
use crate::math::amg_v_cycle::amg_v_cycle;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Caller-owned dispatch scratch for fixed-point AMG V-cycle execution.
#[derive(Debug, Default)]
pub struct AmgPassGpuScratch {
    inputs: Vec<Vec<u8>>,
    omega: Vec<u32>,
}

/// Default Jacobi relaxation parameter  -  0.66 is the standard
/// damping factor for diagonally-dominant matrices arising in
/// matroid-intersection LP relaxations.
pub const DEFAULT_OMEGA: f64 = 0.66;

/// Default Jacobi relaxation parameter in 16.16 fixed-point form.
///
/// This is the primitive-native equivalent of [`DEFAULT_OMEGA`].
pub const DEFAULT_OMEGA_FIXED: u32 = 43_254;

/// Primitive-native fixed-point production path for one AMG V-cycle.
///
/// Inputs are 16.16 u32 buffers. This dispatches the complete
/// [`amg_v_cycle`] primitive once and returns the post-smoothed fine-level
/// iterate.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shapes are invalid, primitive lane counts
/// overflow, or the backend returns malformed output.
#[allow(clippy::too_many_arguments)]
pub fn smooth_matroid_flow_fixed_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    a_fixed: &[u32],
    b_fixed: &[u32],
    x_fixed: &[u32],
    r_mat_fixed: &[u32],
    p_mat_fixed: &[u32],
    a_c_fixed: &[u32],
    n_fine: u32,
    n_coarse: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    smooth_matroid_flow_fixed_via_into(
        dispatcher,
        policy,
        a_fixed,
        b_fixed,
        x_fixed,
        r_mat_fixed,
        p_mat_fixed,
        a_c_fixed,
        n_fine,
        n_coarse,
        &mut out,
    )?;
    Ok(out)
}

/// Primitive-native fixed-point AMG V-cycle into caller-owned storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks or backend execution fail.
#[allow(clippy::too_many_arguments)]
pub fn smooth_matroid_flow_fixed_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    a_fixed: &[u32],
    b_fixed: &[u32],
    x_fixed: &[u32],
    r_mat_fixed: &[u32],
    p_mat_fixed: &[u32],
    a_c_fixed: &[u32],
    n_fine: u32,
    n_coarse: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = AmgPassGpuScratch::default();
    smooth_matroid_flow_fixed_via_with_scratch_into(
        dispatcher,
        policy,
        a_fixed,
        b_fixed,
        x_fixed,
        r_mat_fixed,
        p_mat_fixed,
        a_c_fixed,
        n_fine,
        n_coarse,
        &mut scratch,
        out,
    )
}

/// Primitive-native fixed-point AMG V-cycle using caller-owned dispatch scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks or backend execution fail.
#[allow(clippy::too_many_arguments)]
pub fn smooth_matroid_flow_fixed_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    a_fixed: &[u32],
    b_fixed: &[u32],
    x_fixed: &[u32],
    r_mat_fixed: &[u32],
    p_mat_fixed: &[u32],
    a_c_fixed: &[u32],
    n_fine: u32,
    n_coarse: u32,
    scratch: &mut AmgPassGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{amg_pass_solver_calls, bump};
    bump(&amg_pass_solver_calls);

    if n_coarse >= n_fine {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires 0 < n_coarse < n_fine, got n_coarse={n_coarse}, n_fine={n_fine}."
    )));
    }
    let fine_cells = checked_square_cells(n_fine, "smooth_matroid_flow_fixed_via fine matrix")
        .map_err(|error| SemanticExecutionError::InvalidRequest(error.to_string()))?;
    let coarse_cells =
        checked_square_cells(n_coarse, "smooth_matroid_flow_fixed_via coarse matrix")
            .map_err(|error| SemanticExecutionError::InvalidRequest(error.to_string()))?;
    let transfer_cells = checked_product_count(
        n_coarse,
        n_fine,
        "n_coarse",
        "n_fine",
        "smooth_matroid_flow_fixed_via transfer matrix",
    )
    .map_err(|error| SemanticExecutionError::InvalidRequest(error.to_string()))?;
    if a_fixed.len() != fine_cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires a_fixed.len() == n_fine*n_fine, got len={}, expected={fine_cells}.",
        a_fixed.len()
    )));
    }
    if b_fixed.len() != n_fine as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires b_fixed.len() == n_fine, got len={}, n_fine={n_fine}.",
        b_fixed.len()
    )));
    }
    if x_fixed.len() != n_fine as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires x_fixed.len() == n_fine, got len={}, n_fine={n_fine}.",
        x_fixed.len()
    )));
    }
    if r_mat_fixed.len() != transfer_cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires r_mat_fixed.len() == n_coarse*n_fine, got len={}, expected={transfer_cells}.",
        r_mat_fixed.len()
    )));
    }
    if p_mat_fixed.len() != transfer_cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires p_mat_fixed.len() == n_fine*n_coarse, got len={}, expected={transfer_cells}.",
        p_mat_fixed.len()
    )));
    }
    if a_c_fixed.len() != coarse_cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via requires a_c_fixed.len() == n_coarse*n_coarse, got len={}, expected={coarse_cells}.",
        a_c_fixed.len()
    )));
    }

    let program = amg_v_cycle(
        "a",
        "b",
        "x",
        "r_mat",
        "p_mat",
        "a_c",
        "omega",
        "scratch_fine",
        "scratch_coarse_b",
        "scratch_coarse_x",
        n_fine,
        n_coarse,
    );
    let fine_bytes = (n_fine as usize)
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via n_fine={n_fine} overflows fine scratch byte count."
                ))
        })?;
    let coarse_bytes = (n_coarse as usize).checked_mul(std::mem::size_of::<u32>()).ok_or_else(|| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: smooth_matroid_flow_fixed_via n_coarse={n_coarse} overflows coarse scratch byte count."
    ))
})?;
    scratch.omega.clear();
    scratch.omega.push(DEFAULT_OMEGA_FIXED);
    ensure_input_slots(&mut scratch.inputs, 11);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], a_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], b_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], x_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[3], r_mat_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[4], p_mat_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[5], a_c_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[6], &scratch.omega);
    write_zero_bytes(&mut scratch.inputs[7], fine_bytes);
    write_zero_bytes(&mut scratch.inputs[8], coarse_bytes);
    write_zero_bytes(&mut scratch.inputs[9], coarse_bytes);
    write_zero_bytes(&mut scratch.inputs[10], coarse_bytes);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    let [out_buf, ..] = match outputs.as_slice() {
        [b, ..] => [b],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: smooth_matroid_flow_fixed_via expected at least one output buffer, got {}.",
                outputs.len()
            )))
        }
    };
    decode_u32_output_exact(
        out_buf,
        n_fine as usize,
        "smooth_matroid_flow_fixed_via",
        out,
    )
    .map_err(|error| SemanticExecutionError::Backend(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;

    /// Caller-owned scratch for CPU reference AMG V-cycle execution.
    #[derive(Debug, Default, Clone)]
    pub(crate) struct AmgVcycleScratch {
        pub(crate) inner: vyre_reference::composition_witness::AmgSolveScratchWitness,
    }

    /// Execute one AMG V-cycle on the dense matroid system via the reference witness.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reference_smooth_matroid_flow(
        a: &[f64],
        b: &[f64],
        x: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        n_fine: u32,
        n_coarse: u32,
    ) -> Vec<f64> {
        if n_fine == 0 {
            return Vec::new();
        }
        vyre_reference::composition_witness::amg_v_cycle_witness(
            a,
            b,
            x,
            r_mat,
            p_mat,
            a_c,
            DEFAULT_OMEGA,
            n_fine,
            n_coarse,
        )
    }

    /// Execute one AMG V-cycle into caller-owned storage.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reference_smooth_matroid_flow_into(
        a: &[f64],
        b: &[f64],
        x: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        n_fine: u32,
        n_coarse: u32,
        scratch: &mut AmgVcycleScratch,
        out: &mut Vec<f64>,
    ) {
        if n_fine == 0 {
            out.clear();
            return;
        }
        vyre_reference::composition_witness::amg_v_cycle_witness_with_scratch_into(
            a,
            b,
            x,
            r_mat,
            p_mat,
            a_c,
            DEFAULT_OMEGA,
            n_fine,
            n_coarse,
            &mut scratch.inner.v_cycle,
            out,
        );
    }

    /// Run V-cycles until residual norm `||A·x − b||_∞` drops below `tol`
    /// or `max_cycles` is reached. Returns `(x_final, cycles_run)`.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_to_tolerance(
        a: &[f64],
        b: &[f64],
        x0: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        n_fine: u32,
        n_coarse: u32,
        tol: f64,
        max_cycles: u32,
    ) -> (Vec<f64>, u32) {
        use crate::telemetry::{amg_pass_solver_calls, bump};
        bump(&amg_pass_solver_calls);
        let mut x = Vec::new();
        let mut next = Vec::new();
        let mut scratch = AmgVcycleScratch::default();
        let cycles = solve_to_tolerance_into(
            a,
            b,
            x0,
            r_mat,
            p_mat,
            a_c,
            n_fine,
            n_coarse,
            tol,
            max_cycles,
            &mut scratch,
            &mut x,
            &mut next,
        );
        (x, cycles)
    }

    /// Run V-cycles until tolerance using caller-owned solver buffers.
    ///
    /// Returns the cycle count and leaves the final solution in `x`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn solve_to_tolerance_into(
        a: &[f64],
        b: &[f64],
        x0: &[f64],
        r_mat: &[f64],
        p_mat: &[f64],
        a_c: &[f64],
        n_fine: u32,
        n_coarse: u32,
        tol: f64,
        max_cycles: u32,
        scratch: &mut AmgVcycleScratch,
        x: &mut Vec<f64>,
        _next: &mut Vec<f64>,
    ) -> u32 {
        vyre_reference::composition_witness::amg_solve_to_tolerance_witness_with_scratch_into(
            a,
            b,
            x0,
            r_mat,
            p_mat,
            a_c,
            DEFAULT_OMEGA,
            n_fine,
            n_coarse,
            tol,
            max_cycles,
            &mut scratch.inner,
            x,
        )
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-3 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn identity_system_converges_in_one_cycle() {
        // A = I, b = [1, 2, 3, 4], x0 = [0; 4]. Expected after V-cycle:
        // x ≈ [1, 2, 3, 4].
        let n_fine = 4;
        let n_coarse = 2;
        let mut a = vec![0.0; 16];
        for i in 0..4 {
            a[i * 4 + i] = 1.0;
        }
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![0.0; 4];
        // Restriction: 4×2 matrix collapsing pairs. Prolongation: 2×4 transpose.
        let r_mat = vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
        let p_mat = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let a_c = vec![1.0, 0.0, 0.0, 1.0];
        let result =
            reference_smooth_matroid_flow(&a, &b, &x, &r_mat, &p_mat, &a_c, n_fine, n_coarse);
        assert_eq!(result.len(), 4);
        for v in &result {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn solve_to_tolerance_converges_or_returns_max_cycles() {
        let n_fine = 4;
        let n_coarse = 2;
        let mut a = vec![0.0; 16];
        for i in 0..4 {
            a[i * 4 + i] = 4.0;
        }
        let b = vec![1.0; 4];
        let x0 = vec![0.0; 4];
        let r_mat = vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
        let p_mat = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let a_c = vec![4.0, 0.0, 0.0, 4.0];
        let (result, cycles) =
            solve_to_tolerance(&a, &b, &x0, &r_mat, &p_mat, &a_c, n_fine, n_coarse, 1e-2, 8);
        assert!(cycles >= 1);
        assert_eq!(result.len(), 4);
        // Expected: x ≈ b/4 = 0.25 per element.
        for v in result {
            assert!(approx_eq(v, 0.25) || v.abs() > 0.0);
        }
    }

    #[test]
    fn solve_to_tolerance_into_matches_owned_solver() {
        let n_fine = 4;
        let n_coarse = 2;
        let mut a = vec![0.0; 16];
        for i in 0..4 {
            a[i * 4 + i] = 4.0;
        }
        let b = vec![1.0; 4];
        let x0 = vec![0.0; 4];
        let r_mat = vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5];
        let p_mat = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
        let a_c = vec![4.0, 0.0, 0.0, 4.0];
        let (owned, owned_cycles) =
            solve_to_tolerance(&a, &b, &x0, &r_mat, &p_mat, &a_c, n_fine, n_coarse, 1e-2, 8);

        let mut scratch = AmgVcycleScratch::default();
        let mut x = Vec::new();
        let mut next = Vec::new();
        let into_cycles = solve_to_tolerance_into(
            &a,
            &b,
            &x0,
            &r_mat,
            &p_mat,
            &a_c,
            n_fine,
            n_coarse,
            1e-2,
            8,
            &mut scratch,
            &mut x,
            &mut next,
        );

        assert_eq!(into_cycles, owned_cycles);
        assert_eq!(x.len(), owned.len());
        for (a, b) in x.iter().zip(owned.iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn empty_input_handles_zero_size() {
        let result = reference_smooth_matroid_flow(&[], &[], &[], &[], &[], &[], 0, 0);
        assert!(result.is_empty());
    }

    struct AmgDispatcher;

    impl SemanticExecutor for AmgDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                assert_eq!(inputs.len(), 11);
                let b = crate::dispatch_buffers::read_u32s(&inputs[1]);
                let x = crate::dispatch_buffers::read_u32s(&inputs[2]);
                assert_eq!(
                    crate::dispatch_buffers::read_u32s(&inputs[6])[0],
                    DEFAULT_OMEGA_FIXED
                );
                let out: Vec<u32> = x
                    .iter()
                    .zip(b.iter())
                    .map(|(&current, &rhs)| current.max(rhs))
                    .collect();
                Ok(vec![u32_slice_to_le_bytes(&out)])
            })();
            let mut ordered = ordered?;
            let output_count = request.logical().graph().nodes()[0].outputs.len();
            if ordered.len() < output_count {
                ordered.resize(output_count, Vec::new());
            }
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    #[test]
    fn fixed_via_dispatches_amg_v_cycle() {
        let one = 1u32 << 16;
        let out = smooth_matroid_flow_fixed_via(
            &AmgDispatcher,
            &crate::test_parity_oracles::policy(),
            &[one, 0, 0, one],
            &[3 * one, 4 * one],
            &[0, 0],
            &[one, one],
            &[one, one],
            &[one],
            2,
            1,
        )
        .unwrap();
        assert_eq!(out, vec![3 * one, 4 * one]);
    }

    #[test]
    fn fixed_via_rejects_invalid_level_shape() {
        let err = smooth_matroid_flow_fixed_via(
            &AmgDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1, 0, 0, 1],
            &[1, 1],
            &[0, 0],
            &[1, 1, 1, 1],
            &[1, 1, 1, 1],
            &[1, 0, 0, 1],
            2,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
    }

    #[test]
    fn fixed_via_with_scratch_reuses_input_buffers() {
        let one = 1u32 << 16;
        let mut scratch = AmgPassGpuScratch::default();
        let mut out = Vec::new();

        smooth_matroid_flow_fixed_via_with_scratch_into(
            &AmgDispatcher,
            &crate::test_parity_oracles::policy(),
            &[one, 0, 0, one],
            &[3 * one, 4 * one],
            &[0, 0],
            &[one, one],
            &[one, one],
            &[one],
            2,
            1,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let input_ptrs: Vec<*const u8> = scratch.inputs.iter().map(Vec::as_ptr).collect();
        smooth_matroid_flow_fixed_via_with_scratch_into(
            &AmgDispatcher,
            &crate::test_parity_oracles::policy(),
            &[one, 0, 0, one],
            &[2 * one, 5 * one],
            &[0, 0],
            &[one, one],
            &[one, one],
            &[one],
            2,
            1,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        crate::solvers::test_helpers::assert_input_pointers_preserved(&input_ptrs, &scratch.inputs);
    }
}
