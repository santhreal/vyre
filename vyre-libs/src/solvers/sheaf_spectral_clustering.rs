//! Sheaf-spectral clustering of dispatch graphs.
//!
//! Self-consumer for [`sheaf_laplacian_eigenvalue`](crate::math::sheaf_laplacian_eigenvalue).
//!
//! The dispatch graph's sheaf Laplacian carries spectral information
//! about cluster structure: the dominant eigenvalue corresponds to
//! the longest principal direction of the graph's heterophilic
//! diffusion. Combined with the existing
//! `super::sheaf_heterophilic_dispatch::flag_fusion_incompatible`
//! divergence flagging, this gives:
//!
//! - **Spectral gap**  -  eigenvalue magnitude indicates how cleanly
//!   the graph splits into clusters. Large gap = clean clusters,
//!   safe to fuse within each cluster.
//! - **Suggested cluster count**  -  derived from the eigenvalue
//!   spectrum via the substrate's exact diagonal eigenpair output.
//!
//! Used by the megakernel scheduler when the matroid heuristic
//! produces ambiguous results (many tied gain values)  -  falls back
//! to spectral cluster suggestions for tie-breaking.

#[cfg(test)]
use crate::dispatch_buffers::u32_slice_to_le_bytes;
use crate::dispatch_buffers::{
    checked_product_count, decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    write_zero_bytes,
};
use crate::math::sheaf_laplacian_eigenvalue::sheaf_laplacian_eigenvalue;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Default value for the retained (interface-stability) `iterations` parameter.
///
/// The dominant eigenpair of a DIAGONAL sheaf Laplacian is the exact closed form
/// `(max_i r[i], e_argmax)`: no power iteration and no convergence budget are needed, so the
/// `iterations` argument is a documented no-op. This default is kept so callers that historically
/// passed an iteration count keep compiling and behaving identically.
pub const DEFAULT_SPECTRUM_ITERATIONS: u32 = 32;

/// Reusable buffers for the sheaf-spectral dominant-eigenpair scan.
#[derive(Debug, Default)]
#[cfg(test)]
pub(crate) struct SheafSpectrumScratch {
    v_init: Vec<f64>,
    v: Vec<f64>,
    v_next: Vec<f64>,
}

#[cfg(test)]
impl SheafSpectrumScratch {
    /// Dominant eigenvector from the last spectral solve.
    #[must_use]
    pub(crate) fn eigenvector(&self) -> &[f64] {
        &self.v
    }
}

#[must_use]
#[cfg(test)]
pub(crate) fn dominant_spectrum(restriction_diag: &[f64], iterations: u32) -> (f64, Vec<f64>) {
    vyre_reference::composition_witness::sheaf_dominant_spectrum_witness(
        restriction_diag,
        iterations,
    )
}

#[cfg(test)]
pub(crate) fn dominant_spectrum_with_scratch(
    restriction_diag: &[f64],
    iterations: u32,
    scratch: &mut SheafSpectrumScratch,
) -> f64 {
    scratch.v_init.clear();
    scratch.v_next.clear();
    vyre_reference::composition_witness::sheaf_dominant_spectrum_witness_into(
        restriction_diag,
        iterations,
        &mut scratch.v,
    )
}
/// Caller-owned GPU dispatch scratch for fixed-point sheaf spectra.
#[derive(Debug, Default)]
pub struct SheafSpectrumGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Fixed-point dominant sheaf spectrum returned by the GPU-dispatchable path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedSheafSpectrum {
    /// Dominant eigenvalue/norm signal in primitive-native 16.16 storage.
    pub lambda: u32,
    /// Final eigenvector buffer in primitive-native 16.16 storage.
    pub eigenvector: Vec<u32>,
}

/// Fixed-point production path for sheaf spectral clustering.
///
/// `restriction_diag_fixed` and `v_init_fixed` are primitive-native 16.16
/// buffers with shape `n_nodes * d`. The dispatcher runs
/// [`sheaf_laplacian_eigenvalue`] directly and returns both the lambda output
/// and the mutated eigenvector buffer.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks fail, the primitive lane space
/// is exceeded, or the backend returns malformed output buffers.
pub fn dominant_spectrum_fixed_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    restriction_diag_fixed: &[u32],
    v_init_fixed: &[u32],
    n_nodes: u32,
    d: u32,
    iterations: u32,
) -> Result<FixedSheafSpectrum, SemanticExecutionError> {
    let mut scratch = SheafSpectrumGpuScratch::default();
    let mut eigenvector = Vec::new();
    let lambda = dominant_spectrum_fixed_via_with_scratch_into(
        dispatcher,
        policy,
        restriction_diag_fixed,
        v_init_fixed,
        n_nodes,
        d,
        iterations,
        &mut scratch,
        &mut eigenvector,
    )?;
    Ok(FixedSheafSpectrum {
        lambda,
        eigenvector,
    })
}

/// Fixed-point sheaf spectral clustering into caller-owned eigenvector
/// storage. Returns the fixed-point lambda output.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks or backend execution fail.
pub fn dominant_spectrum_fixed_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    restriction_diag_fixed: &[u32],
    v_init_fixed: &[u32],
    n_nodes: u32,
    d: u32,
    iterations: u32,
    eigenvector_out: &mut Vec<u32>,
) -> Result<u32, SemanticExecutionError> {
    let mut scratch = SheafSpectrumGpuScratch::default();
    dominant_spectrum_fixed_via_with_scratch_into(
        dispatcher,
        policy,
        restriction_diag_fixed,
        v_init_fixed,
        n_nodes,
        d,
        iterations,
        &mut scratch,
        eigenvector_out,
    )
}

/// Fixed-point sheaf spectral clustering with reusable dispatch input storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks or backend execution fail.
pub fn dominant_spectrum_fixed_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    restriction_diag_fixed: &[u32],
    v_init_fixed: &[u32],
    n_nodes: u32,
    d: u32,
    iterations: u32,
    scratch: &mut SheafSpectrumGpuScratch,
    eigenvector_out: &mut Vec<u32>,
) -> Result<u32, SemanticExecutionError> {
    use crate::telemetry::{bump, sheaf_spectral_clustering_calls};
    bump(&sheaf_spectral_clustering_calls);

    let cells = checked_product_count(n_nodes, d, "n_nodes", "d", "dominant_spectrum_fixed_via")?;
    let cells_u32 = u32::try_from(cells).map_err(|_| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: dominant_spectrum_fixed_via n_nodes*d exceeds the primitive u32 lane limit for n_nodes={n_nodes}, d={d}."
    ))
})?;
    if restriction_diag_fixed.len() != cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: dominant_spectrum_fixed_via requires restriction_diag_fixed.len() == n_nodes*d, got len={}, n_nodes={n_nodes}, d={d}, cells={cells}.",
        restriction_diag_fixed.len()
    )));
    }
    if v_init_fixed.len() != cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: dominant_spectrum_fixed_via requires v_init_fixed.len() == n_nodes*d, got len={}, n_nodes={n_nodes}, d={d}, cells={cells}.",
        v_init_fixed.len()
    )));
    }

    let program =
        sheaf_laplacian_eigenvalue("restriction_diag", "v", "lambda", n_nodes, d, iterations);
    // Canonical dispatch input contract (the REAL backend's, per vyre-driver `role_for_buffer`): one
    // input per INPUT-CONSUMING buffer. `ReadOnly` (Input), plain `ReadWrite` (InputOutput, whose
    // zero/initial contents the caller supplies), `Uniform`: in buffer order. The eigenvalue kernel
    // declares four such buffers: `restriction_diag` RO (0), `v` RW (1), `lambda` RW (2), `one_fp_buf`
    // RO (3, the 16.16 unit written into the eigenvector's arg-max slot). `v` and `lambda` are plain
    // ReadWrite outputs, so the backend requires a zero-filled input slot for each (the dominant
    // eigenpair of a diagonal operator is independent of the starting vector, so `v_init_fixed` is
    // validated above but the kernel overwrites it from zeros). Passing fewer than four inputs would
    // fail the backend's strict `validate_input_lengths` count check.
    let v_bytes = cells
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: dominant_spectrum_fixed_via v byte size overflows usize for {cells} cells."
            ))
        })?;
    ensure_input_slots(&mut scratch.inputs, 4);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], restriction_diag_fixed);
    write_zero_bytes(&mut scratch.inputs[1], v_bytes);
    write_zero_bytes(&mut scratch.inputs[2], std::mem::size_of::<u32>());
    scratch.inputs[3].clear();
    scratch.inputs[3].extend_from_slice(&(1u32 << 16).to_le_bytes());
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs[..4],
        policy,
    )
    .map(|output| output.outputs)?;
    // The kernel's writable buffers, in binding order, are exactly `v` (the eigenvector) then
    // `lambda`; the running max/arg-max are loop-carried locals, not storage buffers, so a faithful
    let [eigenvector_out_buf, lambda_out_buf] = match outputs.as_slice() {
    [ev, l] => [ev, l],
    _ => {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: dominant_spectrum_fixed_via expected exactly eigenvector and lambda outputs, got {} buffer(s).",
            outputs.len()
        )))
    }
};

    decode_u32_output_exact(
        eigenvector_out_buf,
        cells,
        "dominant_spectrum_fixed_via eigenvector",
        eigenvector_out,
    )?;
    let mut lambda = Vec::with_capacity(1);
    decode_u32_output_exact(
        lambda_out_buf,
        1,
        "dominant_spectrum_fixed_via lambda",
        &mut lambda,
    )?;
    Ok(lambda[0])
}

/// Convenience: spectral gap signal in `[0, 1]` derived from the
/// dominant eigenvalue. Higher = cleaner cluster separation.
#[must_use]
#[cfg(test)]
pub(crate) fn spectral_gap(restriction_diag: &[f64]) -> f64 {
    vyre_reference::composition_witness::sheaf_spectral_gap_witness(
        restriction_diag,
        DEFAULT_SPECTRUM_ITERATIONS,
    )
}

/// Compute spectral gap using caller-owned dominant-eigenpair scratch.
#[cfg(test)]
pub(crate) fn spectral_gap_into(
    restriction_diag: &[f64],
    scratch: &mut SheafSpectrumScratch,
) -> f64 {
    scratch.v_init.clear();
    scratch.v_next.clear();
    vyre_reference::composition_witness::sheaf_spectral_gap_witness_into(
        restriction_diag,
        DEFAULT_SPECTRUM_ITERATIONS,
        &mut scratch.v,
    )
}

/// Derive a suggested cluster count from the principal eigenvector
/// sign pattern. Items whose eigenvector entry has the same sign
/// belong in the same cluster; flips between consecutive items
/// suggest cluster boundaries. Returns the count of distinct sign
/// runs (≥ 1).
#[must_use]
#[cfg(test)]
pub(crate) fn suggested_cluster_count(restriction_diag: &[f64]) -> u32 {
    let mut scratch = SheafSpectrumScratch::default();
    dominant_spectrum_with_scratch(restriction_diag, DEFAULT_SPECTRUM_ITERATIONS, &mut scratch);
    suggested_cluster_count_into(restriction_diag, &mut scratch)
}

/// Derive suggested cluster count using caller-owned spectral scratch.
#[cfg(test)]
pub(crate) fn suggested_cluster_count_into(
    _restriction_diag: &[f64],
    scratch: &mut SheafSpectrumScratch,
) -> u32 {
    vyre_reference::composition_witness::sheaf_suggested_cluster_count_witness(
        scratch.eigenvector(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-3 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn dominant_eigenvalue_of_uniform_diag_is_diag_value() {
        // restriction = [0.7, 0.7, 0.7, 0.7] → dominant eigenvalue = 0.7.
        let diag = vec![0.7, 0.7, 0.7, 0.7];
        let (lambda, _v) = dominant_spectrum(&diag, 64);
        assert!(approx_eq(lambda, 0.7), "got lambda={lambda}");
    }

    #[test]
    fn dominant_eigenvalue_of_nonuniform_picks_max() {
        // restriction = [0.1, 0.5, 0.9, 0.3] → dominant eigenvalue ≈ 0.9.
        let diag = vec![0.1, 0.5, 0.9, 0.3];
        let (lambda, v) = dominant_spectrum(&diag, 128);
        assert!((lambda - 0.9).abs() < 0.01, "got lambda={lambda}");
        // Eigenvector should localize on index 2 (the 0.9 entry).
        let max_idx = v
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        assert_eq!(max_idx, 2);
    }

    #[test]
    fn empty_input_returns_zero_spectrum() {
        let (lambda, v) = dominant_spectrum(&[], 32);
        assert_eq!(lambda, 0.0);
        assert!(v.is_empty());
    }

    #[test]
    fn spectral_gap_is_one_for_uniform_diag() {
        // Uniform diagonal  -  eigenvalue equals max  -  gap = 1.
        let diag = vec![0.5; 8];
        let gap = spectral_gap(&diag);
        assert!((gap - 1.0).abs() < 1e-3);
    }

    #[test]
    fn scratch_paths_match_owned_spectral_helpers() {
        let diag = vec![0.1, 0.5, 0.9, 0.3];
        let (owned_lambda, owned_v) = dominant_spectrum(&diag, 64);
        let mut scratch = SheafSpectrumScratch::default();
        let borrowed_lambda = dominant_spectrum_with_scratch(&diag, 64, &mut scratch);
        assert!(approx_eq(owned_lambda, borrowed_lambda));
        assert_eq!(scratch.eigenvector().len(), owned_v.len());

        let owned_gap = spectral_gap(&diag);
        let scratch_gap = spectral_gap_into(&diag, &mut scratch);
        assert!(approx_eq(owned_gap, scratch_gap));

        let owned_count = suggested_cluster_count(&diag);
        let scratch_count = suggested_cluster_count_into(&diag, &mut scratch);
        assert_eq!(owned_count, scratch_count);
    }

    #[test]
    fn suggested_cluster_count_at_least_one() {
        let diag = vec![0.7; 4];
        let count = suggested_cluster_count(&diag);
        assert!(count >= 1);
    }

    struct SpectrumDispatcher;

    impl SemanticExecutor for SpectrumDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                // Real-backend input contract: one input per input-consuming buffer in buffer order
                // restriction_diag RO (0), v RW (1, zero slot), lambda RW (2, zero slot), one_fp RO (3).
                // Compute the SAME closed-form diagonal eigenpair the real kernel does. (max r,
                // e_argmax) (so this double stays truthful to the IR under test (Law 6)).

                assert_eq!(inputs.len(), 4);
                let restriction = crate::dispatch_buffers::read_u32s(&inputs[0]);
                let one_fp = crate::dispatch_buffers::read_u32s(&inputs[3])[0];
                assert_eq!(one_fp, 1u32 << 16);
                let mut max_r = 0u32;
                let mut argmax = 0usize;
                for (i, &ri) in restriction.iter().enumerate() {
                    if ri > max_r {
                        max_r = ri;
                        argmax = i;
                    }
                }
                let eigenvector: Vec<u32> = (0..restriction.len())
                    .map(|j| if j == argmax { one_fp } else { 0 })
                    .collect();
                Ok(vec![
                    u32_slice_to_le_bytes(&eigenvector),
                    max_r.to_le_bytes().to_vec(),
                ])
            })();
            let mut ordered = ordered?;
            let output_count = request.logical().graph().nodes()[0].outputs.len();
            if ordered.len() < output_count {
                ordered.resize(output_count, Vec::new());
            }
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    struct ExtraSpectrumDispatcher;

    impl SemanticExecutor for ExtraSpectrumDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                Ok(vec![
                    u32_slice_to_le_bytes(&[1]),
                    u32_slice_to_le_bytes(&[1]),
                    u32_slice_to_le_bytes(&[1]),
                ])
            })();
            let mut ordered = ordered?;
            let output_count = request.logical().graph().nodes()[0].outputs.len();
            if ordered.len() < output_count {
                ordered.resize(output_count, Vec::new());
            }
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    struct TrailingLambdaDispatcher;

    impl SemanticExecutor for TrailingLambdaDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                Ok(vec![u32_slice_to_le_bytes(&[1]), vec![1, 0, 0, 0, 2]])
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
    fn fixed_via_dispatches_sheaf_spectrum() {
        let one = 1u32 << 16;
        // r = [1.0, 0.5]; the dominant eigenpair of diag(r) is (max r, e_argmax) = (1.0, e_0). The
        // initial vector is ignored by the diagonal kernel.
        let spectrum = dominant_spectrum_fixed_via(
            &SpectrumDispatcher,
            &crate::test_parity_oracles::policy(),
            &[one, one / 2],
            &[8 * one, 4 * one],
            2,
            1,
            1,
        )
        .unwrap();
        assert_eq!(spectrum.eigenvector, vec![one, 0]);
        assert_eq!(spectrum.lambda, one);
    }

    #[test]
    fn fixed_via_reuses_dispatch_buffers_and_eigenvector_output() {
        let one = 1u32 << 16;
        // Four input slots in buffer order [restriction_diag, v_zero, lambda_zero, one_fp], the
        // real-backend input-consuming set (2 RO + 2 plain-RW). Pre-sized with ample capacity so a
        // single call reuses them without reallocation (the reuse contract under test).
        let mut scratch = SheafSpectrumGpuScratch {
            inputs: vec![
                Vec::with_capacity(64),
                Vec::with_capacity(64),
                Vec::with_capacity(8),
                Vec::with_capacity(8),
            ],
        };
        let mut eigenvector = Vec::with_capacity(4);
        let input_caps = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let out_ptr = eigenvector.as_ptr();
        let lambda = dominant_spectrum_fixed_via_with_scratch_into(
            &SpectrumDispatcher,
            &crate::test_parity_oracles::policy(),
            &[one, one / 2],
            &[8 * one, 4 * one],
            2,
            1,
            1,
            &mut scratch,
            &mut eigenvector,
        )
        .unwrap();
        assert_eq!(lambda, one);
        assert_eq!(eigenvector, vec![one, 0]);
        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_caps
        );
        assert_eq!(eigenvector.as_ptr(), out_ptr);
    }

    #[test]
    fn fixed_via_rejects_shape_mismatch() {
        let err = dominant_spectrum_fixed_via(
            &SpectrumDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1, 2, 3],
            &[1, 2],
            2,
            2,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
    }

    #[test]
    fn fixed_via_rejects_extra_outputs() {
        let err = dominant_spectrum_fixed_via(
            &ExtraSpectrumDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1],
            &[1],
            1,
            1,
            1,
        )
        .unwrap_err();
        assert!(
            matches!(err, SemanticExecutionError::Backend(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn fixed_via_rejects_trailing_lambda_bytes() {
        let err = dominant_spectrum_fixed_via(
            &TrailingLambdaDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1],
            &[1],
            1,
            1,
            1,
        )
        .unwrap_err();
        assert!(
            matches!(err, SemanticExecutionError::Backend(_)),
            "unexpected error: {err:?}"
        );
    }
}
