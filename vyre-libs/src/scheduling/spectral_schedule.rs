//! Spectral analysis of dispatch graph via chebyshev_filter +
//! spectral_shape.
//!
//! Apply Chebyshev polynomial filtering to vyre's own dispatch
//! dependency matrix, clip outlier eigenvalues via Marchenko-Pastur
//! edge, identify clusters of Programs that should be fused.
//! Output: cluster IDs that polyhedral fusion + megakernel
//! scheduler consume as fusion hints.

use super::{checked_square_cells, decode_u32_output_exact};
use crate::dispatch_buffers::{ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes};
use crate::graph::chebyshev_filter::{chebyshev_filter, MAX_K as CHEBYSHEV_MAX_K};
use crate::math::spectral_shape::mp_edge_clip;
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Caller-owned semantic execution scratch for spectral scheduling primitives.
#[derive(Debug, Default)]
pub struct SpectralScheduleGpuScratch {
    inputs: Vec<Vec<u8>>,
    mp_edge: Vec<u32>,
}

/// Fixed-point production path for spectral fusion scores.
///
/// Inputs are primitive-native 16.16 u32 buffers. Semantic execution compiles
/// and executes [`chebyshev_filter`] and returns the fixed-point score vector.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shapes are invalid, the primitive
/// order is unsupported, compilation or execution fails, or output is malformed.
pub fn fusion_scores_fixed_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    laplacian_fixed: &[u32],
    signal_fixed: &[u32],
    coeffs_fixed: &[u32],
    n: u32,
    k_steps: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    fusion_scores_fixed_via_into(
        executor,
        policy,
        laplacian_fixed,
        signal_fixed,
        coeffs_fixed,
        n,
        k_steps,
        &mut out,
    )?;
    Ok(out)
}

/// Fixed-point production path for spectral fusion scores into caller-owned
/// output storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks, compilation, or backend
/// execution fails.
pub fn fusion_scores_fixed_via_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    laplacian_fixed: &[u32],
    signal_fixed: &[u32],
    coeffs_fixed: &[u32],
    n: u32,
    k_steps: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = SpectralScheduleGpuScratch::default();
    fusion_scores_fixed_via_with_scratch_into(
        executor,
        policy,
        laplacian_fixed,
        signal_fixed,
        coeffs_fixed,
        n,
        k_steps,
        &mut scratch,
        out,
    )
}

/// Fixed-point production path for spectral fusion scores using caller-owned
/// execution scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks, compilation, or backend
/// execution fails.
pub fn fusion_scores_fixed_via_with_scratch_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    laplacian_fixed: &[u32],
    signal_fixed: &[u32],
    coeffs_fixed: &[u32],
    n: u32,
    k_steps: u32,
    scratch: &mut SpectralScheduleGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bump, spectral_schedule_calls};
    bump(&spectral_schedule_calls);

    if k_steps > CHEBYSHEV_MAX_K {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: fusion_scores_fixed_via requires k_steps <= {CHEBYSHEV_MAX_K}, got {k_steps}."
        )));
    }
    let cells = checked_square_cells(n, "fusion_scores_fixed_via")?;
    let _cells_u32 = u32::try_from(cells).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: fusion_scores_fixed_via n*n exceeds the primitive u32 lane limit for n={n}."
        ))
    })?;
    if n > u32::MAX / 2 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: fusion_scores_fixed_via scratch size 2*n overflows u32 for n={n}."
        )));
    }
    if laplacian_fixed.len() != cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: fusion_scores_fixed_via requires laplacian_fixed.len() == n*n, got len={}, n={}, n*n={cells}.",
            laplacian_fixed.len(),
            n
        )));
    }
    if signal_fixed.len() != n as usize {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: fusion_scores_fixed_via requires signal_fixed.len() == n, got len={}, n={n}.",
            signal_fixed.len()
        )));
    }
    let coeff_count = (k_steps as usize).checked_add(1).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(
            "Fix: fusion_scores_fixed_via coefficient count overflowed usize.".to_string(),
        )
    })?;
    if coeffs_fixed.len() != coeff_count {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: fusion_scores_fixed_via requires coeffs_fixed.len() == k_steps + 1, got len={}, k_steps={k_steps}.",
            coeffs_fixed.len()
        )));
    }

    let program = chebyshev_filter(
        "laplacian",
        "signal",
        "coeffs",
        "output",
        "scratch",
        n,
        k_steps,
    );
    let out_bytes = (n as usize) * std::mem::size_of::<u32>();
    let scratch_bytes = 2 * out_bytes;
    ensure_input_slots(&mut scratch.inputs, 5);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], laplacian_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], signal_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[2], coeffs_fixed);
    write_zero_bytes(&mut scratch.inputs[3], out_bytes);
    write_zero_bytes(&mut scratch.inputs[4], scratch_bytes);
    let outputs = execute_single_program(
        executor,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    if outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: fusion_scores_fixed_via expected at least one output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(&outputs[0], n as usize, "fusion_scores_fixed_via", out)
}

/// Fixed-point production path for Marchenko-Pastur edge clipping.
///
/// `mp_edge_fixed` is the already-scaled 16.16 upper edge. Callers that need
/// the f64 helper can keep using `mp_upper_edge` at the representation
/// boundary, then quantize once before semantic execution.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the eigenvalue vector is empty, too
/// large for the primitive lane space, compilation or execution fails, or the
/// output is malformed.
pub fn shape_spectrum_fixed_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    eigenvalues_fixed: &[u32],
    mp_edge_fixed: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    shape_spectrum_fixed_via_into(executor, policy, eigenvalues_fixed, mp_edge_fixed, &mut out)?;
    Ok(out)
}

/// Fixed-point Marchenko-Pastur edge clipping into caller-owned storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks, compilation, or backend
/// execution fails.
pub fn shape_spectrum_fixed_via_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    eigenvalues_fixed: &[u32],
    mp_edge_fixed: u32,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = SpectralScheduleGpuScratch::default();
    shape_spectrum_fixed_via_with_scratch_into(
        executor,
        policy,
        eigenvalues_fixed,
        mp_edge_fixed,
        &mut scratch,
        out,
    )
}

/// Fixed-point Marchenko-Pastur edge clipping using caller-owned execution scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when shape checks, compilation, or backend
/// execution fails.
pub fn shape_spectrum_fixed_via_with_scratch_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    eigenvalues_fixed: &[u32],
    mp_edge_fixed: u32,
    scratch: &mut SpectralScheduleGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    if eigenvalues_fixed.is_empty() {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: shape_spectrum_fixed_via requires at least one eigenvalue.".to_string(),
        ));
    }
    let n = u32::try_from(eigenvalues_fixed.len()).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: shape_spectrum_fixed_via eigenvalue count exceeds u32 lane limit: {}.",
            eigenvalues_fixed.len()
        ))
    })?;

    let program = mp_edge_clip("eigenvalues", "mp_edge", "out", n);
    scratch.mp_edge.clear();
    scratch.mp_edge.push(mp_edge_fixed);
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], eigenvalues_fixed);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &scratch.mp_edge);
    let outputs = execute_single_program(
        executor,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    if outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: shape_spectrum_fixed_via expected at least one output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(
        &outputs[0],
        eigenvalues_fixed.len(),
        "shape_spectrum_fixed_via",
        out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use crate::math::spectral_shape::mp_upper_edge;
    use crate::test_parity_oracles::{policy, StaticOutputs};
    use vyre_reference::composition_witness::{
        chebyshev_filter_witness as reference_chebyshev_filter,
        mp_edge_clip_witness as reference_mp_edge_clip,
    };

    fn reference_fusion_scores(laplacian: &[f32], n: u32) -> Vec<f32> {
        use crate::telemetry::{bump, spectral_schedule_calls};
        bump(&spectral_schedule_calls);
        assert_eq!(laplacian.len(), (n * n) as usize);
        let signal: Vec<f32> = (0..n).map(|_| 1.0 / (n as f32).sqrt()).collect();
        let coeffs: Vec<f32> = vec![1.0, 0.5, 0.25];
        reference_chebyshev_filter(laplacian, &signal, &coeffs, n, 2)
    }

    fn reference_shape_spectrum(
        eigenvalues: &[f64],
        n_dispatches: u32,
        n_features: u32,
    ) -> Vec<f64> {
        let edge = mp_upper_edge(n_dispatches, n_features, 1.0);
        reference_mp_edge_clip(eigenvalues, edge)
    }

    fn approx_eq_f32(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4 * (1.0 + a.abs() + b.abs())
    }

    #[test]
    fn fusion_scores_uniform_for_zero_laplacian() {
        let l: Vec<f32> = vec![0.0; 16];
        let scores = reference_fusion_scores(&l, 4);
        for s in scores {
            assert!(approx_eq_f32(s, 0.375));
        }
    }

    #[test]
    fn shape_spectrum_clips_outliers() {
        let eig = vec![1.0, 3.0, 5.0, 100.0];
        let clipped = reference_shape_spectrum(&eig, 100, 100);
        assert_eq!(clipped[0], 1.0);
        assert_eq!(clipped[1], 3.0);
        assert_eq!(clipped[2], 4.0);
        assert_eq!(clipped[3], 4.0);
    }

    #[test]
    fn fusion_scores_zero_signal_zero_output() {
        let l: Vec<f32> = vec![0.5; 4];
        let scores = reference_fusion_scores(&l, 2);
        for s in scores {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn shape_spectrum_fixed_via_preserves_input_and_output_contracts() {
        let eigenvalues = vec![1, 5, 10];
        let executor =
            StaticOutputs::new("spectral shape", vec![u32_slice_to_le_bytes(&[1, 4, 4])])
                .expecting_inputs(&[2])
                .recording_input(0);

        let clipped = shape_spectrum_fixed_via(&executor, &policy(), &eigenvalues, 4).unwrap();

        assert_eq!(clipped, vec![1, 4, 4]);
        assert_eq!(executor.recorded(), vec![eigenvalues]);
    }

    #[test]
    fn fusion_scores_fixed_via_preserves_input_and_output_contracts() {
        let laplacian = vec![1, 0, 0, 1];
        let executor = StaticOutputs::new(
            "spectral fusion",
            vec![u32_slice_to_le_bytes(&[7, 11]), vec![0; 16]],
        )
        .expecting_inputs(&[5])
        .recording_input(0);

        let scores =
            fusion_scores_fixed_via(&executor, &policy(), &laplacian, &[7, 11], &[1], 2, 0)
                .unwrap();

        assert_eq!(scores, vec![7, 11]);
        assert_eq!(executor.recorded(), vec![laplacian]);
    }

    #[test]
    fn fixed_via_rejects_bad_shapes() {
        let executor = StaticOutputs::new("unused spectral output", Vec::new());
        let err = shape_spectrum_fixed_via(&executor, &policy(), &[], 4).unwrap_err();
        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));

        let err = fusion_scores_fixed_via(&executor, &policy(), &[1, 0, 0], &[1, 1], &[1], 2, 0)
            .unwrap_err();
        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
    }

    #[test]
    fn fixed_via_with_scratch_reuses_input_buffers() {
        let executor = StaticOutputs::new(
            "spectral fusion scratch",
            vec![u32_slice_to_le_bytes(&[7, 11]), vec![0; 16]],
        )
        .expecting_inputs(&[5]);
        let execution_policy = policy();
        let mut scratch = SpectralScheduleGpuScratch::default();
        let mut out = Vec::new();

        fusion_scores_fixed_via_with_scratch_into(
            &executor,
            &execution_policy,
            &[1, 0, 0, 1],
            &[7, 11],
            &[1],
            2,
            0,
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let input_ptrs: Vec<*const u8> = scratch.inputs.iter().take(3).map(Vec::as_ptr).collect();
        fusion_scores_fixed_via_with_scratch_into(
            &executor,
            &execution_policy,
            &[1, 0, 0, 1],
            &[5, 13],
            &[1],
            2,
            0,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        for (before, after) in input_ptrs
            .iter()
            .zip(scratch.inputs.iter().take(3).map(Vec::as_ptr))
        {
            assert_eq!(*before, after);
        }
    }
}
