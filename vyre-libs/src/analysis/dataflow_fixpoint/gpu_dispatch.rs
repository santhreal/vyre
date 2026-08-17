//! Dispatcher-backed wrappers: each closure step and SCC pivot query is routed
//! through vyre dispatch while the host owns the fixed-point loop.

use vyre_foundation::pass_substrate::semiring_closure as foundation_dataflow;

use super::scc_decomposition::write_pivot_bitsets;
use super::{SccComponentsGpuScratch, Semiring, SemiringGemmGpuScratch};
use crate::dispatch_buffers::{
    ceil_div_u32, decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    write_zero_bytes,
};
use crate::graph::scc_decompose::dense_reachability_bitsets;
use crate::plumbing::host::scratch::reserve_vec_capacity;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// GPU dispatch wrapper around the primitive semiring GEMM program for an
/// arbitrary semiring.
///
/// # Errors
///
/// Returns [`vyre_foundation::program_dispatch::DispatchError`] when dimensions
/// overflow, inputs do not match the declared matrix shape, dispatch fails,
/// or readback does not contain the `m * n` output matrix.
#[allow(clippy::too_many_arguments)]
pub fn semiring_gemm_via(
    dispatcher: &dyn ProgramDispatcher,
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
) -> Result<Vec<u32>, DispatchError> {
    let c_words = m.checked_mul(n).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via dimensions overflow m*n: m={m}, n={n}."
        ))
    })? as usize;
    let mut c = Vec::with_capacity(c_words);
    semiring_gemm_via_into(dispatcher, a, b, m, n, k, semiring, &mut c)?;
    Ok(c)
}

/// Multiply matrices over the selected semiring through a dispatcher into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn semiring_gemm_via_into(
    dispatcher: &dyn ProgramDispatcher,
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
    c: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = SemiringGemmGpuScratch::default();
    semiring_gemm_via_with_scratch_into(dispatcher, a, b, m, n, k, semiring, &mut scratch, c)
}

/// Multiply matrices over the selected semiring using caller-owned dispatch scratch.
#[allow(clippy::too_many_arguments)]
pub fn semiring_gemm_via_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
    semiring: Semiring,
    scratch: &mut SemiringGemmGpuScratch,
    c: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let a_words = m.checked_mul(k).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via dimensions overflow m*k: m={m}, k={k}."
        ))
    })? as usize;
    let b_words = k.checked_mul(n).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via dimensions overflow k*n: k={k}, n={n}."
        ))
    })? as usize;
    let c_words_u32 = m.checked_mul(n).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via dimensions overflow m*n: m={m}, n={n}."
        ))
    })?;
    let c_words = c_words_u32 as usize;
    let c_bytes = c_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: semiring_gemm_via output byte count overflows usize for {c_words} words."
            ))
        })?;

    if m == 0 || n == 0 || k == 0 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via requires nonzero dimensions; got m={m}, n={n}, k={k}."
        )));
    }
    if a.len() != a_words {
        return Err(DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via expected a.len() == m*k == {a_words}, got {}.",
            a.len()
        )));
    }
    if b.len() != b_words {
        return Err(DispatchError::BadInputs(format!(
            "Fix: semiring_gemm_via expected b.len() == k*n == {b_words}, got {}.",
            b.len()
        )));
    }

    let program = crate::math::semiring_gemm::semiring_gemm("a", "b", "c", m, n, k, semiring);
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], a);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], b);
    write_zero_bytes(&mut scratch.inputs[2], c_bytes);
    let grid_x = ceil_div_u32(c_words_u32, 256);
    let outputs = dispatcher.dispatch(&program, &scratch.inputs, Some([grid_x, 1, 1]))?;
    let [c_out] = match outputs.as_slice() {
        [c_out] => [c_out],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: semiring_gemm_via expected exactly one c output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(c_out, c_words, "semiring_gemm_via c", c)
}

/// Boolean-OR semiring specialisation of [`semiring_gemm_via`].
pub fn semiring_gemm_via_bool_or(
    dispatcher: &dyn ProgramDispatcher,
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<u32>, DispatchError> {
    semiring_gemm_via(dispatcher, a, b, m, n, k, Semiring::BoolOr)
}

/// Min-plus semiring specialisation of [`semiring_gemm_via`].
pub fn semiring_gemm_via_min_plus(
    dispatcher: &dyn ProgramDispatcher,
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<u32>, DispatchError> {
    semiring_gemm_via(dispatcher, a, b, m, n, k, Semiring::MinPlus)
}

/// Lineage (provenance OR) semiring specialisation of [`semiring_gemm_via`].
pub fn semiring_gemm_via_lineage(
    dispatcher: &dyn ProgramDispatcher,
    a: &[u32],
    b: &[u32],
    m: u32,
    n: u32,
    k: u32,
) -> Result<Vec<u32>, DispatchError> {
    semiring_gemm_via(dispatcher, a, b, m, n, k, Semiring::Lineage)
}

// ─────────────────────────────────────────────────────────────────────
// GPU dispatcher wrappers (`*_via`)
// ─────────────────────────────────────────────────────────────────────
//
// Each wrapper takes an `ProgramDispatcher` and routes closure steps through
// vyre dispatch. The host currently owns the fixed-point loop and convergence
// check; each matrix-power step is backend-dispatched via semiring GEMM.

/// GPU dispatch wrapper around reachability closure.
///
/// # Errors
///
/// Propagates semiring-GEMM dispatch failures.
pub fn reachability_closure_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    max_iters: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    reachability_closure_via_into(dispatcher, adj, n, max_iters, &mut current, &mut next)?;
    Ok(current)
}

/// GPU dispatch wrapper around reachability closure into caller-owned buffers.
///
/// # Errors
///
/// Propagates semiring-GEMM dispatch failures.
pub fn reachability_closure_via_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    _max_iters: u32,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = SemiringGemmGpuScratch::default();
    reachability_closure_via_with_scratch_into(
        dispatcher,
        adj,
        n,
        _max_iters,
        &mut scratch,
        current,
        next,
    )
}

/// GPU dispatch wrapper around reachability closure with caller-owned dispatch scratch.
///
/// # Errors
///
/// Propagates semiring-GEMM dispatch failures.
pub fn reachability_closure_via_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    _max_iters: u32,
    scratch: &mut SemiringGemmGpuScratch,
    current: &mut Vec<u32>,
    next: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    current.clear();
    current.extend_from_slice(adj);
    next.clear();
    reserve_vec_capacity(next, current.len(), "reachability closure next matrix")?;
    for _ in 0..n {
        semiring_gemm_via_with_scratch_into(
            dispatcher,
            current.as_slice(),
            current.as_slice(),
            n,
            n,
            n,
            Semiring::BoolOr,
            scratch,
            next,
        )?;
        if !foundation_dataflow::merge_or_changed(current, next) {
            return Ok(());
        }
    }
    Ok(())
}

/// GPU dispatch wrapper around lineage closure.
///
/// # Errors
///
/// Propagates semiring-GEMM dispatch failures.
pub fn lineage_closure_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    max_iters: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut current = adj.to_vec();
    let mut next = Vec::with_capacity(current.len());
    for _ in 0..max_iters {
        semiring_gemm_via_into(
            dispatcher,
            &current,
            &current,
            n,
            n,
            n,
            Semiring::Lineage,
            &mut next,
        )?;
        if !foundation_dataflow::merge_or_changed(&mut current, &next) {
            return Ok(current);
        }
    }
    Ok(current)
}

/// GPU dispatch wrapper around shortest-path closure.
///
/// # Errors
///
/// Propagates semiring-GEMM dispatch failures.
pub fn shortest_path_closure_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    max_iters: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut current = adj.to_vec();
    let mut next = Vec::with_capacity(current.len());
    for _ in 0..max_iters {
        semiring_gemm_via_into(
            dispatcher,
            &current,
            &current,
            n,
            n,
            n,
            Semiring::MinPlus,
            &mut next,
        )?;
        if !foundation_dataflow::merge_min_changed(&mut current, &next) {
            return Ok(current);
        }
    }
    Ok(current)
}

/// GPU-backed forward/backward reach bitset query for one pivot.
///
/// # Errors
///
/// Propagates reachability closure dispatch failures.
pub fn forward_backward_bitsets_for_pivot_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    pivot: u32,
    n: u32,
) -> Result<(Vec<u32>, Vec<u32>), DispatchError> {
    if n == 0 || pivot >= n {
        return Err(DispatchError::BadInputs(format!(
            "Fix: forward_backward_bitsets_for_pivot_via requires n > 0 and pivot < n; got n={n}, pivot={pivot}."
        )));
    }
    let n_us = n as usize;
    let cells = n_us.checked_mul(n_us).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: forward_backward_bitsets_for_pivot_via n*n overflows usize for n={n}."
        ))
    })?;
    if adj.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: forward_backward_bitsets_for_pivot_via expected adj.len() == n*n == {cells}, got {}.",
            adj.len()
        )));
    }
    let dense_count = n.checked_mul(n).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: forward_backward_bitsets_for_pivot_via n*n exceeds the GPU u32 buffer domain for n={n}."
        ))
    })?;
    let words = ((n + 31) / 32) as usize;

    let fwd_closure = reachability_closure_via(dispatcher, adj, n, n)?;
    let mut transpose = vec![0u32; cells];
    for i in 0..n_us {
        for j in 0..n_us {
            transpose[j * n_us + i] = adj[i * n_us + j];
        }
    }
    let bwd_closure = reachability_closure_via(dispatcher, &transpose, n, n)?;
    let bitset_bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: forward_backward_bitsets_for_pivot_via bitset byte count overflows usize for {words} words."
            ))
        })?;
    let program = dense_reachability_bitsets(
        n,
        dense_count,
        pivot,
        "forward_closure",
        "backward_closure",
        "forward",
        "backward",
    );
    let mut inputs = vec![Vec::new(); 4];
    write_u32_slice_le_bytes(&mut inputs[0], &fwd_closure);
    write_u32_slice_le_bytes(&mut inputs[1], &bwd_closure);
    write_zero_bytes(&mut inputs[2], bitset_bytes);
    write_zero_bytes(&mut inputs[3], bitset_bytes);
    let outputs = dispatcher.dispatch(&program, &inputs, Some([ceil_div_u32(n, 256), 1, 1]))?;
    let [fwd_out, bwd_out] = match outputs.as_slice() {
        [fwd_out, bwd_out] => [fwd_out, bwd_out],
        _ => {
            return Err(DispatchError::BackendError(format!(
                "Fix: forward_backward_bitsets_for_pivot_via expected exactly two bitset output buffers, got {}.",
                outputs.len()
            )));
        }
    };
    let mut forward = Vec::with_capacity(words);
    let mut backward = Vec::with_capacity(words);
    decode_u32_output_exact(
        fwd_out,
        words,
        "forward_backward_bitsets_for_pivot_via forward",
        &mut forward,
    )?;
    decode_u32_output_exact(
        bwd_out,
        words,
        "forward_backward_bitsets_for_pivot_via backward",
        &mut backward,
    )?;
    Ok((forward, backward))
}

/// GPU-backed SCC composition over reachability and SCC-decompose primitives.
///
/// # Errors
///
/// Propagates closure or SCC-decompose dispatch failures.
pub fn scc_components_via_substrate_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut scratch = SccComponentsGpuScratch::default();
    let mut components = Vec::new();
    scc_components_via_substrate_with_scratch_into(
        dispatcher,
        adj,
        n,
        &mut scratch,
        &mut components,
    )?;
    Ok(components)
}

/// GPU-backed SCC composition using caller-owned scratch across closure and pivot dispatches.
///
/// # Errors
///
/// Propagates closure or SCC-decompose dispatch failures.
pub fn scc_components_via_substrate_with_scratch_via(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    scratch: &mut SccComponentsGpuScratch,
) -> Result<Vec<u32>, DispatchError> {
    let mut components = Vec::new();
    scc_components_via_substrate_with_scratch_into(dispatcher, adj, n, scratch, &mut components)?;
    Ok(components)
}

/// Return the SCC-decomposition launch grid for `node_count` lanes.
const fn scc_decompose_dispatch_grid(node_count: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(
        node_count,
        crate::graph::scc_decompose::SCC_DECOMPOSE_WORKGROUP_SIZE[0],
    )
}

/// GPU-backed SCC composition into caller-owned output storage.
///
/// # Errors
///
/// Returns [`DispatchError::BadInputs`] for malformed adjacency storage and
/// propagates closure or SCC-decompose dispatch failures.
pub fn scc_components_via_substrate_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    scratch: &mut SccComponentsGpuScratch,
    components: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    if n == 0 {
        if !adj.is_empty() {
            return Err(DispatchError::BadInputs(format!(
                "Fix: scc_components_via_substrate_via expected adj.len() == 0 for n=0, got {}.",
                adj.len()
            )));
        }
        components.clear();
        return Ok(());
    }
    let n_us = n as usize;
    let cells = n_us.checked_mul(n_us).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: scc_components_via_substrate_via n*n overflows usize for n={n}."
        ))
    })?;
    if adj.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: scc_components_via_substrate_via expected adj.len() == n*n == {cells}, got {}.",
            adj.len()
        )));
    }
    let words = ((n + 31) / 32) as usize;

    reachability_closure_via_with_scratch_into(
        dispatcher,
        adj,
        n,
        n,
        &mut scratch.semiring,
        &mut scratch.fwd_closure,
        &mut scratch.fwd_next,
    )?;
    scratch.transpose.clear();
    scratch.transpose.resize(cells, 0);
    for i in 0..n_us {
        for j in 0..n_us {
            scratch.transpose[j * n_us + i] = adj[i * n_us + j];
        }
    }
    reachability_closure_via_with_scratch_into(
        dispatcher,
        &scratch.transpose,
        n,
        n,
        &mut scratch.semiring,
        &mut scratch.bwd_closure,
        &mut scratch.bwd_next,
    )?;
    scratch.forward.clear();
    scratch.forward.resize(words, 0);
    scratch.backward.clear();
    scratch.backward.resize(words, 0);
    components.clear();
    components.resize(n_us, u32::MAX);
    ensure_input_slots(&mut scratch.inputs, 3);

    for pivot in 0..n {
        match components.get(pivot as usize) {
            Some(&u32::MAX) => {}
            _ => continue,
        }
        write_pivot_bitsets(
            &scratch.fwd_closure,
            &scratch.bwd_closure,
            pivot,
            n_us,
            &mut scratch.forward,
            &mut scratch.backward,
        );
        let program = crate::graph::scc_decompose::scc_decompose(
            n,
            "forward",
            "backward",
            "components",
            pivot,
        );
        write_u32_slice_le_bytes(&mut scratch.inputs[0], &scratch.forward);
        write_u32_slice_le_bytes(&mut scratch.inputs[1], &scratch.backward);
        write_u32_slice_le_bytes(&mut scratch.inputs[2], components);
        let outputs = dispatcher.dispatch(
            &program,
            &scratch.inputs,
            Some(scc_decompose_dispatch_grid(n)),
        )?;
        let [comp_out] = match outputs.as_slice() {
            [comp_out] => [comp_out],
            _ => {
                return Err(DispatchError::BackendError(format!(
                    "Fix: scc_components_via_substrate_via expected exactly one component output, got {}.",
                    outputs.len()
                )));
            }
        };
        decode_u32_output_exact(
            comp_out,
            n_us,
            "scc_components_via_substrate_via components",
            components,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use vyre_foundation::ir::Program;

    use super::super::{SccComponentsGpuScratch, Semiring};
    use super::{
        forward_backward_bitsets_for_pivot_via, scc_components_via_substrate_with_scratch_into,
        scc_decompose_dispatch_grid, semiring_gemm_via, semiring_gemm_via_into,
    };
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use crate::test_parity_oracles::StaticOutputs;
    use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

    struct SequenceDispatcher {
        outputs: Vec<Vec<Vec<u8>>>,
        expected_input_counts: Vec<usize>,
        cursor: Cell<usize>,
    }

    impl ProgramDispatcher for SequenceDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            _grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            let idx = self.cursor.get();
            let expected_inputs =
                self.expected_input_counts
                    .get(idx)
                    .copied()
                    .ok_or_else(|| {
                        DispatchError::BackendError(
                            "Fix: sequence dispatcher exhausted input expectations.".into(),
                        )
                    })?;
            if inputs.len() != expected_inputs {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: sequence test dispatcher expected {expected_inputs} inputs, got {}.",
                    inputs.len()
                )));
            }
            self.cursor.set(idx + 1);
            self.outputs.get(idx).cloned().ok_or_else(|| {
                DispatchError::BackendError("Fix: sequence dispatcher exhausted outputs.".into())
            })
        }
    }

    #[test]
    fn semiring_via_into_decodes_exact_output_into_reused_buffer() {
        let dispatcher =
            StaticOutputs::new("semiring gemm dispatch", vec![u32_slice_to_le_bytes(&[7])])
                .expecting_grid([1, 1, 1])
                .expecting_inputs(&[3]);
        let mut c = Vec::with_capacity(4);
        let ptr = c.as_ptr();
        semiring_gemm_via_into(&dispatcher, &[2], &[3], 1, 1, 1, Semiring::Real, &mut c)
            .expect("Fix: dispatch succeeds");
        assert_eq!(c, vec![7]);
        assert_eq!(c.as_ptr(), ptr);
    }

    #[test]
    fn semiring_via_rejects_extra_outputs() {
        let dispatcher = StaticOutputs::new(
            "semiring gemm dispatch",
            vec![u32_slice_to_le_bytes(&[7]), u32_slice_to_le_bytes(&[0])],
        )
        .expecting_grid([1, 1, 1])
        .expecting_inputs(&[3]);
        let err = semiring_gemm_via(&dispatcher, &[2], &[3], 1, 1, 1, Semiring::Real)
            .expect_err("extra outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn semiring_via_rejects_trailing_output_bytes() {
        let dispatcher = StaticOutputs::new("semiring gemm dispatch", vec![vec![7, 0, 0, 0, 1]])
            .expecting_grid([1, 1, 1])
            .expecting_inputs(&[3]);
        let err = semiring_gemm_via(&dispatcher, &[2], &[3], 1, 1, 1, Semiring::Real)
            .expect_err("trailing output bytes must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn scc_components_gpu_into_reuses_output_storage() {
        let adj = vec![0, 1, 1, 0];
        let semiring_step_a = u32_slice_to_le_bytes(&[1, 0, 0, 1]);
        let semiring_step_b = u32_slice_to_le_bytes(&[1, 1, 1, 1]);
        let components_done = u32_slice_to_le_bytes(&[0, 0]);
        let dispatcher = SequenceDispatcher {
            outputs: vec![
                vec![semiring_step_a.clone()],
                vec![semiring_step_b.clone()],
                vec![semiring_step_a.clone()],
                vec![semiring_step_b.clone()],
                vec![components_done.clone()],
                vec![semiring_step_a.clone()],
                vec![semiring_step_b.clone()],
                vec![semiring_step_a],
                vec![semiring_step_b],
                vec![components_done],
            ],
            expected_input_counts: vec![3; 10],
            cursor: Cell::new(0),
        };
        let mut scratch = SccComponentsGpuScratch::default();
        let mut components = Vec::with_capacity(2);

        scc_components_via_substrate_with_scratch_into(
            &dispatcher,
            &adj,
            2,
            &mut scratch,
            &mut components,
        )
        .unwrap();
        let capacity = components.capacity();
        assert_eq!(components, vec![0, 0]);

        scc_components_via_substrate_with_scratch_into(
            &dispatcher,
            &adj,
            2,
            &mut scratch,
            &mut components,
        )
        .unwrap();
        assert_eq!(components.capacity(), capacity);
        assert_eq!(components, vec![0, 0]);
    }

    #[test]
    fn scc_components_zero_node_validation_precedes_output_mutation() {
        let dispatcher = SequenceDispatcher {
            outputs: vec![],
            expected_input_counts: vec![],
            cursor: Cell::new(0),
        };
        let mut scratch = SccComponentsGpuScratch::default();
        let mut components = vec![9, 8];
        let err = scc_components_via_substrate_with_scratch_into(
            &dispatcher,
            &[1],
            0,
            &mut scratch,
            &mut components,
        )
        .expect_err("non-empty zero-node adjacency must fail");
        assert!(matches!(err, DispatchError::BadInputs(_)));
        assert_eq!(components, [9, 8]);
    }

    #[test]
    fn forward_backward_bitsets_for_pivot_dispatches_dual_closures_and_gpu_packing() {
        let adj = vec![0, 1, 1, 0];
        let semiring_step_a = u32_slice_to_le_bytes(&[1, 0, 0, 1]);
        let semiring_step_b = u32_slice_to_le_bytes(&[1, 1, 1, 1]);
        let packed_bitsets = u32_slice_to_le_bytes(&[0b11]);
        let dispatcher = SequenceDispatcher {
            outputs: vec![
                vec![semiring_step_a.clone()],
                vec![semiring_step_b.clone()],
                vec![semiring_step_a],
                vec![semiring_step_b],
                vec![packed_bitsets.clone(), packed_bitsets],
            ],
            expected_input_counts: vec![3, 3, 3, 3, 4],
            cursor: Cell::new(0),
        };
        let (fwd, bwd) = forward_backward_bitsets_for_pivot_via(&dispatcher, &adj, 0, 2).unwrap();
        assert_eq!(fwd, vec![0b11]);
        assert_eq!(bwd, vec![0b11]);
    }

    #[test]
    fn forward_backward_bitsets_for_pivot_rejects_extra_outputs() {
        let adj = vec![0, 1, 1, 0];
        let semiring_step_a = u32_slice_to_le_bytes(&[1, 0, 0, 1]);
        let semiring_step_b = u32_slice_to_le_bytes(&[1, 1, 1, 1]);
        let packed_bitsets = u32_slice_to_le_bytes(&[0b11]);
        let dispatcher = SequenceDispatcher {
            outputs: vec![
                vec![semiring_step_a.clone()],
                vec![semiring_step_b.clone()],
                vec![semiring_step_a],
                vec![semiring_step_b],
                vec![
                    packed_bitsets.clone(),
                    packed_bitsets.clone(),
                    packed_bitsets,
                ],
            ],
            expected_input_counts: vec![3, 3, 3, 3, 4],
            cursor: Cell::new(0),
        };
        let err = forward_backward_bitsets_for_pivot_via(&dispatcher, &adj, 0, 2).unwrap_err();
        assert!(matches!(err, DispatchError::BackendError(_)));
    }

    #[test]
    fn forward_backward_bitsets_for_pivot_rejects_invalid_pivot() {
        let dispatcher = SequenceDispatcher {
            outputs: vec![],
            expected_input_counts: vec![],
            cursor: Cell::new(0),
        };
        let err =
            forward_backward_bitsets_for_pivot_via(&dispatcher, &[0, 0, 0, 0], 2, 2).unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    #[test]
    fn scc_components_gpu_dispatches_with_workgroup_grid() {
        struct GridCheckDispatcher {
            expected_grid: [u32; 3],
        }
        impl ProgramDispatcher for GridCheckDispatcher {
            fn dispatch(
                &self,
                _program: &Program,
                inputs: &[Vec<u8>],
                grid_override: Option<[u32; 3]>,
            ) -> Result<Vec<Vec<u8>>, DispatchError> {
                if inputs.len() == 3
                    && inputs[2].len() == 8
                    && grid_override == Some(self.expected_grid)
                {
                    Ok(vec![u32_slice_to_le_bytes(&[0, 0])])
                } else {
                    // Semiring closure steps
                    Ok(vec![u32_slice_to_le_bytes(&[1, 1, 1, 1])])
                }
            }
        }
        let dispatcher = GridCheckDispatcher {
            expected_grid: scc_decompose_dispatch_grid(2),
        };
        let adj = vec![0, 1, 1, 0];
        let mut scratch = SccComponentsGpuScratch::default();
        let mut components = Vec::new();
        scc_components_via_substrate_with_scratch_into(
            &dispatcher,
            &adj,
            2,
            &mut scratch,
            &mut components,
        )
        .expect("dispatch with correct grid override succeeds");
        assert_eq!(components, vec![0, 0]);
    }
}
