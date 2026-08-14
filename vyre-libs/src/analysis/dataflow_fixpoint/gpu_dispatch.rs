//! Dispatcher-backed wrappers: each closure step and SCC pivot query is routed
//! through vyre dispatch while the host owns the fixed-point loop.

use vyre_foundation::pass_substrate::dataflow_fixpoint as foundation_dataflow;

use super::scc_decomposition::write_pivot_bitsets;
use super::{SccComponentsGpuScratch, Semiring, SemiringGemmGpuScratch};
use crate::dispatch_buffers::{
    ceil_div_u32, decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    write_zero_bytes,
};
use crate::device::scratch::reserve_vec_capacity;
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

    let program =
        vyre_primitives::math::semiring_gemm::semiring_gemm("a", "b", "c", m, n, k, semiring);
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], a);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], b);
    write_zero_bytes(&mut scratch.inputs[2], c_bytes);
    let grid_x = ceil_div_u32(c_words_u32, 256);
    let outputs = dispatcher.dispatch(&program, &scratch.inputs, Some([grid_x, 1, 1]))?;
    if outputs.len() != 1 {
        return Err(DispatchError::BackendError(format!(
            "Fix: semiring_gemm_via expected exactly one c output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(&outputs[0], c_words, "semiring_gemm_via c", c)
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

    let fwd_closure = reachability_closure_via(dispatcher, adj, n, n)?;
    let mut transpose = vec![0u32; cells];
    for i in 0..n_us {
        for j in 0..n_us {
            transpose[j * n_us + i] = adj[i * n_us + j];
        }
    }
    let bwd_closure = reachability_closure_via(dispatcher, &transpose, n, n)?;
    let words = ((n + 31) / 32) as usize;
    let mut forward = vec![0u32; words];
    let mut backward = vec![0u32; words];
    write_pivot_bitsets(
        &fwd_closure,
        &bwd_closure,
        pivot,
        n_us,
        &mut forward,
        &mut backward,
    );
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

/// GPU-backed SCC composition into caller-owned output storage.
///
/// # Errors
///
/// Propagates closure or SCC-decompose dispatch failures.
pub fn scc_components_via_substrate_with_scratch_into(
    dispatcher: &dyn ProgramDispatcher,
    adj: &[u32],
    n: u32,
    scratch: &mut SccComponentsGpuScratch,
    components: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    if n == 0 {
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
    let words = ((n + 31) / 32) as usize;
    scratch.forward.clear();
    scratch.forward.resize(words, 0);
    scratch.backward.clear();
    scratch.backward.resize(words, 0);
    components.clear();
    components.resize(n_us, u32::MAX);
    ensure_input_slots(&mut scratch.inputs, 3);

    for pivot in 0..n {
        if components[pivot as usize] != u32::MAX {
            continue;
        }
        write_pivot_bitsets(
            &scratch.fwd_closure,
            &scratch.bwd_closure,
            pivot,
            n_us,
            &mut scratch.forward,
            &mut scratch.backward,
        );
        let program = vyre_primitives::graph::scc_decompose::scc_decompose(
            n,
            "forward",
            "backward",
            "components",
            pivot,
        );
        write_u32_slice_le_bytes(&mut scratch.inputs[0], &scratch.forward);
        write_u32_slice_le_bytes(&mut scratch.inputs[1], &scratch.backward);
        write_u32_slice_le_bytes(&mut scratch.inputs[2], components);
        let outputs = dispatcher.dispatch(&program, &scratch.inputs, Some([n, 1, 1]))?;
        if outputs.len() != 1 {
            return Err(DispatchError::BackendError(format!(
                "Fix: scc_components_via_substrate_via expected exactly one component output, got {}.",
                outputs.len()
            )));
        }
        decode_u32_output_exact(
            &outputs[0],
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
        scc_components_via_substrate_with_scratch_into, semiring_gemm_via, semiring_gemm_via_into,
    };
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

    struct SemiringDispatcher {
        outputs: Vec<Vec<u8>>,
    }

    impl ProgramDispatcher for SemiringDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_eq!(grid_override, Some([1, 1, 1]));
            if inputs.len() != 3 {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: semiring test dispatcher expected 3 inputs, got {}.",
                    inputs.len()
                )));
            }
            Ok(self.outputs.clone())
        }
    }

    struct SequenceDispatcher {
        outputs: Vec<Vec<Vec<u8>>>,
        cursor: Cell<usize>,
    }

    impl ProgramDispatcher for SequenceDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            _grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            if inputs.len() != 3 {
                return Err(DispatchError::BadInputs(format!(
                    "Fix: sequence test dispatcher expected 3 inputs, got {}.",
                    inputs.len()
                )));
            }
            let idx = self.cursor.get();
            self.cursor.set(idx + 1);
            self.outputs.get(idx).cloned().ok_or_else(|| {
                DispatchError::BackendError("Fix: sequence dispatcher exhausted outputs.".into())
            })
        }
    }

    #[test]
    fn semiring_via_into_decodes_exact_output_into_reused_buffer() {
        let dispatcher = SemiringDispatcher {
            outputs: vec![u32_slice_to_le_bytes(&[7])],
        };
        let mut c = Vec::with_capacity(4);
        let ptr = c.as_ptr();
        semiring_gemm_via_into(&dispatcher, &[2], &[3], 1, 1, 1, Semiring::Real, &mut c)
            .expect("Fix: dispatch succeeds");
        assert_eq!(c, vec![7]);
        assert_eq!(c.as_ptr(), ptr);
    }

    #[test]
    fn semiring_via_rejects_extra_outputs() {
        let dispatcher = SemiringDispatcher {
            outputs: vec![u32_slice_to_le_bytes(&[7]), u32_slice_to_le_bytes(&[0])],
        };
        let err = semiring_gemm_via(&dispatcher, &[2], &[3], 1, 1, 1, Semiring::Real)
            .expect_err("extra outputs must be rejected");
        assert!(
            matches!(err, DispatchError::BackendError(_)),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn semiring_via_rejects_trailing_output_bytes() {
        let dispatcher = SemiringDispatcher {
            outputs: vec![vec![7, 0, 0, 0, 1]],
        };
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
}
