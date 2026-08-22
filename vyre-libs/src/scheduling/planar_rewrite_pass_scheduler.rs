//! IR rewrite-batch scheduler via planar_rewrite.
//!
//! Closes the recursion thesis: planar grammar rewriting
//! ships to user dialects (visual-programming languages, 2D
//! pattern-matching languages) AND schedules vyre's own batch IR
//! rewrites for non-conflicting parallel application.
//!
//! # The self-use
//!
//! Vyre's optimizer applies many local rewrites per pass  -  fold a
//! constant, inline a region, eliminate a dead store. When N
//! candidate rewrites apply to the same Region, naive sequential
//! application means N kernel launches and N validation steps.
//!
//! Most of those N rewrites operate on disjoint sub-trees of the
//! Region tree. They could batch into ONE parallel application  -
//! the only constraint is "two rewrites can't touch overlapping
//! sub-trees in the same batch."
//!
//! That's exactly the **planar non-overlapping selection** problem
//! that `planar_rewrite_schedule` solves: given a 2D grid of
//! candidate matches, greedily select a maximum non-overlapping
//! subset.
//!
//! # Algorithm
//!
//! ```text
//! 1. lay out the Region tree as a 2D grid (rows = depth, columns =
//!    sibling order)
//! 2. mark `candidates[i, j] = 1` for every (i, j) where a rewrite
//!    pattern matches
//! 3. schedule candidates with the planar rewrite primitive  -  k=2 means
//!    each rewrite "covers" a 2×2 sub-region (parent + immediate
//!    descendant). Returns the maximum non-conflicting subset.
//! 4. apply the selected rewrites in ONE batched dispatch
//! ```
//!
//! # Why this matters
//!
//! At workspace scale, an optimizer pass may have 100k+ candidate
//! rewrites. Sequential application with kernel launch overhead
//! kills throughput. Planar-scheduled batching cuts the dispatch
//! count by orders of magnitude with provably-correct disjointness
//! (the greedy schedule never picks two overlapping rewrites).

use crate::dispatch_buffers::{
    checked_product_count, decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes,
    write_zero_bytes,
};
use crate::parsing::planar_rewrite::planar_rewrite_schedule;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

/// Caller-owned GPU dispatch scratch for planar rewrite scheduling.
#[derive(Debug, Default)]
pub struct PlanarRewriteScheduleGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Schedule a batch of non-conflicting IR rewrites through the dispatcher.
///
/// This dispatches the primitive [`planar_rewrite_schedule`] and returns the
/// selected `h x w` mask. The primitive's contract is single-lane greedy
/// scheduling; callers that need higher-throughput graph coloring should use a
/// separate primitive rather than changing this deterministic order.
///
/// # Errors
///
/// Returns [`DispatchError`] when shape validation fails, `k == 0`, or the
/// backend returns malformed output.
pub fn schedule_disjoint_rewrites_via(
    dispatcher: &impl ProgramDispatcher,
    candidates: &[u32],
    h: u32,
    w: u32,
    k: u32,
) -> Result<Vec<u32>, DispatchError> {
    let mut out = Vec::new();
    schedule_disjoint_rewrites_via_into(dispatcher, candidates, h, w, k, &mut out)?;
    Ok(out)
}

/// Dispatcher-backed planar rewrite scheduling into caller-owned storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn schedule_disjoint_rewrites_via_into(
    dispatcher: &impl ProgramDispatcher,
    candidates: &[u32],
    h: u32,
    w: u32,
    k: u32,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    let mut scratch = PlanarRewriteScheduleGpuScratch::default();
    schedule_disjoint_rewrites_via_with_scratch_into(
        dispatcher,
        candidates,
        h,
        w,
        k,
        &mut scratch,
        out,
    )
}

/// Dispatcher-backed planar rewrite scheduling into caller-owned dispatch and
/// output storage.
///
/// # Errors
///
/// Returns [`DispatchError`] when validation or backend execution fails.
pub fn schedule_disjoint_rewrites_via_with_scratch_into(
    dispatcher: &impl ProgramDispatcher,
    candidates: &[u32],
    h: u32,
    w: u32,
    k: u32,
    scratch: &mut PlanarRewriteScheduleGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), DispatchError> {
    use crate::telemetry::{bump, planar_rewrite_pass_scheduler_calls};
    bump(&planar_rewrite_pass_scheduler_calls);

    if k == 0 {
        return Err(DispatchError::BadInputs(
            "Fix: schedule_disjoint_rewrites_via requires k > 0.".to_string(),
        ));
    }
    let cells = checked_product_count(h, w, "h", "w", "schedule_disjoint_rewrites_via")?;
    if candidates.len() != cells {
        return Err(DispatchError::BadInputs(format!(
            "Fix: schedule_disjoint_rewrites_via requires candidates.len() == h*w, got len={}, h={h}, w={w}, h*w={cells}.",
            candidates.len()
        )));
    }

    let program = planar_rewrite_schedule("candidates", "chosen", h, w, k);
    let output_bytes = cells
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| {
            DispatchError::BadInputs(format!(
                "Fix: schedule_disjoint_rewrites_via output byte count overflows usize for {cells} cells."
            ))
        })?;
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], candidates);
    write_zero_bytes(&mut scratch.inputs[1], output_bytes);
    let outputs = dispatcher.dispatch(&program, &scratch.inputs, Some([1, 1, 1]))?;
    if outputs.is_empty() {
        return Err(DispatchError::BackendError(format!(
            "Fix: schedule_disjoint_rewrites_via expected at least one output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_u32_output_exact(&outputs[0], cells, "schedule_disjoint_rewrites_via", out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use vyre_foundation::ir::Program;
    use vyre_reference::composition_witness::{
        planar_rewrite_schedule_witness as reference_planar_rewrite_schedule,
        reduce_count_non_zero_witness as count_scheduled,
    };

    fn schedule_disjoint_rewrites(candidates: &[u32], h: u32, w: u32, k: u32) -> Vec<u32> {
        use crate::telemetry::{bump, planar_rewrite_pass_scheduler_calls};
        bump(&planar_rewrite_pass_scheduler_calls);
        assert!(k > 0, "Fix: rewrite footprint k must be > 0.");
        reference_planar_rewrite_schedule(candidates, h, w, k)
    }

    #[test]
    fn empty_grid_yields_no_schedule() {
        let candidates = vec![0u32; 16];
        let schedule = schedule_disjoint_rewrites(&candidates, 4, 4, 2);
        assert_eq!(count_scheduled(&schedule), 0);
    }

    #[test]
    fn full_grid_yields_disjoint_subset() {
        // 4x4 grid, all candidates, k=2. Expected: at most ⌈4/2⌉² = 4
        // rewrites picked at positions (0,0), (0,2), (2,0), (2,2).
        let candidates = vec![1u32; 16];
        let schedule = schedule_disjoint_rewrites(&candidates, 4, 4, 2);
        let count = count_scheduled(&schedule);
        // Disjointness with k=2 footprint allows at most 4 rewrites.
        assert!(count >= 1, "at least one rewrite must be schedulable");
        assert!(count <= 4, "at most 4 disjoint k=2 rewrites in a 4x4 grid");
    }

    #[test]
    fn k_one_allows_every_candidate() {
        // k=1 means each rewrite covers a 1x1 sub-region  -  no
        // overlap possible, so every candidate is selectable.
        let candidates = vec![1u32, 1, 1, 1];
        let schedule = schedule_disjoint_rewrites(&candidates, 2, 2, 1);
        assert_eq!(count_scheduled(&schedule), 4);
    }

    struct PlanarDispatcher;

    impl ProgramDispatcher for PlanarDispatcher {
        fn dispatch(
            &self,
            _program: &Program,
            inputs: &[Vec<u8>],
            grid_override: Option<[u32; 3]>,
        ) -> Result<Vec<Vec<u8>>, DispatchError> {
            assert_eq!(grid_override, Some([1, 1, 1]));
            assert_eq!(inputs.len(), 2);
            let candidates = crate::dispatch_buffers::read_u32s(&inputs[0]);
            let n = integer_sqrt(candidates.len());
            let chosen = reference_planar_rewrite_schedule(&candidates, n as u32, n as u32, 2);
            Ok(vec![u32_slice_to_le_bytes(&chosen)])
        }
    }

    #[test]
    fn schedule_disjoint_rewrites_via_dispatches_primitive() {
        let candidates = vec![1u32; 16];
        let via = schedule_disjoint_rewrites_via(&PlanarDispatcher, &candidates, 4, 4, 2).unwrap();
        let reference = schedule_disjoint_rewrites(&candidates, 4, 4, 2);
        assert_eq!(via, reference);
    }

    #[test]
    fn schedule_disjoint_rewrites_via_with_scratch_reuses_dispatch_and_output_storage() {
        let candidates = vec![1u32; 16];
        let mut scratch = PlanarRewriteScheduleGpuScratch::default();
        let mut out = Vec::with_capacity(16);

        schedule_disjoint_rewrites_via_with_scratch_into(
            &PlanarDispatcher,
            &candidates,
            4,
            4,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let out_capacity = out.capacity();

        schedule_disjoint_rewrites_via_with_scratch_into(
            &PlanarDispatcher,
            &candidates,
            4,
            4,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(out, schedule_disjoint_rewrites(&candidates, 4, 4, 2));
    }

    #[test]
    fn schedule_disjoint_rewrites_via_rejects_bad_shape() {
        let err =
            schedule_disjoint_rewrites_via(&PlanarDispatcher, &[1, 0, 1], 2, 2, 2).unwrap_err();
        assert!(matches!(err, DispatchError::BadInputs(_)));
    }

    fn integer_sqrt(n: usize) -> usize {
        let mut root = 0usize;
        while root * root < n {
            root += 1;
        }
        root
    }
}
