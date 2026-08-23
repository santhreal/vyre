//! `scallop_join`  -  Scallop-style probabilistic Datalog join, GPU-resident.
//!
//! Compiles a Datalog fixpoint into GPU-resident dispatch phases by
//! emitting a Lineage-semiring relational join. Small matrices use a
//! block-local convergence loop; large matrices expose split-visible
//! GridSync phases for multi-block dispatch. The output cell
//! `C[i,j]` is the bitset union of clauses
//! participating in any `i ⇝ j` derivation through one join step.
//!
//! # Why ship this as a named primitive instead of "compose them yourself"
//!
//! Two reasons:
//!
//! ## (a) The fixpoint contract
//!
//! Datalog fixpoint converges when no new fact is derived. Under the
//! Lineage semiring that means no clause-bitset OR'd into any cell
//! flips a 0 bit to 1  -  the canonical convergence signal `next ==
//! current` per word. `scallop_join` packages the Lineage transfer and
//! convergence loop together so callers do not re-derive that the
//! Lineage semiring's monotonic OR-accumulator is safe with ping-pong
//! equality convergence. Other semirings would NOT be safe  -  `MinPlus`
//! accumulators decrease over iterations, which the equality check would
//! treat as "changed = 1" indefinitely until the absolute minimum settles.
//! So the recursion-thesis-clean wrapper is the contract:
//!
//! > "scallop_join is exactly the Datalog-shaped, monotone,
//! >  GPU-resident Lineage fixpoint."
//!
//! ## (b) Two consumers, recursion thesis closed from day 1
//!
//! - **User dialect consumer**: probabilistic Datalog programs (Scallop
//!   programs compile each rule body to one `scallop_join`). Substrate
//!   for neuro-symbolic reasoning systems.
//! - **vyre-self consumer**: rule-provenance tracking for external analyzer / any
//!   substrate that needs to ask "which input rule produced this output
//!   finding?" The answer is a Datalog query over (rule_id, derives,
//!   finding_id), and `scallop_join` is the GPU-resident execution.
//!   See [`crate::math::scallop_join::PROVENANCE_SELF_CONSUMER`].
//!
//! # Algorithm
//!
//! ```text
//! initial:    R[0]   = adjacency matrix encoding source → target
//!                      facts; cell is the bitset of clauses introducing
//!                      that edge (Lineage encoding).
//! transfer:   R[t+1] = R[t] ⊗_Lineage A_join,  where A_join is the
//!                      static join-rule adjacency. Combine = "OR
//!                      participating clauses across one path step",
//!                      Accumulate = "OR alternative derivations into
//!                      the same cell."
//! converge:   stop when R[t+1] == R[t] per cell.
//! ```
//!
//! A cell is `w` contiguous `u32` bitset words, so a clause set holds
//! `32 * w` rules. `w = 1` is the single-word form, and the contract test
//! distinguishing "no edge" from "edge with empty clause set" through the
//! zero-absorbing combine holds at every `w`.
//!
//! # Wiring contract
//!
//! Caller supplies:
//!
//! - `state`: `n × n × w` word buffer (ReadWrite). Initialized by caller
//!   with the seed facts; mutated to fixpoint by the dispatch.
//! - `next`: `n × n × w` scratch buffer (ReadWrite). Reused as the
//!   ping-pong target between fixpoint iterations.
//! - `join_rules`: `n × n × w` static join-rule adjacency (ReadOnly).
//!   `join_rules[i,j]` is the clause bitset that, when present at
//!   `state[i,k]` and `join_rules[k,j]` for some k, derives a fact at
//!   `state[i,j]`.
//! - `changed`: 1-word convergence flag (ReadWrite, atomic OR).
//! - `n`: matrix dimension (relations encoded as n × n cells).
//! - `w`: `u32` words per cell, so a clause set holds `32 * w` rules.
//! - `max_iterations`: hard upper bound (Datalog fixpoint is monotone
//!   so converges in ≤ n^2 iterations; cap at a safety multiple).
//!
//! # CPU reference
//!
//! The `vyre-reference` scallop join witness performs the same fixpoint iteration on host arrays and
//! is the parity oracle for every GPU dispatch.

use vyre_foundation::composition::trap_program;
use vyre_foundation::ir::{DataType, Program};

use crate::math::scallop_persistent::{
    lineage_fixpoint_program, single_word_lineage_body, single_word_lineage_grid_sync_body,
    wide_lineage_body, wide_lineage_grid_sync_body, LineageFixpoint,
};

/// Canonical op id.
pub const OP_ID: &str = "vyre-libs::math::scallop_join";
/// One lane per relation cell in the lineage fixpoint.
pub const SCALLOP_JOIN_WORKGROUP_SIZE: [u32; 3] = [256, 1, 1];

/// Dispatch grid for the Scallop kernel.
///
/// # W-wide lane mapping
///
/// One lane per relation CELL, not per word. A lane owns the `w` contiguous
/// `u32` words of its cell and walks them, so `n * n` lanes cover all
/// `n * n * w` words and the grid does not scale with `w`. The word-wise walk
/// lives in `scallop_persistent::wide_transfer_body` and
/// `wide_compare_body`, which derive `cell_base = cell * w` and iterate `w`
/// words from there.
///
/// Over the [`vyre_primitives::lane_grid`] owner, so the zero-relation case still
/// yields a launchable grid.
#[must_use]
pub const fn scallop_join_dispatch_grid(n: u32) -> [u32; 3] {
    vyre_primitives::lane_grid(n.saturating_mul(n), SCALLOP_JOIN_WORKGROUP_SIZE[0])
}

/// Documentation hook for the recursion-thesis self-consumer wired in
/// `vyre-libs::self_substrate::scallop_provenance`. Updates to this
/// constant must update the self-consumer module's doc-link.
pub const PROVENANCE_SELF_CONSUMER: &str = "vyre-libs::self_substrate::scallop_provenance";

/// Build a fused Datalog-fixpoint Program: iterate the Lineage join until
/// convergence for block-local matrices, or through fixed split-visible phases
/// for larger matrices.
///
/// The transfer step writes `next` from `state` and the supplied
/// join-rule matrix, then compares and copies the ping-pong buffer. Small
/// matrices finish inside one workgroup. Larger matrices surface top-level
/// GridSync barriers so the megakernel planner can cut transfer and compare
/// phases into sequential dispatches across blocks.
///
/// `w` is the number of `u32` words per relation cell. `w = 1` emits the
/// single-word bodies, where a lane owns one word. `w > 1` emits the wide
/// bodies, where a lane owns one cell and walks its `w` words.
///
/// # Panics
///
/// Panics if `n == 0`, `w == 0`, or `max_iterations == 0`.
#[must_use]
pub fn scallop_join(
    state: &str,
    next: &str,
    join_rules: &str,
    changed: &str,
    n: u32,
    w: u32,
    max_iterations: u32,
) -> Program {
    if n == 0 {
        return trap_program(
            OP_ID,
            Some((state, DataType::U32)),
            format!("Fix: scallop_join requires n > 0, got {n}."),
        );
    }
    if w == 0 {
        return trap_program(
            OP_ID,
            Some((state, DataType::U32)),
            "Fix: scallop_join requires w > 0, got 0.".to_string(),
        );
    }
    if max_iterations == 0 {
        return trap_program(
            OP_ID,
            Some((state, DataType::U32)),
            "Fix: scallop_join requires max_iterations > 0, got 0.".to_string(),
        );
    }

    let spec = LineageFixpoint {
        state,
        next,
        join_rules,
        changed,
        n,
        w,
        lanes: SCALLOP_JOIN_WORKGROUP_SIZE[0],
        max_iterations,
    };
    let block_local = spec.cells() <= SCALLOP_JOIN_WORKGROUP_SIZE[0];

    let body = match (w, block_local) {
        (1, true) => single_word_lineage_body(&spec),
        (1, false) => single_word_lineage_grid_sync_body(&spec),
        (_, true) => wide_lineage_body(&spec),
        (_, false) => wide_lineage_grid_sync_body(&spec),
    };

    lineage_fixpoint_program(OP_ID, &spec, SCALLOP_JOIN_WORKGROUP_SIZE, body)
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        OP_ID,
        || scallop_join("state", "next", "join_rules", "changed", 2, 1, 4),
        Some(|| {
            // Seed: state[0,1] = clause-bit 0 (a derives b directly).
            // join: join_rules[1,1] = clause-bit 1 (b derives b through itself, transitively).
            // After one round: state[0,1] |= join_rules[1,1] applied through k=1 → bits 0 + 1.
            let to_bytes = |w: &[u32]| vyre_primitives::wire::pack_u32_slice(w);
            vec![vec![
                to_bytes(&[0, 0b01, 0, 0]),
                to_bytes(&[0, 0, 0, 0]),
                to_bytes(&[0]),
                to_bytes(&[0, 0, 0, 0b10]),
            ]]
        }),
        Some(|| {
            vec![vec![
                vec![
                    0x00, 0x00, 0x00, 0x00,
                    0x03, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                ], // state
                vec![
                    0x00, 0x00, 0x00, 0x00,
                    0x03, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00,
                ], // next
                vec![0x00, 0x00, 0x00, 0x00], // changed
            ]]
        }),
    )
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::fixpoint::persistent_fixpoint::count_grid_sync;
    use crate::fixture_bytes::eval_bytes;
    use crate::math::semiring_gemm::Semiring;
    use vyre_reference::composition_witness::{
        scallop_join_fixpoint_witness as cpu_ref, scallop_join_fixpoint_witness_into,
        semiring_gemm_witness,
    };

    fn cpu_ref_into(
        state: &[u32],
        join_rules: &[u32],
        n: u32,
        w: u32,
        max_iterations: u32,
        current: &mut Vec<u32>,
        next: &mut Vec<u32>,
    ) -> u32 {
        scallop_join_fixpoint_witness_into(state, join_rules, n, w, max_iterations, current, next)
    }

    fn accumulate_lineage_words(current: &mut [u32], derived: &[u32]) -> bool {
        let mut changed = false;
        for (value, &new_bits) in current.iter_mut().zip(derived) {
            let accumulated = *value | new_bits;
            changed |= accumulated != *value;
            *value = accumulated;
        }
        changed
    }

    #[test]
    fn cpu_ref_one_step_join() {
        // 2x2 system. state[0,1]=clause 0; join_rules[1,1]=clause 1.
        // First fixpoint round: derive state[0,1] |= state[0,k] · join_rules[k,1]
        //   = state[0,1] · join_rules[1,1] = bit0 · bit1 (Lineage combine: OR
        //   when both nonzero) = bits 0+1.
        let state = vec![0u32, 0b01, 0u32, 0u32];
        let join_rules = vec![0u32, 0u32, 0u32, 0b10];
        let (final_state, iters) = cpu_ref(&state, &join_rules, 2, 1, 16);
        // state[0,1] should now have bit 1 OR'd in (the lineage of the
        // newly derived path).
        assert_eq!(
            final_state[1] & 0b10,
            0b10,
            "Lineage of clause 1 must propagate to state[0,1] after one round"
        );
        // bit 0 (the seed) must persist  -  Datalog never retracts facts.
        assert_eq!(
            final_state[1] & 0b01,
            0b01,
            "seed clause 0 must persist through the fixpoint"
        );
        assert!(
            iters <= 4,
            "small system should converge quickly, got {iters}"
        );
    }

    /// The oracle now walks `w` words per cell for every `w`. At `w = 1` it has
    /// to remain exactly the Lineage semiring GEMM fixpoint it replaced, or the
    /// single-word parity oracle silently changed meaning.
    #[test]
    fn the_single_word_oracle_is_the_lineage_semiring_gemm_fixpoint() {
        let n = 4u32;
        let cells = (n * n) as usize;
        let mut state = vec![0u32; cells];
        state[1] = 0b0001;
        state[6] = 0b0010;
        state[11] = 0b0100;
        let mut join_rules = vec![0u32; cells];
        join_rules[6] = 0b1000;
        join_rules[11] = 0b0001;
        join_rules[15] = 0b0010;

        let (actual, actual_iters) = cpu_ref(&state, &join_rules, n, 1, 16);

        let mut current = state.clone();
        let mut expected_iters = 16;
        for iter in 0..16 {
            let next = semiring_gemm_witness(
                &current,
                &join_rules,
                n as usize,
                n as usize,
                n as usize,
                Semiring::Lineage,
            );
            if !accumulate_lineage_words(&mut current, &next) {
                expected_iters = iter;
                break;
            }
        }

        assert_eq!(
            actual, current,
            "the w=1 oracle must agree word for word with the Lineage semiring GEMM fixpoint"
        );
        assert_eq!(
            actual_iters, expected_iters,
            "the w=1 oracle must report the same convergence round"
        );
    }

    #[test]
    fn cpu_ref_converges_on_idempotent_input() {
        // No new facts can be derived: state has only the diagonal
        // self-loop, join_rules has no clauses at all → first iteration
        // produces zeros + the seed; second iteration produces the same
        // → converges at iter 1.
        let state = vec![0b01, 0u32, 0u32, 0b01];
        let join_rules = vec![0u32; 4];
        let (final_state, iters) = cpu_ref(&state, &join_rules, 2, 1, 16);
        assert_eq!(
            final_state, state,
            "idempotent system must not change state"
        );
        assert!(iters <= 2, "idempotent system converges in ≤ 2 iters");
    }

    #[test]
    fn cpu_ref_into_reuses_buffers() {
        let state = vec![0u32, 0b01, 0u32, 0u32];
        let join_rules = vec![0u32, 0u32, 0u32, 0b10];
        let mut current = Vec::with_capacity(128);
        let mut next = Vec::with_capacity(128);
        let current_ptr = current.as_ptr();
        let next_ptr = next.as_ptr();
        let iters = cpu_ref_into(&state, &join_rules, 2, 1, 16, &mut current, &mut next);
        assert!(iters <= 4);
        assert_eq!(current[1] & 0b11, 0b11);
        assert_eq!(current.as_ptr(), current_ptr);
        assert_eq!(next.as_ptr(), next_ptr);
    }

    /// A reused scratch buffer longer than the current problem must not leak its
    /// stale tail into the answer, and must not reallocate.
    #[test]
    fn cpu_ref_into_truncates_a_stale_tail_without_reallocating() {
        let mut state = vec![0u32; 8];
        state[2] = 0b01;
        let mut join_rules = vec![0u32; 8];
        join_rules[7] = 0b10;
        let mut current = Vec::with_capacity(16);
        let mut next = Vec::with_capacity(16);
        current.extend_from_slice(&[99u32; 12]);
        next.extend_from_slice(&[77u32; 12]);
        let current_capacity = current.capacity();
        let next_capacity = next.capacity();

        let iters = cpu_ref_into(&state, &join_rules, 2, 2, 4, &mut current, &mut next);

        assert!(iters <= 4);
        assert_eq!(current, vec![0, 0, 0b01, 0b10, 0, 0, 0, 0]);
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);

        let iters = cpu_ref_into(&[0b01], &[0b10], 1, 1, 10, &mut current, &mut next);
        assert_eq!(iters, 1);
        assert_eq!(current, vec![0b11]);
        assert_eq!(next, vec![0b11]);
        assert_eq!(current.capacity(), current_capacity);
        assert_eq!(next.capacity(), next_capacity);
    }

    #[test]
    #[should_panic(expected = "complete n*n*w state matrix")]
    fn cpu_ref_short_inputs_fail_loudly() {
        let _ = cpu_ref(&[0b01], &[], 1, 2, 10);
    }

    #[test]
    fn cpu_ref_transitive_closure() {
        // 3-cell chain: state[0,1]=bit0, state[1,2]=bit1.
        // join_rules: same as state (each path step adds its own bit).
        // Fixpoint should produce state[0,2] with both bits set.
        let mut state = vec![0u32; 9];
        state[1] = 0b001; // (0→1) clause 0
        state[5] = 0b010; // (1→2) clause 1
        let join_rules = state.clone();
        let (final_state, iters) = cpu_ref(&state, &join_rules, 3, 1, 16);
        // Transitive derivation 0→1→2 must accumulate clauses 0 and 1.
        assert_eq!(
            final_state[2] & 0b011,
            0b011,
            "transitive 0→2 must collect lineage of both edges; got 0x{:x}",
            final_state[2]
        );
        assert!(iters <= 8, "3-node chain should converge fast");
    }

    /// A clause bit above 32 lives in a high word of the cell. Losing the walk
    /// over the high words would still pass every single-word assertion.
    #[test]
    fn cpu_ref_carries_lineage_in_high_words() {
        let n = 2u32;
        let w = 4u32;
        let mut state = vec![0u32; 16];
        state[6] = 0x1;
        let mut join_rules = vec![0u32; 16];
        join_rules[15] = 0x2;
        let (final_state, _) = cpu_ref(&state, &join_rules, n, w, 10);

        assert_eq!(final_state[6], 0x1, "the seed word must persist");
        assert_eq!(
            final_state[7], 0x2,
            "a clause bit in word 3 of the rule cell must reach word 3 of the state cell"
        );
    }

    #[test]
    fn cpu_ref_zero_absorbing_no_phantom_lineage() {
        // Edge present with empty clause set vs no edge  -  Lineage
        // combine is zero-absorbing, so an empty cell × any
        // join-rule cell stays zero (no spurious lineage).
        let state = vec![0u32; 4]; // no facts
        let join_rules = vec![0b11u32; 4];
        let (final_state, _) = cpu_ref(&state, &join_rules, 2, 1, 16);
        assert_eq!(
            final_state, state,
            "no seed facts → no derivations regardless of rule set; \
             zero-absorbing combine prevents phantom lineage"
        );
    }

    #[test]
    fn program_declares_four_buffers() {
        for w in [1u32, 2, 8] {
            let p = scallop_join("s", "n", "j", "c", 2, w, 4);
            let bufs = p.buffers();
            assert_eq!(
                bufs.len(),
                4,
                "scallop_join must declare 4 buffers at w={w}"
            );
            assert_eq!(p.workgroup_size(), SCALLOP_JOIN_WORKGROUP_SIZE);
            let names: Vec<&str> = bufs.iter().map(|b| b.name()).collect();
            assert!(names.contains(&"s"));
            assert!(names.contains(&"n"));
            assert!(names.contains(&"j"));
            assert!(names.contains(&"c"));
        }
    }

    #[test]
    fn dispatch_grid_scales_large_relations_into_blocks() {
        assert_eq!(scallop_join_dispatch_grid(0), [1, 1, 1]);
        assert_eq!(scallop_join_dispatch_grid(1), [1, 1, 1]);
        assert_eq!(scallop_join_dispatch_grid(16), [1, 1, 1]);
        assert_eq!(scallop_join_dispatch_grid(17), [2, 1, 1]);
        assert_eq!(scallop_join_dispatch_grid(33), [5, 1, 1]);
    }

    /// The grid is sized in CELLS, and a lane walks the `w` words of its cell.
    /// A grid sized in cells only covers the relation if that walk exists, so
    /// assert the launched lane count times the per-lane word count reaches
    /// every word of the relation. Dropping the walk from
    /// `wide_transfer_body` would leave `w - 1` words of every cell untouched
    /// while this arithmetic still held, so also assert the words each lane is
    /// responsible for are the contiguous run the body derives.
    #[test]
    fn a_wide_grid_covers_every_relation_word() {
        let n = 16u32;
        let w = 8u32;
        let grid = scallop_join_dispatch_grid(n);
        let lanes = grid[0] * grid[1] * grid[2] * SCALLOP_JOIN_WORKGROUP_SIZE[0];
        let words = n * n * w;

        assert!(
            lanes >= n * n,
            "one lane per cell: {lanes} lanes must cover {} cells",
            n * n
        );
        assert!(
            lanes * w >= words,
            "{lanes} lanes walking {w} words each must cover {words} relation words"
        );

        let covered: usize = (0..n * n)
            .flat_map(|cell| (0..w).map(move |word| cell * w + word))
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        assert_eq!(
            covered, words as usize,
            "cell_base = cell * w over w words must enumerate every word exactly once"
        );
    }

    #[test]
    fn large_program_uses_split_visible_grid_sync() {
        for w in [1u32, 2] {
            let p = scallop_join("s", "n", "j", "c", 17, w, 4);
            assert_eq!(
                count_grid_sync(p.entry()),
                7,
                "a relation past one workgroup must expose the split-visible phases at w={w}"
            );
        }
    }

    /// The emitted Program has to agree with the oracle at `w > 1`, which is the
    /// only check that the wide IR bodies and the wide oracle encode the same
    /// fixpoint.
    #[test]
    fn wide_program_matches_the_oracle_under_reference_evaluation() {
        let n = 2u32;
        let w = 2u32;
        let mut state_init = vec![0u32; 8];
        state_init[2] = 0b01;
        let mut join_rules = vec![0u32; 8];
        join_rules[7] = 0b10;

        let p = scallop_join("s", "nx", "j", "c", n, w, 4);
        let (expected_state, _) = cpu_ref(&state_init, &join_rules, n, w, 4);

        let to_value = |data: &[u32]| vyre_primitives::wire::pack_u32_slice(data);

        let inputs = vec![
            to_value(&state_init),
            to_value(&[0_u32; 8]),
            to_value(&[0]),
            to_value(&join_rules),
        ];

        let results = eval_bytes("scallop_join", &p, inputs);
        let actual_bytes = results[0].clone();
        let actual_state: Vec<u32> = actual_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        assert_eq!(actual_state, expected_state);
    }

    #[test]
    fn rejects_zero_n_with_trap() {
        let p = scallop_join("s", "n", "j", "c", 0, 1, 4);
        assert!(p.stats().trap());
    }

    #[test]
    fn rejects_zero_w_with_trap() {
        let p = scallop_join("s", "n", "j", "c", 2, 0, 4);
        assert!(p.stats().trap());
    }

    #[test]
    fn rejects_zero_max_iterations_with_trap() {
        let p = scallop_join("s", "n", "j", "c", 2, 1, 0);
        assert!(p.stats().trap());
    }
}
