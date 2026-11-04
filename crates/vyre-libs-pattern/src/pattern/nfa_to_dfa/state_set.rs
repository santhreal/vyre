//! NFA state-set bitsets and the epsilon closures over them.
//!
//! A DFA state is a set of NFA states, so the construction is bitset
//! arithmetic over a fixed-width word array. The layout constants describe
//! that array, so they live beside the operations on it.

/// Lanes-per-subgroup the state-major NFA tables are laid out for.
///
/// Contractually equal to `crate::nfa::subgroup_nfa::LANES_PER_SUBGROUP`
/// (= 32). Hard-coded here so `matching::nfa_to_dfa` can compile without
/// the `feature = "nfa"` gate - this primitive only consumes the
/// state-major bit-table layout, it doesn't invoke the NFA scan kernel.
/// The `layout_matches_nfa_module` test asserts the equality so a future
/// change in `subgroup_nfa::LANES_PER_SUBGROUP` produces a CI failure
/// here, not a silent layout mismatch at runtime.
pub(super) const LANES: usize = 32;

/// Width of one NFA state-set bitset, in u32 words. `LANES × 32` bit
/// positions per word × bits per state covers the
/// `MAX_STATES_PER_SUBGROUP = 1024` cap.
pub(super) const STATE_BITSET_WORDS: usize = LANES;

/// Per-NFA-state-set bitset. Bit `(lane * 32 + i)` set ⇔ NFA state
/// `lane * 32 + i` is live in this set.
pub(super) type StateSet = [u32; STATE_BITSET_WORDS];

pub(super) const EMPTY_SET: StateSet = [0u32; STATE_BITSET_WORDS];

// ── bitset helpers ────────────────────────────────────────────────────

#[inline]
pub(super) fn set_bit(set: &mut StateSet, state: u32) {
    let lane = (state / 32) as usize;
    let bit = state % 32;
    set[lane] |= 1u32 << bit;
}

#[inline]
pub(super) fn test_bit(set: &StateSet, state: u32) -> bool {
    let lane = (state / 32) as usize;
    let bit = state % 32;
    (set[lane] & (1u32 << bit)) != 0
}

pub(super) fn for_each_set_bit(set: &StateSet, mut f: impl FnMut(u32)) {
    for (lane, &word) in set.iter().enumerate() {
        let mut w = word;
        while w != 0 {
            let bit = w.trailing_zeros();
            f((lane as u32) * 32 + bit);
            w &= w - 1;
        }
    }
}

// ── ε-closure ─────────────────────────────────────────────────────────

/// Per-state ε-closure: for each NFA state s, the set of NFA states
/// reachable from s via zero or more ε edges (including s itself).
pub(super) fn build_epsilon_closures(num_states: usize, epsilon_table: &[u32]) -> Vec<StateSet> {
    let mut closures = vec![EMPTY_SET; num_states];
    // Seed each closure with the state itself + its direct ε successors,
    // then run BFS until no new states are added. Standard fixpoint.
    for state in 0..num_states {
        let mut closure = EMPTY_SET;
        set_bit(&mut closure, state as u32);
        let mut frontier_word = EMPTY_SET;
        for lane in 0..LANES {
            frontier_word[lane] = epsilon_table[state * LANES + lane];
        }
        // Union direct ε successors into the closure to seed BFS.
        for lane in 0..LANES {
            closure[lane] |= frontier_word[lane];
        }
        // BFS: walk every newly-added state, OR in its ε successors,
        // continue until the closure stops growing.
        loop {
            let mut next_frontier = EMPTY_SET;
            for_each_set_bit(&frontier_word, |s| {
                for lane in 0..LANES {
                    let bits = epsilon_table[(s as usize) * LANES + lane];
                    let new_bits = bits & !closure[lane];
                    next_frontier[lane] |= new_bits;
                }
            });
            if next_frontier == EMPTY_SET {
                break;
            }
            for lane in 0..LANES {
                closure[lane] |= next_frontier[lane];
            }
            frontier_word = next_frontier;
        }
        closures[state] = closure;
    }
    closures
}

/// ε-close a state set: union the precomputed per-state closures of
/// every live state in `set`.
pub(super) fn closure_of_set(set: &StateSet, per_state_closures: &[StateSet]) -> StateSet {
    let mut out = EMPTY_SET;
    for_each_set_bit(set, |state| {
        let closure = &per_state_closures[state as usize];
        for lane in 0..LANES {
            out[lane] |= closure[lane];
        }
    });
    out
}
