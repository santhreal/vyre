//! The subset construction: caller-supplied NFA tables in, dense DFA out.

use std::collections::HashMap;

use crate::matching::dfa_compile::CompiledDfa;

use super::error::NfaToDfaError;
use super::state_set::{
    build_epsilon_closures, closure_of_set, for_each_set_bit, set_bit, test_bit, StateSet,
    EMPTY_SET, LANES,
};

/// Caller-supplied NFA bit-tables, in the exact layout
/// `vyre_libs::scan::compile_regex_set` and `nfa_scan_with_plan` emit.
///
/// `transition_table[src * 256 * LANES + byte * LANES + lane]` is the
/// u32 bitmask of NFA states (`lane * 32 + i` for `i ∈ 0..32`) reachable
/// from `src` on `byte`.
///
/// `epsilon_table[src * LANES + lane]` is the same shape minus the byte
/// dimension.
///
/// `accept_state_ids[i]` is the NFA state that fires accept index `i`;
/// `accept_pattern_ids[i]` is the consumer's pattern id for that accept.
/// `max_pattern_len` is the max accepted match length and propagates
/// straight onto [`CompiledDfa::max_pattern_len`].
#[derive(Debug, Clone)]
pub struct NfaTables<'tables> {
    /// NFA state count.
    pub num_states: u32,
    /// Lane-major `[num_states × 256 × LANES]` u32 byte-transition table.
    pub transition_table: &'tables [u32],
    /// Lane-major `[num_states × LANES]` u32 epsilon-transition table.
    pub epsilon_table: &'tables [u32],
    /// One entry per accept; the NFA state id that accepts.
    pub accept_state_ids: &'tables [u32],
    /// One entry per accept; the consumer's pattern id for that accept.
    /// Must be the same length as `accept_state_ids`.
    pub accept_pattern_ids: &'tables [u32],
    /// Max match length over the pattern set. Forwarded to
    /// `CompiledDfa::max_pattern_len`; consumers (e.g. AC scan kernels
    /// with per-position replay windows) use it to bound work.
    pub max_pattern_len: u32,
}

/// Compile an NFA into the dense [`CompiledDfa`] via subset construction.
///
/// `max_dfa_states` is a hard cap on output state count - exceeding it
/// returns [`NfaToDfaError::StateExplosion`] rather than ballooning
/// memory. Typical regex sets (literal-ish + bounded character classes
/// + bounded repetition) produce a small constant multiple of the input
/// NFA state count; pathological alternations of large classes can blow
/// up exponentially and need either a higher cap or to stay on the NFA
/// scan path.
///
/// # Errors
/// * [`NfaToDfaError::ShapeMismatch`] if input table lengths disagree
///   with `num_states`.
/// * [`NfaToDfaError::StateExplosion`] when the cap is exceeded.
///
/// # Panics
/// Panics when the output record count overflows `u32`. The GPU kernel indexes those
/// records with u32 offsets, so a wrapped length would yield corrupt pattern ids;
/// shard the pattern set instead.
pub fn nfa_to_dfa(
    tables: &NfaTables<'_>,
    max_dfa_states: usize,
) -> Result<CompiledDfa, NfaToDfaError> {
    let n = tables.num_states as usize;
    if n > LANES * 32 {
        return Err(NfaToDfaError::ShapeMismatch {
            reason: "num_states exceeds LANES * 32 bit-set capacity",
        });
    }
    // Cap max_dfa_states to u32::MAX so DFA state IDs stored as u32 never wrap.
    // State IDs are used as array indices (transitions, accept, output_offsets),
    // so wrapping would alias existing states and corrupt the automaton silently.
    if max_dfa_states > u32::MAX as usize {
        return Err(NfaToDfaError::StateExplosion {
            produced: 0,
            cap: max_dfa_states,
        });
    }
    if tables.transition_table.len() != n * 256 * LANES {
        return Err(NfaToDfaError::ShapeMismatch {
            reason: "transition_table length != num_states * 256 * LANES",
        });
    }
    if tables.epsilon_table.len() != n * LANES {
        return Err(NfaToDfaError::ShapeMismatch {
            reason: "epsilon_table length != num_states * LANES",
        });
    }
    if tables.accept_state_ids.len() != tables.accept_pattern_ids.len() {
        return Err(NfaToDfaError::ShapeMismatch {
            reason: "accept_state_ids and accept_pattern_ids length disagree",
        });
    }
    // Validate every accept_state_id is within [0, num_states). Without this,
    // a state id >= num_states reaches test_bit() with lane = state/32 that
    // indexes outside the [u32; LANES] StateSet array, causing an OOB panic
    // instead of a structured ShapeMismatch error.
    for &id in tables.accept_state_ids {
        if id >= n as u32 {
            return Err(NfaToDfaError::ShapeMismatch {
                reason: "accept_state_ids entry >= num_states",
            });
        }
    }
    // accept[state] encodes the accepting pattern id as `pid + 1` (0 = no match),
    // so a pattern id of u32::MAX would wrap to 0 and silently hide the match.
    // Reject it upfront as a structured error rather than panicking deep in subset
    // construction (mirrors the accept_state_ids bound check above).
    for &pid in tables.accept_pattern_ids {
        if pid == u32::MAX {
            return Err(NfaToDfaError::ShapeMismatch {
                reason: "accept_pattern_ids entry u32::MAX cannot be encoded as pid+1; pattern ids must be <= u32::MAX - 1",
            });
        }
    }

    // Per-NFA-state ε-closure, precomputed once. Subset construction
    // looks up `epsilon_closure[s]` for every state in every byte step,
    // so the BFS cost stays bounded by num_states rather than reused
    // per DFA-state-transition.
    let epsilon_closures = build_epsilon_closures(n, tables.epsilon_table);

    // DFA state 0 = ε-closure of NFA entry state 0. Same convention as
    // `compile_regex_set`, where state 0 is the shared entry.
    let mut entry_set = EMPTY_SET;
    set_bit(&mut entry_set, 0);
    let start_set = closure_of_set(&entry_set, &epsilon_closures);

    let mut dfa_state_index: HashMap<StateSet, u32> = HashMap::new();
    let mut dfa_state_sets: Vec<StateSet> = Vec::new();
    let mut transitions: Vec<u32> = Vec::new();

    dfa_state_index.insert(start_set, 0);
    dfa_state_sets.push(start_set);
    transitions.extend(std::iter::repeat_n(0u32, 256));

    // Worklist-driven BFS over DFA states. We push the start state, then
    // for each unprocessed DFA state expand its 256 byte transitions -
    // adding any newly-discovered DFA state to the worklist. Stops when
    // every produced state has had its transitions filled in.
    let mut next_to_process: usize = 0;
    while next_to_process < dfa_state_sets.len() {
        let dfa_state_id = next_to_process;
        let current_set = dfa_state_sets[dfa_state_id];
        next_to_process += 1;

        for byte in 0u32..256 {
            let mut target_set = EMPTY_SET;
            // Walk only live NFA states in `current_set`; for each, OR
            // in the lane-major transition row for this byte. The row
            // already encodes "which states does s reach on b" so the
            // result is the union of byte-targets across all live s.
            for_each_set_bit(&current_set, |src_state| {
                let row_start = (src_state as usize) * 256 * LANES + (byte as usize) * LANES;
                for lane in 0..LANES {
                    target_set[lane] |= tables.transition_table[row_start + lane];
                }
            });
            // ε-close the union. Most NFA frontends connect alternation
            // / repetition via ε edges, so this step is what stitches
            // the pattern's full state graph back together.
            let closed = closure_of_set(&target_set, &epsilon_closures);
            let next_dfa_state = if closed == EMPTY_SET {
                // Reject - convention: state 0 is the start state and is
                // not a sink, so we model rejection as "stay at a dead
                // state". Allocate one dead state lazily the first time
                // it's needed.
                ensure_dead_state(
                    &mut dfa_state_index,
                    &mut dfa_state_sets,
                    &mut transitions,
                    max_dfa_states,
                )?
            } else if let Some(&existing) = dfa_state_index.get(&closed) {
                existing
            } else {
                if dfa_state_sets.len() >= max_dfa_states {
                    return Err(NfaToDfaError::StateExplosion {
                        produced: dfa_state_sets.len(),
                        cap: max_dfa_states,
                    });
                }
                // Safe: max_dfa_states <= u32::MAX (guarded at function entry), and
                // the StateExplosion guard above ensures len < max_dfa_states <= u32::MAX.
                let new_id = dfa_state_sets.len() as u32;
                dfa_state_index.insert(closed, new_id);
                dfa_state_sets.push(closed);
                transitions.extend(std::iter::repeat_n(0u32, 256));
                new_id
            };
            transitions[(dfa_state_id) * 256 + byte as usize] = next_dfa_state;
        }
    }

    // Accept + output_records: for each DFA state, walk every NFA accept
    // and emit the consumer's pattern_id if that accept's NFA state is
    // in the DFA state's bitset. Stable order in `accept_state_ids` →
    // stable order in `output_records` slice per state, which matches
    // the contract `dfa_compile` exposes.
    // Safe: max_dfa_states <= u32::MAX (guarded at function entry), and
    // dfa_state_sets.len() <= max_dfa_states at this point.
    let state_count = dfa_state_sets.len() as u32;
    let mut accept: Vec<u32> = vec![0; state_count as usize];
    let mut output_offsets: Vec<u32> = Vec::with_capacity(state_count as usize + 1);
    let mut output_records: Vec<u32> = Vec::new();
    output_offsets.push(0);
    for dfa_state_id in 0..state_count {
        let set = &dfa_state_sets[dfa_state_id as usize];
        let mut first_accept_pid: Option<u32> = None;
        for (i, &nfa_state) in tables.accept_state_ids.iter().enumerate() {
            if test_bit(set, nfa_state) {
                let pid = tables.accept_pattern_ids[i];
                if first_accept_pid.is_none() {
                    first_accept_pid = Some(pid);
                }
                output_records.push(pid);
            }
        }
        // accept[state] encodes the first accepting pattern id as `pid + 1` so that 0
        // means "no match". pid == u32::MAX (which would wrap to 0 and silently hide the
        // match) was already rejected with a structured ShapeMismatch at function entry,
        // so `pid + 1` cannot overflow here.
        accept[dfa_state_id as usize] = first_accept_pid.map(|pid| pid + 1).unwrap_or(0);
        // output_records.len() as u32: safe because max_dfa_states <= u32::MAX (guarded at
        // function entry), and each DFA state contributes at most num_states accept records,
        // so output_records.len() <= dfa_state_sets.len() * num_states <= u32::MAX * 1024.
        // That exceeds u32::MAX, so use a checked conversion to catch overflow.
        let output_offset = u32::try_from(output_records.len()).unwrap_or_else(|_| {
            panic!(
                "output_records length {} overflowed u32; the GPU kernel indexes output_records \
                 via u32 offsets so this would produce corrupt pattern ids. Fix: shard the \
                 pattern set or reduce num_states before calling nfa_to_dfa.",
                output_records.len()
            )
        });
        output_offsets.push(output_offset);
    }

    Ok(CompiledDfa {
        transitions,
        accept,
        state_count,
        max_pattern_len: tables.max_pattern_len,
        output_offsets,
        output_records,
    })
}

// ── dead state ────────────────────────────────────────────────────────

/// Lazily allocate a single dead state that self-loops on every byte.
/// Avoids consuming `max_dfa_states` budget when the NFA has no
/// rejecting paths but lets us address "no transition" uniformly.
fn ensure_dead_state(
    index: &mut HashMap<StateSet, u32>,
    sets: &mut Vec<StateSet>,
    transitions: &mut Vec<u32>,
    max_dfa_states: usize,
) -> Result<u32, NfaToDfaError> {
    if let Some(&existing) = index.get(&EMPTY_SET) {
        return Ok(existing);
    }
    if sets.len() >= max_dfa_states {
        return Err(NfaToDfaError::StateExplosion {
            produced: sets.len(),
            cap: max_dfa_states,
        });
    }
    // Safe: the StateExplosion guard above ensures sets.len() < max_dfa_states,
    // and max_dfa_states <= u32::MAX is guarded at nfa_to_dfa entry.
    let dead_id = sets.len() as u32;
    index.insert(EMPTY_SET, dead_id);
    sets.push(EMPTY_SET);
    // Self-loops: every byte stays at the dead state. Use the id we
    // just assigned (not 0 - 0 is the start state).
    transitions.extend(std::iter::repeat_n(dead_id, 256));
    Ok(dead_id)
}
