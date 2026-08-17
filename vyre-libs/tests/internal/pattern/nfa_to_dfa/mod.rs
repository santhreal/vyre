//! Contracts for the subset construction and the dedup table.
//!
//! This suite is inline rather than in the crate's `tests/` directory because
//! the surface it pins is unreachable from outside the crate. `nfa_to_dfa` is
//! `pub(crate)`, and the state-set half of the contract lives in
//! [`super::state_set`], a module this one re-exports nothing from:
//! `build_epsilon_closures`, `closure_of_set`, `for_each_set_bit`, `set_bit`,
//! `test_bit`, `EMPTY_SET` and `LANES` have no path an integration test can
//! name. Moving the suite out would mean widening that visibility for the
//! test's benefit, which is a worse trade than the placement exception.

use crate::pattern::dfa_compile::CompiledDfa;

use super::dedup::{dfa_fingerprint, DfaDedupTable};
use super::error::NfaToDfaError;
use super::state_set::{
    build_epsilon_closures, closure_of_set, for_each_set_bit, set_bit, test_bit, EMPTY_SET, LANES,
};
use super::subset::{nfa_to_dfa, NfaTables};

/// Lock the layout-constant invariant: our local LANES must equal
/// `crate::nfa::subgroup_nfa::LANES_PER_SUBGROUP`. The nfa
/// module is feature-gated, so only run the check when both features
/// (matching is implicit here; nfa is opt-in) are on.
#[cfg(feature = "nfa")]
#[test]
fn layout_matches_nfa_module() {
    assert_eq!(
        LANES,
        crate::nfa::subgroup_nfa::LANES_PER_SUBGROUP,
        "nfa_to_dfa's local LANES must mirror subgroup_nfa::LANES_PER_SUBGROUP - a drift means the bit-table layout in this primitive no longer matches what `compile_regex_set` / `nfa_scan_with_plan` emit. Fix: update LANES here and re-run the matching test suite."
    );
}

/// Build NFA tables for a single literal pattern "abc" by hand.
/// Mirrors what `crate::nfa::nfa::compile` (vyre-libs side) would
/// produce: 4 states (entry + 3 byte states), no ε edges.
fn literal_abc_tables() -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
    // 4 states: 0=entry, 1=after 'a', 2=after 'ab', 3=after 'abc' (accept)
    let num_states = 4_usize;
    let mut transition = vec![0u32; num_states * 256 * LANES];
    for (src, b, dst) in [(0usize, b'a', 1u32), (1, b'b', 2), (2, b'c', 3)] {
        let dst_lane = (dst / 32) as usize;
        let dst_bit = 1u32 << (dst % 32);
        let idx = src * 256 * LANES + (b as usize) * LANES + dst_lane;
        transition[idx] |= dst_bit;
    }
    let epsilon = vec![0u32; num_states * LANES];
    let accept_state_ids = vec![3u32];
    let accept_pattern_ids = vec![0u32];
    (transition, epsilon, accept_state_ids, accept_pattern_ids)
}

struct GeneratedNfa {
    num_states: u32,
    transition: Vec<u32>,
    epsilon: Vec<u32>,
    accept_states: Vec<u32>,
    accept_pids: Vec<u32>,
    max_pattern_len: u32,
    primary_word: Vec<u8>,
    alternate_word: Vec<u8>,
}

impl GeneratedNfa {
    fn tables(&self) -> NfaTables<'_> {
        NfaTables {
            num_states: self.num_states,
            transition_table: &self.transition,
            epsilon_table: &self.epsilon,
            accept_state_ids: &self.accept_states,
            accept_pattern_ids: &self.accept_pids,
            max_pattern_len: self.max_pattern_len,
        }
    }
}

fn generated_nfa(seed: u32) -> GeneratedNfa {
    let num_states = 3 + (seed as usize % 4);
    let mut transition = vec![0u32; num_states * 256 * LANES];
    let mut epsilon = vec![0u32; num_states * LANES];
    let mut primary_word = Vec::with_capacity(num_states.saturating_sub(1));

    for src in 0..num_states.saturating_sub(1) {
        let byte = generated_byte(seed, src as u32);
        primary_word.push(byte);
        add_transition(&mut transition, src, byte, (src + 1) as u32);
        if src > 0 {
            add_transition(
                &mut transition,
                src,
                generated_byte(seed ^ 0x5a5a_1337, src as u32),
                src as u32,
            );
        }
    }

    let alternate_target = num_states.saturating_sub(2).max(1);
    let alternate_byte = generated_byte(seed ^ 0x9e37_79b9, 0);
    add_transition(&mut transition, 0, alternate_byte, alternate_target as u32);
    let alternate_word = vec![alternate_byte];

    if seed % 3 == 0 && num_states > 3 {
        add_epsilon(&mut epsilon, 1, 2);
    }
    if seed % 5 == 0 {
        add_epsilon(&mut epsilon, 0, 1);
    }
    if seed % 11 == 0 && num_states > 4 {
        add_epsilon(&mut epsilon, 2, (num_states - 1) as u32);
    }

    let mut accept_states = vec![(num_states - 1) as u32];
    let mut accept_pids = vec![seed % 31];
    if seed % 7 == 0 {
        accept_states.push(alternate_target as u32);
        accept_pids.push(100 + (seed % 97));
    }

    GeneratedNfa {
        num_states: num_states as u32,
        transition,
        epsilon,
        accept_states,
        accept_pids,
        max_pattern_len: primary_word.len() as u32,
        primary_word,
        alternate_word,
    }
}

fn generated_byte(seed: u32, lane: u32) -> u8 {
    let mixed = seed
        .wrapping_mul(1_664_525)
        .wrapping_add(lane.wrapping_mul(1_013_904_223))
        .wrapping_add(0x045d_9f3b);
    b'a' + (mixed % 23) as u8
}

fn add_transition(table: &mut [u32], src: usize, byte: u8, dst: u32) {
    let lane = (dst / 32) as usize;
    let bit = 1u32 << (dst % 32);
    let idx = src * 256 * LANES + (byte as usize) * LANES + lane;
    table[idx] |= bit;
}

fn add_epsilon(table: &mut [u32], src: usize, dst: u32) {
    let lane = (dst / 32) as usize;
    let bit = 1u32 << (dst % 32);
    table[src * LANES + lane] |= bit;
}

fn nfa_outputs_for(nfa: &GeneratedNfa, input: &[u8]) -> Vec<u32> {
    let closures = build_epsilon_closures(nfa.num_states as usize, &nfa.epsilon);
    let mut current = EMPTY_SET;
    set_bit(&mut current, 0);
    current = closure_of_set(&current, &closures);

    for &byte in input {
        let mut target = EMPTY_SET;
        for_each_set_bit(&current, |src_state| {
            let row_start = (src_state as usize) * 256 * LANES + (byte as usize) * LANES;
            for lane in 0..LANES {
                target[lane] |= nfa.transition[row_start + lane];
            }
        });
        current = closure_of_set(&target, &closures);
    }

    let mut out = Vec::new();
    for (idx, &state) in nfa.accept_states.iter().enumerate() {
        if test_bit(&current, state) {
            out.push(nfa.accept_pids[idx]);
        }
    }
    out
}

fn dfa_outputs_for(dfa: &CompiledDfa, input: &[u8]) -> Vec<u32> {
    let mut state = 0usize;
    for &byte in input {
        state = dfa.transitions[state * 256 + byte as usize] as usize;
    }
    let start = dfa.output_offsets[state] as usize;
    let end = dfa.output_offsets[state + 1] as usize;
    dfa.output_records[start..end].to_vec()
}

#[test]
fn generated_nfa_to_dfa_matches_reference_nfa_for_thousands_of_inputs() {
    let mut checked = 0usize;
    for seed in 0..1024u32 {
        let nfa = generated_nfa(seed);
        let dfa = nfa_to_dfa(&nfa.tables(), 4096)
            .expect("Fix: generated sparse NFA must stay inside the DFA cap");
        let mut mutated_primary = nfa.primary_word.clone();
        if let Some(last) = mutated_primary.last_mut() {
            *last = last.wrapping_add(1);
        }
        let primary_prefix = nfa.primary_word[..nfa.primary_word.len().saturating_sub(1)].to_vec();
        let mut reversed_primary = nfa.primary_word.clone();
        reversed_primary.reverse();
        let generated_noise = vec![
            generated_byte(seed, 9),
            generated_byte(seed, 10),
            generated_byte(seed, 11),
            generated_byte(seed, 12),
        ];
        let alternate_twice =
            [nfa.alternate_word.as_slice(), nfa.alternate_word.as_slice()].concat();
        let corpus = [
            Vec::new(),
            nfa.primary_word.clone(),
            primary_prefix,
            reversed_primary,
            mutated_primary,
            nfa.alternate_word.clone(),
            alternate_twice,
            generated_noise.clone(),
            vec![generated_byte(seed ^ 0xa5a5_a5a5, 1)],
            [generated_byte(seed, 13)].into(),
            [generated_byte(seed, 14), generated_byte(seed, 15)].into(),
            [nfa.alternate_word.as_slice(), nfa.primary_word.as_slice()].concat(),
            [nfa.primary_word.as_slice(), nfa.alternate_word.as_slice()].concat(),
            [generated_byte(seed, 16)]
                .into_iter()
                .chain(nfa.primary_word.iter().copied())
                .collect(),
            nfa.primary_word
                .iter()
                .copied()
                .chain([generated_byte(seed, 17)])
                .collect(),
            [
                nfa.alternate_word.as_slice(),
                &generated_noise,
                nfa.primary_word.as_slice(),
            ]
            .concat(),
        ];
        for input in corpus {
            assert_eq!(
                dfa_outputs_for(&dfa, &input),
                nfa_outputs_for(&nfa, &input),
                "seed {seed} input {input:?} must produce identical accept records"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 16_384);
}

#[test]
fn generated_malformed_nfa_shapes_report_structured_errors() {
    let mut checked = 0usize;
    for seed in 0..1024u32 {
        let nfa = generated_nfa(seed);

        let mut short_transition = nfa.transition.clone();
        short_transition.pop();
        assert!(matches!(
            nfa_to_dfa(
                &NfaTables {
                    transition_table: &short_transition,
                    ..nfa.tables()
                },
                4096,
            ),
            Err(NfaToDfaError::ShapeMismatch { .. })
        ));
        checked += 1;

        let mut short_epsilon = nfa.epsilon.clone();
        short_epsilon.pop();
        assert!(matches!(
            nfa_to_dfa(
                &NfaTables {
                    epsilon_table: &short_epsilon,
                    ..nfa.tables()
                },
                4096,
            ),
            Err(NfaToDfaError::ShapeMismatch { .. })
        ));
        checked += 1;

        let mut extra_pid = nfa.accept_pids.clone();
        extra_pid.push(seed);
        assert!(matches!(
            nfa_to_dfa(
                &NfaTables {
                    accept_pattern_ids: &extra_pid,
                    ..nfa.tables()
                },
                4096,
            ),
            Err(NfaToDfaError::ShapeMismatch { .. })
        ));
        checked += 1;

        assert!(matches!(
            nfa_to_dfa(
                &NfaTables {
                    num_states: 1025,
                    ..nfa.tables()
                },
                4096,
            ),
            Err(NfaToDfaError::ShapeMismatch { .. })
        ));
        checked += 1;
    }
    assert_eq!(checked, 4096);
}

#[test]
fn generated_dfa_fingerprints_are_stable_and_content_addressed() {
    let mut checked = 0usize;
    for seed in 0..1024u32 {
        let nfa = generated_nfa(seed);
        let first = match nfa_to_dfa(&nfa.tables(), 4096) {
            Ok(dfa) => dfa,
            Err(err) => panic!("Fix: generated NFA must lower for seed {seed}: {err}"),
        };
        let second = match nfa_to_dfa(&nfa.tables(), 4096) {
            Ok(dfa) => dfa,
            Err(err) => {
                panic!("Fix: generated NFA must lower on replay for seed {seed}: {err}")
            }
        };
        assert_eq!(
            dfa_fingerprint(&first),
            dfa_fingerprint(&second),
            "seed {seed} must produce a stable content fingerprint"
        );

        let mut changed = first.clone();
        changed.max_pattern_len = changed.max_pattern_len.wrapping_add(1);
        assert_ne!(
            dfa_fingerprint(&first),
            dfa_fingerprint(&changed),
            "seed {seed} max-pattern metadata must perturb the fingerprint"
        );

        if let Some(first_transition) = changed.transitions.first_mut() {
            *first_transition = first_transition.wrapping_add(1);
            assert_ne!(
                dfa_fingerprint(&first),
                dfa_fingerprint(&changed),
                "seed {seed} transition metadata must perturb the fingerprint"
            );
        }
        checked += 3;
    }
    assert_eq!(checked, 3072);
}

#[test]
fn generated_dfa_dedup_table_canonicalizes_repeated_automata() {
    let mut table = DfaDedupTable::default();
    let mut checked = 0usize;
    for seed in 0..1024u32 {
        let nfa = generated_nfa(seed);
        let first = nfa_to_dfa(&nfa.tables(), 4096)
            .unwrap_or_else(|err| panic!("Fix: generated NFA must lower for seed {seed}: {err}"));
        let replay = nfa_to_dfa(&nfa.tables(), 4096).unwrap_or_else(|err| {
            panic!("Fix: generated NFA must lower on replay for seed {seed}: {err}")
        });

        let first_result = table.insert(first.clone());
        let replay_result = table.insert(replay);
        assert!(
            first_result.inserted,
            "seed {seed} first insert must create a canonical DFA"
        );
        assert!(
            !replay_result.inserted,
            "seed {seed} replay insert must deduplicate"
        );
        assert_eq!(
            first_result.canonical_index, replay_result.canonical_index,
            "seed {seed} replay must resolve to the first canonical slot"
        );
        assert_eq!(
            first_result.fingerprint, replay_result.fingerprint,
            "seed {seed} replay must keep the same content fingerprint"
        );

        let mut changed = first;
        changed.max_pattern_len = changed.max_pattern_len.wrapping_add(1);
        let changed_result = table.insert(changed);
        assert!(
            changed_result.inserted,
            "seed {seed} changed DFA metadata must not deduplicate"
        );
        assert_ne!(
            changed_result.canonical_index, first_result.canonical_index,
            "seed {seed} changed DFA must get a distinct canonical slot"
        );
        checked += 3;
    }
    assert_eq!(checked, 3072);
    assert_eq!(table.len(), 2048);
}

#[test]
fn generated_dfa_batch_dedup_preserves_input_order_and_stats() {
    let mut table = DfaDedupTable::default();
    let mut input = Vec::new();
    for seed in 0..512u32 {
        let nfa = generated_nfa(seed);
        let dfa = nfa_to_dfa(&nfa.tables(), 4096)
            .unwrap_or_else(|err| panic!("Fix: generated NFA must lower for seed {seed}: {err}"));
        input.push(dfa.clone());
        input.push(dfa.clone());
        let mut changed = dfa;
        changed.max_pattern_len = changed.max_pattern_len.wrapping_add(1);
        input.push(changed);
    }

    let batch = table.insert_many(input);
    assert_eq!(batch.stats.input_count, 1536);
    assert_eq!(batch.stats.inserted_count, 1024);
    assert_eq!(batch.stats.duplicate_count, 512);
    assert_eq!(batch.stats.table_len_after, 1024);
    assert_eq!(
        batch.stats.input_wire_bytes,
        batch.stats.inserted_wire_bytes + batch.stats.saved_wire_bytes
    );
    assert_eq!(
        batch.stats.inserted_wire_bytes,
        table.canonical_wire_bytes()
    );
    assert!(
        batch.stats.saved_wire_bytes > 0,
        "batch dedup must report saved wire bytes for replayed automata"
    );
    assert!(
        batch.saved_wire_ppm() > 0,
        "batch dedup must report a nonzero deterministic saved-byte ratio"
    );
    assert_eq!(batch.results.len(), 1536);

    for chunk in batch.results.chunks_exact(3) {
        assert!(chunk[0].inserted);
        assert!(!chunk[1].inserted);
        assert!(chunk[2].inserted);
        assert_eq!(chunk[0].canonical_index, chunk[1].canonical_index);
        assert_ne!(chunk[0].canonical_index, chunk[2].canonical_index);
        assert_eq!(chunk[0].fingerprint, chunk[1].fingerprint);
        assert_ne!(chunk[0].fingerprint, chunk[2].fingerprint);
    }
}

#[test]
fn generated_dfa_table_merge_deduplicates_cross_shard_plans() {
    let mut left = DfaDedupTable::default();
    let mut right = DfaDedupTable::default();
    for seed in 0..256u32 {
        let nfa = generated_nfa(seed);
        let dfa = nfa_to_dfa(&nfa.tables(), 4096)
            .unwrap_or_else(|err| panic!("Fix: generated NFA must lower for seed {seed}: {err}"));
        left.insert(dfa.clone());
        right.insert(dfa);
    }
    for seed in 256..512u32 {
        let nfa = generated_nfa(seed);
        let dfa = nfa_to_dfa(&nfa.tables(), 4096)
            .unwrap_or_else(|err| panic!("Fix: generated NFA must lower for seed {seed}: {err}"));
        right.insert(dfa);
    }

    let before_len = left.len();
    let before_bytes = left.canonical_wire_bytes();
    let batch = left.merge_from(&right);

    assert_eq!(before_len, 256);
    assert_eq!(batch.stats.input_count, 512);
    assert_eq!(batch.stats.inserted_count, 256);
    assert_eq!(batch.stats.duplicate_count, 256);
    assert_eq!(batch.stats.table_len_after, 512);
    assert!(batch.stats.saved_wire_bytes > 0);
    assert!(batch.saved_wire_ppm() > 0);
    assert!(left.canonical_wire_bytes() > before_bytes);
    assert_eq!(left.len(), 512);

    for result in batch.results.iter().take(256) {
        assert!(!result.inserted);
        assert!(result.canonical_index < before_len);
    }
    for result in batch.results.iter().skip(256) {
        assert!(result.inserted);
        assert!(result.canonical_index >= before_len);
    }
}

#[test]
fn literal_pattern_lowers_to_acceptor_dfa() {
    let (transition, epsilon, accepts, pids) = literal_abc_tables();
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &accepts,
        accept_pattern_ids: &pids,
        max_pattern_len: 3,
    };
    let dfa = nfa_to_dfa(&tables, 1024).expect("Fix: literal NFA must lower cleanly");
    assert!(
        dfa.state_count >= 4,
        "literal 'abc' needs at least entry + 3 progress states; got {}",
        dfa.state_count
    );
    // Trace 'a' 'b' 'c' from state 0 and assert the final state accepts.
    let s_a = dfa.transitions[0 * 256 + b'a' as usize];
    let s_ab = dfa.transitions[(s_a as usize) * 256 + b'b' as usize];
    let s_abc = dfa.transitions[(s_ab as usize) * 256 + b'c' as usize];
    // The accept field encodes pid+1, so pattern 0 → value 1. Asserting
    // != 0 only proves some pattern accepted; asserting == 1 proves the
    // correct pattern id is stored. A pid transposition or the pid+1 wrap
    // bug (finding #3: pid=u32::MAX→0) would pass != 0 but fail == 1.
    assert_eq!(
        dfa.accept[s_abc as usize], 1,
        "DFA state after 'abc' must accept pattern 0 encoded as pid+1=1; \
         got {} (0=no match, n=pattern n-1 accepted)",
        dfa.accept[s_abc as usize]
    );
    // Verify output_records carries the correct pid for the full-match path.
    let rec_start = dfa.output_offsets[s_abc as usize] as usize;
    let rec_end = dfa.output_offsets[s_abc as usize + 1] as usize;
    assert_eq!(
        &dfa.output_records[rec_start..rec_end],
        &[0u32],
        "output_records for the abc-accept state must contain exactly [0] (pid=0)"
    );
    // Negative twin: 'a' 'b' 'x' must not accept.
    let s_x = dfa.transitions[(s_ab as usize) * 256 + b'x' as usize];
    assert_eq!(
        dfa.accept[s_x as usize], 0,
        "'abx' is not 'abc' - must not accept"
    );
}

#[test]
fn empty_input_returns_one_state_dfa_with_dead_self_loop() {
    // 1 NFA state, no transitions, no accepts. The DFA we get back
    // must still have at least the start state and a dead state
    // such that every byte from start lands on the dead state.
    let transition = vec![0u32; 1 * 256 * LANES];
    let epsilon = vec![0u32; 1 * LANES];
    let tables = NfaTables {
        num_states: 1,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &[],
        accept_pattern_ids: &[],
        max_pattern_len: 0,
    };
    let dfa = nfa_to_dfa(&tables, 16).expect("Fix: trivial NFA must lower");
    let dead = dfa.transitions[0 * 256 + b'a' as usize];
    assert_eq!(
        dfa.transitions[dead as usize * 256 + b'a' as usize],
        dead,
        "dead state must self-loop on every byte"
    );
    assert_eq!(dfa.accept[dead as usize], 0, "dead state must not accept");
}

#[test]
fn state_explosion_reports_structured_error() {
    // Force the cap to 1 → any reachable byte produces state 2 and
    // hits the cap. We just need the error variant, not a specific
    // exploding pattern.
    let (transition, epsilon, accepts, pids) = literal_abc_tables();
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &accepts,
        accept_pattern_ids: &pids,
        max_pattern_len: 3,
    };
    let err = nfa_to_dfa(&tables, 1).expect_err("cap=1 must trip state explosion");
    match err {
        NfaToDfaError::StateExplosion { cap, produced } => {
            assert_eq!(cap, 1);
            assert!(produced >= 1);
        }
        other => panic!("expected StateExplosion, got {other:?}"),
    }
}

#[test]
fn shape_mismatch_caught_before_construction() {
    // num_states=4 declared but transition table sized for 1 - the
    // shape guard must catch this without panicking inside the loop.
    let transition = vec![0u32; 1 * 256 * LANES];
    let epsilon = vec![0u32; 1 * LANES];
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &[],
        accept_pattern_ids: &[],
        max_pattern_len: 0,
    };
    let err = nfa_to_dfa(&tables, 16)
        .expect_err("declared num_states != table length must error, not panic");
    match err {
        NfaToDfaError::ShapeMismatch { reason } => {
            assert!(reason.contains("transition_table"));
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

/// Finding #4 (P0): accept_state_ids entries are not validated to be < num_states.
/// Before the fix, nfa_state=1024 produced lane=32 which is OOB on StateSet=[u32;32],
/// causing a panic instead of a structured ShapeMismatch error.
#[test]
fn accept_state_id_ge_num_states_is_shape_mismatch() {
    let (transition, epsilon, _, _) = literal_abc_tables();
    // Declare 4 states but put accept_state_id=1024 (way out of range).
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &[1024],
        accept_pattern_ids: &[0],
        max_pattern_len: 3,
    };
    let result = nfa_to_dfa(&tables, 64);
    assert!(
        matches!(result, Err(NfaToDfaError::ShapeMismatch { .. })),
        "accept_state_id >= num_states must return ShapeMismatch, not panic; got {result:?}"
    );
}

/// Finding #4: also verify accept_state_id exactly equal to num_states is rejected.
#[test]
fn accept_state_id_equal_to_num_states_is_shape_mismatch() {
    let (transition, epsilon, _, _) = literal_abc_tables();
    // num_states=4, so valid ids are 0..3; id=4 is OOB.
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &[4],
        accept_pattern_ids: &[0],
        max_pattern_len: 3,
    };
    let result = nfa_to_dfa(&tables, 64);
    assert!(
        matches!(result, Err(NfaToDfaError::ShapeMismatch { .. })),
        "accept_state_id == num_states must be rejected; got {result:?}"
    );
}

/// Finding #2+8 (P0/P1): max_dfa_states > u32::MAX must return StateExplosion
/// rather than allowing dfa_state_sets.len() to grow past u32::MAX and wrap
/// the as u32 cast to 0, aliasing existing DFA states and corrupting the automaton.
#[test]
fn max_dfa_states_above_u32_max_returns_state_explosion() {
    let (transition, epsilon, accepts, pids) = literal_abc_tables();
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &accepts,
        accept_pattern_ids: &pids,
        max_pattern_len: 3,
    };
    // (u32::MAX as usize) + 1 is the first value that exceeds the u32 domain.
    let result = nfa_to_dfa(&tables, (u32::MAX as usize) + 1);
    assert!(
        matches!(result, Err(NfaToDfaError::StateExplosion { .. })),
        "max_dfa_states > u32::MAX must return StateExplosion before any state is allocated; \
         got {result:?}"
    );
}

/// Finding #3 (P0): the accept field encodes pid+1 so that 0 means no match.
/// For pattern 0, the encoded value must be exactly 1, not some other nonzero value.
/// This test replaces the existing decoration test that only checked != 0.
#[test]
fn literal_abc_accept_field_is_exactly_one() {
    let (transition, epsilon, accepts, pids) = literal_abc_tables();
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &accepts,
        accept_pattern_ids: &pids,
        max_pattern_len: 3,
    };
    let dfa = nfa_to_dfa(&tables, 1024).expect("Fix: literal NFA must lower cleanly");
    let s_a = dfa.transitions[b'a' as usize];
    let s_ab = dfa.transitions[s_a as usize * 256 + b'b' as usize];
    let s_abc = dfa.transitions[s_ab as usize * 256 + b'c' as usize];
    // Pattern 0 is encoded as pid+1 = 0+1 = 1. Any other nonzero value means
    // the wrong pattern id is reported on the fast-path field.
    assert_eq!(
        dfa.accept[s_abc as usize], 1,
        "pattern 0 must be encoded as pid+1=1 in the accept fast-path field; \
         got {} (accept field encodes pid+1, 0=no match)",
        dfa.accept[s_abc as usize]
    );
    // Also verify output_records carries the correct pid for the full-match path.
    let start = dfa.output_offsets[s_abc as usize] as usize;
    let end = dfa.output_offsets[s_abc as usize + 1] as usize;
    assert_eq!(
        &dfa.output_records[start..end],
        &[0u32],
        "output_records for the abc-accept state must contain exactly [0] (pid=0)"
    );
}

/// VL-002 (P0): a caller-supplied accept_pattern_id of u32::MAX cannot be encoded
/// as pid+1 (it would wrap to 0, silently hiding the match). nfa_to_dfa must reject
/// it with a structured ShapeMismatch at function entry, NOT panic deep in subset
/// construction (the prior behaviour) and NOT silently encode 0.
#[test]
fn accept_pattern_id_u32_max_returns_shape_mismatch_not_panic() {
    let (transition, epsilon, accepts, _pids) = literal_abc_tables();
    let pids = vec![u32::MAX; accepts.len()];
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &accepts,
        accept_pattern_ids: &pids,
        max_pattern_len: 3,
    };
    let result = nfa_to_dfa(&tables, 1024);
    assert!(
        matches!(&result, Err(NfaToDfaError::ShapeMismatch { reason }) if reason.contains("u32::MAX")),
        "accept_pattern_ids=[u32::MAX] must return ShapeMismatch naming u32::MAX, got {result:?}"
    );
}

/// VL-002 negative twin: pid = u32::MAX - 1 is the largest ENCODABLE pattern id;
/// it must lower cleanly and encode as accept = (u32::MAX - 1) + 1 = u32::MAX, with
/// the raw pid preserved in output_records.
#[test]
fn accept_pattern_id_u32_max_minus_one_is_valid() {
    let (transition, epsilon, accepts, _pids) = literal_abc_tables();
    let pids = vec![u32::MAX - 1; accepts.len()];
    let tables = NfaTables {
        num_states: 4,
        transition_table: &transition,
        epsilon_table: &epsilon,
        accept_state_ids: &accepts,
        accept_pattern_ids: &pids,
        max_pattern_len: 3,
    };
    let dfa =
        nfa_to_dfa(&tables, 1024).expect("Fix: pid=u32::MAX-1 is encodable and must lower cleanly");
    let s_a = dfa.transitions[b'a' as usize];
    let s_ab = dfa.transitions[s_a as usize * 256 + b'b' as usize];
    let s_abc = dfa.transitions[s_ab as usize * 256 + b'c' as usize];
    assert_eq!(
        dfa.accept[s_abc as usize],
        u32::MAX,
        "pid=u32::MAX-1 must encode as pid+1=u32::MAX in the accept fast-path field, got {}",
        dfa.accept[s_abc as usize]
    );
    let start = dfa.output_offsets[s_abc as usize] as usize;
    let end = dfa.output_offsets[s_abc as usize + 1] as usize;
    assert_eq!(
        &dfa.output_records[start..end],
        &[u32::MAX - 1],
        "output_records must carry the raw pid u32::MAX-1, not the encoded value"
    );
}
