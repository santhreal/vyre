//! Internal tests and contracts for anchored-window regex extraction.

use super::*;
use crate::pattern::classic_ac::test_dispatch_and_decode::with_reference_dispatch_lanes;
use crate::pattern::regex_dfa::build_regex_dfa_pipeline;

const MAX_MATCHES: u32 = 4096;
const MAX_DFA_STATES: usize = 16_384;

fn validator_for(patterns: &[&str]) -> CompiledDfa {
    build_regex_dfa_pipeline(patterns, MAX_MATCHES, MAX_DFA_STATES)
        .expect("Fix: test patterns must compile to an anchored regex DFA")
        .dfa
}

/// THE anchoring contract: a match is emitted ONLY when the candidate origin
/// is exactly where the pattern starts. A candidate one byte early or one
/// byte late, even though the pattern is present in the window, yields
/// nothing. This is precisely what distinguishes anchored-window extraction
/// from an unanchored "match somewhere in the region" scan.
#[test]
fn matches_only_at_exact_candidate_origin() {
    let dfa = validator_for(&["abc"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"..abc..";

    assert_eq!(
        validator.validate_candidates(haystack, &[2]),
        vec![ByteRange::new(0, 2, 5)],
        "candidate at the true start must extract the match with start==origin"
    );
    assert!(
        validator.validate_candidates(haystack, &[1]).is_empty(),
        "candidate one byte before the match start must NOT match (anchored, not unanchored)"
    );
    assert!(
        validator.validate_candidates(haystack, &[3]).is_empty(),
        "candidate one byte after the match start must NOT match"
    );
}

/// A short match at the origin must not let the DFA re-accept later in the
/// window: after "abc" accepts at end 3, the trailing bytes drive the walk
/// into the dead sink, which never accepts. Proves the dead-state stop is
/// what keeps a long window anchored.
#[test]
fn short_match_does_not_re_accept_deeper_in_window() {
    let dfa = validator_for(&["abc"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    // Long tail after the match, well past max_pattern_len were it unbounded.
    let haystack = b"abcabcabc";
    assert_eq!(
        validator.validate_candidates(haystack, &[0]),
        vec![ByteRange::new(0, 0, 3)],
        "only the anchored match at the origin may surface; later 'abc's start at other origins"
    );
}

/// One origin, multiple accept lengths: two patterns sharing a prefix both
/// accept at the same origin at their own end offsets, and both surface via
/// the accepting states' output records.
#[test]
fn shared_prefix_patterns_emit_every_accept_length_at_one_origin() {
    let dfa = validator_for(&["abc", "abcde"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"abcde";
    let got = validator.validate_candidates(haystack, &[0]);
    assert_eq!(
        got,
        vec![ByteRange::new(0, 0, 3), ByteRange::new(1, 0, 5)],
        "both the length-3 and length-5 pattern must extract at the shared origin"
    );
}

/// A variable-length pattern (bounded repetition) extracts a faithful,
/// anchored, non-vacuous match: whatever accept ends the compiled DFA
/// carries, the validator surfaces them all, each starting exactly at the
/// origin. We compute the expectation by walking the *same* DFA directly
/// (faithfulness), never by assuming a particular multi-length semantics
/// vyre's AC-at-end DFA reports a bounded repetition at a single canonical
/// length, not one match per length (recorded in BACKLOG for the regex-DFA
/// owner). Asserting the direct-walk truth keeps this test correct
/// regardless of that choice.
#[test]
fn bounded_repetition_pattern_extracts_faithful_anchored_matches() {
    let dfa = validator_for(&["a{2,4}"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"aaaaa";
    // Compare over the SAME candidate set the oracle walks (every origin),
    // else the sets legitimately differ (the oracle finds "aa" at each
    // origin, not just origin 0).
    let origins: Vec<u32> = (0..haystack.len() as u32).collect();
    let got = validator.validate_candidates(haystack, &origins);

    let expected = direct_walk_all_origins(&dfa, haystack);
    assert_eq!(
        got, expected,
        "validator must extract exactly what a direct walk of the same DFA accepts"
    );
    assert!(
        !got.is_empty(),
        "a bounded repetition anchored at a matching origin must extract at least one match"
    );
    // Every extracted match starts at one of the supplied origins and is a
    // genuine run of 'a's of the accepted length.
    assert!(
        got.iter()
            .all(|m| haystack[m.start as usize..m.end as usize]
                .iter()
                .all(|&b| b == b'a')),
        "every anchored-window match must be a real run of the repeated byte"
    );
    // The bounded-repetition lowering fix (BACKLOG items 18/27) records the
    // MAXIMUM match length, so the replay window now covers the full range:
    // `a{2,4}` has max_pattern_len == 4 and the raw fan-out surfaces every
    // admissible length 2..=4 (the ε skip edges make the fragment end
    // reachable after 2, 3, or 4 copies). Before the fix the window was
    // capped at the MINIMUM (2), so the longer accepts were never visited.
    assert_eq!(
        dfa.max_pattern_len, 4,
        "the {{n,m}} lowering must size the window to the MAX repetition (4), \
         not the min (2), so the windowed walk can reach the longer accepts"
    );
    // At origin 0 over "aaaa" the raw fan-out accepts at lengths 2, 3, AND 4.
    let origin0_ends: Vec<u32> = got
        .iter()
        .filter(|m| m.start == 0)
        .map(|m| m.end - m.start)
        .collect();
    assert_eq!(
        origin0_ends,
        vec![2, 3, 4],
        "raw fan-out must now surface every admissible {{2,4}} length at origin 0"
    );
    // Leftmost-longest extraction collapses those to the single maximal
    // match (the whole 4-'a' run, not three overlapping partial hits).
    assert_eq!(
        validator.validate_candidates_leftmost_longest(haystack, &[0]),
        vec![ByteRange::new(0, 0, 4)],
        "leftmost-longest must emit exactly the longest {{2,4}} match at origin 0"
    );
}

/// Direct, un-optimized reference walk of `dfa` over every origin of
/// `haystack` (no dead-state early-out), returning the canonical extracted
/// set. This is the faithfulness oracle: the validator must equal it.
fn direct_walk_all_origins(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<ByteRange> {
    let mut out = Vec::new();
    for origin in 0..haystack.len() {
        let window = (dfa.max_pattern_len as usize).min(haystack.len() - origin);
        let mut state = 0u32;
        for step in 0..window {
            let trans_idx = crate::builder::TableStateMachineComposer::flat_byte_index(
                state,
                haystack[origin + step],
            );
            state = dfa.transitions[trans_idx];
            let lo = dfa.output_offsets[state as usize] as usize;
            let hi = dfa.output_offsets[state as usize + 1] as usize;
            for &pid in &dfa.output_records[lo..hi] {
                out.push(ByteRange::new(
                    pid,
                    origin as u32,
                    (origin + step + 1) as u32,
                ));
            }
        }
    }
    out.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
    out.dedup();
    out
}

/// Distinct patterns anchored at their own origins each extract exactly once;
/// candidate origins are validated independently.
#[test]
fn distinct_patterns_extract_at_their_own_origins() {
    let dfa = validator_for(&["abc", "bcd"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"abcd";
    assert_eq!(
        validator.validate_candidates(haystack, &[0, 1]),
        vec![ByteRange::new(0, 0, 3), ByteRange::new(1, 1, 4)],
        "each pattern extracts at the origin where it starts"
    );
}

/// Boundary safety: an origin at or past EOF is ignored, and an origin whose
/// window is truncated by EOF only extracts matches that fit, no panic, no
/// out-of-bounds read.
#[test]
fn origins_at_and_past_eof_are_safe_and_windows_truncate() {
    let dfa = validator_for(&["abcd"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"xxabc"; // "abcd" does not fit starting at index 2 (only "abc" remains)
    assert!(
        validator.validate_candidates(haystack, &[2]).is_empty(),
        "a pattern that runs off the end of the haystack must not match"
    );
    assert!(
        validator
            .validate_candidates(haystack, &[haystack.len() as u32])
            .is_empty(),
        "origin == haystack.len() must be ignored, not indexed"
    );
    assert!(
        validator
            .validate_candidates(haystack, &[haystack.len() as u32 + 9])
            .is_empty(),
        "origin past EOF must be ignored"
    );
}

/// Duplicate origins collapse: validating the same origin twice yields the
/// same match once, so the batch result is a clean set.
#[test]
fn duplicate_origins_collapse_to_a_set() {
    let dfa = validator_for(&["abc"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"abc";
    assert_eq!(
        validator.validate_candidates(haystack, &[0, 0, 0]),
        vec![ByteRange::new(0, 0, 3)],
        "repeated origins must not duplicate the extracted match"
    );
}

#[test]
fn empty_candidate_batch_is_empty() {
    let dfa = validator_for(&["abc"]);
    let validator = AnchoredWindowValidator::new(&dfa);
    assert!(validator.validate_candidates(b"abcabc", &[]).is_empty());
}

/// Decode `(pattern_id, start, end)` triples from a `[match_count, matches]`
/// reference-output pair (little-endian u32 words).
fn decode_match_triples(outputs: &[vyre_reference::value::Value]) -> Vec<(u32, u32, u32)> {
    let words = |value: &vyre_reference::value::Value| -> Vec<u32> {
        value
            .to_bytes()
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let count = words(&outputs[0])[0] as usize;
    let matches = words(&outputs[1]);
    matches[..count.saturating_mul(3)]
        .chunks_exact(3)
        .map(|chunk| (chunk[0], chunk[1], chunk[2]))
        .collect()
}

/// Proves the emitted kernel and reference oracle implement identical anchored-window semantics.
#[test]
fn extract_program_reference_eval_matches_reference_oracle() {
    use crate::pattern::haystack::pack_haystack_u32;
    use vyre_primitives::wire::pack_u32_slice;

    let patterns = ["abc", "abcde", "bcd", "x"];
    let dfa = validator_for(&patterns);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"zabcdex bcd abc x abcde";
    // Candidate origins: a mix of real match starts, near-misses, and EOF-
    // adjacent positions (the program must reject the non-starts).
    let candidates: Vec<u32> = vec![0, 1, 2, 8, 12, 16, 18, haystack.len() as u32 - 1];

    // Oracle.
    let mut expected = validator.validate_candidates(haystack, &candidates);
    expected.sort_unstable_by_key(|m| (m.start, m.end, m.tag));

    // Reference dispatch of the emitted program (one lane per candidate).
    let num_candidates = candidates.len() as u32;
    let max_matches = 4096u32;
    let program = with_reference_dispatch_lanes(
        anchored_window_extract_program(
            "haystack",
            "transitions",
            "output_offsets",
            "output_records",
            "candidates",
            "candidate_count",
            "haystack_len",
            "match_count",
            "matches",
            dfa.state_count,
            dfa.output_records.len() as u32,
            num_candidates,
            max_matches,
            dfa.max_pattern_len,
        ),
        num_candidates,
    );
    let inputs = vec![
        vyre_reference::value::Value::from(pack_haystack_u32(haystack)),
        vyre_reference::value::Value::from(pack_u32_slice(&dfa.transitions)),
        vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_offsets)),
        vyre_reference::value::Value::from(pack_u32_slice(&dfa.output_records)),
        vyre_reference::value::Value::from(pack_u32_slice(&candidates)),
        vyre_reference::value::Value::from(pack_u32_slice(&[num_candidates])),
        vyre_reference::value::Value::from(pack_u32_slice(&[haystack.len() as u32])),
        vyre_reference::value::Value::from(vec![0_u8; num_candidates as usize * 4]),
    ];
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("Fix: anchored-window extract program must evaluate in the reference backend");

    let mut actual: Vec<ByteRange> = decode_match_triples(&outputs)
        .into_iter()
        .map(|(pid, start, end)| ByteRange::new(pid, start, end))
        .collect();
    actual.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
    actual.dedup();

    assert_eq!(
        actual, expected,
        "reference-eval of the anchored-window program must equal the CPU oracle's extraction"
    );
    assert!(
        !expected.is_empty(),
        "parity test is vacuous: the oracle extracted no matches for these candidates"
    );
}

/// Rigorous differential: for literal patterns, the anchored-window
/// extraction over EVERY position must equal an independent naive
/// "does pattern P start exactly at position p" substring oracle. A
/// deterministic LCG builds the haystack so the test is reproducible without
/// an RNG dependency.
#[test]
fn differential_vs_naive_anchored_substring_oracle() {
    let patterns = ["ab", "abc", "bcx", "x", "cab"];
    let dfa = validator_for(&patterns);
    let validator = AnchoredWindowValidator::new(&dfa);

    // Deterministic haystack over a small alphabet that plants the patterns
    // densely (LCG (pure, reproducible, no rand crate)).
    let alphabet = b"abcx";
    let mut state: u32 = 0x1234_5678;
    let mut haystack = Vec::with_capacity(600);
    for _ in 0..600 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        haystack.push(alphabet[(state >> 24) as usize % alphabet.len()]);
    }

    // Candidates = every position (the extractor must be exact everywhere).
    let origins: Vec<u32> = (0..haystack.len() as u32).collect();
    let got = validator.validate_candidates(&haystack, &origins);

    // Independent oracle: naive anchored substring test per pattern.
    let mut oracle: Vec<ByteRange> = Vec::new();
    for (pid, pat) in patterns.iter().enumerate() {
        let pb = pat.as_bytes();
        if pb.len() <= haystack.len() {
            for start in 0..=haystack.len() - pb.len() {
                if &haystack[start..start + pb.len()] == pb {
                    oracle.push(ByteRange::new(
                        pid as u32,
                        start as u32,
                        (start + pb.len()) as u32,
                    ));
                }
            }
        }
    }
    oracle.sort_unstable_by_key(|m| (m.start, m.end, m.tag));
    oracle.dedup();

    assert_eq!(
        got, oracle,
        "anchored-window extraction must equal the naive anchored-substring oracle at every position"
    );
    // Guard against a vacuous pass: the dense haystack must actually plant
    // matches for several distinct patterns.
    assert!(
        oracle.len() > 50,
        "differential is vacuous: oracle found only {} matches",
        oracle.len()
    );
    let distinct_pids: std::collections::BTreeSet<u32> = oracle.iter().map(|m| m.tag).collect();
    assert!(
        distinct_pids.len() >= 4,
        "differential should exercise most patterns; saw pids {distinct_pids:?}"
    );
}

/// The dead-state early-out must not change results: a validator that stops
/// at the dead sink extracts the same set as a full-window walk that never
/// stops early. We reconstruct the un-optimized walk inline and compare.
#[test]
fn dead_state_early_out_equals_full_window_walk() {
    let patterns = ["abc", "abcde", "bx"];
    let dfa = validator_for(&patterns);
    let validator = AnchoredWindowValidator::new(&dfa);
    let haystack = b"abcdefabxabc";
    let origins: Vec<u32> = (0..haystack.len() as u32).collect();
    let optimized = validator.validate_candidates(haystack, &origins);

    // Reference: identical walk WITHOUT the dead-state break (shared helper).
    let full = direct_walk_all_origins(&dfa, haystack);

    assert_eq!(
        optimized, full,
        "dead-state early-out must be a pure optimization, identical extraction to the full walk"
    );
}

/// The DFA has exactly one detectable dead sink, and it is neither the start
/// state nor an accepting state (guards the detector against misclassifying a
/// live state).
#[test]
fn detected_dead_state_is_non_start_non_accepting_self_loop() {
    let dfa = validator_for(&["abc"]);
    let dead =
        detect_dead_state(&dfa).expect("an anchored DFA with a rejecting path has a dead sink");
    assert_ne!(dead, 0, "the start state must not be classified as dead");
    assert_eq!(dfa.accept[dead as usize], 0, "dead state must not accept");
    for byte in 0..=255u16 {
        assert_eq!(
            dfa.transitions[dead as usize * 256 + byte as usize],
            dead,
            "dead state must self-loop on every byte"
        );
    }
}

#[test]
fn literal_repeat_collapses_to_single_longest_match() {
    // `a{2,4}` over "aaaa" seeded at origin 0.
    let pipeline = dfa_for("a{2,4}");
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);
    let haystack = b"aaaa";

    // Raw fan-out: one hit per admissible length 2, 3, 4.
    let raw = validator.validate_candidates(haystack, &[0]);
    assert_eq!(
        triples(&raw),
        vec![(0, 0, 2), (0, 0, 3), (0, 0, 4)],
        "raw fan-out must surface every admissible {{2,4}} length at origin 0"
    );

    // Leftmost-longest: exactly the longest match (the whole 4-'a' run).
    let ll = validator.validate_candidates_leftmost_longest(haystack, &[0]);
    assert_eq!(
        triples(&ll),
        vec![(0, 0, 4)],
        "leftmost-longest must collapse the run to its single maximal match"
    );
}

#[test]
fn class_repeat_takes_maximal_body_and_stops_at_terminator() {
    // `k[0-9]{2,4}`: prefix `k`, then 2..4 digits. max_pattern_len == 1 + 4 == 5.
    let pipeline = dfa_for("k[0-9]{2,4}");
    assert_eq!(
        pipeline.dfa.max_pattern_len, 5,
        "window must size to the MAX repetition (k + 4 digits)"
    );
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);

    // Exactly 2 digits then a non-digit: single match `k12` (end 3).
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(b"k12x", &[0])),
        vec![(0, 0, 3)],
        "a 2-digit body terminated by a non-digit is the whole (minimal-length) token"
    );

    // 4 digits (== max): single match `k1234` (end 5).
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(b"k1234", &[0])),
        vec![(0, 0, 5)],
        "a 4-digit body is consumed whole"
    );

    // 6 digits (> max): maximal munch takes only 4 → `k1234` (end 5), NOT the
    // whole 6-digit run. The trailing digits are not part of a `{2,4}` match.
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(b"k123456", &[0])),
        vec![(0, 0, 5)],
        "maximal munch caps the body at m == 4 digits even when more digits follow"
    );

    // 1 digit (< min): no match at all.
    assert!(
        validator
            .validate_candidates_leftmost_longest(b"k1x", &[0])
            .is_empty(),
        "a 1-digit body is below the {{2,4}} minimum and must not match"
    );
}

#[test]
fn fixed_repeat_is_unchanged_by_leftmost_longest() {
    // A fixed `{n}` pattern accepts at exactly one length, so leftmost-longest
    // and the raw fan-out must agree (the fix must not perturb fixed patterns).
    let pipeline = dfa_for("ghp_[A-Za-z0-9]{4}");
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);
    let haystack = b"ghp_aB3d";

    let raw = triples(&validator.validate_candidates(haystack, &[0]));
    let ll = triples(&validator.validate_candidates_leftmost_longest(haystack, &[0]));
    assert_eq!(
        raw,
        vec![(0, 0, 8)],
        "fixed token accepts once at its full length"
    );
    assert_eq!(
        ll, raw,
        "leftmost-longest must equal the fan-out for fixed patterns"
    );
}

#[test]
fn two_variable_tokens_each_collapse_at_their_own_origin() {
    // Two `{2,4}` tokens; feed each token's start origin. Each yields exactly one
    // whole-token match (no cross-token bleed, no per-length duplicates).
    let pipeline = dfa_for("v[0-9]{2,4}");
    let validator = AnchoredWindowValidator::new(&pipeline.dfa);
    //                0123456789012
    let haystack = b"v123 xx v4567";
    // Token A: origin 0, `v123` (3 digits) -> end 4.
    // Token B: origin 8, `v4567` (4 digits) -> end 13.
    assert_eq!(
        triples(&validator.validate_candidates_leftmost_longest(haystack, &[0, 8])),
        vec![(0, 0, 4), (0, 8, 13)],
        "each variable token collapses to one maximal match at its own origin"
    );
}

#[test]
fn open_ended_repeat_window_uses_the_bounded_replay_policy() {
    let plus = dfa_for("k[0-9]+");
    assert_eq!(
        plus.dfa.max_pattern_len,
        crate::pattern::DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
        "open-ended `+` must use the finite default replay budget"
    );
    let lower_bounded = dfa_for("k[0-9]{3,}");
    assert_eq!(
        lower_bounded.dfa.max_pattern_len,
        crate::pattern::DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
        "open-ended `{{3,}}` must use the same finite default replay budget"
    );
}

fn dfa_for(pattern: &str) -> crate::pattern::RegexDfaPipeline {
    build_regex_dfa_pipeline(&[pattern], 4096, 16_384)
        .unwrap_or_else(|e| panic!("pattern {pattern:?} must compile to an anchored DFA: {e:?}"))
}

fn triples(matches: &[ByteRange]) -> Vec<(u32, u32, u32)> {
    let mut v: Vec<(u32, u32, u32)> = matches.iter().map(|m| (m.tag, m.start, m.end)).collect();
    v.sort_unstable();
    v
}

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(800))]

    #[test]
    fn planted_bounded_token_yields_single_maximal_match(
        k in 2usize..=6,
        pre in 0usize..6,
        body_seed in "[a-z]{6}",
    ) {
        let pattern = "q_[a-z]{2,6}";
        let pipeline = dfa_for(pattern);
        let validator = AnchoredWindowValidator::new(&pipeline.dfa);

        let body: String = body_seed.chars().take(k).collect();
        let mut haystack = String::new();
        haystack.push_str(&" ".repeat(pre)); // filler: cannot start `q`
        let origin = haystack.len() as u32;
        haystack.push_str("q_");
        haystack.push_str(&body);
        haystack.push('9'); // digit terminator: not in [a-z], cannot extend body

        let expected_end = origin + 2 + k as u32; // q_ (2) + k body bytes
        prop_assert_eq!(
            triples(&validator.validate_candidates_leftmost_longest(haystack.as_bytes(), &[origin])),
            vec![(0, origin, expected_end)],
            "planted token {:?} at origin {} must yield one maximal match ending at {}",
            haystack, origin, expected_end
        );
    }

    #[test]
    fn planted_below_minimum_body_yields_no_match(
        pre in 0usize..6,
        c in "[a-z]",
    ) {
        let pipeline = dfa_for("q_[a-z]{2,6}");
        let validator = AnchoredWindowValidator::new(&pipeline.dfa);

        let mut haystack = String::new();
        haystack.push_str(&" ".repeat(pre));
        let origin = haystack.len() as u32;
        haystack.push_str("q_");
        haystack.push_str(&c);
        haystack.push('9'); // terminator so only 1 body byte is available

        prop_assert!(
            validator
                .validate_candidates_leftmost_longest(haystack.as_bytes(), &[origin])
                .is_empty(),
            "a single-byte body is below the {{2,6}} minimum; token {:?} must not match",
            haystack
        );
    }
}
