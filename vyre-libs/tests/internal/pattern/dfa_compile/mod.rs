//! Compiler, wire round-trip, and budget contracts for the DFA.

use super::*;

#[test]
fn single_string_matches_only_its_suffix() {
    let dfa = dfa_compile(&[b"abc"]);
    let input = b"xxabcxx";

    // Walk to the state immediately after scanning "xxabc" (before the trailing xx).
    // We can't stop mid-scan in a loop; trace the exact 5-byte prefix instead.
    let mut s = 0usize;
    for &b in b"xxabc" {
        s = dfa.transitions[s * 256 + b as usize] as usize;
    }
    // Pattern 0 encodes as accept = pid+1 = 0+1 = 1. Asserting == 1 catches both
    // "no match" (accept=0) and wrong pid (accept != 1), including the pid+1 wrap
    // bug where pid=u32::MAX would encode as 0 and silence the match.
    assert_eq!(
        dfa.accept[s], 1,
        "after 'xxabc' the DFA must be in a state that accepts pattern 0 (encoded as 1); \
         got accept[{s}] = {}",
        dfa.accept[s]
    );
    // Verify output_records carries the correct pid for the full-match path.
    let rec_start = dfa.output_offsets[s] as usize;
    let rec_end = dfa.output_offsets[s + 1] as usize;
    assert_eq!(
        &dfa.output_records[rec_start..rec_end],
        &[0u32],
        "output_records for the accept state must contain exactly [0] (pid=0)"
    );

    // Negative: after trailing 'x' the DFA must have left the accept state.
    let s_after_x = dfa.transitions[s * 256 + b'x' as usize] as usize;
    assert_eq!(
        dfa.accept[s_after_x], 0,
        "after trailing 'x' the DFA must not accept; pattern 'abc' ends before it"
    );
}

/// Walk `dfa` over `haystack` and return every `(pattern_id, end_pos)` match,
/// the plain-Rust oracle used to prove case-insensitive folding.
fn scan_ends(dfa: &CompiledDfa, haystack: &[u8]) -> std::collections::BTreeSet<(u32, u32)> {
    let mut state = 0usize;
    let mut out = std::collections::BTreeSet::new();
    for (pos, &b) in haystack.iter().enumerate() {
        state = dfa.transitions[state * 256 + b as usize] as usize;
        let begin = dfa.output_offsets[state] as usize;
        let end = dfa.output_offsets[state + 1] as usize;
        for &pid in &dfa.output_records[begin..end] {
            out.insert((pid, pos as u32));
        }
    }
    out
}

#[test]
fn case_insensitive_matches_every_case_variant() {
    let dfa = dfa_compile_case_insensitive(&[b"key"]);
    // Every case variant of "key" ends at position 2 in its 3-byte window.
    for variant in [b"KEY", b"Key", b"kEy", b"keY", b"kEY", b"key"] {
        let hits = scan_ends(&dfa, variant);
        assert!(
            hits.contains(&(0, 2)),
            "case-insensitive DFA must match {:?} as pattern 0 ending at 2, got {hits:?}",
            std::str::from_utf8(variant).unwrap()
        );
    }
    // A genuinely different string must NOT match.
    assert!(
        scan_ends(&dfa, b"kez").is_empty(),
        "case-insensitive folding must not match a non-variant string"
    );
}

#[test]
fn case_insensitive_is_identical_to_host_folded_case_sensitive() {
    // The correctness contract the plan names: a case-insensitive scan of the
    // RAW mixed-case haystack must equal a case-SENSITIVE scan of the
    // host-lowercased haystack with lowercased patterns. Randomized differential.
    let alphabet = b"aAbBkK_9/";
    let mut seed = 0x9E37_79B1u64;
    let mut next = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as u32
    };
    for _ in 0..500 {
        // 1..=4 patterns, each 1..=5 bytes over the mixed-case alphabet.
        let pat_count = 1 + (next() % 4) as usize;
        let patterns_owned: Vec<Vec<u8>> = (0..pat_count)
            .map(|_| {
                let len = 1 + (next() % 5) as usize;
                (0..len)
                    .map(|_| alphabet[(next() as usize) % alphabet.len()])
                    .collect()
            })
            .collect();
        let patterns: Vec<&[u8]> = patterns_owned.iter().map(Vec::as_slice).collect();

        let hay_len = 4 + (next() % 40) as usize;
        let haystack: Vec<u8> = (0..hay_len)
            .map(|_| alphabet[(next() as usize) % alphabet.len()])
            .collect();

        // Case-insensitive DFA over the raw haystack.
        let ci = dfa_compile_case_insensitive(&patterns);
        let ci_hits = scan_ends(&ci, &haystack);

        // Host-folded reference: lowercase patterns + lowercase haystack,
        // case-SENSITIVE DFA. This is exactly the pass W2-1 replaces.
        let lowered_pat: Vec<Vec<u8>> = patterns_owned
            .iter()
            .map(|p| p.iter().map(|b| b.to_ascii_lowercase()).collect())
            .collect();
        let lowered_refs: Vec<&[u8]> = lowered_pat.iter().map(Vec::as_slice).collect();
        let lowered_hay: Vec<u8> = haystack.iter().map(|b| b.to_ascii_lowercase()).collect();
        let reference = dfa_compile(&lowered_refs);
        let ref_hits = scan_ends(&reference, &lowered_hay);

        assert_eq!(
            ci_hits,
            ref_hits,
            "case-insensitive DFA over raw haystack must equal host-folded case-sensitive scan\n\
             patterns={patterns_owned:?}\n\
             haystack={:?}",
            String::from_utf8_lossy(&haystack)
        );
    }
}

#[test]
fn overlapping_patterns_both_accept() {
    let patterns: [&[u8]; 4] = [b"he", b"she", b"his", b"hers"];
    let dfa = dfa_compile(&patterns);
    let mut state = 0u32;
    let mut matches = Vec::new();
    for &b in b"ushers" {
        state = dfa.transitions[(state as usize) * 256 + (b as usize)];
        let accept = dfa.accept[state as usize];
        if accept != 0 {
            matches.push(accept - 1);
        }
    }
    assert!(matches.contains(&1), "must accept `she`");
    assert!(
        matches.contains(&0) || matches.contains(&3),
        "must accept `he` or `hers`"
    );
}

#[test]
fn duplicate_literals_preserve_distinct_output_records() {
    let dfa = dfa_compile(&[b"B".as_slice(), b"B".as_slice(), b"AB".as_slice()]);
    let state_b = dfa.transitions[b'B' as usize] as usize;
    let state_ab = {
        let state_a = dfa.transitions[b'A' as usize] as usize;
        dfa.transitions[state_a * 256 + b'B' as usize] as usize
    };

    let b_start = dfa.output_offsets[state_b] as usize;
    let b_end = dfa.output_offsets[state_b + 1] as usize;
    assert_eq!(
        &dfa.output_records[b_start..b_end],
        &[0, 1],
        "Fix: exact duplicate literals must keep both consumer pattern ids in output_records."
    );

    let ab_start = dfa.output_offsets[state_ab] as usize;
    let ab_end = dfa.output_offsets[state_ab + 1] as usize;
    assert_eq!(
        &dfa.output_records[ab_start..ab_end],
        &[0, 1, 2],
        "Fix: suffix inheritance must preserve duplicate suffix pattern ids plus the local longer pattern."
    );
}

#[test]
fn empty_pattern_list_yields_trivial_dfa() {
    let dfa = dfa_compile(&[]);
    assert_eq!(dfa.state_count, 1);
    assert_eq!(dfa.transitions.len(), 256);
    assert!(dfa.transitions.iter().all(|&t| t == 0));
    assert_eq!(dfa.accept, vec![0]);
}

#[test]
fn budget_exhaustion_returns_structured_error() {
    let err = dfa_compile_with_budget(&[b"ab", b"cd"], 1024).unwrap_err();
    match err {
        DfaCompileError::TooLarge {
            requested_bytes,
            budget_bytes,
            state_count,
        } => {
            assert!(
                requested_bytes > budget_bytes,
                "TooLarge must carry requested > budget"
            );
            assert_eq!(budget_bytes, 1024);
            assert!(state_count >= 1);
        }
        DfaCompileError::TrieStateCapExceeded { state_cap } => {
            assert!(state_cap <= 1024);
        }
        other => panic!("a two-pattern set under a 1 KiB budget is a size failure: {other}"),
    }
}

/// WHY: a compile failure is read by whoever supplied the patterns, so each one
/// owes the numbers that caused it and the action that resolves it. Every
/// variant is listed here rather than the one a caller happens to hit, so a new
/// failure added without a corrective sentence turns this red.
#[test]
fn every_compile_failure_names_its_numbers_and_its_fix() {
    let failures = [
        DfaCompileError::TooLarge {
            requested_bytes: 4096,
            budget_bytes: 1024,
            state_count: 7,
        },
        DfaCompileError::TrieStateCapExceeded { state_cap: 64 },
        DfaCompileError::TooManyPatterns {
            pattern_count: 5,
            limit: 4,
        },
    ];
    for failure in &failures {
        let rendered = failure.to_string();
        assert!(
            rendered.contains("Fix:"),
            "{failure:?} must state a corrective action: {rendered}"
        );
        let numbers: Vec<char> = rendered.chars().filter(char::is_ascii_digit).collect();
        assert!(
            !numbers.is_empty(),
            "{failure:?} must carry the measured numbers: {rendered}"
        );
    }
}

#[test]
fn generous_budget_succeeds() {
    let dfa = dfa_compile_with_budget(&[b"abc"], DEFAULT_DFA_BUDGET_BYTES)
        .expect("Fix: generous budget must succeed; restore this invariant before continuing.");
    assert!(dfa.state_count >= 1);
}

#[test]
fn zero_budget_rejects_every_nonempty_dfa() {
    let err = dfa_compile_with_budget(&[b"a"], 0).unwrap_err();
    assert!(matches!(
        err,
        DfaCompileError::TooLarge { .. } | DfaCompileError::TrieStateCapExceeded { .. }
    ));
}

/// Finding #13 (P2): accept field last-writer-wins bug.
/// When two patterns share a final trie node (duplicate literals or suffix patterns),
/// the accept fast-path field must store the FIRST (lowest) pattern id, not the last.
/// Before the fix, accept[state_B] = 2 (pid=1, last writer) instead of 1 (pid=0, first).
/// Finding #14 (P2): from_bytes incorrectly rejected DFAs compiled from
/// zero-length patterns because max_pattern_len==0 with accept states was
/// treated as "corrupt sentinel" rather than "empty-pattern accept".
#[test]
fn empty_pattern_dfa_round_trips() {
    let dfa = dfa_compile(&[b"".as_slice()]);
    // The root state must accept (empty string matches everywhere).
    assert_eq!(
        dfa.accept[0], 1,
        "dfa_compile(&[b\"\"]) root state must accept pattern 0 (accept=1)"
    );
    assert_eq!(
        dfa.max_pattern_len, 0,
        "empty pattern must produce max_pattern_len=0"
    );
    let bytes = dfa
        .to_bytes()
        .expect("Fix: serialization must succeed for empty-pattern DFA");
    let dfa2 = CompiledDfa::from_bytes(&bytes)
        .expect("Fix: round-trip must succeed for empty-pattern DFA");
    assert_eq!(
        dfa2.accept[0], 1,
        "deserialized DFA must preserve accept[0]=1 for empty-pattern compile"
    );
    assert_eq!(
        dfa2.max_pattern_len, 0,
        "deserialized DFA must preserve max_pattern_len=0"
    );
}

#[test]
fn from_bytes_rejects_zero_max_pattern_len_with_non_root_accept() {
    // dfa_compile(&[b"AKIA"]) produces a non-root accept state (the state reached
    // after consuming A-K-I-A) and max_pattern_len == 4. A blob that claims
    // max_pattern_len == 0 while still carrying that deeper accept is internally
    // inconsistent, the canonical symptom of a corrupted cache whose length scalar
    // was zeroed. Decoding it would yield a DFA whose under-sized replay/segmentation
    // window silently drops cross-boundary matches, so from_bytes must fail closed.
    let mut dfa = dfa_compile(&[b"AKIA".as_slice()]);
    assert!(
        dfa.max_pattern_len >= 1,
        "precondition: AKIA must compile to max_pattern_len >= 1, got {}",
        dfa.max_pattern_len
    );
    assert!(
        dfa.accept.iter().skip(1).any(|&state| state != 0),
        "precondition: AKIA must have a non-root accept state"
    );
    // Forge the corruption by zeroing only the max_pattern_len scalar; every other
    // table stays consistent, so the rejection is attributable solely to the new check.
    dfa.max_pattern_len = 0;
    let bytes = dfa.to_bytes().expect("encode forged DFA wire blob");
    let err = CompiledDfa::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(
            err,
            DfaWireError::ShapeMismatch {
                reason: "max_pattern_len == 0 but a non-root state accepts"
            }
        ),
        "expected ShapeMismatch with the non-root-accept reason, got {err:?}"
    );
}

#[test]
fn from_bytes_rejects_out_of_range_transition_target() {
    // Every transition value is consumed as the next state index
    // (`transitions[state * 256 + byte]`), so a target >= state_count would
    // OOB-index on the following step. A corrupt/stale cache blob must fail
    // closed at decode, not panic (or read a garbage state) mid-scan.
    let mut dfa = dfa_compile(&[b"abc".as_slice()]);
    assert!(
        dfa.state_count >= 2,
        "precondition: fixture must have real states"
    );
    assert!(
        dfa.transitions
            .iter()
            .all(|&t| (t as usize) < dfa.state_count as usize),
        "precondition: an honest compile keeps every transition target in range"
    );
    // state_count itself is the first out-of-range state id (valid ids are 0..state_count).
    // Forge only this one target; every length/offset table stays consistent, so the
    // rejection is attributable solely to the new bounds check.
    dfa.transitions[0] = dfa.state_count;
    let bytes = dfa.to_bytes().expect("encode forged DFA wire blob");
    let err = CompiledDfa::from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(
            err,
            DfaWireError::ShapeMismatch {
                reason: "transition target out of range for state_count"
            }
        ),
        "expected the transition-target range violation, got {err:?}"
    );
}

#[test]
fn duplicate_literal_accept_field_contains_first_pattern() {
    // dfa_compile(&[b"B", b"B"]): both patterns share trie state 1 (after b'B').
    // pid=0 is inserted first → accept[state_B] must be 1 (0+1).
    // pid=1 is inserted second → must not overwrite → accept[state_B] stays 1.
    let dfa = dfa_compile(&[b"B".as_slice(), b"B".as_slice()]);
    let state_b = dfa.transitions[b'B' as usize] as usize;
    assert_eq!(
        dfa.accept[state_b],
        1,
        "first duplicate literal (pid=0) must win the accept fast-path field (encoded as pid+1=1); \
         last-writer-wins would give 2 (pid=1)"
    );
    // The output_records must still carry both pids for the full-match path.
    let start = dfa.output_offsets[state_b] as usize;
    let end = dfa.output_offsets[state_b + 1] as usize;
    assert_eq!(
        &dfa.output_records[start..end],
        &[0u32, 1u32],
        "duplicate literals must both appear in output_records"
    );
}

#[test]
fn from_bytes_wire_program_builder_round_trip() {
    let dfa = dfa_compile(&[b"test".as_slice()]);
    let bytes = dfa.to_bytes().expect("encode DFA wire blob");
    let prog = crate::pattern::aho_corasick::aho_corasick_program_from_dfa_wire(
        &bytes,
        "haystack",
        "transitions",
        "accept",
        "matches",
        32,
    )
    .expect("decode DFA wire blob into Program");
    assert_eq!(prog.workgroup_size, [64, 1, 1]);
}

#[test]
fn from_bytes_wire_program_builder_rejects_bad_magic() {
    let mut bytes = dfa_compile(&[b"test".as_slice()])
        .to_bytes()
        .expect("encode DFA wire blob");
    bytes[0] = 0;
    let err = crate::pattern::aho_corasick::aho_corasick_program_from_dfa_wire(
        &bytes,
        "haystack",
        "transitions",
        "accept",
        "matches",
        32,
    )
    .unwrap_err();
    assert!(matches!(err, DfaWireError::BadMagic));
}
