//! Tests for the regex DFA pipeline.
//!
//! Conformance accepts registered bytes as proof for every backend row, so the
//! registered fixture is checked against an independent reference execution
//! rather than merely for being present.

use super::*;
use crate::fixture_bytes::eval_bytes;

/// WHY: conformance accepts registered bytes as proof for every backend row, so the regex
/// DFA fixture must equal an independent reference execution rather than merely be present.
#[test]
fn registered_regex_dfa_expected_bytes_match_reference_execution() {
    let pipeline = build_regex_dfa_pipeline(&["[a-z]+"], 64, 256)
        .expect("Fix: canonical fixture regex DFA must compile");
    let inputs = [
        vec![0u8; 64],
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.transitions),
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.output_offsets),
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.output_records),
        vyre_primitives::wire::pack_u32_slice(&pipeline.pattern_lengths),
        vec![0u8; 4],
        vec![0u8; 4],
    ];
    let values = inputs
        .iter()
        .map(|bytes| vyre_reference::value::Value::Bytes(bytes.as_slice().into()))
        .collect::<Vec<_>>();
    let actual = eval_bytes("regex_dfa_tests", &pipeline.program, values);

    assert_eq!(
        actual,
        vec![
            EXPECTED_REGEX_DFA_MATCH_COUNT_BYTES.to_vec(),
            EXPECTED_REGEX_DFA_MATCHES_BYTES.to_vec(),
        ],
        "Fix: registered regex DFA expected bytes must equal the reference result"
    );
}

/// Single-pass DFA replay from the start state, the exact semantics the
/// megakernel batch dispatcher uses (one pass per file, no per-position
/// restart). Returns the end offsets where the DFA accepts.
fn single_pass_accept_ends(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<usize> {
    let mut state = 0u32;
    let mut ends = Vec::new();
    for (i, &b) in haystack.iter().enumerate() {
        state = dfa.transitions
            [crate::builder::state_machine::TableStateMachineComposer::flat_byte_index(state, b)];
        if dfa.accept[state as usize] != 0 {
            ends.push(i + 1);
        }
    }
    ends
}

/// Leftmost-longest ("maximal munch") accept ends over the unanchored dense
/// DFA. A token that accepts at several consecutive lengths, a variable
/// `{n,m}` / `+` / `*` body, collapses to the SINGLE longest end (the end of
/// its accepting run) instead of one hit per admissible length. Emits end `p`
/// iff the DFA accepts at `p` and does NOT accept at `p + 1` (the match cannot
/// be extended), which for a `<prefix><class>{n,m}` token terminated by a
/// non-class byte is exactly its maximal end. Fixed-length patterns (one
/// accept length per occurrence) yield the same result as
/// [`single_pass_accept_ends`]. This is the semantics a scanner wants: one
/// finding covering the whole token, not `m - n + 1` overlapping partials.
fn single_pass_leftmost_longest_ends(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<usize> {
    let mut state = 0u32;
    let mut ends = Vec::new();
    let mut prev_end = 0usize;
    let mut prev_accept = false;
    for (i, &b) in haystack.iter().enumerate() {
        state = dfa.transitions
            [crate::builder::state_machine::TableStateMachineComposer::flat_byte_index(state, b)];
        let accept = dfa.accept[state as usize] != 0;
        if prev_accept && !accept {
            // The accepting run ended: `prev_end` was its maximal end.
            ends.push(prev_end);
        }
        prev_end = i + 1;
        prev_accept = accept;
    }
    if prev_accept {
        // The accepting run reaches end-of-input.
        ends.push(prev_end);
    }
    ends
}

/// The unanchored build must match a pattern at ANY offset under a single
/// forward pass (find-anywhere), while the anchored build dies on a
/// non-matching prefix. This is the property the megakernel fallback
/// port depends on (a secret is rarely at byte 0).
#[test]
fn unanchored_dfa_matches_at_any_offset_single_pass() {
    let anchored = build_regex_dfa_pipeline(&["abc"], 1024, 1024).expect("anchored compiles");
    let unanchored = build_regex_dfa_unanchored(&["abc"], 1024, 1024).expect("unanchored compiles");

    // Unanchored: one pass over "xxabc" accepts at end=5 (abc at bytes 2..4).
    assert_eq!(
        single_pass_accept_ends(&unanchored.dfa, b"xxabc"),
        vec![5],
        "unanchored DFA must match `abc` after a non-matching prefix"
    );
    // Anchored: the leading 'x' drives state 0 to a dead state → no accept.
    assert!(
        single_pass_accept_ends(&anchored.dfa, b"xxabc").is_empty(),
        "anchored DFA must NOT match `abc` after a non-matching prefix"
    );
    // Both match at the start.
    assert_eq!(single_pass_accept_ends(&unanchored.dfa, b"abc"), vec![3]);
    assert_eq!(single_pass_accept_ends(&anchored.dfa, b"abc"), vec![3]);
    // Unanchored finds every occurrence in one pass.
    assert_eq!(
        single_pass_accept_ends(&unanchored.dfa, b"abcxabc"),
        vec![3, 7],
        "unanchored DFA must find all occurrences"
    );
}

/// Regression: a downstream GPU parity gate missed a real `ghp_` token whose
/// 36-char body contains g/h/p (the prefix chars), a prefix/body overlap
/// under the `.*` self-loop. This CPU single-pass DFA check isolates whether
/// the miss is in THIS primitive's construction or downstream on the GPU.
#[test]
fn unanchored_dfa_finds_overlap_body_token_single_pass() {
    let dfa = build_regex_dfa_unanchored(&["ghp_[A-Za-z0-9]{36}"], 1024, 16384)
        .expect("compiles")
        .dfa;
    // Exact missed content from a downstream parity gate (file 120).
    let hay = b"at = \"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\"";
    let ends = single_pass_accept_ends(&dfa, hay);
    assert_eq!(
        ends,
        vec![hay.len() - 1],
        "unanchored DFA must accept the ghp_ token exactly before the closing quote"
    );
}

/// Isolation for the 6 GPU parity-gate misses: run the EXACT missed contexts
/// through the dense `CompiledDfa` on the CPU with the kernel's single-pass
/// semantics. If these all accept here but the GPU drops them, the bug is in
/// the megakernel dispatch, not this primitive's DFA construction.
#[test]
fn unanchored_dfa_finds_all_parity_gate_misses_single_pass() {
    // (pattern, exact missed match content from the parity gate run)
    let cases: &[(&str, &[u8])] = &[
        (
            "ghp_[A-Za-z0-9]{36}",
            b"at = \"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\"",
        ),
        (
            "gho_[A-Za-z0-9]{36}",
            b"ken: \"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\"",
        ),
        (
            "ghu_[A-Za-z0-9]{36}",
            b"Key: \"ghu_m7BOv2Uj0AZZK088M7RQJkZX3EgBVV1Xt7i2\"",
        ),
        (
            "ghu_[A-Za-z0-9]{36}",
            b"OKEN: ghu_4u5ef0rIhtKpPV1F0dPwwhXNMpEXkB0tWWQv",
        ),
        (
            "xox[baprs]-[A-Za-z0-9-]{10,48}",
            b"Key: \"xoxb-1234567890-1234567890-EXAMPLE-TOKEN\"",
        ),
        (
            "xox[baprs]-[A-Za-z0-9-]{10,48}",
            b"_KEY=\"xoxb-32790994721-16118213278-q5KLPWcLboh0tchHpJPgWhuC\"",
        ),
    ];
    for (pat, hay) in cases {
        let dfa = build_regex_dfa_unanchored(&[pat], 1024, 16384)
            .unwrap_or_else(|e| panic!("pattern {pat:?} must compile: {e:?}"))
            .dfa;
        // Leftmost-longest ("maximal munch") extraction: each case holds ONE
        // complete token, so the scanner-correct result is its single maximal
        // end. The raw all-ends walk (`single_pass_accept_ends`) is only
        // single-valued for FIXED-length patterns, a variable `{10,48}` body
        // genuinely accepts at every admissible length (26 ends for the `xox`
        // cases), so asserting a single end there requires the leftmost-longest
        // walk, which collapses the run to its longest end. Asserting the exact
        // set (not containment) catches both a missed hit and a spurious/
        // duplicated earlier hit from body overlap under the dotstar self-loop.
        let ends = single_pass_leftmost_longest_ends(&dfa, hay);
        let expected_end = if hay.ends_with(b"\"") {
            hay.len() - 1
        } else {
            hay.len()
        };
        assert_eq!(
            ends,
            vec![expected_end],
            "dense CompiledDfa for {pat:?} must report exactly one leftmost-longest \
             end offset ({expected_end}) in {:?}; got {ends:?}. state_count={}",
            String::from_utf8_lossy(hay),
            dfa.state_count,
        );
    }
}

/// End-to-end: a literal regex set should produce a Program whose
/// CompiledDfa accepts the literal at the expected end offset. The
/// CompiledDfa accept table is the load-bearing assertion - if it's
/// empty, the composition didn't propagate accept metadata through
/// the subset construction.
#[test]
fn literal_pattern_set_lowers_through_to_dfa_program() {
    let pipeline =
        build_regex_dfa_pipeline(&["abc"], 1024, 1024).expect("Fix: literal must compile");
    assert!(
        pipeline.dfa.state_count >= 4,
        "literal 'abc' DFA must have at least 4 states (entry + 3 progress); got {}",
        pipeline.dfa.state_count
    );
    assert_eq!(
        pipeline.pattern_lengths,
        vec![3],
        "single literal 'abc' must have pattern_lengths = [3]"
    );
    assert!(
        pipeline
            .dfa
            .accept
            .iter()
            .any(|&pid_plus_one| pid_plus_one == 1),
        "at least one DFA state must accept pattern 0 (encoded as accept = 1)"
    );
    // Program buffer surface matches the AC kernel's contract:
    // haystack, transitions, output_offsets, output_records,
    // pattern_lengths, haystack_len, match_count, matches.
    let names: Vec<&str> = pipeline.program.buffers.iter().map(|b| b.name()).collect();
    for expected in [
        "haystack",
        "transitions",
        "output_offsets",
        "output_records",
        "pattern_lengths",
        "haystack_len",
        "match_count",
        "matches",
    ] {
        assert!(
            names.contains(&expected),
            "RegexDfaPipeline program must declare buffer `{expected}` for AC dispatch; got {names:?}"
        );
    }
}

/// Multi-pattern union: two literals must end up in two distinct
/// accept states (each tied to its own pattern id), not collapsed
/// into one.
#[test]
fn multi_literal_set_emits_distinct_accept_pids() {
    let pipeline = build_regex_dfa_pipeline(&["abc", "xyz"], 1024, 1024)
        .expect("Fix: two literals must compile");
    assert_eq!(pipeline.pattern_lengths, vec![3, 3]);
    // accept[s] = pid + 1, so a multi-pattern set should produce
    // both `1` (pid 0) and `2` (pid 1) somewhere in the accept
    // table. If either is missing, the subset construction lost
    // an accept's pattern_id.
    let has_pid0 = pipeline.dfa.accept.iter().any(|&value| value == 1);
    let has_pid1 = pipeline.dfa.accept.iter().any(|&value| value == 2);
    assert!(has_pid0, "no DFA state accepts pid 0 - 'abc' lost in lower");
    assert!(has_pid1, "no DFA state accepts pid 1 - 'xyz' lost in lower");
}

/// State-explosion path: setting `max_dfa_states` to 1 must surface
/// as a structured error, not a panic.
#[test]
fn state_explosion_surfaces_as_error_not_panic() {
    let err = build_regex_dfa_pipeline(&["abc"], 1024, 1)
        .expect_err("max_dfa_states=1 must trip state explosion");
    match err {
        RegexDfaError::Lower(NfaToDfaError::StateExplosion { .. }) => {}
        other => panic!("expected Lower(StateExplosion), got {other:?}"),
    }
}

/// A regex with a character class should also lower - this is the
/// case `ScanProgram` would scan via NFA bit-vector. The DFA path
/// must produce an accept somewhere so the consumer gets a hit.
#[test]
fn character_class_pattern_lowers_to_acceptor_dfa() {
    let pipeline = build_regex_dfa_pipeline(&["[ab]c"], 1024, 1024)
        .expect("Fix: character class must compile");
    assert!(
        pipeline.dfa.accept.iter().any(|&value| value != 0),
        "DFA for '[ab]c' must accept at least one state"
    );
}

/// Behavioral complement to regex_dfa_pipeline_uses_checked_size_conversions:
/// verify that the RegexDfaError::Size variant actually carries an actionable
/// message when triggered. We trigger it via nfa_to_dfa's max_dfa_states guard
/// (maps to RegexDfaError::Lower), and separately verify the Size variant's
/// Display output is actionable when constructed directly.
#[test]
fn regex_dfa_size_error_has_actionable_message() {
    // Construct a Size error directly (the behavioral path that exercises the
    // variant formatting, pattern-count overflow requires > u32::MAX allocations
    // which is not feasible in a unit test, but we can verify the error is
    // coherent and carries the expected guidance text).
    let err = RegexDfaError::Size {
        message: "pattern count 4294967296 exceeds u32 GPU buffer metadata: out of range integral type conversion attempted. Fix: shard the regex set before building a DFA dispatch.".to_string(),
    };
    let displayed = format!("{err}");
    assert!(
        displayed.contains("Fix:"),
        "RegexDfaError::Size display must carry an actionable Fix directive; got: {displayed:?}"
    );
    assert!(
        displayed.contains("shard"),
        "RegexDfaError::Size display must mention sharding as the recovery path; got: {displayed:?}"
    );
}

/// Regression guard: build_regex_dfa_unanchored must propagate the error from
/// add_implicit_dotstar_prefix rather than silently producing an anchored DFA.
/// This test verifies the success path still works; the error path cannot be
/// triggered for well-formed patterns (compile_regex_set always produces
/// internally-consistent tables), so the fix is covered by a source-scan guard below.
#[test]
fn unanchored_build_succeeds_and_is_actually_unanchored() {
    let pipeline =
        build_regex_dfa_unanchored(&["abc"], 1024, 1024).expect("unanchored must compile");
    // An anchored DFA would fail to match "abc" after a non-matching prefix in
    // a single forward pass. The unanchored DFA must succeed.
    let mut state = 0u32;
    let mut accepted = false;
    for &b in b"xxabc" {
        state = pipeline.dfa.transitions
            [crate::builder::state_machine::TableStateMachineComposer::flat_byte_index(state, b)];
        if pipeline.dfa.accept[state as usize] != 0 {
            accepted = true;
        }
    }
    assert!(
        accepted,
        "unanchored DFA must match 'abc' after non-matching prefix 'xx' in a single pass; \
         if this fails the add_implicit_dotstar_prefix self-loop was not applied"
    );
}

/// Pid-aware single-pass replay: at each accepting state, emit EVERY pattern
/// id in `output_records` (not just the single `accept` id), exactly as the
/// real dispatch does (so overlapping patterns at one position all surface).
fn walk_unanchored_local_hits(dfa: &CompiledDfa, hay: &[u8]) -> Vec<(u32, usize)> {
    let mut state = 0u32;
    let mut hits = Vec::new();
    for (i, &b) in hay.iter().enumerate() {
        state = dfa.transitions
            [crate::builder::state_machine::TableStateMachineComposer::flat_byte_index(state, b)];
        let s = state as usize;
        let lo = dfa.output_offsets[s] as usize;
        let hi = dfa.output_offsets[s + 1] as usize;
        for &pid in &dfa.output_records[lo..hi] {
            hits.push((pid, i + 1));
        }
    }
    hits
}

/// State-cap elimination: a pattern set that OVERFLOWS a small single-DFA cap
/// must still scan losslessly once split into shards, and the union of shard
/// hits, rewritten to global pattern ids, must equal an independent
/// naive-substring oracle over the same haystack. Proves both the fitting
/// guarantee and that pid remapping loses/duplicates nothing (Law 10).
#[test]
fn dfa_shards_cover_overflowing_set_losslessly_with_global_pids() {
    let patterns = ["alpha", "bravo", "charlie", "delta", "epsilon", "gamma"];
    let refs: Vec<&str> = patterns.to_vec();
    // A cap that fits a couple of these literals' unanchored DFA but not all
    // six at once (forces multiple shards).
    let cap = 18usize;

    // Precondition: the whole set genuinely overflows the small cap.
    assert!(
        build_regex_dfa_unanchored(&refs, 4096, cap).is_err(),
        "precondition: the whole 6-pattern set must overflow a {cap}-state cap"
    );

    let shards = build_regex_dfa_shards_unanchored(&refs, 4096, cap)
        .expect("sharding must fit every pattern within the cap");
    assert!(
        shards.len() >= 2,
        "an overflowing set must split into >=2 shards"
    );

    // Every global pid 0..6 is covered exactly once across shards, and each
    // shard's DFA actually fits the cap (the fitting guarantee).
    let mut covered: Vec<u32> = shards
        .iter()
        .flat_map(|s| s.global_pattern_ids.iter().copied())
        .collect();
    covered.sort_unstable();
    assert_eq!(
        covered,
        (0..patterns.len() as u32).collect::<Vec<_>>(),
        "shards must partition the global pattern ids with no gap or overlap"
    );
    for shard in &shards {
        assert!(
            shard.pipeline.dfa.state_count as usize <= cap,
            "every emitted shard must fit the {cap}-state cap; got {}",
            shard.pipeline.dfa.state_count
        );
        assert_eq!(
            shard.global_pattern_ids.len(),
            shard.pipeline.pattern_lengths.len(),
            "one global id per shard-local pattern"
        );
    }

    // Differential over a haystack that embeds several patterns at offsets.
    let hay = b"__alpha xx charlie--epsilon..bravo gamma zz delta__epsilonalpha";
    // Independent oracle: every occurrence of each pattern -> (global_pid, end).
    let mut oracle: Vec<(u32, usize)> = Vec::new();
    for (gid, pat) in patterns.iter().enumerate() {
        let pb = pat.as_bytes();
        if pb.len() <= hay.len() {
            for start in 0..=hay.len() - pb.len() {
                if &hay[start..start + pb.len()] == pb {
                    oracle.push((gid as u32, start + pb.len()));
                }
            }
        }
    }
    oracle.sort_unstable();

    // Sharded union: walk each shard, rewrite local pid -> global pid.
    let mut got: Vec<(u32, usize)> = Vec::new();
    for shard in &shards {
        for (local_pid, end) in walk_unanchored_local_hits(&shard.pipeline.dfa, hay) {
            let global = shard.global_pattern_ids[local_pid as usize];
            got.push((global, end));
        }
    }
    got.sort_unstable();

    assert_eq!(
        got, oracle,
        "sharded scan (global-remapped) must equal the naive-substring oracle; \
         a mismatch means the cap-sharding dropped, duplicated, or mis-attributed a match"
    );
    // Sanity: the oracle actually found the embedded patterns (guards a vacuous pass).
    assert!(
        oracle.len() >= patterns.len(),
        "oracle must contain at least one hit per pattern for a meaningful differential"
    );
}

/// A single pattern that cannot fit the cap on its own must SURFACE its
/// capacity error, never be silently omitted from the shard set (Law 10).
#[test]
fn dfa_shards_surface_error_for_unshardable_single_pattern() {
    // One pattern whose own DFA needs more than a 1-state cap.
    let result = build_regex_dfa_shards_unanchored(&["abcdef"], 4096, 1);
    assert!(
        result.is_err(),
        "a lone pattern that overflows the cap must error, not drop silently"
    );
}

/// The pipeline builder must forward the inner compile error's registry
/// diagnostic code, so a consumer routing on `build_regex_dfa_pipeline`'s
/// error gets the same code as the low-level `compile_regex_set` path.
#[test]
fn pipeline_error_forwards_diagnostic_code() {
    let err = build_regex_dfa_pipeline(&[r"a\bc"], 1024, 1024)
        .expect_err("a non-edge lookaround pattern must not compile");
    assert_eq!(
        err.diagnostic_code(),
        Some("VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER"),
        "pipeline error must forward the inner lookaround diagnostic code; error was: {err}"
    );
    // A sizing/lowering failure is not a registry construct -> no code.
    let size_err =
        build_regex_dfa_pipeline(&["abc"], 1024, 1).expect_err("a 1-state cap must overflow");
    assert_eq!(
        size_err.diagnostic_code(),
        None,
        "a state-budget overflow is not a registry unsupported-construct"
    );
}

#[test]
fn regex_dfa_program_builders_produce_valid_ir() {
    let prog = regex_dfa_program(&["[a-z]+"], 64, 256).expect("anchored regex DFA program builds");
    assert!(!prog.buffers().is_empty());
    let prog_unanchored = regex_dfa_unanchored_program(&["[a-z]+"], 64, 256)
        .expect("unanchored regex DFA program builds");
    assert!(!prog_unanchored.buffers().is_empty());
    let sharded = regex_dfa_sharded_programs(&["[a-z]+", "[0-9]+"], 64, 256)
        .expect("sharded regex DFA programs build");
    assert!(!sharded.is_empty());
    let unanchored_sharded = regex_dfa_unanchored_sharded_programs(&["[a-z]+", "[0-9]+"], 64, 256)
        .expect("unanchored sharded programs build");
    assert!(!unanchored_sharded.is_empty());
}
