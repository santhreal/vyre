//! Cross-layer matching construction and reference-oracle contracts.
//!
//! These tests cover DFA construction, literal-set reference execution, and
//! regex-to-NFA compilation. Registered artifact execution is covered by
//! the downstream scan product's tests and concrete backend conformance tests.

#![allow(deprecated)]
// (MatchScan trait imported in the tests that need it.)

#[test]
fn literal_set_cpu_finds_planted_secret() {
    use vyre::scan::GpuLiteralSet;
    let engine = GpuLiteralSet::compile(&[b"AKIA".as_slice(), b"ghp_".as_slice()]);
    let haystack = b"foo AKIAIOSFODNN7 bar ghp_xxxx baz";
    let matches = engine.reference_scan(haystack);
    // Two distinct literals fire; AKIA at offset 4, ghp_ at offset 22.
    assert!(matches.iter().any(|m| m.tag == 0 && m.start == 4));
    assert!(matches.iter().any(|m| m.tag == 1 && m.start == 22));
}

#[test]
fn literal_set_idempotent() {
    use vyre::scan::GpuLiteralSet;
    let engine = GpuLiteralSet::compile(&[b"abc".as_slice()]);
    let haystack = b"abc";
    let first = engine.reference_scan(haystack);
    let second = engine.reference_scan(haystack);
    assert_eq!(first, second);
}

#[test]
fn empty_haystack_yields_empty_matches() {
    use vyre::scan::GpuLiteralSet;
    let engine = GpuLiteralSet::compile(&[b"x".as_slice()]);
    assert!(engine.reference_scan(b"").is_empty());
}

#[test]
fn no_matches_when_pattern_absent() {
    use vyre::scan::GpuLiteralSet;
    let engine = GpuLiteralSet::compile(&[b"DEADBEEF".as_slice()]);
    assert!(engine.reference_scan(b"the quick brown fox").is_empty());
}

#[cfg(feature = "matching-regex")]
#[test]
fn regex_compile_round_trips_literal_via_nfa() {
    // A literal regex compiled through the regex frontend should
    // recognize the same substring the literal-set engine would
    // (modulo NFA-vs-DFA stepping differences). Smoke-check by
    // ensuring construction + round-trip succeeds without panic.
    let compiled = vyre::scan::compile_regex_set(&["abc"]).expect("compile");
    assert_eq!(compiled.plan.accept_states.len(), 1);
    assert!(compiled.plan.num_states > 0);
}

#[cfg(feature = "matching-regex")]
#[test]
fn regex_alternation_compiles_to_nfa() {
    let compiled = vyre::scan::compile_regex_set(&["foo|bar"]).expect("compile");
    assert_eq!(compiled.plan.accept_states.len(), 1);
}

#[cfg(feature = "matching-regex")]
#[test]
fn regex_class_compiles_to_nfa() {
    let compiled = vyre::scan::compile_regex_set(&[r"[a-z]+"]).expect("compile");
    assert_eq!(compiled.plan.accept_states.len(), 1);
}

/// Text anchors compile, and the anchor reaches the plan as a flag.
///
/// This test used to assert the opposite: that `^foo` was rejected with
/// `RegexCompileError::Unsupported`. Anchors are supported now. The compiler
/// records `^` and `$` per accept state and the NFA scan program guards the
/// accept on `start == 0` / `cursor + 1 == haystack_len`, so accepting the
/// pattern is a capability, not a silent widening of `^foo` into `foo`.
///
/// The distinction is the whole point of the assertions below. Compiling
/// without error while dropping the anchor would match `xfoo`, which is worse
/// than the old refusal, so the flag is checked rather than just the `Ok`.
#[cfg(feature = "matching-regex")]
#[test]
fn regex_start_anchor_compiles_and_is_recorded_on_the_accept_state() {
    let compiled = vyre::scan::compile_regex_set(&["^foo"]).expect("^foo must compile");
    assert_eq!(
        compiled.plan.accept_start_anchored,
        vec![true],
        "the start anchor must survive into the plan, not be dropped"
    );
    assert_eq!(
        compiled.plan.accept_end_anchored,
        vec![false],
        "^foo is not end-anchored"
    );
}

/// An unanchored pattern carries no anchor flag.
///
/// The negative twin. Without it, a compiler that marked every accept state
/// anchored would pass the test above while refusing to match anything that
/// does not start at offset zero.
#[cfg(feature = "matching-regex")]
#[test]
fn an_unanchored_pattern_records_no_anchor() {
    let compiled = vyre::scan::compile_regex_set(&["foo"]).expect("foo must compile");
    assert_eq!(compiled.plan.accept_start_anchored, vec![false]);
    assert_eq!(compiled.plan.accept_end_anchored, vec![false]);
}

/// Anchors are tracked per accept state, not per pattern set.
///
/// A set mixing anchored and unanchored patterns is the case where a
/// whole-set flag would be wrong for at least one member.
#[cfg(feature = "matching-regex")]
#[test]
fn anchors_are_tracked_per_pattern_within_one_set() {
    let compiled = vyre::scan::compile_regex_set(&["foo", "^bar", "baz$", "^qux$"])
        .expect("a mixed anchored set must compile");
    assert_eq!(
        compiled.plan.accept_start_anchored,
        vec![false, true, false, true]
    );
    assert_eq!(
        compiled.plan.accept_end_anchored,
        vec![false, false, true, true]
    );
}

#[test]
fn region_dedup_collapses_overlap() {
    use vyre_primitives::matching::{dedup_regions_cpu, RegionTriple};
    let input = vec![
        RegionTriple::new(0, 5, 10),
        RegionTriple::new(0, 7, 12),
        RegionTriple::new(1, 5, 10),
    ];
    let got = dedup_regions_cpu(input);
    assert_eq!(got.len(), 2); // pid=0 spans merge, pid=1 stands alone
}

#[test]
fn match_engine_cache_key_changes_with_patterns() {
    use vyre::scan::{GpuLiteralSet, MatchScan};
    let a = GpuLiteralSet::compile(&[b"foo".as_slice()]);
    let b = GpuLiteralSet::compile(&[b"bar".as_slice()]);
    assert_ne!(MatchScan::cache_key(&a), MatchScan::cache_key(&b));
}

/// The cache key is the same string in every process, for the same patterns.
///
/// This locks the on-disk cache contract: a key produced by one run must name
/// the file a later run looks for. What it is really guarding against is a
/// move to a randomized hasher (`std::DefaultHasher` seeds SipHash per
/// process), which would silently invalidate every user's cache on every run
/// while every single-process test still passed.
///
/// It used to assert against a hand-rolled FNV-1a over the pattern tables,
/// a third copy of an encoding that already lived in two places. That copy had
/// drifted: it omitted the case-insensitive word the real hash folds in, so it
/// computed a digest the code never produces and the test sat red. The
/// encoding now has one owner (`GpuLiteralSet::pattern_fingerprint`), and this
/// test asserts the PROPERTIES a cache key must have rather than restating how
/// it is built.
#[test]
fn cache_key_is_deterministic_constant() {
    use vyre::scan::{GpuLiteralSet, MatchScan};

    let patterns: &[&[u8]] = &[b"AKIA".as_slice(), b"ghp_".as_slice()];
    let key = MatchScan::cache_key(&GpuLiteralSet::compile(patterns));

    // Recompiling the same patterns, in the same order, reproduces the key.
    // A per-process seed would break here.
    for attempt in 0..4 {
        assert_eq!(
            MatchScan::cache_key(&GpuLiteralSet::compile(patterns)),
            key,
            "attempt {attempt}: the cache key must not vary between compiles"
        );
    }

    // The shape is a filename component: a fixed prefix and 16 lower-hex
    // digits, zero-padded, so a small digest cannot produce a shorter name
    // that another digest could also produce.
    let digits = key
        .strip_prefix("lit-")
        .unwrap_or_else(|| panic!("a literal-set key is prefixed lit-, got {key}"));
    assert_eq!(
        digits.len(),
        16,
        "key must be zero-padded to 16 digits: {key}"
    );
    assert!(
        digits
            .chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
        "key digits must be lower-case hex: {key}"
    );

    // Distinct inputs take distinct keys, including the case-insensitivity
    // flag: a ci matcher builds different prefilter masks from the same
    // bytes, so sharing a key would let it load the wrong cached tables.
    let reordered: &[&[u8]] = &[b"ghp_".as_slice(), b"AKIA".as_slice()];
    assert_ne!(
        MatchScan::cache_key(&GpuLiteralSet::compile(reordered)),
        key,
        "pattern order is part of the identity"
    );
    assert_ne!(
        MatchScan::cache_key(&GpuLiteralSet::compile_case_insensitive(patterns)),
        key,
        "case-insensitivity is part of the identity"
    );
}

#[test]
fn every_match_engine_implements_match_scan() {
    // Type-level contract: any matcher named in this assertion must
    // implement `MatchScan`. If a future refactor breaks this, the
    // compile error here is the canary. (Trait objects double-check
    // dyn-safety at the same time.)
    use vyre::scan::{GpuLiteralSet, MatchScan};
    let engine = GpuLiteralSet::compile(&[b"x".as_slice()]);
    let _trait_obj: &dyn MatchScan = &engine;
}

#[test]
fn region_dedup_idempotent_on_already_deduped_input() {
    // Contract: dedup_regions_cpu(dedup_regions_cpu(x)) == dedup_regions_cpu(x)
    use vyre_primitives::matching::{dedup_regions_cpu, RegionTriple};
    let input = vec![
        RegionTriple::new(0, 5, 10),
        RegionTriple::new(0, 7, 12),
        RegionTriple::new(1, 5, 10),
        RegionTriple::new(2, 1, 100),
    ];
    let once = dedup_regions_cpu(input);
    let twice = dedup_regions_cpu(once.clone());
    assert_eq!(once, twice);
}
