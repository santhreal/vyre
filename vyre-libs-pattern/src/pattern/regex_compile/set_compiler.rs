//! Pattern-set compilation: parse each pattern, budget-check it, lower it, and
//! emit the shared plan plus state-major tables.

use crate::pattern::nfa::NfaPlan;

use super::compile_error::FEATURE_BACKREFERENCE;
use super::construct_budget::{
    pattern_uses_backreference, scan_constructs, ConstructScan, STATE_CAP,
};
use super::hir_lowering::build_pattern_hir;
use super::match_extent::analyze_match_extent;
use super::nfa_builder::{reserve_vec, NfaBuilder};
use super::{
    regex_construct_diagnostic_code, CompiledRegexSet, RegexCompileError, RegexConstruct,
    RegexPatternExtent, RegexReplayPolicy,
};

impl CompiledRegexSet {
    /// The `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` code a consumer routes on when
    /// it needs submatch (capture) spans this whole-match GPU engine does not
    /// prove, or `None` when the compiled set has no capture groups.
    ///
    /// This is NOT an error: the set compiled and scans correctly for the
    /// whole-match decision. The code tells a consumer that wants capture
    /// offsets to run the scalar capture verifier for these patterns.
    #[must_use]
    pub fn capture_extraction_diagnostic_code(&self) -> Option<&'static str> {
        self.captures_present
            .then_some(regex_construct_diagnostic_code(
                RegexConstruct::CaptureExtraction,
            ))
    }
}

pub(super) fn compile_regex_set_inner(
    patterns: &[&str],
    replay_policy: RegexReplayPolicy,
) -> Result<CompiledRegexSet, RegexCompileError> {
    let mut builder = NfaBuilder::new();
    let _pattern_count =
        u32::try_from(patterns.len()).map_err(|_| RegexCompileError::PatternCountOverflow {
            count: patterns.len(),
        })?;
    let mut accept_states = Vec::new();
    reserve_vec(&mut accept_states, patterns.len(), "accept state")?;
    let mut accept_state_ids = Vec::new();
    reserve_vec(&mut accept_state_ids, patterns.len(), "accept state id")?;
    let mut accept_start_anchored = Vec::new();
    reserve_vec(
        &mut accept_start_anchored,
        patterns.len(),
        "accept start-anchor flag",
    )?;
    let mut accept_end_anchored = Vec::new();
    reserve_vec(
        &mut accept_end_anchored,
        patterns.len(),
        "accept end-anchor flag",
    )?;
    let mut pattern_extents = Vec::new();
    reserve_vec(&mut pattern_extents, patterns.len(), "pattern extent")?;
    let entry = builder.fresh_state()?; // shared entry state 0
    let mut captures_present = false;

    // Use the byte-oriented parser configuration: `unicode(false)` +
    // `utf8(false)` makes `\d` / `\w` / `\s` ASCII-only, which is what
    // this primitive's byte-state automaton can represent.
    // `regex_syntax::parse(pat)` defaults to Unicode classes that
    // explode into hundreds of byte ranges and trip our `> 0x7F` guard.
    for (pid, pat) in patterns.iter().enumerate() {
        // Two-phase parse: byte-mode first (keeps `\d`/`\w`/`\s` ASCII
        // so they don't explode into hundreds of Unicode codepoint
        // ranges), then unicode-mode as a fallback when the source
        // contains a non-ASCII codepoint inside a character class
        // (e.g. homoglyph-expanded `[hнһｈ]`). The unicode-mode HIR
        // gets the same `build_class` lowering - non-ASCII members
        // expand into UTF-8 byte-sequence alternations.
        let hir = match regex_syntax::ParserBuilder::new()
            .unicode(false)
            .utf8(false)
            .build()
            .parse(pat)
        {
            Ok(h) => h,
            Err(byte_mode_err) => match regex_syntax::ParserBuilder::new()
                .unicode(true)
                .utf8(false)
                .build()
                .parse(pat)
            {
                Ok(h) => h,
                Err(_unicode_err) => {
                    // Both grammars rejected it. Classify a backreference
                    // which `regex-syntax` never supports, as its DISTINCT
                    // unsupported construct instead of a generic parse error,
                    // so a consumer can route on the registry code. Everything
                    // else keeps the byte-mode diagnostic (the narrow grammar
                    // the kernel actually supports; the unicode retry only
                    // widens the character-class path).
                    if pattern_uses_backreference(pat) {
                        return Err(RegexCompileError::Unsupported {
                            pattern_index: pid,
                            feature: FEATURE_BACKREFERENCE,
                        });
                    }
                    return Err(RegexCompileError::Parse {
                        pattern_index: pid,
                        message: format!("{byte_mode_err}"),
                    });
                }
            },
        };
        // Validate construct budgets (huge alternation / nested repeats) with a
        // DISTINCT diagnostic before lowering collapses them into a generic
        // `TooManyStates`, and note capture presence (a non-error signal).
        let mut construct_scan = ConstructScan {
            captures_present: false,
        };
        scan_constructs(&hir, pid, &mut construct_scan)?;
        captures_present |= construct_scan.captures_present;
        let extent = analyze_match_extent(&hir, pid)?;
        let (frag, anchors) = build_pattern_hir(&mut builder, &hir, pid)?;
        // Connect the shared entry to this pattern's start via epsilon.
        builder.add_epsilon(entry, frag.start);
        let pid_u32 = u32::try_from(pid).map_err(|_| RegexCompileError::PatternCountOverflow {
            count: patterns.len(),
        })?;
        let min_bytes =
            u32::try_from(extent.min).map_err(|_| RegexCompileError::MatchLengthOverflow {
                pattern_index: pid,
                len: extent.min,
            })?;
        let max_bytes = extent
            .max
            .map(|max| {
                u32::try_from(max).map_err(|_| RegexCompileError::MatchLengthOverflow {
                    pattern_index: pid,
                    len: max,
                })
            })
            .transpose()?;
        let replay_limit_bytes = match max_bytes {
            Some(max) => max,
            None => {
                let required = min_bytes.max(1);
                if replay_policy.open_ended_limit_bytes < required {
                    return Err(RegexCompileError::OpenEndedReplayLimitTooSmall {
                        pattern_index: pid,
                        minimum: required,
                        limit: replay_policy.open_ended_limit_bytes,
                    });
                }
                replay_policy.open_ended_limit_bytes
            }
        };
        accept_states.push((pid_u32, replay_limit_bytes));
        pattern_extents.push(RegexPatternExtent {
            min_bytes,
            max_bytes,
            replay_limit_bytes,
        });
        accept_state_ids.push(frag.end);
        accept_start_anchored.push(anchors.start);
        accept_end_anchored.push(anchors.end);
    }

    if builder.state_count() > STATE_CAP {
        return Err(RegexCompileError::TooManyStates {
            states: builder.state_count(),
            cap: STATE_CAP,
        });
    }

    let plan = NfaPlan {
        num_states: u32::try_from(builder.state_count()).map_err(|_| {
            RegexCompileError::TooManyStates {
                states: builder.state_count(),
                cap: STATE_CAP,
            }
        })?,
        input_len: 0,
        accept_states,
        accept_state_ids,
        accept_start_anchored,
        accept_end_anchored,
    };
    let (transition_table, epsilon_table) = builder.emit_state_major_tables()?;
    Ok(CompiledRegexSet {
        plan,
        transition_table,
        epsilon_table,
        pattern_extents,
        captures_present,
    })
}

#[cfg(test)]
mod tests {
    use super::super::construct_budget::STATE_CAP;
    use super::super::{
        build_scan_program_from_regex, compile_regex_set, RegexCompileError, LANES,
    };

    fn states_of(s: &str) -> u32 {
        compile_regex_set(&[s]).unwrap().plan.num_states
    }

    #[test]
    fn literal_compiles() {
        let r = compile_regex_set(&["abc"]).unwrap();
        // 1 entry + 1 literal-start + 3 letter states = 5
        assert_eq!(r.plan.num_states, 5);
        assert_eq!(r.plan.accept_states.len(), 1);
    }

    #[test]
    fn alternation_compiles() {
        let r = compile_regex_set(&["a|b"]).unwrap();
        // entry + fork + join + 2*(start + 1 byte) = 1+1+1+2+2 = 7
        // (exact count depends on builder; just sanity-check it's >0).
        assert!(r.plan.num_states > 0);
        assert_eq!(r.plan.accept_states.len(), 1);
    }

    #[test]
    fn class_compiles() {
        let r = compile_regex_set(&["[a-z]"]).unwrap();
        assert!(r.plan.num_states > 0);
        // Sanity: 26 lowercase bytes hit the same destination state.
        // We don't introspect the table here  -  just ensure it builds.
    }

    #[test]
    fn text_anchors_compile_to_accept_flags() {
        let r = compile_regex_set(&["^foo$"]).unwrap();
        assert_eq!(r.plan.accept_start_anchored, vec![true]);
        assert_eq!(r.plan.accept_end_anchored, vec![true]);
    }

    #[test]
    fn bounded_repetition_above_old_cap_compiles_under_state_cap() {
        let r = compile_regex_set(&["a{0,128}"]).unwrap();
        assert!(r.plan.num_states > 64);
        assert!(r.plan.num_states <= STATE_CAP as u32);
    }

    #[test]
    fn regex_compile_preserves_accept_metadata_through_checked_paths() {
        let r = compile_regex_set(&["a", "bc", "^de$"]).unwrap();

        assert_eq!(r.plan.accept_states, vec![(0, 1), (1, 2), (2, 2)]);
        assert_eq!(r.plan.accept_state_ids.len(), 3);
        assert_eq!(r.plan.accept_start_anchored, vec![false, false, true]);
        assert_eq!(r.plan.accept_end_anchored, vec![false, false, true]);
        assert_eq!(
            r.transition_table.len(),
            r.plan.num_states as usize * 256 * LANES
        );
        assert_eq!(r.epsilon_table.len(), r.plan.num_states as usize * LANES);
    }

    #[test]
    fn regex_pipeline_uses_compiled_plan_instead_of_literal_source_plan() {
        let compiled = compile_regex_set(&["a|bc"]).unwrap();
        let pipeline = build_scan_program_from_regex(&["a|bc"], "input", "hits", 64).unwrap();

        assert_eq!(pipeline.plan.num_states, compiled.plan.num_states);
        assert_eq!(
            pipeline.plan.accept_state_ids,
            compiled.plan.accept_state_ids
        );
        assert_eq!(
            pipeline.epsilon_table.iter().any(|word| *word != 0),
            compiled.epsilon_table.iter().any(|word| *word != 0)
        );
        assert_ne!(
            pipeline.plan.num_states,
            crate::pattern::nfa::compile(&["a|bc"]).num_states,
            "regex pipeline must not rebuild the scan program from literal regex source bytes"
        );
    }

    #[test]
    fn states_count_grows_with_concat() {
        let one = states_of("a");
        let two = states_of("ab");
        let three = states_of("abc");
        assert!(two > one);
        assert!(three > two);
    }

    #[test]
    fn state_cap_enforced() {
        // Build a regex that would exceed the per-pipeline state cap.
        // A literal of LANES*32+1 = 1025 chars exceeds the cap.
        let huge: String = (0..(STATE_CAP + 4)).map(|_| 'a').collect();
        let err = compile_regex_set(&[&huge]).unwrap_err();
        assert!(matches!(err, RegexCompileError::TooManyStates { .. }));
    }

    /// Capture groups must NOT become a compile error (whole-match acceleration
    /// still works); instead the compiled set reports capture presence so a
    /// consumer that needs submatch spans can route to the verifier.
    #[test]
    fn captures_compile_and_surface_the_verifier_diagnostic() {
        // A pattern with a capture group compiles (whole-match works) and reports
        // its presence + the verifier diagnostic code.
        let with_cap = compile_regex_set(&[r"(abc)def"]).expect("captures compile for whole-match");
        assert!(with_cap.captures_present, "the capture group must be noted");
        assert_eq!(
            with_cap.capture_extraction_diagnostic_code(),
            Some("VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER"),
            "a captured pattern must surface the capture-verifier code without erroring"
        );

        // A capture-free pattern compiles with no capture signal.
        let no_cap = compile_regex_set(&[r"abcdef"]).expect("plain pattern compiles");
        assert!(!no_cap.captures_present);
        assert_eq!(no_cap.capture_extraction_diagnostic_code(), None);

        // A non-capturing group is not a capture.
        let noncap = compile_regex_set(&[r"(?:abc)def"]).expect("non-capturing group compiles");
        assert!(
            !noncap.captures_present,
            "a (?:…) non-capturing group must not be flagged as a capture"
        );
    }
}
