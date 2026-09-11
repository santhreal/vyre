//! Error rendering and the feature-string to diagnostic-code mapping.

use super::{regex_construct_diagnostic_code, RegexCompileError, RegexConstruct};

impl RegexCompileError {
    /// The canonical `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` diagnostic code for
    /// this error, or `None` when the error does not correspond to a tracked
    /// unsupported-construct in that registry.
    ///
    /// A consumer routes on this code, e.g. a `*_REQUIRES_VERIFIER` code means
    /// "send this detector to the scalar verifier", while a `*_UNSUPPORTED_*`
    /// code means "reject or rewrite". It returns `Some` only for constructs the
    /// GPU-NFA frontend can distinctly identify AND that have a registry code
    /// today (ASCII lookaround assertions and over-budget Unicode classes); it
    /// invents no codes. `Parse` errors, state-budget overflow, and ABI-sizing
    /// failures return `None`: they are not registry constructs. As the frontend
    /// learns to distinguish more constructs (backreferences, captures, huge
    /// alternations, nested repeats), map them here against their registry codes.
    ///
    /// The `feature` strings matched below are this crate's own construction-site
    /// constants (not upstream parser text), so the mapping is stable; the
    /// `regex_compile_diagnostic_codes` test locks the real compile path to them.
    #[must_use]
    pub fn diagnostic_code(&self) -> Option<&'static str> {
        match self {
            // The `feature` strings are this crate's own construction-site
            // constants (below), so the feature→construct→code chain has ONE
            // owner: `regex_feature_construct` + `regex_construct_diagnostic_code`.
            Self::Unsupported { feature, .. } => {
                regex_feature_construct(feature).map(regex_construct_diagnostic_code)
            }
            _ => None,
        }
    }
}

// Feature strings carried by `RegexCompileError::Unsupported`. Defined ONCE here
// and used at every construction site AND by `regex_feature_construct`, so the
// error text and the diagnostic mapping cannot drift apart.
pub(super) const FEATURE_LOOKAROUND: &str = "non-edge lookaround assertion";
pub(super) const FEATURE_UNICODE_CLASS_CAP: &str = "unicode character class exceeded expansion cap";
pub(super) const FEATURE_BACKREFERENCE: &str = "backreference";
pub(super) const FEATURE_HUGE_ALTERNATION: &str = "huge alternation exceeds budget";
pub(super) const FEATURE_NESTED_REPEATS: &str = "nested repeat exceeds budget";

/// Map an `Unsupported { feature }` string back to its construct. Returns `None`
/// for feature strings that are real GPU-NFA limits but have no registry code
/// (e.g. the empty/byte-class expansion caps), so `diagnostic_code` invents none.
fn regex_feature_construct(feature: &str) -> Option<RegexConstruct> {
    match feature {
        FEATURE_LOOKAROUND => Some(RegexConstruct::Lookaround),
        FEATURE_UNICODE_CLASS_CAP => Some(RegexConstruct::UnicodeClassesGpu),
        FEATURE_BACKREFERENCE => Some(RegexConstruct::Backreference),
        FEATURE_HUGE_ALTERNATION => Some(RegexConstruct::HugeAlternation),
        FEATURE_NESTED_REPEATS => Some(RegexConstruct::NestedRepeats),
        _ => None,
    }
}

impl std::fmt::Display for RegexCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse {
                pattern_index,
                message,
            } => write!(
                f,
                "regex_compile: pattern {pattern_index} parse error: {message}. \
                 Fix: review the regex syntax."
            ),
            Self::Unsupported {
                pattern_index,
                feature,
            } => write!(
                f,
                "regex_compile: pattern {pattern_index} uses unsupported feature `{feature}`. \
                 Fix: rewrite the detector into supported GPU-NFA syntax or split it into GPU-compatible rules."
            ),
            Self::TooManyStates { states, cap } => write!(
                f,
                "regex_compile: NFA needs {states} states; per-pipeline cap is {cap}. \
                 Fix: split the pattern set across multiple pipelines."
            ),
            Self::PatternCountOverflow { count } => write!(
                f,
                "regex_compile: pattern count {count} exceeds u32 capacity. Fix: shard the pattern set before GPU regex compilation."
            ),
            Self::MatchLengthOverflow {
                pattern_index,
                len,
            } => write!(
                f,
                "regex_compile: pattern {pattern_index} match length {len} exceeds u32 capacity. Fix: bound or shard the regex before GPU compilation."
            ),
            Self::MatchLengthArithmeticOverflow { pattern_index } => write!(
                f,
                "regex_compile: pattern {pattern_index} match-length arithmetic overflowed host usize. Fix: reduce repetition bounds before GPU compilation."
            ),
            Self::OpenEndedReplayLimitTooSmall {
                pattern_index,
                minimum,
                limit,
            } => write!(
                f,
                "regex_compile: pattern {pattern_index} needs at least {minimum} byte(s), but the open-ended replay limit is {limit}. Fix: raise RegexReplayPolicy::open_ended_limit_bytes to at least {minimum}."
            ),
            Self::TableWordCountOverflow { table } => write!(
                f,
                "regex_compile: {table} table word count overflows host usize. Fix: shard the regex pattern set before table construction."
            ),
            Self::StorageReserveFailed {
                field,
                requested,
                message,
            } => write!(
                f,
                "regex_compile: could not reserve {requested} {field} slot(s): {message}. Fix: shard the regex pattern set before GPU compilation."
            ),
        }
    }
}

impl std::error::Error for RegexCompileError {}

#[cfg(test)]
mod tests {
    use super::super::construct_budget::MAX_ALTERNATION_ARMS;
    use super::super::{compile_regex_set, RegexCompileError};

    #[test]
    fn unsupported_regex_diagnostic_does_not_route_to_cpu_backend() {
        let err = compile_regex_set(&[r"\bsecret\b"]).unwrap_err();
        let message = err.to_string().to_ascii_lowercase();
        assert!(
            !message.contains("cpu"),
            "unsupported GPU-NFA regex diagnostics must not recommend host-side routing: {message}"
        );
        assert!(
            message.contains("gpu"),
            "unsupported GPU-NFA regex diagnostics must name the GPU-compatible rewrite contract: {message}"
        );
    }

    /// The real compile path must emit the canonical registry diagnostic code for
    /// each construct the frontend distinctly identifies, so a consumer can route
    /// precisely (verifier vs reject) instead of parsing free-text `feature`.
    #[test]
    fn regex_compile_diagnostic_codes() {
        // A non-edge lookaround (word boundary) routes to the verifier.
        let look_err = compile_regex_set(&[r"a\bc"]).expect_err("word boundary is unsupported");
        assert_eq!(
            look_err.diagnostic_code(),
            Some("VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER"),
            "non-edge lookaround must map to its verifier diagnostic code; error was: {look_err}"
        );

        // An over-cap Unicode class routes to the Unicode-mode-GPU rejection.
        let uni_err =
            compile_regex_set(&["[\u{0100}-\u{0200}]"]).expect_err("over-cap unicode class");
        assert_eq!(
            uni_err.diagnostic_code(),
            Some("VYRE_SCAN_UNSUPPORTED_UNICODE_MODE_GPU"),
            "over-cap unicode class must map to its diagnostic code; error was: {uni_err}"
        );

        // Edge anchors are SUPPORTED (Look::Start/End), so no error at all.
        assert!(
            compile_regex_set(&["^abc$"]).is_ok(),
            "start/end anchors must compile, not be flagged as unsupported lookaround"
        );

        // A pure syntax error is not a registry construct -> no code.
        let parse_err = compile_regex_set(&["("]).expect_err("unbalanced group is a parse error");
        assert_eq!(
            parse_err.diagnostic_code(),
            None,
            "a parse error must not claim a registry diagnostic code"
        );

        // W2-3: a backreference is classified as its DISTINCT construct, not a
        // generic parse error, so a consumer can route on the registry code.
        let backref_err =
            compile_regex_set(&[r"(a)\1"]).expect_err("backreferences are unsupported");
        assert_eq!(
            backref_err.diagnostic_code(),
            Some("VYRE_SCAN_UNSUPPORTED_BACKREFERENCE"),
            "a backreference must map to its distinct code, not fall back to Parse; error was: {backref_err}"
        );

        // W2-3: a huge alternation gets its own budget code instead of collapsing
        // into a generic TooManyStates.
        let huge: String = (0..(MAX_ALTERNATION_ARMS + 8))
            .map(|i| format!("v{i}"))
            .collect::<Vec<_>>()
            .join("|");
        let alt_err = compile_regex_set(&[huge.as_str()]).expect_err("over-budget alternation");
        assert_eq!(
            alt_err.diagnostic_code(),
            Some("VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET"),
            "a huge alternation must map to its budget code, not TooManyStates; error was: {alt_err}"
        );

        // W2-3: nested bounded repeats whose unroll product exceeds the budget get
        // their own code, distinct from a flat over-cap repeat.
        let nested_err =
            compile_regex_set(&[r"(?:a{40}){40}"]).expect_err("nested-repeat unroll blowup");
        assert_eq!(
            nested_err.diagnostic_code(),
            Some("VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET"),
            "nested bounded repeats must map to their budget code; error was: {nested_err}"
        );
    }

    /// W8-2 (structured diagnostics quality): every capability refusal must carry
    /// the `regex_compile:` owner prefix AND a `Fix:` clause naming the remedy
    /// the engineering standard that error messages include context and the fix.
    /// The `variants` array below is enforced COMPLETE by the exhaustive match in
    /// `assert_covers_every_variant`: adding a `RegexCompileError` variant without
    /// listing it here fails to COMPILE (the refusal cannot ship un-audited), and
    /// the per-variant assertions fail if any Display drops its owner or fix path.
    #[test]
    fn every_compile_error_variant_names_its_owner_and_fix_path() {
        let variants = [
            RegexCompileError::Parse {
                pattern_index: 0,
                message: "unclosed group".to_string(),
            },
            RegexCompileError::Unsupported {
                pattern_index: 1,
                feature: "backreference",
            },
            RegexCompileError::TooManyStates {
                states: 5_000,
                cap: 1_024,
            },
            RegexCompileError::PatternCountOverflow { count: usize::MAX },
            RegexCompileError::MatchLengthOverflow {
                pattern_index: 2,
                len: usize::MAX,
            },
            RegexCompileError::MatchLengthArithmeticOverflow { pattern_index: 3 },
            RegexCompileError::OpenEndedReplayLimitTooSmall {
                pattern_index: 4,
                minimum: 12,
                limit: 8,
            },
            RegexCompileError::TableWordCountOverflow {
                table: "transition",
            },
            RegexCompileError::StorageReserveFailed {
                field: "epsilon",
                requested: 9,
                message: "allocator refused".to_string(),
            },
        ];

        // Exhaustiveness guard: the match has no wildcard arm, so a new variant
        // breaks the build here until it is added to `variants` above and given a
        // fix path in `Display` (this test is in the defining crate, where a
        // `#[non_exhaustive]` enum can still be matched exhaustively).
        fn assert_covers_every_variant(error: &RegexCompileError) {
            match error {
                RegexCompileError::Parse { .. }
                | RegexCompileError::Unsupported { .. }
                | RegexCompileError::TooManyStates { .. }
                | RegexCompileError::PatternCountOverflow { .. }
                | RegexCompileError::MatchLengthOverflow { .. }
                | RegexCompileError::MatchLengthArithmeticOverflow { .. }
                | RegexCompileError::OpenEndedReplayLimitTooSmall { .. }
                | RegexCompileError::TableWordCountOverflow { .. }
                | RegexCompileError::StorageReserveFailed { .. } => {}
            }
        }

        for error in &variants {
            assert_covers_every_variant(error);
            let rendered = error.to_string();
            assert!(
                rendered.starts_with("regex_compile:"),
                "a RegexCompileError variant lacks the `regex_compile:` owner prefix: {rendered}"
            );
            assert!(
                rendered.contains("Fix:"),
                "a RegexCompileError variant lacks a `Fix:` remedy clause: {rendered}"
            );
        }
    }
}
