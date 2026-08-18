//! Regex AST → `NfaPlan` frontend.
//!
//! `nfa::compile` ships a literal-only NFA (one byte per state). This
//! module is its regex-aware counterpart: parse a regex string with
//! `regex-syntax`, lower the AST into a Thompson NFA over byte
//! transitions, emit the same `(NfaPlan, transition_table,
//! epsilon_table)` triple the literal compiler produces.
//!
//! # Why a separate module instead of widening `nfa::compile`
//!
//! The literal compiler is hot-path simple  -  every byte is a single
//! state. Bolting alternation / repetition / character classes onto it
//! would either bloat the literal path or fork the construction code.
//! The lego-block fix is a SECOND construction module that emits the
//! SAME output shape, so every downstream component (`nfa_scan`
//! Program, `scan_program::build`, `ScanProgram`) works unmodified.
//!
//! # Supported regex subset
//!
//! Targets the ~85% of vyre's expected detector regex shapes:
//!
//!   - Concatenation (default)
//!   - Alternation `a|b`
//!   - Character classes `[abc]`, `[a-z]`, `[^abc]`
//!   - Builtin escapes `\d \D \w \W \s \S` (ASCII semantics)
//!   - Bounded repetition `*`, `+`, `?`, `{n}`, `{n,m}`
//!   - Text anchors `^` and `$`
//!   - Escape literals `\.`, `\\`, `\(`, `\[`
//!
//! Explicitly NOT supported (returns `RegexCompileError::Unsupported`):
//!
//!   - Backreferences `\1` (NFA cannot represent)
//!   - Word-boundary and line-boundary lookarounds
//!   - Unicode character classes outside the ASCII range

mod capture_mode;
mod char_class;
mod compile_error;
mod construct_budget;
mod hir_lowering;
mod match_extent;
mod nfa_builder;
mod set_compiler;

use crate::pattern::nfa::NfaPlan;
use vyre_foundation::ir::Program;

use self::construct_budget::STATE_CAP;
use self::set_compiler::compile_regex_set_inner;

const LANES: usize = crate::nfa::subgroup_nfa::LANES_PER_SUBGROUP;

/// Default whole-match replay budget for a pattern containing `*`, `+`, or
/// `{n,}`. Open-ended regexes have no finite maximum, so accelerator extraction
/// is exact only through this many bytes unless the caller supplies a larger
/// [`RegexReplayPolicy`].
pub const DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES: u32 = 4096;

/// Runtime work bound for open-ended regex extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegexReplayPolicy {
    /// Maximum bytes replayed from one candidate origin for an open-ended
    /// pattern. It must cover the pattern's finite minimum and be nonzero.
    pub open_ended_limit_bytes: u32,
}

impl Default for RegexReplayPolicy {
    fn default() -> Self {
        Self {
            open_ended_limit_bytes: DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
        }
    }
}

/// Static byte extent and effective replay limit for one compiled pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegexPatternExtent {
    /// Smallest byte length the pattern can accept.
    pub min_bytes: u32,
    /// Largest accepted byte length, or `None` for an open-ended pattern.
    pub max_bytes: Option<u32>,
    /// Finite byte budget used by accelerator replay.
    pub replay_limit_bytes: u32,
}

/// Capture output mode a consumer requests for a regex scan, wiring
/// `docs/optimization/REGEX_CAPTURE_MODE_CONTRACTS.toml` to code.
///
/// A consumer routes on this instead of parsing the TOML: [`accelerator_eligible`]
/// says whether the GPU DFA/AC path can serve the request directly, and
/// [`verifier_required`] says whether a scalar (CPU-semantics) verifier must own
/// the output. The three whole-match modes run entirely on the accelerator; the
/// three group-extraction modes need the verifier (the byte-DFA has no capture
/// stack). Keeping this a typed enum with one `contract_row` owner means the
/// routing decision has one home and cannot silently disagree with the contract.
///
/// [`accelerator_eligible`]: CaptureMode::accelerator_eligible
/// [`verifier_required`]: CaptureMode::verifier_required
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptureMode {
    /// Whole match only, no spans (`whole_match_only`). Accelerator path.
    NonCapture,
    /// Match count per pattern (`match_count_per_pattern`). Accelerator path.
    Count,
    /// Whole-match `(start, end)` span (`whole_match_span`). Accelerator path.
    Span,
    /// Named group span records (`named_group_span_records`). Verifier-bound;
    /// unmatched group → null.
    NamedCapture,
    /// Ordered list of spans for a repeated group (`ordered_group_span_list`).
    /// Verifier-bound; an empty repeat yields an empty list.
    RepeatedCapture,
    /// Row × group value table (`row_group_value_table`). Verifier-bound;
    /// unmatched group → null.
    GroupExtraction,
}

/// Static per-mode contract row mirroring one `[[mode]]` entry of
/// `REGEX_CAPTURE_MODE_CONTRACTS.toml`. The [`CaptureMode::contract_row`] table
/// is the single code-side owner; `regex_capture_mode_contracts.rs` locks it to
/// the TOML so the two cannot drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureModeContract {
    /// Stable `mode_id` string, identical to the TOML.
    pub mode_id: &'static str,
    /// `output_shape` string, identical to the TOML.
    pub output_shape: &'static str,
    /// Whether the GPU accelerator path can serve this mode directly.
    pub accelerator_eligible: bool,
    /// Whether a scalar verifier must own the output for this mode.
    pub verifier_required: bool,
    /// `null_policy` string, identical to the TOML.
    pub null_policy: &'static str,
}

/// Failure modes for [`compile_regex_set`]. Variants are non-exhaustive
/// so future regex features can be added without a breaking change.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RegexCompileError {
    /// `regex-syntax` rejected the pattern. Carries the parser's own
    /// diagnostic so callers can forward it.
    Parse {
        /// Index into the input slice that failed to parse.
        pattern_index: usize,
        /// `regex-syntax`'s error message.
        message: String,
    },
    /// The pattern uses a regex feature this GPU NFA frontend does not
    /// support. Callers must reject or rewrite the detector into supported
    /// GPU-NFA rule data.
    Unsupported {
        /// Index into the input slice that uses the unsupported feature.
        pattern_index: usize,
        /// One-line description of what isn't supported (e.g. "anchors").
        feature: &'static str,
    },
    /// The compiled NFA exceeds `LANES * 32` states (the state-major
    /// transition table addresses states with one bit per lane).
    /// Mitigation: split the pattern set across multiple pipelines.
    TooManyStates {
        /// Number of states the AST would have produced.
        states: usize,
        /// Per-pipeline maximum.
        cap: usize,
    },
    /// Pattern count does not fit the GPU ABI's `u32` pattern id field.
    PatternCountOverflow {
        /// Number of patterns supplied by the caller.
        count: usize,
    },
    /// A compiled regex match length does not fit the `u32` match ABI.
    MatchLengthOverflow {
        /// Index into the input slice that produced the oversized match.
        pattern_index: usize,
        /// Longest matched byte length for the pattern.
        len: usize,
    },
    /// Match-extent arithmetic overflowed before it could be represented.
    MatchLengthArithmeticOverflow {
        /// Index into the input slice that overflowed.
        pattern_index: usize,
    },
    /// An open-ended pattern's finite replay budget cannot reach its minimum.
    OpenEndedReplayLimitTooSmall {
        /// Index into the input slice that cannot fit the policy.
        pattern_index: usize,
        /// Smallest byte length the pattern can accept.
        minimum: u32,
        /// Configured open-ended replay budget.
        limit: u32,
    },
    /// Transition or epsilon table word count overflowed host `usize`.
    TableWordCountOverflow {
        /// Table being built.
        table: &'static str,
    },
    /// Compiler staging allocation failed.
    StorageReserveFailed {
        /// Scratch vector being reserved.
        field: &'static str,
        /// Requested target capacity.
        requested: usize,
        /// Allocator failure details.
        message: String,
    },
}

/// A regex construct vyre's GPU-NFA frontend distinctly detects AND that has a
/// canonical `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` diagnostic code.
///
/// This enum is the ONE owner of the construct→code mapping. Both
/// [`RegexCompileError::diagnostic_code`] (for the constructs that surface as a
/// compile error) and [`CompiledRegexSet::capture_extraction_diagnostic_code`]
/// (for the non-error capture case) route through
/// [`regex_construct_diagnostic_code`], so a code string is never written twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RegexConstruct {
    /// `\1` / `\k<name>` / `(?P=name)`: not a regular language; rejected.
    Backreference,
    /// A non-edge lookaround assertion (`\b`, `(?=…)`, …) (verifier-routed).
    Lookaround,
    /// A Unicode character class over the byte-mode GPU expansion budget.
    UnicodeClassesGpu,
    /// A capture group whose submatch spans a whole-match engine cannot prove.
    /// NOT a compile error (whole-match still accelerates; verifier-routed).
    CaptureExtraction,
    /// An alternation with more arms than the state budget can ever hold.
    HugeAlternation,
    /// Nested bounded repeats whose unroll product exceeds the state budget.
    NestedRepeats,
}

impl RegexConstruct {
    /// Every construct this enum names.
    ///
    /// The enum is `#[non_exhaustive]`, so a consumer outside this crate cannot
    /// match it exhaustively and cannot discover a variant added later. This
    /// slice is how a consumer enumerates the construct space, and
    /// `all_lists_every_construct` below keeps it closed: it matches
    /// exhaustively, which only this crate may do, so a new variant that is not
    /// listed here fails to compile.
    pub const ALL: &'static [Self] = &[
        Self::Backreference,
        Self::Lookaround,
        Self::UnicodeClassesGpu,
        Self::CaptureExtraction,
        Self::HugeAlternation,
        Self::NestedRepeats,
    ];
}

/// The canonical `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` code for a construct, the
/// single source of truth for these strings.
#[must_use]
pub fn regex_construct_diagnostic_code(construct: RegexConstruct) -> &'static str {
    match construct {
        RegexConstruct::Backreference => "VYRE_SCAN_UNSUPPORTED_BACKREFERENCE",
        RegexConstruct::Lookaround => "VYRE_SCAN_APPROXIMATED_LOOKAROUND_REQUIRES_VERIFIER",
        RegexConstruct::UnicodeClassesGpu => "VYRE_SCAN_UNSUPPORTED_UNICODE_MODE_GPU",
        RegexConstruct::CaptureExtraction => "VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER",
        RegexConstruct::HugeAlternation => "VYRE_SCAN_UNSUPPORTED_HUGE_ALTERNATION_BUDGET",
        RegexConstruct::NestedRepeats => "VYRE_SCAN_UNSUPPORTED_NESTED_REPEAT_BUDGET",
    }
}

/// Output of [`compile_regex_set`]  -  same triple shape as the literal
/// `nfa::compile` returns plus the GPU side-tables `nfa::nfa_scan`
/// expects, so consumers can plug this into `ScanProgram` without
/// changing the dispatch path.
#[derive(Debug, Clone)]
pub struct CompiledRegexSet {
    /// State graph + accept-state metadata.
    pub plan: NfaPlan,
    /// Lane-major byte→bitset transition table:
    /// `[num_states × 256 × LANES_PER_SUBGROUP]` u32s.
    pub transition_table: Vec<u32>,
    /// Lane-major epsilon (free) transition table:
    /// `[num_states × LANES_PER_SUBGROUP]` u32s.
    pub epsilon_table: Vec<u32>,
    /// Per-pattern accepted extent and finite accelerator replay budget.
    ///
    /// `max_bytes == None` makes the truncation boundary explicit for an
    /// open-ended regex rather than presenting its minimum as a false maximum.
    pub pattern_extents: Vec<RegexPatternExtent>,
    /// `true` when at least one source pattern contained a capture group.
    ///
    /// The GPU NFA is a WHOLE-MATCH multimatch engine: it accelerates the
    /// match decision but does NOT prove submatch (capture) spans, capture
    /// groups are stripped during lowering (whole-match still compiles and
    /// runs correctly). A consumer that needs submatch offsets must route
    /// these patterns to the scalar verifier; this flag is the distinct signal
    /// for the `VYRE_SCAN_CAPTURE_EXTRACTION_REQUIRES_VERIFIER` diagnostic
    /// (see [`regex_construct_diagnostic_code`]) WITHOUT rejecting the pattern
    /// (making captures a compile error would regress whole-match acceleration).
    pub captures_present: bool,
}

/// Compile with the default bounded replay policy for open-ended patterns.
///
/// # Errors
/// See [`RegexCompileError`].
pub fn compile_regex_set(patterns: &[&str]) -> Result<CompiledRegexSet, RegexCompileError> {
    compile_regex_set_with_policy(patterns, RegexReplayPolicy::default())
}

/// Compile with an explicit finite replay budget for open-ended patterns.
///
/// # Errors
/// See [`RegexCompileError`].
pub fn compile_regex_set_with_policy(
    patterns: &[&str],
    replay_policy: RegexReplayPolicy,
) -> Result<CompiledRegexSet, RegexCompileError> {
    compile_regex_set_inner(patterns, replay_policy)
}

/// Build a [`crate::pattern::ScanProgram`] directly from regex
/// sources. Convenience for consumers who don't need the
/// `CompiledRegexSet` intermediate. `input_len` matches the contract
/// of `scan_program::build` (haystack byte count the dispatch will scan).
///
/// # Errors
/// Forwards [`RegexCompileError`].
pub fn build_scan_program_from_regex(
    patterns: &[&str],
    input_buf: &str,
    hit_buf: &str,
    input_len: u32,
) -> Result<crate::pattern::ScanProgram, RegexCompileError> {
    let compiled = compile_regex_set(patterns)?;
    let has_epsilon = compiled.epsilon_table.iter().any(|word| *word != 0);
    let program = crate::pattern::nfa::nfa_scan_with_plan(
        &compiled.plan,
        has_epsilon,
        input_buf,
        hit_buf,
        input_len,
    )
    .map_err(|_| RegexCompileError::TooManyStates {
        states: compiled.plan.num_states as usize,
        cap: STATE_CAP,
    })?;
    Ok(crate::pattern::ScanProgram {
        program,
        transition_table: compiled.transition_table,
        epsilon_table: compiled.epsilon_table,
        plan: compiled.plan.for_input_len(input_len),
    })
}

/// Build a regex scan [`Program`] directly from regex sources.
///
/// Convenience wrapper over [`build_scan_program_from_regex`] returning the
/// compiled IR [`Program`] directly.
///
/// # Errors
/// Forwards [`RegexCompileError`].
pub fn regex_scan_program(
    patterns: &[&str],
    input_buf: &str,
    hit_buf: &str,
    input_len: u32,
) -> Result<Program, RegexCompileError> {
    build_scan_program_from_regex(patterns, input_buf, hit_buf, input_len).map(|s| s.program)
}

static EXPECTED_REGEX_SCAN_HITS_BYTES: [u8; 120_004] = [0; 120_004];

/// Canonical registration fixture program for regex scan.
///
/// # Panics
///
/// Panics if the canonical regex scan fixture pattern fails to compile.
fn canonical_regex_scan_program() -> Program {
    regex_scan_program(&["[a-z]+"], "input", "hits", 64)
        .expect("Fix: canonical fixture regex scan must compile")
}

/// Canonical registration fixture inputs for regex scan.
///
/// # Panics
///
/// Panics if the canonical regex scan fixture pattern fails to compile.
fn canonical_regex_scan_inputs() -> Vec<Vec<Vec<u8>>> {
    let compiled =
        compile_regex_set(&["[a-z]+"]).expect("Fix: canonical fixture regex scan must compile");
    vec![vec![
        vec![0u8; 64],
        vyre_primitives::wire::pack_u32_slice(&compiled.transition_table),
        vyre_primitives::wire::pack_u32_slice(&compiled.epsilon_table),
        EXPECTED_REGEX_SCAN_HITS_BYTES.to_vec(),
        vec![0u8; 4],
        vec![0u8; 4],
    ]]
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        crate::pattern::nfa::REGEX_SCAN_OP_ID,
        canonical_regex_scan_program,
        Some(canonical_regex_scan_inputs),
        Some(|| vec![vec![EXPECTED_REGEX_SCAN_HITS_BYTES.to_vec()]]),
    )
}

#[cfg(test)]
mod tests {
    use super::{regex_construct_diagnostic_code, RegexConstruct};
    use std::collections::BTreeSet;

    /// WHY: conformance accepts registered bytes as proof for every backend row, so the regex
    /// scan fixture must equal an independent reference execution rather than merely be present.
    #[test]
    fn registered_regex_scan_expected_bytes_match_reference_execution() {
        let compiled = super::compile_regex_set(&["[a-z]+"])
            .expect("Fix: canonical fixture regex scan must compile");
        let program = super::regex_scan_program(&["[a-z]+"], "input", "hits", 64)
            .expect("Fix: canonical fixture regex scan must compile");
        let inputs = [
            vec![0u8; 64],
            vyre_primitives::wire::pack_u32_slice(&compiled.transition_table),
            vyre_primitives::wire::pack_u32_slice(&compiled.epsilon_table),
            super::EXPECTED_REGEX_SCAN_HITS_BYTES.to_vec(),
            vec![0u8; 4],
            vec![0u8; 4],
        ];
        let values = inputs
            .iter()
            .map(|bytes| vyre_reference::value::Value::Bytes(bytes.as_slice().into()))
            .collect::<Vec<_>>();
        let actual = vyre_reference::reference_eval(&program, &values)
            .expect("Fix: canonical regex scan fixture must execute in the reference interpreter")
            .into_iter()
            .map(|value| value.to_bytes())
            .collect::<Vec<_>>();

        assert_eq!(
            actual,
            vec![super::EXPECTED_REGEX_SCAN_HITS_BYTES.to_vec()],
            "Fix: registered regex scan expected bytes must equal the reference result"
        );
    }

    /// `RegexConstruct::ALL` is what a consumer outside this crate enumerates,
    /// and `#[non_exhaustive]` means such a consumer cannot notice a variant the
    /// slice omits. The match below is exhaustive, which only this crate may
    /// write, so a new variant makes this file fail to compile until it is
    /// listed. Anything weaker lets a construct be added with no diagnostic code
    /// and no gate that sees it.
    #[test]
    fn all_lists_every_construct() {
        for construct in RegexConstruct::ALL {
            match construct {
                RegexConstruct::Backreference
                | RegexConstruct::Lookaround
                | RegexConstruct::UnicodeClassesGpu
                | RegexConstruct::CaptureExtraction
                | RegexConstruct::HugeAlternation
                | RegexConstruct::NestedRepeats => {}
            }
        }
        assert_eq!(
            RegexConstruct::ALL.len(),
            6,
            "Fix: a construct was added to RegexConstruct without listing it in ALL, \
             so every consumer that enumerates the construct space silently skips it."
        );
    }

    /// A code shared by two constructs makes a diagnostic ambiguous: the reader
    /// of `VYRE_SCAN_...` cannot tell which construct refused the pattern.
    #[test]
    fn every_construct_has_a_distinct_diagnostic_code() {
        let mut codes = BTreeSet::new();
        for construct in RegexConstruct::ALL {
            let code = regex_construct_diagnostic_code(*construct);
            assert!(
                code.starts_with("VYRE_SCAN_"),
                "Fix: {construct:?} maps to `{code}`, which is outside the VYRE_SCAN_ namespace."
            );
            assert!(
                codes.insert(code),
                "Fix: `{code}` is the diagnostic code of two constructs, so the code no longer \
                 names which construct refused the pattern."
            );
        }
    }

    #[test]
    fn regex_scan_program_builds_valid_ir() {
        let program = super::regex_scan_program(&["[a-z]+", "token=[0-9]+"], "input", "hits", 128)
            .expect("valid regex scan program must build");
        assert!(!program.buffers().is_empty());
        assert!(!program.entry().is_empty());
    }
}
