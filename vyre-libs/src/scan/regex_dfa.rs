//! Regex set → dense DFA GPU pipeline.
//!
//! Composes three existing primitives end-to-end:
//!
//! 1. `compile_regex_set` (this crate) - regex sources → bit-vector NFA.
//! 2. `vyre_primitives::matching::nfa_to_dfa` - NFA → dense
//!    `CompiledDfa` via subset construction.
//! 3. `build_ac_bounded_ranges_program` (this crate) - `CompiledDfa`
//!    → GPU `Program` with O(1)-per-byte AC scan semantics.
//!
//! The output [`RegexDfaPipeline`] is the regex-aware counterpart of
//! the literal-pattern `ClassicAcAutomaton`: same `Program` shape, same
//! per-byte transition cost, same `(pid, start, end)` hit triple
//! output. Consumers that already dispatch
//! `classic_ac_bounded_ranges_program` can swap their literal pattern
//! source for a regex set and pay no extra per-byte cost on the GPU.
//!
//! # When this beats `RulePipeline`
//!
//! `RulePipeline` (the bit-vector NFA scan kernel) is O(LANES²) per
//! byte regardless of pattern set size - fine for short-buffer per-
//! anchor confirmation, catastrophic for whole-buffer scans at >1 MiB.
//! `RegexDfaPipeline` is O(1) per byte but pays a state-explosion risk
//! in the subset construction. Use it when:
//!
//! * The pattern set's subset-constructed DFA stays under
//!   `max_dfa_states` (typical for high-volume detector regexes:
//!   bounded character classes + bounded repetitions).
//! * The haystack is large enough that per-byte cost dominates per-
//!   dispatch overhead.
//!
//! Fall back to `RulePipeline` when the subset construction returns
//! [`vyre_primitives::matching::NfaToDfaError::StateExplosion`].

use std::error::Error;
use std::fmt;

use vyre_foundation::ir::Program;

use vyre_primitives::matching::{nfa_to_dfa, CompiledDfa, NfaTables, NfaToDfaError};

use crate::scan::classic_ac::try_build_ac_bounded_ranges_program_ext;
use crate::scan::regex_compile::{compile_regex_set, CompiledRegexSet, RegexCompileError};

/// Ready-to-dispatch regex DFA pipeline.
///
/// `program` is built by `build_ac_bounded_ranges_program` and has the
/// same buffer contract as a literal-AC dispatch: `haystack`,
/// `transitions`, `output_offsets`, `output_records`, `pattern_lengths`,
/// `haystack_len`, `match_count`, `matches`. Upload the corresponding
/// fields of [`RegexDfaPipeline::dfa`] and [`RegexDfaPipeline::pattern_lengths`]
/// to those buffers and dispatch.
#[derive(Debug, Clone)]
pub struct RegexDfaPipeline {
    /// Dispatchable GPU program. Same shape as
    /// `classic_ac_bounded_ranges_program` output - caller wires the
    /// existing AC kernel host-side, no new dispatch path needed.
    pub program: Program,
    /// Dense DFA produced by NFA → DFA subset construction. Owns the
    /// transition / accept / output_offsets / output_records buffers
    /// the GPU program reads from.
    pub dfa: CompiledDfa,
    /// One entry per input regex; `pattern_lengths[pid]` is the longest
    /// possible match length for that pattern. Uploaded to the
    /// `pattern_lengths` buffer the AC kernel uses to derive each
    /// match's `start` offset from the accepting `end` offset.
    pub pattern_lengths: Vec<u32>,
}

/// Failures from [`build_regex_dfa_pipeline`].
#[derive(Debug)]
#[non_exhaustive]
pub enum RegexDfaError {
    /// Regex parsing or NFA construction rejected a pattern.
    Compile(RegexCompileError),
    /// Subset construction couldn't lower the NFA - typically state
    /// explosion. The caller should either raise `max_dfa_states`,
    /// shard the pattern set, or fall back to `RulePipeline`.
    Lower(NfaToDfaError),
    /// Regex/DFA metadata exceeded the GPU program's u32 ABI or host-side
    /// staging allocation budget.
    Size {
        /// Actionable sizing diagnostic.
        message: String,
    },
}

impl fmt::Display for RegexDfaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => write!(formatter, "regex NFA compile failed: {error}"),
            Self::Lower(error) => {
                write!(formatter, "NFA → DFA subset construction failed: {error}")
            }
            Self::Size { message } => write!(formatter, "regex DFA sizing failed: {message}"),
        }
    }
}

impl RegexDfaError {
    /// The canonical `REGEX_UNSUPPORTED_DIAGNOSTICS.toml` diagnostic code for
    /// this pipeline error, forwarded from the inner [`RegexCompileError`] when
    /// the failure is an unsupported construct, else `None`. Lets a consumer of
    /// the higher-level pipeline builder route on the same registry code as the
    /// low-level `compile_regex_set` path (one owner for the mapping).
    #[must_use]
    pub fn diagnostic_code(&self) -> Option<&'static str> {
        match self {
            Self::Compile(error) => error.diagnostic_code(),
            Self::Lower(_) | Self::Size { .. } => None,
        }
    }
}

impl Error for RegexDfaError {}

impl From<RegexCompileError> for RegexDfaError {
    fn from(error: RegexCompileError) -> Self {
        Self::Compile(error)
    }
}

impl From<NfaToDfaError> for RegexDfaError {
    fn from(error: NfaToDfaError) -> Self {
        Self::Lower(error)
    }
}

/// Build a [`RegexDfaPipeline`] from a list of regex sources.
///
/// `max_matches` is the per-dispatch hit-buffer cap (passed through to
/// `build_ac_bounded_ranges_program`). `max_dfa_states` is the subset-
/// construction state cap (see
/// [`vyre_primitives::matching::nfa_to_dfa`]). The default of 16k
/// states matches `DEFAULT_DFA_BUDGET_BYTES = 16 MiB` (16k × 256 × 4 B).
///
/// The match-append strategy is the default `append_match_subgroup`
/// (I.17 - one atomic per subgroup leader). On backends that can't
/// lower `subgroup_ballot` / `subgroup_shuffle` yet (currently
/// `vyre-driver-cuda`) use [`build_regex_dfa_pipeline_ext`] with
/// `use_subgroup_coalesce = false`.
///
/// # Errors
/// See [`RegexDfaError`].
pub fn build_regex_dfa_pipeline(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    build_regex_dfa_pipeline_ext(patterns, max_matches, max_dfa_states, true)
}

/// [`build_regex_dfa_pipeline`] with explicit `use_subgroup_coalesce`
/// control. Pass `false` on backends whose IR lowering cannot yet emit
/// `subgroup_ballot` + `subgroup_shuffle` - currently `vyre-driver-cuda`
/// rejects the subgroup form during canonical pre-emit lowering. Either
/// flag produces bit-identical match output; the difference is purely
/// the atomic-coalescing strategy at hit-buffer append time.
///
/// # Errors
/// See [`RegexDfaError`].
pub fn build_regex_dfa_pipeline_ext(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
    use_subgroup_coalesce: bool,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    let regex_set = compile_regex_set(patterns)?;
    finish_regex_dfa_pipeline(
        regex_set,
        patterns,
        max_matches,
        max_dfa_states,
        use_subgroup_coalesce,
    )
}

/// **Unanchored (find-anywhere)** counterpart of [`build_regex_dfa_pipeline`].
///
/// [`build_regex_dfa_pipeline`] compiles an *anchored* DFA: it only matches a
/// pattern starting at the scan origin (a secret at byte 9 of a file is missed).
/// This variant adds the implicit `.*` prefix at the **NFA-table level**: it
/// self-loops the NFA start state on every byte so the automaton stays live at
/// every position (Aho-Corasick semantics), then runs the same subset
/// construction. Match offsets are reported at the match END, exactly as the
/// literal AC path.
///
/// This is done on the bit-table, NOT by prepending `(?s).*?` to the regex
/// source: the regex-text approach explodes NFA/DFA construction for complex
/// patterns (measured OOM across a 1.7k-pattern set), while the start self-loop
/// is O(256) and leaves the rest of the automaton untouched.
///
/// # Errors
/// See [`RegexDfaError`].
pub fn build_regex_dfa_unanchored(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    let mut regex_set = compile_regex_set(patterns)?;
    add_implicit_dotstar_prefix(
        &mut regex_set.transition_table,
        regex_set.plan.num_states as usize,
    )?;
    finish_regex_dfa_pipeline(regex_set, patterns, max_matches, max_dfa_states, true)
}

/// One shard of a state-cap-sharded regex DFA set: a self-contained,
/// independently dispatchable [`RegexDfaPipeline`] plus the map from its
/// local pattern ids back to the caller's global pattern indices.
///
/// A shard's DFA reports matches with LOCAL pattern ids `0..global_pattern_ids.len()`;
/// the consumer rewrites each hit's pid to `global_pattern_ids[local_pid]` before
/// merging shard results, so the union is expressed in the caller's original
/// pattern numbering.
#[derive(Debug, Clone)]
pub struct RegexDfaShard {
    /// Dispatchable pipeline for this shard's pattern subset.
    pub pipeline: RegexDfaPipeline,
    /// `global_pattern_ids[local_pid]` = index of this shard's pattern in the
    /// original `patterns` slice passed to the shard builder.
    pub global_pattern_ids: Vec<u32>,
}

/// True when `error` is a *capacity* failure that splitting the pattern group
/// can resolve (the DFA/table was too big), as opposed to a *per-pattern*
/// failure (bad syntax, unsupported construct) that no amount of sharding fixes.
fn regex_dfa_error_is_capacity(error: &RegexDfaError) -> bool {
    match error {
        // Subset construction blew its state budget, or the metadata exceeded
        // the GPU program's ABI/staging budget: fewer patterns per shard fixes both.
        RegexDfaError::Lower(_) | RegexDfaError::Size { .. } => true,
        // The NFA itself needed more states than the per-pipeline cap.
        RegexDfaError::Compile(RegexCompileError::TooManyStates { .. }) => true,
        // Parse / Unsupported / ABI-count overflow are per-pattern: return them.
        RegexDfaError::Compile(_) => false,
    }
}

/// Recursively compile `indexed` into fitting shards, bisecting on any capacity
/// overflow. Each emitted shard is a proven-fitting DFA (its build returned Ok).
fn compile_or_split(
    indexed: &[(u32, &str)],
    max_matches: u32,
    max_dfa_states: usize,
    compile: fn(&[&str], u32, usize) -> Result<RegexDfaPipeline, RegexDfaError>,
    out: &mut Vec<RegexDfaShard>,
) -> Result<(), RegexDfaError> {
    if indexed.is_empty() {
        return Ok(());
    }
    let pats: Vec<&str> = indexed.iter().map(|(_, p)| *p).collect();
    match compile(&pats, max_matches, max_dfa_states) {
        Ok(pipeline) => {
            out.push(RegexDfaShard {
                pipeline,
                global_pattern_ids: indexed.iter().map(|(g, _)| *g).collect(),
            });
            Ok(())
        }
        // A single pattern that still overflows cannot be split further: surface
        // its error so the caller raises the cap or drops that pattern, never a
        // silent omission (Law 10).
        Err(error) if indexed.len() > 1 && regex_dfa_error_is_capacity(&error) => {
            let mid = indexed.len() / 2;
            compile_or_split(&indexed[..mid], max_matches, max_dfa_states, compile, out)?;
            compile_or_split(&indexed[mid..], max_matches, max_dfa_states, compile, out)
        }
        Err(error) => Err(error),
    }
}

/// Compile a pattern set into one-or-more [`RegexDfaShard`]s, each of whose DFA
/// fits within `max_dfa_states`: eliminating the single-DFA state cap as a hard
/// limit on how many patterns a consumer can admit in one scan phase.
///
/// Why not just size-account the NFA (`plan_shards`)? Subset construction can
/// explode the DFA far past the NFA state count, so NFA accounting cannot
/// *guarantee* a fitting DFA. This builder instead COMPILES each candidate group
/// and, on a capacity overflow, bisects and recompiles, so every emitted shard is
/// a proven-fitting DFA. A single pattern that cannot fit on its own surfaces its
/// compile error rather than being silently dropped.
///
/// The default builds **anchored** shards (mirrors [`build_regex_dfa_pipeline`]);
/// use [`build_regex_dfa_shards_unanchored`] for the find-anywhere consumer path.
///
/// # Errors
/// The first per-pattern compile error (bad syntax / unsupported construct), or a
/// capacity error for a lone pattern that cannot fit `max_dfa_states`.
pub fn build_regex_dfa_shards(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<Vec<RegexDfaShard>, RegexDfaError> {
    build_regex_dfa_shards_with(
        patterns,
        max_matches,
        max_dfa_states,
        build_regex_dfa_pipeline,
    )
}

/// Unanchored (find-anywhere) counterpart of [`build_regex_dfa_shards`], shards
/// the `.*`-prefixed DFA the megakernel batch path uses.
///
/// # Errors
/// See [`build_regex_dfa_shards`].
pub fn build_regex_dfa_shards_unanchored(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<Vec<RegexDfaShard>, RegexDfaError> {
    build_regex_dfa_shards_with(
        patterns,
        max_matches,
        max_dfa_states,
        build_regex_dfa_unanchored,
    )
}

fn build_regex_dfa_shards_with(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
    compile: fn(&[&str], u32, usize) -> Result<RegexDfaPipeline, RegexDfaError>,
) -> Result<Vec<RegexDfaShard>, RegexDfaError> {
    let mut indexed: Vec<(u32, &str)> = Vec::with_capacity(patterns.len());
    for (index, pattern) in patterns.iter().enumerate() {
        let global = u32::try_from(index).map_err(|_| {
            RegexDfaError::Compile(RegexCompileError::PatternCountOverflow {
                count: patterns.len(),
            })
        })?;
        indexed.push((global, *pattern));
    }
    let mut shards = Vec::new();
    compile_or_split(&indexed, max_matches, max_dfa_states, compile, &mut shards)?;
    Ok(shards)
}

/// Add an implicit `.*` prefix to a subgroup-NFA transition table: self-loop the
/// start state (state 0, lane 0, bit 0) on every byte so it remains active at
/// each input position. This is the standard unanchored/Aho-Corasick transform,
/// applied to the lane-major `[num_states × 256 × LANES]` table where entry
/// `trans[src*256*LANES + byte*LANES + lane]` holds the destination-state bits
/// lane `lane` owns. For `src = 0, lane = 0` over every byte we OR in bit 0.
/// Returns `Err(RegexDfaError::Size)` when `transition_table.len()` is not
/// divisible by `num_states * 256`, which would produce a silently-anchored DFA
/// (the self-loop cannot be applied, so the caller's `build_regex_dfa_unanchored`
/// would succeed but return an anchored DFA (every match at offset > 0 dropped)).
fn add_implicit_dotstar_prefix(
    transition_table: &mut [u32],
    num_states: usize,
) -> Result<(), RegexDfaError> {
    if num_states == 0 {
        return Ok(());
    }
    // LANES = table_len / (num_states * 256); derive it so this stays correct if
    // LANES_PER_SUBGROUP ever changes, with no extra feature import.
    let denom = num_states.saturating_mul(256);
    if denom == 0 || transition_table.len() % denom != 0 {
        // A malformed table means the self-loop cannot be applied. Returning
        // Ok(()) here would leave the table anchored, causing build_regex_dfa_unanchored
        // to return an anchored DFA (silently dropping every match at offset > 0).
        return Err(RegexDfaError::Size {
            message: format!(
                "add_implicit_dotstar_prefix: transition_table length {} is not divisible \
                 by num_states({num_states}) * 256 = {denom}; cannot apply unanchored \
                 start-state self-loop. Fix: ensure the NFA table is well-formed before \
                 calling build_regex_dfa_unanchored.",
                transition_table.len()
            ),
        });
    }
    let lanes = transition_table.len() / denom;
    for byte in 0..256usize {
        // src = 0, lane = 0  →  index = 0*256*lanes + byte*lanes + 0
        let idx = byte * lanes;
        if idx < transition_table.len() {
            transition_table[idx] |= 1; // bit 0 = state 0 (start) self-loop
        }
    }
    Ok(())
}

/// Shared tail of the regex→DFA build: turn a compiled NFA regex set into a
/// dispatchable [`RegexDfaPipeline`] (subset construction + AC program). Called
/// by both the anchored and unanchored entry points.
fn finish_regex_dfa_pipeline(
    regex_set: CompiledRegexSet,
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
    use_subgroup_coalesce: bool,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    // The NFA `plan` carries accept_states as `(pattern_id, match_len)`
    // tuples. nfa_to_dfa wants the pattern ids and the max len
    // separately; max_pattern_len doubles as the AC kernel's per-
    // position replay window cap.
    let mut accept_pattern_ids: Vec<u32> = Vec::new();
    reserve_regex_vec(
        &mut accept_pattern_ids,
        regex_set.plan.accept_states.len(),
        "accept pattern id table",
    )?;
    accept_pattern_ids.extend(regex_set.plan.accept_states.iter().map(|(pid, _)| *pid));
    let max_pattern_len = regex_set
        .plan
        .accept_states
        .iter()
        .map(|(_, len)| *len)
        .max()
        .unwrap_or(0);
    // pattern_lengths is per-pattern indexed; build it from the accept
    // table. A pattern with multiple accept states (alternation) takes
    // the longest match length - same convention `dfa_compile` uses.
    let pattern_count = u32::try_from(patterns.len()).map_err(|source| RegexDfaError::Size {
        message: format!(
            "pattern count {} exceeds u32 GPU buffer metadata: {source}. Fix: shard the regex set before building a DFA dispatch.",
            patterns.len()
        ),
    })?;
    let mut pattern_lengths = Vec::new();
    reserve_regex_vec(&mut pattern_lengths, patterns.len(), "pattern length table")?;
    pattern_lengths.resize(patterns.len(), 0);
    for (pid, len) in &regex_set.plan.accept_states {
        let idx = usize::try_from(*pid).map_err(|source| RegexDfaError::Size {
            message: format!(
                "accept pattern id {pid} cannot fit usize for pattern-length indexing: {source}. Fix: shard the regex set before building a DFA dispatch."
            ),
        })?;
        if idx < pattern_lengths.len() && *len > pattern_lengths[idx] {
            pattern_lengths[idx] = *len;
        }
    }

    let tables = NfaTables {
        num_states: regex_set.plan.num_states,
        transition_table: &regex_set.transition_table,
        epsilon_table: &regex_set.epsilon_table,
        accept_state_ids: &regex_set.plan.accept_state_ids,
        accept_pattern_ids: &accept_pattern_ids,
        max_pattern_len,
    };
    let dfa = nfa_to_dfa(&tables, max_dfa_states)?;

    let program = try_build_ac_bounded_ranges_program_ext(
        &dfa,
        pattern_count,
        max_matches,
        use_subgroup_coalesce,
    )
    .map_err(|message| RegexDfaError::Size { message })?;

    Ok(RegexDfaPipeline {
        program,
        dfa,
        pattern_lengths,
    })
}

fn reserve_regex_vec<T>(
    vec: &mut Vec<T>,
    requested: usize,
    label: &'static str,
) -> Result<(), RegexDfaError> {
    vyre_foundation::allocation::try_reserve_vec_to_capacity(vec, requested).map_err(|source| {
        RegexDfaError::Size {
            message: format!(
                "regex DFA {label} reservation failed for {requested} item(s): {source}. Fix: shard the regex set or lower the DFA budget before dispatch."
            ),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-pass DFA replay from the start state, the exact semantics the
    /// megakernel batch dispatcher uses (one pass per file, no per-position
    /// restart). Returns the end offsets where the DFA accepts.
    fn single_pass_accept_ends(dfa: &CompiledDfa, haystack: &[u8]) -> Vec<usize> {
        let mut state = 0u32;
        let mut ends = Vec::new();
        for (i, &b) in haystack.iter().enumerate() {
            state = dfa.transitions[state as usize * 256 + b as usize];
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
            state = dfa.transitions[state as usize * 256 + b as usize];
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
        let unanchored =
            build_regex_dfa_unanchored(&["abc"], 1024, 1024).expect("unanchored compiles");

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
        // Exact missed content from a downstream cpu_parity gate (file 120).
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
        // (pattern, exact missed match content from the cpu_parity gate run)
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
    /// case `RulePipeline` would scan via NFA bit-vector. The DFA path
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

    #[test]
    fn regex_dfa_pipeline_uses_checked_size_conversions() {
        let production = include_str!("regex_dfa.rs")
            .split("\n#[cfg(test)]\nmod tests")
            .next()
            .expect("Fix: regex DFA production section should precede tests");

        assert!(
            production.contains("RegexDfaError::Size"),
            "Fix: regex DFA sizing failures must be structured errors, not panics or unchecked casts."
        );
        assert!(
            production.contains("u32::try_from(patterns.len())"),
            "Fix: regex DFA pattern count must use checked conversion for GPU ABI metadata."
        );
        assert!(
            production.contains("usize::try_from(*pid)"),
            "Fix: regex DFA accept pattern ids must use checked host indexing conversion."
        );
        assert!(
            production.contains("try_build_ac_bounded_ranges_program_ext"),
            "Fix: regex DFA must call the fallible AC program builder."
        );
        assert!(
            !production.contains("patterns.len() as u32"),
            "Fix: regex DFA must not narrow pattern counts with unchecked casts."
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
            state = pipeline.dfa.transitions[state as usize * 256 + b as usize];
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
            state = dfa.transitions[state as usize * 256 + b as usize];
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
}
