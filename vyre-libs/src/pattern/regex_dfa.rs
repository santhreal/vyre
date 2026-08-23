//! Regex set → dense DFA GPU pipeline.
//!
//! The builder composes three primitives:
//!
//! 1. [`compile_regex_set`] lowers regex sources to a bit-vector NFA.
//! 2. [`nfa_to_dfa()`] performs subset construction.
//! 3. The exact-range program replays the anchored DFA forward from each byte
//!    origin and emits `(pattern_id, origin, end)`.
//!
//! This keeps one dense transition lookup per replayed byte. The full-buffer
//! program costs `O(haystack_len * replay_limit)`, but unlike the historical
//! suffix replay it reports exact starts for bounded and open-ended
//! variable-length patterns. Use [`crate::pattern::regex_anchored_window`] when a
//! prefilter already supplies the candidate origins.
//!
//! Subset construction can exceed `max_dfa_states`. In that case, shard the
//! pattern set or use `ScanProgram`.

use std::error::Error;
use std::fmt;

use crate::pattern::{nfa_to_dfa, CompiledDfa, NfaTables, NfaToDfaError};
use vyre_foundation::composition::tag_program;
use vyre_foundation::ir::Program;

use crate::pattern::classic_ac::bounded_ranges::AcInputBindings;
use crate::pattern::classic_ac::{
    regex_exact_ranges_program, try_build_ac_bounded_ranges_program_with_subgroup_coalesce,
};
use crate::pattern::regex_compile::{
    compile_regex_set, compile_regex_set_with_policy, CompiledRegexSet, RegexCompileError,
    RegexReplayPolicy,
};

/// Ready-to-dispatch regex DFA pipeline.
///
/// Pipelines built by [`build_regex_dfa_pipeline`] and its policy variants
/// preserve the literal-AC buffer ABI (`haystack`, `transitions`,
/// `output_offsets`, `output_records`, `pattern_lengths`, `haystack_len`,
/// `match_count`, `matches`) while deriving starts from each replay origin.
/// [`build_regex_dfa_unanchored`] retains its documented end-oriented
/// single-pass DFA semantics.
#[derive(Debug, Clone)]
pub struct RegexDfaPipeline {
    /// Dispatchable whole-buffer regex program. The buffer layout remains
    /// compatible with `classic_ac_bounded_ranges_program`.
    pub program: Program,
    /// Dense DFA produced by NFA → DFA subset construction. Owns the
    /// transition / accept / output_offsets / output_records buffers
    /// the GPU program reads from.
    pub dfa: CompiledDfa,
    /// One entry per input regex. Bounded patterns store their maximum length;
    /// open-ended patterns store the finite replay budget selected by
    /// [`RegexReplayPolicy`]. The compatibility buffer remains in the program
    /// ABI, but exact starts do not derive from these values.
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
    /// shard the pattern set, or fall back to `ScanProgram`.
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
/// [`crate::pattern::nfa_to_dfa()`]). The default of 16k
/// states matches `DEFAULT_DFA_BUDGET_BYTES = 16 MiB` (16k × 256 × 4 B).
///
/// The match-append strategy is the default `append_match_subgroup`
/// (I.17 - one atomic per subgroup leader). On backends that can't
/// lower `subgroup_ballot` / `subgroup_shuffle` yet, use
/// [`build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce`] with
/// `use_subgroup_coalesce = false`.
///
/// # Errors
/// See [`RegexDfaError`].
pub fn build_regex_dfa_pipeline(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    build_regex_dfa_pipeline_with_policy(
        patterns,
        max_matches,
        max_dfa_states,
        RegexReplayPolicy::default(),
    )
}
/// Build a regex DFA pipeline with an explicit open-ended replay budget.
///
/// # Errors
/// See [`RegexDfaError`].
pub fn build_regex_dfa_pipeline_with_policy(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
    replay_policy: RegexReplayPolicy,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce(
        patterns,
        max_matches,
        max_dfa_states,
        replay_policy,
        true,
    )
}

/// [`build_regex_dfa_pipeline_with_policy`] with explicit match-append strategy.
///
/// # Errors
/// See [`RegexDfaError`].
pub fn build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
    replay_policy: RegexReplayPolicy,
    use_subgroup_coalesce: bool,
) -> Result<RegexDfaPipeline, RegexDfaError> {
    let regex_set = compile_regex_set_with_policy(patterns, replay_policy)?;
    finish_regex_dfa_pipeline(
        regex_set,
        patterns,
        max_matches,
        max_dfa_states,
        use_subgroup_coalesce,
        true,
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
    finish_regex_dfa_pipeline(
        regex_set,
        patterns,
        max_matches,
        max_dfa_states,
        true,
        false,
    )
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

/// Canonical op id for regex DFA scan.
pub const REGEX_DFA_OP_ID: &str = "vyre-libs::pattern::regex_dfa";

/// Build a regex DFA scan [`Program`] from regex sources.
///
/// Convenience wrapper over [`build_regex_dfa_pipeline`] returning the
/// compiled IR [`Program`] directly.
///
/// # Errors
/// Forwards [`RegexDfaError`].
pub fn regex_dfa_program(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<Program, RegexDfaError> {
    build_regex_dfa_pipeline(patterns, max_matches, max_dfa_states).map(|p| p.program)
}

/// Build an unanchored regex DFA scan [`Program`] from regex sources.
///
/// Convenience wrapper over [`build_regex_dfa_unanchored`] returning the
/// compiled IR [`Program`] directly.
///
/// # Errors
/// Forwards [`RegexDfaError`].
pub fn regex_dfa_unanchored_program(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<Program, RegexDfaError> {
    build_regex_dfa_unanchored(patterns, max_matches, max_dfa_states).map(|p| p.program)
}

/// Build a list of regex DFA scan [`Program`]s for sharded pattern sets.
///
/// Convenience wrapper over [`build_regex_dfa_shards`] returning the
/// compiled IR [`Program`]s directly.
///
/// # Errors
/// Forwards [`RegexDfaError`].
pub fn regex_dfa_sharded_programs(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<Vec<Program>, RegexDfaError> {
    let shards = build_regex_dfa_shards(patterns, max_matches, max_dfa_states)?;
    Ok(shards
        .into_iter()
        .map(|shard| shard.pipeline.program)
        .collect())
}

/// Build a list of unanchored regex DFA scan [`Program`]s for sharded pattern sets.
///
/// Convenience wrapper over [`build_regex_dfa_shards_unanchored`] returning the
/// compiled IR [`Program`]s directly.
///
/// # Errors
/// Forwards [`RegexDfaError`].
pub fn regex_dfa_unanchored_sharded_programs(
    patterns: &[&str],
    max_matches: u32,
    max_dfa_states: usize,
) -> Result<Vec<Program>, RegexDfaError> {
    let shards = build_regex_dfa_shards_unanchored(patterns, max_matches, max_dfa_states)?;
    Ok(shards
        .into_iter()
        .map(|shard| shard.pipeline.program)
        .collect())
}

const EXPECTED_REGEX_DFA_MATCH_COUNT_BYTES: [u8; 4] = [0; 4];
const EXPECTED_REGEX_DFA_MATCHES_BYTES: [u8; 768] = [0; 768];

/// Canonical registration fixture program for regex DFA scanning.
///
/// # Panics
///
/// Panics if the canonical regex DFA fixture pattern fails to compile.
fn canonical_regex_dfa_program() -> Program {
    regex_dfa_program(&["[a-z]+"], 64, 256).expect("Fix: canonical fixture regex DFA must compile")
}

/// Canonical registration fixture inputs for regex DFA scanning.
///
/// # Panics
///
/// Panics if the canonical regex DFA fixture pipeline fails to build.
fn canonical_regex_dfa_inputs() -> Vec<Vec<Vec<u8>>> {
    let pipeline = build_regex_dfa_pipeline(&["[a-z]+"], 64, 256)
        .expect("Fix: canonical fixture regex DFA must compile");
    vec![vec![
        vec![0u8; 64],
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.transitions),
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.output_offsets),
        vyre_primitives::wire::pack_u32_slice(&pipeline.dfa.output_records),
        vyre_primitives::wire::pack_u32_slice(&pipeline.pattern_lengths),
        vec![0u8; 4],
        vec![0u8; 4],
    ]]
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library_unconstrained(
        REGEX_DFA_OP_ID,
        canonical_regex_dfa_program,
        Some(canonical_regex_dfa_inputs),
        Some(|| vec![vec![
            EXPECTED_REGEX_DFA_MATCH_COUNT_BYTES.to_vec(),
            EXPECTED_REGEX_DFA_MATCHES_BYTES.to_vec(),
        ]]),
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
/// applied to the state-major `[num_states × 256 × LANES]` table where entry
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
    exact_starts: bool,
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

    let program = if exact_starts {
        let output_records_len =
            u32::try_from(dfa.output_records.len()).map_err(|source| RegexDfaError::Size {
                message: format!(
                    "regex DFA output record count {} exceeds u32 GPU buffer metadata: {source}. Fix: shard the pattern set or lower the DFA budget before dispatch.",
                    dfa.output_records.len()
                ),
            })?;
        regex_exact_ranges_program(
            AcInputBindings {
                haystack: "haystack",
                transitions: "transitions",
                output_offsets: "output_offsets",
                output_records: "output_records",
                pattern_lengths: "pattern_lengths",
                haystack_len: "haystack_len",
                state_count: dfa.state_count,
                output_records_len,
                pattern_count,
            },
            "match_count",
            "matches",
            max_matches,
            dfa.max_pattern_len,
            use_subgroup_coalesce,
        )
    } else {
        try_build_ac_bounded_ranges_program_with_subgroup_coalesce(
            &dfa,
            pattern_count,
            max_matches,
            use_subgroup_coalesce,
        )
        .map_err(|message| RegexDfaError::Size { message })?
    };

    Ok(RegexDfaPipeline {
        program: tag_program(REGEX_DFA_OP_ID, program),
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
#[path = "regex_dfa_tests.rs"]
mod tests;
