//! Byte and text scan helpers  -  substring search, DFA / Aho–Corasick. Used
//! as components inside full `vyre::Program` values (decode, graph, heuristics).
//!
//! Sub-dialects:
//! - `substring`  -  brute-force single-string scanner
//! - `dfa`  -  DFA compiler + Aho-Corasick multi-string scanner
//!
//! Flat re-exports preserved for back-compat.
//!
//! # API index
//!
//! Every public surface in this module is enumerated in `API_INDEX`
//! as a stable `(name, kind, feature)` triple. Consumers that need to
//! discover the engine surface programmatically  -  consumer engine listings,
//! the conformance harness's coverage check, the cargo-doc completeness test
//! below  -  read this single const instead
//! of grepping the module tree.

/// Stable index of public exports under `vyre_libs::scan`. Each
/// entry is a `(symbol, kind, feature_gate)` triple. `feature_gate`
/// is `None` for unconditional exports and `Some("flag-name")` for
/// items behind a Cargo feature.
///
/// Keep this in sync with the `pub use` lines below. The
/// `api_index_covers_every_export` test in `tests/api_index.rs`
/// verifies that every name in `API_INDEX` resolves to a real
/// import path so a refactor that removes or renames a public symbol
/// fails CI loudly instead of silently leaving the index stale.
pub const API_INDEX: &[(&str, ApiKind, Option<&str>)] = &[
    // Unconditional dispatch primitives.
    ("byte_scan_dispatch_config", ApiKind::Function, None),
    ("candidate_start_dispatch_config", ApiKind::Function, None),
    ("haystack_len_u32", ApiKind::Function, None),
    ("pack_haystack_u32", ApiKind::Function, None),
    ("pack_u32_slice", ApiKind::Function, None),
    ("scan_guard", ApiKind::Function, None),
    ("u32_words_as_le_bytes", ApiKind::Function, None),
    ("unpack_match_triples", ApiKind::Function, None),
    ("DEFAULT_MAX_SCAN_BYTES", ApiKind::Const, None),
    // Engine traits + helpers.
    ("MatchScan", ApiKind::Trait, None),
    ("MatchEngineCache", ApiKind::Trait, None),
    ("ScanResult", ApiKind::Struct, None),
    ("cached_load_or_compile", ApiKind::Function, None),
    ("engine_cache_path", ApiKind::Function, None),
    // Hit-buffer helpers.
    ("compact_hits", ApiKind::Function, None),
    ("compact_hits_with_layout", ApiKind::Function, None),
    ("emit_hit", ApiKind::Function, None),
    ("emit_hit_then_compact", ApiKind::Function, None),
    ("emit_hit_then_compact_with_layout", ApiKind::Function, None),
    ("emit_hit_with_layout", ApiKind::Function, None),
    ("HIT_BUFFER_LIVE_LENGTH", ApiKind::Const, None),
    ("HIT_BUFFER_OVERFLOW_COUNT", ApiKind::Const, None),
    // Literal-set engine  -  unconditional.
    ("GpuLiteralSet", ApiKind::Struct, None),
    ("LiteralMatch", ApiKind::TypeAlias, None),
    ("LiteralSetPreparedCount", ApiKind::Struct, None),
    ("LiteralSetPreparedPresenceByRegion", ApiKind::Struct, None),
    ("LiteralSetPreparedScan", ApiKind::Struct, None),
    ("LiteralSetScanScratch", ApiKind::Struct, None),
    ("LiteralSetWireError", ApiKind::Enum, None),
    ("PendingFusedRegion", ApiKind::Struct, None),
    ("PendingMatches", ApiKind::Struct, None),
    ("PendingPresence", ApiKind::Struct, None),
    ("PendingPresenceByRegion", ApiKind::Struct, None),
    ("ScanAllTimed", ApiKind::Struct, None),
    ("ResidentLiteralScan", ApiKind::Struct, None),
    ("ResidentFusedRegionScan", ApiKind::Struct, None),
    ("ResidentPresencePipeline", ApiKind::Struct, None),
    ("scan_paged_fused", ApiKind::Function, None),
    ("scan_paged_fused_timed", ApiKind::Function, None),
    ("scan_paged_fused_async", ApiKind::Function, None),
    ("scan_sharded_fused", ApiKind::Function, None),
    ("scan_sharded_fused_weighted", ApiKind::Function, None),
    ("scan_sharded_fused_timed", ApiKind::Function, None),
    ("scan_pattern_sharded", ApiKind::Function, None),
    ("PatternShard", ApiKind::Struct, None),
    ("ShardTiming", ApiKind::Struct, None),
    ("ShardedScanTiming", ApiKind::Struct, None),
    ("scan_paths_paged", ApiKind::Function, None),
    ("scan_paths_paged_prefetched", ApiKind::Function, None),
    ("PagedScanResult", ApiKind::Struct, None),
    ("PagedScanTiming", ApiKind::Struct, None),
    ("GlobalMatch", ApiKind::Struct, None),
    ("LITERAL_SET_COUNT_RESOURCE_INDEX", ApiKind::Const, None),
    (
        "LITERAL_SET_PRESENCE_BY_REGION_OUTPUT_RESOURCE_INDEX",
        ApiKind::Const,
        None,
    ),
    (
        "LITERAL_SET_COUNT_RESET_RESOURCE_INDICES",
        ApiKind::Const,
        None,
    ),
    (
        "LITERAL_SET_COUNT_SCAN_RESOURCE_INDICES",
        ApiKind::Const,
        None,
    ),
    (
        "LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX",
        ApiKind::Const,
        None,
    ),
    ("LITERAL_SET_MATCHES_RESOURCE_INDEX", ApiKind::Const, None),
    ("LITERAL_SET_RESET_RESOURCE_INDICES", ApiKind::Const, None),
    ("LITERAL_SET_SCAN_RESOURCE_INDICES", ApiKind::Const, None),
    // Cross-program fusion (re-exported from vyre-foundation).
    ("fuse_programs", ApiKind::Function, None),
    ("fuse_programs_vec", ApiKind::Function, None),
    ("FusionError", ApiKind::Enum, None),
    // matching-substring.
    (
        "substring_search",
        ApiKind::Function,
        Some("matching-substring"),
    ),
    // matching-dfa.
    ("aho_corasick", ApiKind::Function, Some("matching-dfa")),
    ("dfa_compile", ApiKind::Function, Some("matching-dfa")),
    (
        "dfa_compile_with_budget",
        ApiKind::Function,
        Some("matching-dfa"),
    ),
    ("CompiledDfa", ApiKind::Struct, Some("matching-dfa")),
    ("DfaCompileError", ApiKind::Enum, Some("matching-dfa")),
    (
        "DEFAULT_DFA_BUDGET_BYTES",
        ApiKind::Const,
        Some("matching-dfa"),
    ),
    ("DirectGpuScanner", ApiKind::Struct, Some("matching-dfa")),
    // matching-nfa.
    (
        "build_rule_pipeline",
        ApiKind::Function,
        Some("matching-nfa"),
    ),
    ("PipelineWireError", ApiKind::Enum, Some("matching-nfa")),
    ("RulePipeline", ApiKind::Struct, Some("matching-nfa")),
    (
        "ResidentRulePipeline",
        ApiKind::Struct,
        Some("matching-nfa"),
    ),
    // matching-regex.
    (
        "build_rule_pipeline_from_regex",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "compile_regex_set",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    ("CompiledRegexSet", ApiKind::Struct, Some("matching-regex")),
    ("RegexCompileError", ApiKind::Enum, Some("matching-regex")),
    ("RegexConstruct", ApiKind::Enum, Some("matching-regex")),
    (
        "regex_construct_diagnostic_code",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    ("CaptureMode", ApiKind::Enum, Some("matching-regex")),
    (
        "CaptureModeContract",
        ApiKind::Struct,
        Some("matching-regex"),
    ),
    // regex-set → dense DFA → existing AC kernel composition.
    // Gated on both matching-regex (for compile_regex_set) and
    // matching-dfa (for build_ac_bounded_ranges_program). The single
    // entry is reported under matching-regex so the existing index
    // tooling that filters by one feature still finds it.
    (
        "build_regex_dfa_pipeline",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "build_regex_dfa_unanchored",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "build_regex_dfa_shards",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "build_regex_dfa_shards_unanchored",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    ("RegexDfaPipeline", ApiKind::Struct, Some("matching-regex")),
    ("RegexDfaShard", ApiKind::Struct, Some("matching-regex")),
    ("RegexDfaError", ApiKind::Enum, Some("matching-regex")),
    (
        "AnchoredWindowValidator",
        ApiKind::Struct,
        Some("matching-regex"),
    ),
    (
        "anchored_window_extract_program",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "ANCHORED_WINDOW_MATCH_COUNT_BINDING",
        ApiKind::Const,
        Some("matching-regex"),
    ),
    (
        "ANCHORED_WINDOW_MATCHES_BINDING",
        ApiKind::Const,
        Some("matching-regex"),
    ),
    (
        "regex_admission_by_region_program",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "regex_admission_by_region_reference",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "regex_admission_presence_words",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    ("region_of", ApiKind::Function, Some("matching-regex")),
    (
        "fused_region_evidence_program",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "fused_region_evidence_reference",
        ApiKind::Function,
        Some("matching-regex"),
    ),
    (
        "FusedRegionEvidence",
        ApiKind::Struct,
        Some("matching-regex"),
    ),
    (
        "FUSED_EVIDENCE_PRESENCE_BINDING",
        ApiKind::Const,
        Some("matching-regex"),
    ),
    (
        "FUSED_EVIDENCE_MATCHES_BINDING",
        ApiKind::Const,
        Some("matching-regex"),
    ),
    (
        "RegionEvidencePipeline",
        ApiKind::Struct,
        Some("matching-regex"),
    ),
    ("RegionEvidenceError", ApiKind::Enum, Some("matching-regex")),
];

/// Item-kind tag for entries in `API_INDEX`. Coarse on purpose  -
/// the goal is "what's the symbol shape?" not full reflection.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ApiKind {
    /// Free function or method exported at module root.
    Function,
    /// `pub struct` or unit struct.
    Struct,
    /// `pub enum`.
    Enum,
    /// `pub trait`.
    Trait,
    /// `pub const`.
    Const,
    /// `pub type` alias.
    TypeAlias,
}

pub mod builders;
pub mod hit_buffer;

/// Shared GPU dispatch primitives for matching engines.
///
/// Centralises haystack-packing, length validation, dispatch geometry,
/// and match-triple unpacking so every new matcher (literal-set,
/// regex pipeline, future taint scan) reuses the same byte-level
/// plumbing instead of re-implementing it.
pub mod dispatch_io;

/// Common scan + cache traits for every matcher in this crate.
///
/// Engines implement `MatchScan` (object-safe) and `MatchEngineCache`
/// (typed errors). Consumers use `cached_load_or_compile` to wire on-
/// disk caches generically  -  the per-engine cache wiring scan consumer
/// previously hand-rolled is now a one-line call.
pub mod engine;
pub use dispatch_io::{
    byte_scan_dispatch_config, candidate_start_dispatch_config, haystack_len_u32,
    pack_haystack_u32, pack_u32_slice, scan_guard, u32_words_as_le_bytes, unpack_match_triples,
    DEFAULT_MAX_SCAN_BYTES,
};
pub use engine::{
    cache_path as engine_cache_path, cached_load_or_compile, MatchEngineCache, MatchScan,
    ScanResult,
};

#[cfg(feature = "matching-substring")]
pub mod substring;

#[cfg(feature = "matching-dfa")]
pub mod dfa;

/// Classic Aho-Corasick with precomputed flat `output_links`.
/// Scans in O(matches) per position, not O(states × n).
#[cfg(feature = "matching-dfa")]
pub mod classic_ac;

/// Subgroup-cooperative NFA scan helper (G1). Composes
/// `vyre_primitives::nfa::subgroup_nfa::nfa_step` into a multi-byte /
/// multi-pattern scan. Feature-gated behind `matching-nfa` so consumers
/// opt in when they need NFAs up to 1024 states with subgroup-shuffle
/// epsilon closure.
#[cfg(feature = "matching-nfa")]
pub mod nfa;

pub mod literal_set;

/// W2-4 paged-corpus scanning: scan a corpus larger than one resident window as a
/// sequence of resident-window dispatches with stable global region ids and u64
/// global positions, identical to a single-shot scan of the concatenated corpus.
/// Public entry [`paged_corpus::scan_paged_fused`]; the window planner is a private
/// helper.
pub mod paged_corpus;

/// Match post-processing: dedup, entropy, and confidence in one reference pass.
pub mod post_process;

/// Generic engine + post-processor pipeline. Pairs any `MatchScan`
/// implementer with the canonical post-processing contract.
pub mod pipeline;

/// Canonical literal/regex/haystack fixture corpus shared by every
/// integration test in this crate. Public when the consumer opts into
/// `feature = "test-fixtures"`; always available inside the in-tree
/// test compilation.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod test_fixtures;

#[cfg(feature = "matching-dfa")]
pub mod direct_gpu;

/// Mega-scan integrator (G-stack). The single authoritative entry
/// point that produces one `RulePipeline` object program-analysis
/// consumers dispatch. Currently only G1 (subgroup-cooperative NFA
/// scan) is wired end-to-end; G2-G10 (rule fusion, decode-scan
/// handoff, speculative commit, persistent-engine work items,
/// content-hash cache key, adaptive CSR/dense traversal, CHD perfect
/// hash, differential scan file selection) are planned composition
/// hooks that land here as they are implemented.
#[cfg(feature = "matching-nfa")]
pub mod mega_scan;

/// Resident-buffer dispatch session for [`mega_scan::RulePipeline`]. Uploads the
/// immutable NFA transition/epsilon tables into backend-resident resources once,
/// so repeated scans transfer only the haystack instead of re-uploading the
/// multi-MiB tables on every dispatch (the borrowed `RulePipeline::scan` cost).
#[cfg(feature = "matching-nfa")]
pub mod resident;

/// Resident-buffer dispatch session for [`literal_set::GpuLiteralSet`]
/// region-presence scans. Uploads the immutable DFA + suffix-prefilter tables into
/// backend-resident resources once, so repeated coalesced-batch presence scans
/// transfer only the per-file haystack and a presence-prefix reset instead of
/// re-uploading the multi-MiB tables on every dispatch (the borrowed
/// `GpuLiteralSet::scan_presence_by_region` cost).
pub mod resident_presence;

/// Regex AST → NfaPlan frontend. Lowers a regex string into the same
/// `(NfaPlan, transition_table, epsilon_table)` triple that
/// [`nfa::compile`] produces for literals, so every downstream component
/// (`nfa_scan` Program, `mega_scan::build`, `RulePipeline`) runs
/// unmodified. Behind `matching-regex` so consumers without the regex
/// frontend skip the `regex-syntax` dep.
#[cfg(feature = "matching-regex")]
pub mod regex_compile;

/// Regex set → dense `CompiledDfa` GPU pipeline. Composes
/// `compile_regex_set` (NFA build) → `nfa_to_dfa` (subset construction,
/// vyre-primitives) → `build_ac_bounded_ranges_program` (existing AC
/// kernel) so regex pattern sets dispatch through the same O(1)-per-byte
/// kernel that literal AC uses. Behind `matching-regex` + `matching-dfa`
/// because both halves are required.
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub mod regex_dfa;

/// Anchored-window regex extraction (plan W2-3, line 179): validate candidate
/// origins from the positions pass against an anchored [`regex_dfa`] DFA,
/// turning literal-prefilter *admission* into full-match *extraction*
/// (confirm + locate). CPU primitive + parity oracle for the GPU kernel.
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub mod regex_anchored_window;

/// Regex-DFA per-region admission (plan W2-2, line 153's third evidence family):
/// a per-region presence bitmap of which regex patterns start a match in each
/// region, the regex counterpart of literal presence-by-region. Shipped as a
/// SEPARATE occupancy-cheap pass because the "single fused launch" is
/// measured-refuted (see the module docs / BACKLOG).
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub mod regex_region_admission;

/// Fused single-launch phase-1 evidence (plan W2-2, line 153): ONE dispatch, ONE
/// anchored DFA walk per byte, producing all three families keyhog otherwise
/// assembles from three dispatches, per-region presence, position triples for a
/// designated subset, and per-region admission bits. A correctness-equivalent
/// primitive; the fast path stays the separate specialized passes because kernel
/// fusion is measured-refuted on this substrate (see the module docs / BACKLOG).
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub mod fused_region_evidence;

/// Region-evidence pipeline (plan W2-2, line 158): the successor to the vestigial
/// [`mega_scan::RulePipeline`]. One type, one call, returning the full phase-1
/// evidence bundle (presence + positions + admission), a fast two-dispatch path
/// ([`region_evidence_pipeline::RegionEvidencePipeline::scan`]) and the
/// single-launch capability ([`region_evidence_pipeline::RegionEvidencePipeline::scan_fused`]).
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub mod region_evidence_pipeline;

#[cfg(feature = "matching-dfa")]
pub use dfa::{
    aho_corasick, dfa_compile, dfa_compile_with_budget, CompiledDfa, DfaCompileError,
    DEFAULT_DFA_BUDGET_BYTES,
};
#[cfg(feature = "matching-dfa")]
pub use direct_gpu::DirectGpuScanner;
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use fused_region_evidence::{
    fused_region_evidence_program, fused_region_evidence_reference, FusedRegionEvidence,
    FUSED_EVIDENCE_ADMISSION_BINDING, FUSED_EVIDENCE_MATCHES_BINDING,
    FUSED_EVIDENCE_MATCH_COUNT_BINDING, FUSED_EVIDENCE_PRESENCE_BINDING,
};
pub use hit_buffer::{
    compact_hits, compact_hits_with_layout, emit_hit, emit_hit_then_compact,
    emit_hit_then_compact_with_layout, emit_hit_with_layout, HIT_BUFFER_LIVE_LENGTH,
    HIT_BUFFER_OVERFLOW_COUNT,
};
pub use literal_set::{
    GpuLiteralSet, LiteralSetPreparedCount, LiteralSetPreparedPresenceByRegion,
    LiteralSetPreparedScan, LiteralSetScanScratch, LiteralSetWireError, Match as LiteralMatch,
    PendingFusedRegion, PendingMatches, PendingPresence, PendingPresenceByRegion,
    ResidentFusedRegionScan, ResidentLiteralScan, ScanAllTimed,
    LITERAL_SET_COUNT_RESET_RESOURCE_INDICES, LITERAL_SET_COUNT_RESOURCE_INDEX,
    LITERAL_SET_COUNT_SCAN_RESOURCE_INDICES, LITERAL_SET_MATCHES_RESOURCE_INDEX,
    LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX, LITERAL_SET_PRESENCE_BY_REGION_OUTPUT_RESOURCE_INDEX,
    LITERAL_SET_RESET_RESOURCE_INDICES, LITERAL_SET_SCAN_RESOURCE_INDICES,
};
#[cfg(feature = "matching-nfa")]
pub use mega_scan::{build as build_rule_pipeline, PipelineWireError, RulePipeline};
pub use paged_corpus::{
    scan_paged_fused, scan_paged_fused_async, scan_paged_fused_timed, scan_paths_paged,
    scan_paths_paged_prefetched, scan_pattern_sharded, scan_sharded_fused,
    scan_sharded_fused_timed, scan_sharded_fused_weighted, GlobalMatch, PagedScanResult,
    PagedScanTiming, PatternShard, ShardTiming, ShardedScanTiming,
};
pub use pipeline::{Pipeline, PostProcessFn};
#[cfg(any(test, feature = "cpu-parity"))]
pub use post_process::{
    reference_post_process, shannon_entropy_bits_per_byte, try_reference_post_process,
    try_reference_post_process_into,
};
pub use post_process::{PostProcessError, PostProcessedMatch};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_anchored_window::{
    anchored_window_extract_program, AnchoredWindowValidator, ANCHORED_WINDOW_MATCHES_BINDING,
    ANCHORED_WINDOW_MATCH_COUNT_BINDING,
};
#[cfg(feature = "matching-regex")]
pub use regex_compile::{
    build_rule_pipeline_from_regex, compile_regex_set, regex_construct_diagnostic_code,
    CaptureMode, CaptureModeContract, CompiledRegexSet, RegexCompileError, RegexConstruct,
};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_dfa::{
    build_regex_dfa_pipeline, build_regex_dfa_shards, build_regex_dfa_shards_unanchored,
    build_regex_dfa_unanchored, RegexDfaError, RegexDfaPipeline, RegexDfaShard,
};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_region_admission::{
    regex_admission_by_region_program, regex_admission_by_region_reference,
    regex_admission_presence_words, region_of,
};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use region_evidence_pipeline::{RegionEvidenceError, RegionEvidencePipeline};
#[cfg(feature = "matching-nfa")]
pub use resident::ResidentRulePipeline;
pub use resident_presence::ResidentPresencePipeline;
#[cfg(feature = "matching-substring")]
pub use substring::{substring_search, SCAN_SUBSTRING_OP_ID};
// Re-export the cross-program fusion API at the matching layer so consumers
// don't have to reach into `vyre-foundation` directly.
pub use vyre_foundation::execution_plan::fusion::{fuse_programs, fuse_programs_vec, FusionError};

#[cfg(feature = "cpu-parity")]
use vyre_primitives::matching::region::dedup_regions_cpu as primitive_dedup_regions_cpu;
#[cfg(any(test, feature = "cpu-parity"))]
pub use vyre_primitives::matching::region::dedup_regions_inplace;
/// Re-export the region-dedup GPU program builders through the scan layer
/// so consumers get the canonical span-coalescing helpers without taking a
/// separate dependency on `vyre-primitives`.
pub use vyre_primitives::matching::region::{dedup_regions_flag_program, RegionTriple};

/// Reference/parity region deduplication helper.
///
/// Production scan APIs avoid CPU-named symbols; this helper is explicitly a
/// reference contract for tests, examples, and conformance comparisons.
#[cfg(feature = "cpu-parity")]
#[must_use]
pub fn dedup_regions_reference(input: Vec<RegionTriple>) -> Vec<RegionTriple> {
    primitive_dedup_regions_cpu(input)
}
