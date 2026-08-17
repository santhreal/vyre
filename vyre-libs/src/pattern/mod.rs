//! Pattern analysis, compilation, and matching domain.
//!
//! Unifies substring search, Aho-Corasick, DFA/NFA compilation, regex replay,
//! region deduplication, bracket matching, and state-machine engines under
//! one coherent domain.

pub mod builders;
pub(crate) mod haystack;
pub(crate) mod hit_buffer;

#[cfg(feature = "pattern-dfa")]
pub mod classic_ac;
#[cfg(feature = "pattern-dfa")]
pub(crate) mod dfa;
#[cfg(feature = "pattern-nfa")]
pub mod nfa;
pub(crate) mod post_process;
#[cfg(feature = "pattern-nfa")]
pub(crate) mod scan_program;
#[cfg(feature = "pattern-substring")]
pub(crate) mod substring;

#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub(crate) mod fused_region_evidence;
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub(crate) mod regex_anchored_window;
#[cfg(feature = "pattern-regex")]
pub(crate) mod regex_compile;
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub(crate) mod regex_dfa;
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub(crate) mod regex_region_admission;

pub(crate) mod anchor_dfa;
pub(crate) mod bracket_match;
pub(crate) mod dfa_compile;
pub(crate) mod nfa_to_dfa;
pub(crate) mod region;
mod region_programs;
#[cfg(test)]
mod region_tests;

pub use anchor_dfa::{
    build_anchor_dfa_plan, AnchorDfaCandidate, AnchorDfaLiteral, AnchorDfaPlan, AnchorDfaPlanError,
    ANCHOR_DFA_PLAN_SCHEMA_VERSION,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use bracket_match::bracket_match_cpu_ref;
#[cfg(any(test, feature = "cpu-parity"))]
pub use bracket_match::bracket_match_cpu_ref_into;
pub use bracket_match::BRACKET_MATCH_OP_ID;
pub use bracket_match::{
    bracket_match, bracket_match_dispatch_grid, BRACKET_KIND_CLOSE, BRACKET_KIND_OPEN,
    BRACKET_KIND_OTHER, BRACKET_MATCH_NONE, BRACKET_MATCH_PARALLEL_WORKGROUP_SIZE,
};
pub use dfa_compile::{
    dfa_compile, dfa_compile_case_insensitive, dfa_compile_case_insensitive_with_budget,
    dfa_compile_with_budget, CompiledDfa, DfaCompileError, DfaWireError, DEFAULT_DFA_BUDGET_BYTES,
};
pub use nfa_to_dfa::{
    dfa_fingerprint, dfa_wire_bytes, nfa_to_dfa, DfaDedupBatch, DfaDedupResult, DfaDedupStats,
    DfaDedupTable, NfaTables, NfaToDfaError,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use region::cap_regions_per_pattern_survivors_cpu;
#[cfg(any(test, feature = "cpu-parity"))]
pub use region::dedup_regions_cpu;
#[cfg(any(test, feature = "cpu-parity"))]
pub use region::dedup_regions_inplace;
pub use region::{
    cap_regions_per_pattern_flag_program, dedup_regions_cluster_program,
    dedup_regions_flag_program, region_dedup_dispatch_grid, RegionTriple,
    CAP_REGIONS_PER_PATTERN_OP_ID, REGION_DEDUP_WORKGROUP_SIZE,
};
pub use region::{
    compact_first_per_region_pattern_flag_program, region_sort_program,
    COMPACT_FIRST_PER_REGION_PATTERN_OP_ID, DEDUP_REGIONS_CLUSTER_OP_ID, DEDUP_REGIONS_FLAG_OP_ID,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use region::{compact_first_per_region_pattern_survivors_cpu, sort_regions_cpu};

#[cfg(feature = "pattern-dfa")]
pub use dfa::aho_corasick;
#[cfg(feature = "pattern-dfa")]
pub use dfa::{aho_corasick_bounded, cooperative_dfa_scan, cooperative_dfa_scan_body_with_store};
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub use fused_region_evidence::{
    fused_region_evidence_program, fused_region_evidence_reference, FusedRegionEvidence,
    FUSED_EVIDENCE_ADMISSION_BINDING, FUSED_EVIDENCE_MATCHES_BINDING,
    FUSED_EVIDENCE_MATCH_COUNT_BINDING, FUSED_EVIDENCE_PRESENCE_BINDING,
};
pub use haystack::pack_haystack_u32;
pub use hit_buffer::{
    compact_hits, compact_hits_with_layout, emit_hit, emit_hit_then_compact,
    emit_hit_then_compact_with_layout, emit_hit_with_layout, HIT_BUFFER_LIVE_LENGTH,
    HIT_BUFFER_OVERFLOW_COUNT,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use post_process::{
    reference_post_process, shannon_entropy_bits_per_byte, try_reference_post_process,
    try_reference_post_process_into,
};
pub use post_process::{PostProcessError, PostProcessedMatch};
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub use regex_anchored_window::{
    anchored_window_extract_program, AnchoredWindowValidator, ANCHORED_WINDOW_MATCHES_BINDING,
    ANCHORED_WINDOW_MATCH_COUNT_BINDING,
};
#[cfg(feature = "pattern-regex")]
pub use regex_compile::{
    build_scan_program_from_regex, compile_regex_set, compile_regex_set_with_policy,
    regex_construct_diagnostic_code, CaptureMode, CaptureModeContract, CompiledRegexSet,
    RegexCompileError, RegexConstruct, RegexPatternExtent, RegexReplayPolicy,
    DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
};
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub use regex_dfa::build_regex_dfa_pipeline_with_subgroup_coalesce;
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub use regex_dfa::{
    build_regex_dfa_pipeline, build_regex_dfa_pipeline_with_policy,
    build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce, build_regex_dfa_shards,
    build_regex_dfa_shards_unanchored, build_regex_dfa_unanchored, RegexDfaError, RegexDfaPipeline,
    RegexDfaShard,
};
#[cfg(all(feature = "pattern-regex", feature = "pattern-dfa"))]
pub use regex_region_admission::{
    regex_admission_by_region_program, regex_admission_by_region_reference,
    regex_admission_presence_words, region_of,
};
#[cfg(feature = "pattern-nfa")]
pub use scan_program::{build as build_scan_program, ScanProgram};
#[cfg(feature = "pattern-substring")]
pub use substring::{substring_search, SCAN_SUBSTRING_OP_ID};

pub use vyre_foundation::execution_plan::fusion::{fuse_programs, fuse_programs_vec, FusionError};

#[cfg(feature = "cpu-parity")]
use crate::pattern::dedup_regions_cpu as primitive_dedup_regions_cpu;

/// Reference region deduplication helper for parity tests.
#[cfg(feature = "cpu-parity")]
#[must_use]
pub fn dedup_regions_reference(input: Vec<RegionTriple>) -> Vec<RegionTriple> {
    primitive_dedup_regions_cpu(input)
}
