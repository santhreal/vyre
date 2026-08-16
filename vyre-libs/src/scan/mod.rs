//! Substrate-neutral byte and text scan compositions.
//!
//! Builders in this module return typed programs or immutable compilation
//! artifacts. Dispatch, resident-resource, timing, and readback adapters are
//! deliberately owned by upper integration crates.

pub mod builders;
pub(crate) mod haystack;
pub(crate) mod hit_buffer;

#[cfg(feature = "matching-dfa")]
pub mod classic_ac;
#[cfg(feature = "matching-dfa")]
pub(crate) mod dfa;
#[cfg(feature = "matching-nfa")]
pub mod nfa;
pub(crate) mod post_process;
#[cfg(feature = "matching-nfa")]
pub(crate) mod scan_program;
#[cfg(feature = "matching-substring")]
pub(crate) mod substring;

#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub(crate) mod fused_region_evidence;
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub(crate) mod regex_anchored_window;
#[cfg(feature = "matching-regex")]
pub(crate) mod regex_compile;
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub(crate) mod regex_dfa;
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub(crate) mod regex_region_admission;

pub use dfa::aho_corasick;
#[cfg(feature = "matching-dfa")]
pub use dfa::{aho_corasick_bounded, cooperative_dfa_scan, cooperative_dfa_scan_body_with_store};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
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
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_anchored_window::{
    anchored_window_extract_program, AnchoredWindowValidator, ANCHORED_WINDOW_MATCHES_BINDING,
    ANCHORED_WINDOW_MATCH_COUNT_BINDING,
};
#[cfg(feature = "matching-regex")]
pub use regex_compile::{
    build_scan_program_from_regex, compile_regex_set, compile_regex_set_with_policy,
    regex_construct_diagnostic_code, CaptureMode, CaptureModeContract, CompiledRegexSet,
    RegexCompileError, RegexConstruct, RegexPatternExtent, RegexReplayPolicy,
    DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_dfa::build_regex_dfa_pipeline_with_subgroup_coalesce;
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_dfa::{
    build_regex_dfa_pipeline, build_regex_dfa_pipeline_with_policy,
    build_regex_dfa_pipeline_with_policy_and_subgroup_coalesce, build_regex_dfa_shards,
    build_regex_dfa_shards_unanchored, build_regex_dfa_unanchored, RegexDfaError, RegexDfaPipeline,
    RegexDfaShard,
};
#[cfg(all(feature = "matching-regex", feature = "matching-dfa"))]
pub use regex_region_admission::{
    regex_admission_by_region_program, regex_admission_by_region_reference,
    regex_admission_presence_words, region_of,
};
#[cfg(feature = "matching-nfa")]
pub use scan_program::{build as build_scan_program, ScanProgram};
#[cfg(feature = "matching-substring")]
pub use substring::{substring_search, SCAN_SUBSTRING_OP_ID};

pub use vyre_foundation::execution_plan::fusion::{fuse_programs, fuse_programs_vec, FusionError};

#[cfg(feature = "cpu-parity")]
use crate::matching::dedup_regions_cpu as primitive_dedup_regions_cpu;
#[cfg(feature = "cpu-parity")]
use crate::matching::RegionTriple;

/// Reference region deduplication helper for parity tests.
#[cfg(feature = "cpu-parity")]
#[must_use]
pub fn dedup_regions_reference(input: Vec<RegionTriple>) -> Vec<RegionTriple> {
    primitive_dedup_regions_cpu(input)
}
