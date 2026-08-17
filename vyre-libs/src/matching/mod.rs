//! Byte and text scan kernels: DFA, substring, filters.
//!
//! The path IS the interface. Callers write
//! `crate::matching::bracket_match(...)`  -
//! explicit paths; no wildcard re-exports.

/// Anchor-DFA plan shared by software and accelerator experiments.
pub(crate) mod anchor_dfa;
/// Bounded-stack bracket-pair detector.
pub(crate) mod bracket_match;

mod region_programs;

/// Span-region dedup primitive. Collapses same-pid overlapping or
/// touching `(pid, start, end)` triples into a representative span.
/// Every multimatch consumer in the workspace was reimplementing this
///  -  one primitive replaces all of them.
pub(crate) mod region;
#[cfg(test)]
mod region_tests;

mod dfa_compile;

/// NFA → CompiledDfa subset construction. Composes with
/// `dfa_compile`'s output type so any consumer of the dense AC kernel
/// (`vyre_libs::scan::classic_ac_bounded_ranges_program`) can scan
/// regex pattern sets too - not just literal AC.
pub(crate) mod nfa_to_dfa;

pub use anchor_dfa::{
    build_anchor_dfa_plan, AnchorDfaCandidate, AnchorDfaLiteral, AnchorDfaPlan, AnchorDfaPlanError,
    ANCHOR_DFA_PLAN_SCHEMA_VERSION,
};
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
pub use region::{
    cap_regions_per_pattern_flag_program, dedup_regions_cluster_program,
    dedup_regions_flag_program, region_dedup_dispatch_grid, RegionTriple,
    CAP_REGIONS_PER_PATTERN_OP_ID, REGION_DEDUP_WORKGROUP_SIZE,
};
pub use region::{
    compact_first_per_region_pattern_flag_program, region_sort_program,
    COMPACT_FIRST_PER_REGION_PATTERN_OP_ID, DEDUP_REGIONS_CLUSTER_OP_ID, DEDUP_REGIONS_FLAG_OP_ID,
};
