//! Execution adapters for substrate-neutral scan artifacts from `vyre-libs`.

pub use vyre_libs::scan::build_scan_program;
pub use vyre_libs::scan::{
    aho_corasick, build_regex_dfa_pipeline, build_regex_dfa_pipeline_with_policy,
    build_regex_dfa_pipeline_with_policy_ext, build_regex_dfa_shards,
    build_regex_dfa_shards_unanchored, build_regex_dfa_unanchored, compile_regex_set,
    compile_regex_set_with_policy, dfa_compile, dfa_compile_with_budget, fuse_programs,
    fuse_programs_vec, regex_construct_diagnostic_code, substring_search, AnchoredWindowValidator,
    ApiKind, CaptureMode, CaptureModeContract, CompiledDfa, CompiledRegexSet, DfaCompileError,
    FusionError, PostProcessError, PostProcessedMatch, RegexCompileError, RegexConstruct,
    RegexDfaError, RegexDfaPipeline, RegexDfaShard, RegexPatternExtent, RegexReplayPolicy,
    RegionTriple, ScanProgram, API_INDEX, DEFAULT_OPEN_ENDED_REPLAY_LIMIT_BYTES,
};
pub use vyre_libs::scan::{
    builders, classic_ac, dfa, fused_region_evidence, hit_buffer, nfa, post_process,
    regex_anchored_window, regex_compile, regex_dfa, regex_region_admission, substring,
};

pub mod artifact_session;
pub mod direct_gpu;
pub mod dispatch_io;
pub mod engine;
pub mod literal_set;
pub mod paged_corpus;
pub mod pipeline;
pub mod region_evidence_pipeline;
pub mod resident;
pub mod resident_presence;
pub mod session;

pub use artifact_session::{ScanArtifactError, ScanArtifactSession};
pub use direct_gpu::DirectGpuScanner;
pub use dispatch_io::{
    byte_scan_dispatch_config, candidate_start_dispatch_config, haystack_len_u32,
    pack_haystack_u32, pack_u32_slice, scan_guard, u32_words_as_le_bytes, unpack_match_triples,
    DEFAULT_MAX_SCAN_BYTES,
};
pub use engine::{
    cache_path as engine_cache_path, cached_load_or_compile, MatchEngineCache, MatchScan,
    ScanResult,
};
pub use literal_set::{
    GpuLiteralSet, LiteralSetPreparedCount, LiteralSetPreparedPresenceByRegion,
    LiteralSetPreparedScan, LiteralSetScanScratch, LiteralSetWireError, Match as LiteralMatch,
    PendingFusedRegion, PendingMatches, PendingPresence, PendingPresenceByRegion,
    PendingResidentFusedRegion, ResidentFusedRegionScan, ResidentFusedTiming, ResidentLiteralScan,
    ScanAllTimed, LITERAL_SET_COUNT_RESET_RESOURCE_INDICES, LITERAL_SET_COUNT_RESOURCE_INDEX,
    LITERAL_SET_COUNT_SCAN_RESOURCE_INDICES, LITERAL_SET_MATCHES_RESOURCE_INDEX,
    LITERAL_SET_MATCH_COUNT_RESOURCE_INDEX, LITERAL_SET_PRESENCE_BY_REGION_OUTPUT_RESOURCE_INDEX,
    LITERAL_SET_RESET_RESOURCE_INDICES, LITERAL_SET_SCAN_RESOURCE_INDICES,
};
pub use paged_corpus::{
    scan_paged_fused, scan_paged_fused_async, scan_paged_fused_timed, scan_paths_paged,
    scan_paths_paged_prefetched, scan_pattern_sharded, scan_sharded_fused,
    scan_sharded_fused_timed, scan_sharded_fused_weighted, scan_sharded_fused_weighted_timed,
    GlobalMatch, PagedScanResult, PagedScanTiming, PatternShard, ShardTiming, ShardedScanTiming,
};
pub use pipeline::{Pipeline, PostProcessFn};
pub use region_evidence_pipeline::{RegionEvidenceError, RegionEvidencePipeline};
pub use resident::ResidentScanSession;
pub use resident_presence::ResidentPresencePipeline;
pub use session::{
    build as build_scan_session, MaterializedScanSession, PipelineWireError, ScanSession,
};
