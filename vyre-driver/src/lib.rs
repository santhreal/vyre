//! vyre-driver  -  substrate-agnostic backend machinery.
//!
//! Registry, runtime, pipeline, routing, diagnostics, and the VyreBackend
//! trait. Concrete backend crates depend on this crate and contribute
//! lowerings via the inventory collection mechanism.

// missing_docs is enforced workspace-wide via [workspace.lints.rust].
// vyre-driver inherits that floor; do not re-allow it here.

/// Backend-neutral checked arithmetic and atomic accounting primitives.
pub mod accounting;
/// Backend-neutral atomic accounting primitives.
pub(crate) mod accounting_atomic;
/// Backend-neutral fallible allocation reservation helpers.
pub mod allocation;
/// Backend-neutral ahead-of-time emission registry.
pub(crate) mod aot;
/// Independent-arm detection for queue-parallel dispatch.
/// Pure set arithmetic over (reads, writes) summaries; the dispatcher
/// uses `can_dispatch_concurrently` to decide whether two megakernel
/// arms can launch on independent backend queues or streams.
pub mod arm_independence;
/// Async-copy / kernel-overlap decision policy. Pure
/// per-slot read/write conflict check that decides whether an H2D
/// copy can run on a side stream concurrently with a downstream
/// kernel.
pub mod async_copy_overlap;
/// Persistent autotuning record store.
pub mod autotune_store;
/// VyreBackend trait, BackendError, capability records, validation.
pub(crate) mod backend;
/// Backend-neutral benchmark-driven optimization pass selection.
pub mod benchmark_pass_selection;
/// Backend-neutral program binding plans.
pub(crate) mod binding;
/// Stable fingerprints and descriptor-layout sharing for binding plans.
pub(crate) mod binding_layout;
/// Bindless buffers / textures decision policy. Decides
/// whether to use a bindless descriptor array or traditional per-
/// resource bindings, given the kernel's resource count and the
/// backend's bindless support level (Full / Static / Unsupported).
pub mod bindless_policy;
/// Backend-neutral cache eviction policy.
pub mod cache_eviction;
/// N5 substrate: spec-cache eviction with frequency × recency heat
/// decay. Used by F1/F3 cache layers when capacity pressure
/// triggers  -  `entries_to_evict(stats, capacity, now)` returns the
/// evictable IDs in eviction order (lowest heat first).
pub mod cache_eviction_heat;
/// Backend-neutral cache invalidation policy.
pub mod cache_invalidation;
/// Pre-recorded command reuse decision policy. Decides
/// whether to record a native command sequence once and replay it for
/// repeated identical dispatches, based on per-launch overhead vs
/// record + replay overhead.
pub mod command_reuse_policy;
/// Device-conditioned e-graph extraction helpers.
/// Backend-neutral device-side convergence planning.
pub mod device_convergence;
/// Backend-neutral device diagnostic aggregation planning.
pub mod device_diagnostic_aggregation;
pub(crate) mod device_extraction;
/// Backend-neutral device capability profile and projections.
pub(crate) mod device_profile;
/// Tier-B device signature TOML loader.
pub(crate) mod device_signature;
/// Backend-neutral device-side work queue planning.
pub mod device_work_queue;
/// Structured, machine-readable diagnostic rendering.
pub(crate) mod diagnostics;
/// Bundled D-series + I2 policy invocation. One-shot eval of every
/// dispatch-side decision substrate so the runtime threads a single
/// `DispatchPolicyVerdict` instead of six per-substrate verdicts.
pub mod dispatch_policy;
/// Backend-neutral dispatch-shape comparison helpers.
pub(crate) mod dispatch_shape;
/// Backend-neutral bounded fan-out for durability work on a path set.
pub mod durable_fanout;
/// Device-profile-aware extraction cost helpers.
pub mod extraction_cost;
/// Backend-neutral fixpoint-iteration resolution.
pub(crate) mod fixpoint_iterations;
/// Cross-dispatch fusion decision types and pure analysis.
pub mod fusion;
/// Backend-neutral replayable graph-capture binding planning.
pub mod graph_capture;
/// Hostile-input closure obligations shared by every backend's adversarial gate.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod hostile_input_closure;
/// Backend-neutral exact-input identity keys for replay caches.
pub mod input_identity;
/// Backend-neutral target-payload admission shared by every concrete driver.
pub mod materialize;
/// Target payload admission against neutral artifacts.
pub(crate) mod materialize_admission;
/// Materialized instance execution and resident submission paths.
pub(crate) mod materialize_instance;
/// Backend-neutral monotonic ordering helpers for staging hot paths.
pub mod ordering;
/// Backend-neutral fallible output-slot vector management.
pub mod output_slots;
/// Push-constant / tiny-param inlining decision policy.
/// Backends consume `decide_param_inlining` to choose between inlined
/// launch metadata and a uniform buffer upload, based on a per-backend
/// [`crate::param_inlining::ParamInliningPolicy`].
pub mod param_inlining;
/// Elementwise parity-gate scaffolding shared by the concrete driver crates.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod parity_harness;
/// Backend-neutral peer-transfer capability contracts and checked accounting.
pub(crate) mod peer_transfer;
/// Compiled-pipeline cache, dispatch config, batched dispatch.
pub(crate) mod pipeline;
/// N4 substrate: cross-pipeline disjoint-binding fusion analysis.
/// Lifts D2's in-megakernel-arm independence check to the
/// cross-dispatch boundary so consecutive pipelines with disjoint
/// reads/writes can fuse into one launch with a workgroup-bounded
/// fence instead of a full grid-sync.
pub mod pipeline_fusion;
/// Read-only semantic operation projections, migrations, and policy.
pub(crate) mod registry;
/// Backend-neutral reservation policy adapters.
pub mod reservation_policy;
/// Backend-neutral resident-resource reuse telemetry.
pub(crate) mod residency;
/// Canonical resident transfer fusion test model shared by the neutral fusion
/// tests and the concrete driver crates' adapter gates.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod resident_transfer_fixtures;
/// Backend-neutral resident transfer interval fusion.
pub mod resident_transfer_fusion;
/// Backend-neutral compact result readback planning.
pub mod result_compaction;
/// Runtime routing: profile-guided variant selection, algorithm heuristics.
pub(crate) mod routing;
/// Canonical self-hosted optimizer scaling bench shared by every concrete
/// driver crate's scaling suite.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod self_optimizer_bench;
/// N8 substrate: predicted-next-shape fingerprint API. Records
/// recent dispatch fingerprints and predicts the next via repeat /
/// short-cycle detection so the async dispatch path can prefetch
/// the predicted pipeline cache key during the GPU wait window.
pub mod shape_prediction;
/// Backend-neutral shader specialization values and cache key inputs.
pub(crate) mod specialization;
/// N2 substrate (foundation half): per-rewrite speculation-as-substrate
/// decision policy. Given baseline + speculative dispatch observations
/// + side-compile cost, returns Adopt / Reject / KeepRacing.
pub mod speculation_verdict;
/// Canonical subgroup operation taxonomy and capability records.
pub(crate) mod subgroup;
/// Target-compiler shell shared by every backend's dialect.
/// Stable compilation and emission target identifiers.
pub(crate) mod target;
pub mod target_dialect;
/// Trace-based JIT specialization decision policy.
/// Decides whether the dispatcher should fire a speculative
/// pre-spec on a predicted shape, weighted by recent hit count and
/// prediction confidence vs the speculative spec cost.
pub mod trace_jit_policy;
/// Backend-neutral checked transfer accounting policy.
pub mod transfer_accounting;
/// Device-side trap record layout shared by every backend that reports one.
pub mod trap_record;
/// Shared validation caches and launch-geometry contracts.
pub mod validation;

/// Backend-specific lowering strategies (Layer 2 of the two-layer
/// optimization architecture). Target-dependent emission decisions
/// that don't change what a program computes but change how it's
/// emitted for a specific chip/API.
///
/// See the [module docs](strategy/index.html) for the full architecture.
pub mod strategy;

/// Pure [`vyre_foundation::ir::Program`] analysis shared by all backends.
pub(crate) mod program_walks;

/// Driver-tier observability surface (P-OBS-1). Substrate-call
/// counters, cache hit rates, and a Prometheus exposition format.
pub mod observability;

/// G6: speculative rule evaluation with commit/rollback. Runs the
/// expensive confirmer on every tile, commits only tiles whose
/// pre-filter passed. Hides gather latency + improves subgroup
/// uniformity. Scaffold.
pub(crate) mod speculate;

/// Cross-grid synchronization: kernel-split fallback for backends
/// that lack a native cooperative-launch grid barrier. Splits a
/// `Program` at every `Node::Barrier { ordering: GridSync }` and
/// dispatches the segments in sequence  -  the kernel-launch boundary
/// itself is the grid-level fence.
pub mod grid_sync;
/// Backend-neutral launch preparation and program fingerprint wrappers.
pub(crate) mod launch;
/// The complete launch one dispatch runs, re-exported as `LaunchDirective`.
mod launch_directive;
/// Measured launch facts, and realization of the geometry an artifact recorded.
pub(crate) mod launch_facts;
/// Canonical launch-geometry limits shared by the launch preparation, validation,
/// and launch fact tests.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod launch_fixtures;
/// Backend-neutral adjacent-stage launch fusion planning.
pub mod launch_fusion;
/// Backend-neutral megakernel wave barrier planning.
pub mod megakernel_barrier;
/// Backend-neutral persistent megakernel execution planning.
pub mod megakernel_execution;
/// Canonical megakernel wave-policy corpora shared by the neutral planner tests
/// and the concrete driver crates' parity gates.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod megakernel_fixtures;
/// Backend-neutral megakernel frontier memory planning.
pub mod megakernel_frontier;
/// Backend-neutral resident-graph multi-query execution planning.
pub mod multi_query_execution;
/// Backend-neutral numeric boundary conversions.
pub mod numeric;
/// G7: persistent-thread engine + device-side work queue.
/// Eliminates per-file kernel-launch overhead for streams of
/// many small scan jobs.
pub mod persistent;

pub use aot::AotTargetId;
pub use aot::{
    emit_aot_launcher_target, registered_aot_launcher_emitters, AotLauncherEmitter,
    AotLauncherFiles, AotLauncherRequest, LauncherDependency,
};
/// Error-code catalog rendering for the driver-tier diagnostics surface.
pub use backend::error_catalog;
/// Backend-neutral lowering entry points concrete drivers compose.
pub use backend::lowering;
/// Marker trait that seals the driver traits against outside implementations.
pub use backend::sealed;
pub use backend::{
    acquire, acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence,
    backend_registration, core_supported_ops, default_supported_ops,
    default_supported_ops_with_trap, dialect_and_language_supported_ops,
    dialect_only_supported_ops, node_op_id, registered_backends, registered_backends_by_precedence,
    registered_backends_by_precedence_slice, registered_target_operation_facets,
    replace_output_buffers_preserving_slots_with_memory_stats,
    replace_output_buffers_preserving_slots_with_stats, validate_program, Backend,
    BackendCapability, BackendPrecedence, ErrorCode, OutputReplacementStats, OutputSlotByteStats,
    OutputSlotStats, RegexAcceleratorCapability, RegexAcceleratorClass, RegexAcceleratorEvidence,
    RegexAcceleratorMatchSchema, RegexAcceleratorStreamMode,
    REGEX_ACCELERATOR_EVIDENCE_SCHEMA_VERSION,
};
pub use backend::{
    borrowed_input_slices, default_dispatch_with_device_buffers,
    replace_output_buffers_preserving_slots, validate_buffer_ownership, ArtifactInstance,
    ArtifactMaterializer, BackendError, BackendRegistration, BatchOutputs, BindingSet,
    BoundResource, CompiledPipeline, Completion, Device, DeviceBuffer, DeviceIdentity,
    DispatchConfig, HostShimBuffer, OutputBuffers, PendingDispatch, ResidentDispatchStep,
    ResidentHandle, ResidentOwner, ResidentReadRange, ResidentSequenceTiming, Resource, Submission,
    TimedDispatchResult, TypedDispatchExt, VyreBackend, DEVICE_BUFFER_FEATURE,
};
pub use binding::{
    binding_plans_share_layout, dynamic_element_count_from_bytes, BackendLayoutClass,
    BackendLayoutFingerprint, BackendLayoutSlot, Binding, BindingPlan, BindingRole,
    BindingSetFingerprint,
};
pub use device_extraction::{
    extract_best_for_device, extract_best_for_devices, DeviceExtraction, ExtractionDevice,
};
pub use device_profile::{DeviceProfile, DeviceTimingQuality};
pub use device_signature::{DeviceSignature, DeviceSignatureTable};
pub use diagnostics::{Diagnostic, DiagnosticCode, OpLocation, Severity};
pub use diagnostics::{DiagnosticCause, DiagnosticStage, RetryClass};
pub use dispatch_shape::{
    borrowed_input_batch_shapes_match, borrowed_input_shapes_match,
    dispatch_configs_share_launch_shape,
};
pub use fixpoint_iterations::{resolve_fixpoint_iterations, resolve_fixpoint_iterations_usize};
pub use launch::{
    launch_width_measurements, record_launch_measurement, resolve_launch_workgroup,
    resolve_launch_workgroup_for_geometry, LaunchGeometry,
};
pub use launch::{program_vsa_fingerprint, program_vsa_fingerprint_words, LaunchPlan};
pub use launch_directive::LaunchDirective;
pub use peer_transfer::{
    PeerAccessCapability, PeerLinkKind, PeerTopology, PeerTransferAccounting, PeerTransferError,
    PeerTransferPlan, PeerTransferPlanner, PeerTransferRequest,
};
pub use pipeline::{
    dispatch_policy_cache_digest, dispatch_policy_cache_string, normalized_program_cache_digest,
    pipeline_cache_limits_from_env, push_lower_hex, try_normalized_program_cache_digest,
    update_dispatch_policy_cache_hash, PipelineCacheAudit, PipelineCacheAuditReport,
    DEFAULT_1D_WORKGROUP_SIZE, DEFAULT_PIPELINE_CACHE_BYTES, DEFAULT_PIPELINE_CACHE_ENTRIES,
};
pub use pipeline::{
    hex_encode, hex_short, DiskPipelineCache, PipelineCacheIdentity, PipelineCacheKey,
    PipelineCacheMissEvidence, PipelineCacheMissReason, PipelineCacheSnapshot,
    PipelineDeviceFingerprint, PipelineFeatureFlags, CURRENT_PIPELINE_CACHE_KEY_VERSION,
};
pub use program_walks::{
    admit_dispatch_grid, coerce_to_pow2_with_tail_mask, dispatch_element_count,
    dispatch_element_count_for_program, dispatch_param_words_into, element_size_bytes,
    enforce_actual_output_budget, find_indirect_dispatch, infer_dispatch_grid,
    infer_dispatch_grid_for_count, output_binding_layout, output_binding_layout_parts,
    output_binding_layouts, output_layout_from_program, try_coerce_to_pow2_with_tail_mask,
    try_dispatch_param_words, try_dispatch_param_words_into, IndirectDispatch, OutputBindingLayout,
    OutputLayout, TailMaskPolicy,
};
pub use program_walks::{auto_grid, enforce_output_budget, output_binding_layouts_into};
pub use registry::DEPRECATED_OP_CODE;
pub use registry::{
    deprecation_diagnostic, AttrMap, AttrValue, Deprecation, Migration, MigrationError,
    MigrationRegistry, Semver,
};
pub use registry::{
    validate_intrinsic_lowering, Chain, EnforceGate, EnforceVerdict, IntrinsicRegistrationError,
    MutationClass,
};
pub use residency::{ResidentGraphReuseTelemetry, ResidentGraphReuseTelemetryError};
pub use routing::pgo;
pub use routing::{select_sort_backend, Distribution, RoutingTable, SortBackend};
pub use specialization::{versioned_specialization_artifact_key, vsa_specialization_key};
pub use specialization::{SpecCacheKey, SpecMap, SpecValue};
pub use speculate::{
    dispatch_prefilter_confirm, encode_counter_tail, parse_counter_tail, AdaptiveSpeculator,
    SpeculationMode, SpeculationReport, SpeculativeDispatchOutcome, SpeculativeDispatchPlan,
    COUNTER_TAIL_BYTES, DEFAULT_THRESHOLD_PCT,
};
pub use speculate::{
    record_speculative_variant_race, SpeculativeVariantDecision, SpeculativeVariantKeys,
    SpeculativeVariantKind, SpeculativeVariantRace,
};
pub use subgroup::{
    reduction_offsets, reduction_offsets_into, try_reduction_offsets, try_reduction_offsets_into,
};
pub use subgroup::{SubgroupCaps, SubgroupOp};
pub use target::Target;
