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
/// Persistent autotuning record store.
pub mod autotune_store;
/// VyreBackend trait, BackendError, capability records, validation.
pub(crate) mod backend;
/// Backend-neutral program binding plans.
pub(crate) mod binding;
/// Stable fingerprints and descriptor-layout sharing for binding plans.
pub(crate) mod binding_layout;
/// Backend-neutral cache eviction policy.
pub mod cache_eviction;
/// N5 substrate: spec-cache eviction with frequency × recency heat
/// decay. Used by F1/F3 cache layers when capacity pressure
/// triggers  -  `try_entries_to_evict(stats, capacity, now)` returns the
/// evictable IDs in eviction order (lowest heat first).
pub mod cache_eviction_heat;
/// Backend-neutral cache invalidation policy.
pub mod cache_invalidation;
/// Backend-neutral device capability profile and projections.
pub(crate) mod device_profile;
/// Tier-B device signature TOML loader.
pub(crate) mod device_signature;
/// Structured, machine-readable diagnostic rendering.
pub(crate) mod diagnostics;
/// Backend-neutral dispatch-shape comparison helpers.
pub(crate) mod dispatch_shape;
/// Bounded, backend-neutral durability flush for a set of written files.
pub mod durable_fanout;
/// Backend-neutral external resource import shared by every concrete driver.
pub mod external_import;
/// The external-import contract every concrete driver's importer answers.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod external_import_contracts;
/// Backend-neutral fixpoint-iteration resolution.
pub(crate) mod fixpoint_iterations;
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
/// Target-binding input staging shared by every materializer.
pub(crate) mod materialize_input_slots;
/// Materialized instance execution and resident submission paths.
pub(crate) mod materialize_instance;
mod materialize_macros;
/// Backend-neutral monotonic ordering helpers for staging hot paths.
pub mod ordering;
/// Backend-neutral fallible output-slot vector management.
pub mod output_slots;
/// Elementwise parity-gate scaffolding shared by the concrete driver crates.
#[cfg(any(test, feature = "test-fixtures"))]
pub mod parity_harness;
/// Compiled-pipeline cache, dispatch config, batched dispatch.
pub(crate) mod pipeline;
/// The dispatch-boundary rule that makes a read-only binding declaration true.
pub(crate) mod read_only_alias;
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
/// Backend-neutral checked transfer accounting policy.
pub mod transfer_accounting;
/// Device-side trap record layout shared by every backend that reports one.
pub mod trap_record;
/// Shared validation caches and launch-geometry contracts.
pub mod validation;


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
/// The failure domain of a poisoned lock, and the two answers an owner may give.
pub mod lock_policy;
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
/// Backend-neutral numeric boundary conversions.
pub mod numeric;
/// G7: persistent-thread engine + device-side work queue.
/// Eliminates per-file kernel-launch overhead for streams of
/// many small scan jobs.
pub mod persistent;
/// Domain-neutral semantic resource ABI, logical image/view types, and zero-copy timeline synchronization.
mod semantic_resource_abi;

/// The `inventory` crate `register_backend!` submits through. A concrete
/// driver expands the macro and needs no `inventory` dependency of its own.
pub use ::inventory;

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
    ArtifactMaterializer, BackendError, BackendRegistration, BindingSet, BoundResource,
    CompiledPipeline, Completion, Device, DeviceBuffer, DeviceIdentity, DispatchConfig,
    HostShimBuffer, OutputBuffers, PendingDispatch, ResidentDispatchStep, ResidentHandle,
    ResidentOwner, ResidentReadRange, ResidentSequenceTiming, Resource, Submission,
    TimedDispatchResult, TypedDispatchExt, VyreBackend, DEVICE_BUFFER_FEATURE,
};
pub use binding::{
    binding_plans_share_layout, dynamic_element_count_from_bytes, BackendLayoutClass,
    BackendLayoutFingerprint, BackendLayoutSlot, Binding, BindingPlan, BindingRole,
    BindingSetFingerprint,
};
pub use device_profile::{DeviceProfile, DeviceTimingQuality};
pub use device_signature::{DeviceSignature, DeviceSignatureTable};
pub use diagnostics::{Diagnostic, DiagnosticCode, OpLocation, Severity};
pub use diagnostics::{DiagnosticCause, DiagnosticStage, RetryClass};
pub use dispatch_shape::dispatch_configs_share_launch_shape;
pub use fixpoint_iterations::{resolve_fixpoint_iterations, resolve_fixpoint_iterations_usize};
pub use launch::{
    launch_width_measurements, record_launch_measurement, resolve_launch_workgroup,
    resolve_launch_workgroup_for_geometry, LaunchGeometry,
};
pub use launch::{program_vsa_fingerprint, program_vsa_fingerprint_words, LaunchPlan};
pub use launch_directive::LaunchDirective;
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
    infer_dispatch_grid_for_count, infer_launch_grid, output_binding_layout,
    output_binding_layout_parts, output_binding_layouts, output_layout_from_program,
    try_coerce_to_pow2_with_tail_mask, try_dispatch_param_words, try_dispatch_param_words_into,
    IndirectDispatch, OutputBindingLayout, OutputLayout, TailMaskPolicy,
};
pub use program_walks::{auto_grid, output_binding_layouts_into};
pub use read_only_alias::ReadOnlyAliasCheck;
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
pub use semantic_resource_abi::{
    all_address_modes, all_alias_set_kinds, all_border_colors, all_color_interpretations,
    all_compare_functions, all_external_event_kinds, all_external_memory_kinds, all_filter_modes,
    all_format_classes, all_image_formats, all_image_view_kinds, all_layout_states,
    all_lifetime_state_kinds, all_mipmap_filter_modes, all_plane_kinds, all_provenance_kinds,
    all_swizzle_components, all_sync_protocols, all_usage_flags, authenticate_external_import,
    AddressMode, AdmissionState, AdmittedResourceRecord, AdmittedResourceRecordExt, BorderColor,
    ColorInterpretation, CompareFunction, ComponentSwizzle, DeviceLossInvalidationReport,
    ExternalEventCapability, ExternalEventKind, ExternalMemoryCapability, ExternalMemoryKind,
    ExternalResourceRegistry, FilterMode, FormatClass, ImageDimensions, ImageFormat, ImagePlane,
    ImageViewDescriptor, ImageViewKind, ImportedResource, MipmapFilterMode, PlaneKind,
    ResourceAbiError, ResourceAliasSet, ResourceLayoutState, ResourceLifetimeState,
    ResourceOwnershipState, ResourcePermittedUsages, ResourceProvenance,
    ResourceTransitionSchedule, ResourceUsageTransition, SamplerCapability, SamplerDescriptor,
    SubresourceRange, SwizzleComponent, TimelineSyncProtocol, TransitionExecutionReport,
};
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
