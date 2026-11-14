//! Backend contracts, artifact execution, resident work queues, and zero-copy IO.
//!
//! The backend-neutral layer defines the device, target compiler registration,
//! artifact materialization, binding, submission, and completion contracts every
//! backend registers into. Runtime construction starts from an authenticated
//! [`artifact_admission::ArtifactSession`]. Immutable compiler artifacts are
//! materialized through registered target devices; the runtime holds bindings,
//! retained state, queueing, recovery, and IO.

// In-crate modules and macro expansions name this crate by its own name,
// which resolves only through this alias.
extern crate self as vyre_runtime;

// The `PipelineError` variant-space closure. An exhaustive match over a
// `#[non_exhaustive]` enum is legal only inside the crate that defines it.
#[cfg(test)]
mod pipeline_error_closure;

mod error;
pub(crate) use error::closed_enum;
pub use error::*;

/// Canonical artifact-envelope authentication and exact-format admission.
pub mod artifact_admission;
mod semantic_execution;
pub use semantic_execution::RegisteredSemanticExecutor;

/// Resident work-queue protocols, scheduling policy, and runtime IO.
pub mod resident_work_queue;

/// Authenticated persistent execution over retained artifact bindings.
pub mod persistent_executor;
/// Content-addressed authenticated artifact cache.
pub mod pipeline_cache;

/// Structured artifact-session recovery without message parsing or recompilation.
pub mod recovery;

/// Linux io_uring integration. Compiled out on macOS / Windows.
#[cfg(target_os = "linux")]
pub mod uring;

/// Failure-domain and recovery classification shared with drivers.
pub use vyre_foundation::failure_domain::{
    FailureDomain, RecoveryClass, RecoveryDisposition, StateOwnerRecovery, TypedRecoveryError,
};

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
/// Backend-neutral reservation policy adapters.
pub mod reservation_policy;
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
/// Backend-neutral numeric boundary conversions.
pub mod numeric;
/// G7: persistent-thread engine + device-side work queue.
/// Eliminates per-file kernel-launch overhead for streams of
/// many small scan jobs.
pub mod persistent;

/// The `inventory` crate `register_backend!` submits through. A concrete
/// driver expands the macro and needs no `inventory` dependency of its own.
pub use ::inventory;

pub use aot::{
    emit_aot_launcher_target, AotLauncherEmitter, AotLauncherFiles, AotLauncherRequest,
    LauncherDependency,
};
/// Error-code catalog rendering for the driver-tier diagnostics surface.
pub use backend::error_catalog;
/// Marker trait that seals the driver traits against outside implementations.
pub use backend::sealed;
pub use backend::{
    acquire, acquire_preferred_dispatch_backend, backend_dispatches, backend_precedence,
    backend_registration, core_supported_ops, default_supported_ops,
    default_supported_ops_with_trap, dialect_only_supported_ops, node_op_id, registered_backends,
    registered_backends_by_precedence_slice, registered_target_operation_facets,
    replace_output_buffers_preserving_slots_with_memory_stats, validate_program, Backend,
    BackendCapability, BackendPrecedence, ErrorCode, OutputReplacementStats, OutputSlotByteStats,
    OutputSlotStats,
};
pub use backend::{
    borrowed_input_slices, default_dispatch_with_device_buffers,
    replace_output_buffers_preserving_slots, validate_buffer_ownership, ArtifactInstance,
    ArtifactMaterializer, BackendError, BackendRegistration, BindingSet, BoundResource,
    CompiledPipeline, Completion, Device, DeviceBuffer, DeviceIdentity, DispatchConfig,
    HostShimBuffer, OutputBuffers, PendingDispatch, ResidentDispatchStep, ResidentHandle,
    ResidentOwner, ResidentReadRange, ResidentSequenceTiming, Resource, Submission,
    TimedDispatchResult, VyreBackend, DEVICE_BUFFER_FEATURE,
};
pub use binding::{
    dynamic_element_count_from_bytes, BackendLayoutClass, BackendLayoutFingerprint,
    BackendLayoutSlot, Binding, BindingPlan, BindingRole,
};
pub use device_profile::{DeviceProfile, DeviceTimingQuality};
pub use device_signature::DeviceSignatureTable;
pub use diagnostics::{Diagnostic, DiagnosticCode, OpLocation, Severity};
pub use diagnostics::{DiagnosticCause, DiagnosticStage, RetryClass};
pub use dispatch_shape::dispatch_configs_share_launch_shape;
pub use fixpoint_iterations::{resolve_fixpoint_iterations, resolve_fixpoint_iterations_usize};
pub use launch::{program_vsa_fingerprint, program_vsa_fingerprint_words, LaunchPlan};
pub use launch::{
    record_launch_measurement, resolve_launch_workgroup, resolve_launch_workgroup_for_geometry,
    LaunchGeometry,
};
pub use launch_directive::LaunchDirective;
pub use pipeline::{
    dispatch_policy_cache_digest, dispatch_policy_cache_string, pipeline_cache_limits_from_env,
    push_lower_hex, try_normalized_program_cache_digest, update_dispatch_policy_cache_hash,
};
pub use pipeline::{
    hex_encode, hex_short, PipelineCacheIdentity, PipelineCacheMissReason, PipelineCacheSnapshot,
    PipelineDeviceFingerprint, PipelineFeatureFlags,
};
pub use program_walks::{
    admit_dispatch_grid, dispatch_element_count_for_program, element_size_bytes,
    enforce_actual_output_budget, find_indirect_dispatch, infer_dispatch_grid,
    infer_dispatch_grid_for_count, infer_launch_grid, output_binding_layout_parts,
    output_binding_layouts, output_layout_from_program, try_dispatch_param_words_into,
    IndirectDispatch, OutputBindingLayout, OutputLayout,
};
pub use read_only_alias::ReadOnlyAliasCheck;
pub use specialization::SpecCacheKey;
pub use speculate::{
    dispatch_prefilter_confirm, AdaptiveSpeculator, SpeculationMode, SpeculativeDispatchOutcome,
    SpeculativeDispatchPlan,
};
pub use speculate::{
    record_speculative_variant_race, SpeculativeVariantDecision, SpeculativeVariantKeys,
    SpeculativeVariantKind, SpeculativeVariantRace,
};
pub use subgroup::SubgroupCaps;
pub use target::Target;
