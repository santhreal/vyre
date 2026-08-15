//! Resident work-queue protocol, host mirrors, scheduling policy, and IO.
//!
//! Artifact compilation and target selection live in `vyre-megakernel`.
//! Authenticated execution and recovery live in
//! [`crate::artifact_admission::ArtifactSession`].
//! This module owns only mutable queue policy and wire state.

#[cfg(feature = "megakernel-batch")]
pub mod advanced;
pub mod atomic_relaxed;
pub mod automata_worklist;
#[cfg(test)]
mod body_preorder;
pub mod builder;
pub mod descriptor;
pub mod handlers;
pub mod io;
mod lru_tick_cache;
pub mod mixed_work;
pub mod planner;
pub mod policy;
pub mod protocol;
mod protocol_api;
pub mod readback;
pub mod resident;
pub mod ring;
#[cfg(feature = "megakernel-batch")]
pub mod rule_catalog;
pub mod scaling;
pub mod scheduler;
pub mod speculation;
mod staging_reserve;
pub mod task;
pub mod telemetry;
pub mod workspace_adapter;
pub mod workspace_layout;

/// Stateless owner of resident work-queue encoding and decoding operations.
#[derive(Clone, Copy, Debug, Default)]
pub struct ResidentWorkQueue;
pub use automata_worklist::{
    AutomataStateIndex, AutomataWorklistEvidence, AutomataWorklistMode, AutomataWorklistPolicy,
    AutomataWorklistRecommendation, AutomataWorklistRequest,
    AUTOMATA_WORKLIST_EVIDENCE_SCHEMA_VERSION,
};
#[cfg(test)]
pub use builder::build_program_with_self_loading_miss_handler;
pub use builder::{
    build_program, build_program_jit, build_program_jit_slots, build_program_priority,
    build_program_priority_slots, build_program_sharded, build_program_sharded_no_io,
    build_program_sharded_once_slots, build_program_sharded_once_slots_control_report_shared,
    build_program_sharded_once_slots_shared, build_program_sharded_slots,
    build_program_sharded_slots_shared, build_program_sharded_with_io_polling,
    build_program_sharded_with_workspace_adapter, persistent_body, persistent_body_jit,
    persistent_body_priority, persistent_body_priority_slots,
    try_build_program_with_self_loading_miss_handler,
};
pub use descriptor::{
    BatchDescriptor, BuiltinOpcode, PackedOpDescriptor, SlotDescriptor, SlotOpcode, WindowClass,
    WindowDescriptor,
};
pub use handlers::OpcodeHandler;
pub use io::{IoCompletion, IoRequest, ResidentIoQueue, IO_SLOT_COUNT, IO_SLOT_WORDS};
pub use mixed_work::{
    mixed_work_protocol_evidence, validate_mixed_work_protocol, MixedWorkProtocolError,
    MixedWorkProtocolEvidence, MixedWorkProtocolPlan, MixedWorkQueueClass, MixedWorkUnit,
    MixedWorkUnitType, OutputSlabId, ResidentArtifactId, MIXED_WORK_PROTOCOL_SCHEMA_VERSION,
};
#[cfg(feature = "libs-compositions")]
pub use planner::{
    build_bellman_tn_order_program, build_kfac_autotune_step_program,
    build_persistent_fixpoint_program, build_scallop_provenance_wide_program,
    build_sinkhorn_clustering_program, build_sinkhorn_full_clustering_program,
};
pub use planner::{
    dispatch_grid_for, padded_slot_count, worker_workgroup_size, ResidentGridLimits,
    ResidentGridPlan, ResidentGridRequest, ResidentLaunchGeometry, ResidentQueueCapabilities,
    ResidentQueueConfig, ResidentQueueReport, ResidentQueueTelemetry, ResidentSizingPolicy,
    ResidentWorkItem, ResidentWorkloadHints,
};
#[cfg(test)]
pub use policy::{diffuse_priority_across_siblings, diffuse_priority_across_siblings_into};
pub use policy::{
    try_diffuse_priority_across_siblings, try_diffuse_priority_across_siblings_into,
    PriorityDrainReason, PriorityDrainRecommendation, PriorityRequeueAccounting,
    ResidentExecutionMode, ResidentGraphBlasSwitchClass, ResidentLaunchCacheStats,
    ResidentLaunchPolicy, ResidentLaunchRecommendation, ResidentLaunchRequest,
    ResidentPromotionEvidence, ResidentPromotionRoute, ResidentQueuePressure,
    ResidentQueueTopology, ResidentTopologyEvidence, HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION,
    PRIORITY_COUNTER_DRAIN_FIX, PRIORITY_COUNTER_DRAIN_HEADROOM, TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
};
pub use protocol::{
    control, control_byte_len, count_done_ring_slots, debug, debug_log_byte_len, encode_control,
    encode_empty_debug_log, encode_empty_ring, opcode, read_debug_log, read_done_count, read_epoch,
    read_metrics, read_observable, ring_byte_len, slot, try_count_done_ring_slots,
    try_encode_control, try_encode_control_into, try_encode_empty_debug_log,
    try_encode_empty_debug_log_into, try_encode_empty_ring, try_encode_empty_ring_into,
    try_read_debug_log, try_read_done_count, try_read_epoch, try_read_metrics, try_read_observable,
    DebugRecord, ProtocolError, ARG0_WORD, ARGS_PER_SLOT, CONTROL_MIN_WORDS, OPCODE_WORD,
    PRIORITY_WORD, SLOT_WORDS, STATUS_WORD, TENANT_WORD,
};
pub use protocol_api::RingSlotTransition;
pub use readback::{ResidentQueueReadback, ResidentReadbackCounters};
pub use resident::ResidentQueueBuffers;
#[cfg(feature = "megakernel-batch")]
pub use rule_catalog::{BatchRuleProgram, BatchRuleRejection};
pub use scheduler::{
    default_priority_offsets_array, priority_partition_active_lane_count,
    priority_partition_probe_budget, priority_partition_probe_count, priority_scan_body,
    priority_scan_body_with_stride, write_default_priority_offsets,
};
pub use speculation::{PairedSpeculationSample, PairedSpeculationUpdate, PairedSpeculationWindow};
pub use task::{TaskPriority, TaskQueueSnapshot, TaskState, TaskWorkItem};
pub use telemetry::{
    ControlSnapshot, CountMinSketch, ResidentRuntimeCounters, RingOccupancy, RingSlotSnapshot,
    RingStatus, RingTelemetry, SketchTelemetry, WindowTelemetry,
};
pub use telemetry::{
    ResidentRuntimeEvidence, RuntimeEvidenceMetricCoverage, RuntimeEvidenceMetricFamily,
    TelemetryDecodeCapacityEvidence, TelemetryDecodeScratch, RUNTIME_IO_EVIDENCE_SCHEMA_VERSION,
    TELEMETRY_DECODE_CAPACITY_SCHEMA_VERSION,
};
pub use workspace_adapter::ResidentWorkspaceAdapter;
pub use workspace_layout::{
    build_workspace_regions, first_workspace_region, next_record_workspace_region,
    next_workspace_region, workspace_record_words, ResidentWorkspaceLayoutError,
    ResidentWorkspaceRegion, ResidentWorkspaceRegionSpec,
};
