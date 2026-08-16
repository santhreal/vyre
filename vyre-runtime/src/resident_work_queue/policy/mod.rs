//! Resident megakernel launch policy and queue-pressure decisions.

use vyre_driver::BackendError;

mod cache;
use super::planner::{ResidentGridLimits, ResidentGridRequest, ResidentLaunchGeometry};
use super::staging_reserve::try_reserve_vec_capacity;

/// Host-side pressure classification for one megakernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentQueuePressure {
    /// No logical slots are queued.
    Empty,
    /// The queue is below the available worker lanes.
    Light,
    /// The queue is large enough to keep the submitted workers occupied.
    Balanced,
    /// The queue is several waves deep or already showing requeue pressure.
    Saturated,
}

/// Interpreter/JIT route selected by the launch policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentExecutionMode {
    /// Use the generic opcode interpreter.
    Interpreter,
    /// Use a fused payload processor for hot windows or opcodes.
    Jit,
}

/// Scale-aware execution topology selected for one megakernel launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentQueueTopology {
    /// Nothing is queued.
    Empty,
    /// Low frontier density; prefer sparse frontier expansion and avoid
    /// block-wide dense scans.
    SparseFrontier,
    /// Mid-density frontier; combine sparse frontier queues with dense block
    /// tiles instead of forcing either extreme.
    HybridFrontier,
    /// High frontier density; prefer dense block propagation with coalesced
    /// scans.
    DenseFrontier,
    /// High-density graph with enough hot structure to justify fused waves.
    FusedDense,
    /// Memory pressure is high enough that bounded occupancy is more important
    /// than maximizing active waves.
    MemoryConstrained,
}

/// Schema version for topology evidence emitted by the megakernel launch policy.
pub const TOPOLOGY_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Schema version for hot opcode/window promotion evidence emitted by policy.
pub const HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// GraphBLAS-style sparse/dense switch class for a selected launch topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentGraphBlasSwitchClass {
    /// Nothing is queued.
    Empty,
    /// Sparse frontier expansion is preferred.
    Sparse,
    /// Sparse and dense paths should both remain available.
    Hybrid,
    /// Dense propagation is preferred.
    Dense,
    /// Memory pressure overrides the sparse/dense frontier choice.
    MemoryConstrained,
}

impl ResidentGraphBlasSwitchClass {
    /// Stable label for reports and bench output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Sparse => "sparse",
            Self::Hybrid => "hybrid",
            Self::Dense => "dense",
            Self::MemoryConstrained => "memory_constrained",
        }
    }
}

/// Evidence envelope that makes topology selection auditable by runtime benches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentTopologyEvidence {
    /// Evidence schema version.
    pub schema_version: u32,
    /// Queue pressure that participated in the launch recommendation.
    pub queue_pressure: ResidentQueuePressure,
    /// Active frontier density in basis points after policy-side inference.
    pub frontier_density_bps: u16,
    /// Semiring frontier density input used for GraphBLAS-style switching.
    pub semiring_frontier_density_bps: u16,
    /// Concrete topology selected by the launch policy.
    pub selected_topology: ResidentQueueTopology,
    /// Sparse/dense switch class corresponding to the selected topology.
    pub graphblas_switch_class: ResidentGraphBlasSwitchClass,
    /// Resident device bytes reported by the caller after policy-side inference.
    pub resident_device_bytes: u64,
    /// Estimated peak resident bytes for the selected launch plan.
    pub estimated_peak_device_bytes: u64,
    /// True when benches must compare output parity across topology variants.
    pub output_parity_required: bool,
}

impl ResidentTopologyEvidence {
    /// Return true when the evidence envelope contains bounded, versioned
    /// fields that a parity bench can report without consulting hidden policy
    /// state.
    #[must_use]
    pub fn is_complete(self) -> bool {
        self.schema_version == TOPOLOGY_EVIDENCE_SCHEMA_VERSION
            && self.frontier_density_bps <= 10_000
            && self.semiring_frontier_density_bps <= 10_000
            && self.output_parity_required
    }
}

/// Interpreter/JIT promotion route selected from queue and hot-window signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidentPromotionRoute {
    /// Stay on the generic interpreter path.
    Interpreter,
    /// Use JIT because the queue is large enough to amortize fused execution.
    QueueJit,
    /// Use JIT because opcode counters crossed the promotion threshold.
    OpcodeJit,
    /// Use JIT because repeated descriptor windows crossed the promotion threshold.
    WindowJit,
    /// Use JIT because both opcode and window promotion thresholds were crossed.
    OpcodeAndWindowJit,
}

impl ResidentPromotionRoute {
    /// Stable label for reports and lowerer evidence.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interpreter => "interpreter",
            Self::QueueJit => "queue_jit",
            Self::OpcodeJit => "opcode_jit",
            Self::WindowJit => "window_jit",
            Self::OpcodeAndWindowJit => "opcode_and_window_jit",
        }
    }
}

/// Evidence envelope that makes hot opcode/window promotion auditable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentPromotionEvidence {
    /// Evidence schema version.
    pub schema_version: u32,
    /// Logical ring slots or work items queued for this launch.
    pub queue_len: u32,
    /// Queue length threshold that can trigger JIT without hot counters.
    pub jit_queue_len_threshold: u32,
    /// Hot opcode counter supplied to the policy.
    pub hot_opcode_count: u32,
    /// Hot opcode threshold configured on the policy.
    pub hot_opcode_threshold: u32,
    /// Hot descriptor-window counter supplied to the policy.
    pub hot_window_count: u32,
    /// Hot descriptor-window threshold configured on the policy.
    pub hot_window_threshold: u32,
    /// Interpreter or JIT route selected by the policy.
    pub execution_mode: ResidentExecutionMode,
    /// True when opcode counters require fused opcode promotion.
    pub promote_hot_opcodes: bool,
    /// True when window counters require fused descriptor-window promotion.
    pub promote_hot_windows: bool,
    /// Stable promotion class for reports and lowerer input.
    pub promotion_route: ResidentPromotionRoute,
    /// True when the lowerer should materialize fused descriptor windows.
    pub fused_descriptor_window_required: bool,
    /// True when benches must compare interpreter and fused-window outputs.
    pub output_parity_required: bool,
}

impl ResidentPromotionEvidence {
    /// Return true when the promotion evidence carries all thresholds and
    /// route fields needed by a lowerer or parity bench.
    #[must_use]
    pub fn is_complete(self) -> bool {
        self.schema_version == HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION
            && self.jit_queue_len_threshold != 0
            && self.hot_opcode_threshold != 0
            && self.hot_window_threshold != 0
            && self.fused_descriptor_window_required == self.promote_hot_windows
            && self.output_parity_required
    }
}

/// Thread-local launch recommendation cache telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentLaunchCacheStats {
    /// Live cache entries retained in the current thread.
    pub entries: usize,
    /// Cache hits served without recomputing launch geometry.
    pub hits: u64,
    /// Cache misses that required policy recomputation.
    pub misses: u64,
}

/// Inputs for one launch-policy recommendation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentLaunchRequest {
    /// Logical ring slots or work items queued for this launch.
    pub queue_len: u32,
    /// Caller-requested worker workgroup ceiling. Zero means derive from occupancy.
    pub requested_worker_groups: u32,
    /// Adapter maximum workgroup size in the x dimension.
    pub max_workgroup_size_x: u32,
    /// Adapter maximum compute workgroups per dimension.
    pub max_compute_workgroups_per_dimension: u32,
    /// Adapter maximum invocations per compute workgroup.
    pub max_compute_invocations_per_workgroup: u32,
    /// Caller-requested sparse-hit capacity. Zero means derive from queue shape.
    pub requested_hit_capacity: u32,
    /// Expected sparse hits per queued item when deriving hit capacity.
    pub expected_hits_per_item: u32,
    /// Count of opcodes observed hot enough for promotion.
    pub hot_opcode_count: u32,
    /// Count of ticketed route windows observed hot enough for promotion.
    pub hot_window_count: u32,
    /// Slots requeued by priority scheduling since the last recommendation.
    pub requeue_count: u64,
    /// Maximum priority age observed since the last recommendation.
    pub max_priority_age: u32,
    /// Nodes in the resident dependency graph. Zero means the caller has no
    /// graph-shape telemetry for this launch.
    pub graph_node_count: u32,
    /// Edges in the resident dependency graph. Zero means the caller has no
    /// graph-shape telemetry for this launch.
    pub graph_edge_count: u32,
    /// Active frontier density in basis points relative to graph nodes.
    pub frontier_density_bps: u16,
    /// Device-memory pressure in basis points relative to the active budget.
    pub memory_pressure_bps: u16,
    /// Device-resident bytes already required by this dispatch family.
    pub resident_device_bytes: u64,
    /// Hard device-memory budget for this launch. Zero means unbounded.
    pub device_memory_budget_bytes: u64,
}

impl ResidentLaunchRequest {
    /// Construct a direct-dispatch request with conservative defaults.
    #[must_use]
    pub const fn direct(
        queue_len: u32,
        requested_worker_groups: u32,
        max_workgroup_size_x: u32,
    ) -> Self {
        Self {
            queue_len,
            requested_worker_groups,
            max_workgroup_size_x,
            max_compute_workgroups_per_dimension: requested_worker_groups,
            max_compute_invocations_per_workgroup: max_workgroup_size_x,
            requested_hit_capacity: 0,
            expected_hits_per_item: 1,
            hot_opcode_count: 0,
            hot_window_count: 0,
            requeue_count: 0,
            max_priority_age: 0,
            graph_node_count: 0,
            graph_edge_count: 0,
            frontier_density_bps: 0,
            memory_pressure_bps: 0,
            resident_device_bytes: 0,
            device_memory_budget_bytes: 0,
        }
    }
}

/// Policy output consumed by runtime dispatchers and batch builders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentLaunchRecommendation {
    /// Padded launch geometry for the ring protocol.
    pub geometry: ResidentLaunchGeometry,
    /// Worker workgroups selected for the dispatch.
    pub worker_groups: u32,
    /// Sparse-hit capacity selected for the dispatch.
    pub hit_capacity: u32,
    /// Queue pressure classification.
    pub pressure: ResidentQueuePressure,
    /// Interpreter or JIT route selected from telemetry.
    pub execution_mode: ResidentExecutionMode,
    /// Scale-aware dispatch topology selected from graph shape, frontier
    /// density, and memory pressure.
    pub topology: ResidentQueueTopology,
    /// True when hot opcode counters justify fused opcode promotion.
    pub promote_hot_opcodes: bool,
    /// True when ticketed route windows justify fused window promotion.
    pub promote_hot_windows: bool,
    /// True when aged/requeued priority work should be lifted on the next publish.
    pub age_priority_work: bool,
    /// Estimated peak device bytes needed by the resident launch plan.
    pub estimated_peak_device_bytes: u64,
    /// Hard device-memory budget applied to this recommendation. Zero means unbounded.
    pub device_memory_budget_bytes: u64,
}

/// Requeue and aging counters produced by priority-aware schedulers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PriorityRequeueAccounting {
    /// Number of slots requeued due to contention or quota pressure.
    pub requeue_count: u64,
    /// Number of slots promoted because their priority age crossed policy.
    pub aged_promotions: u64,
    /// Largest age observed for any queued priority slot.
    pub max_priority_age: u32,
}

/// Counter headroom at or below which schedulers should drain telemetry.
pub const PRIORITY_COUNTER_DRAIN_HEADROOM: u64 = 1024;

/// Stable operator fix for priority counter drain recommendations.
pub const PRIORITY_COUNTER_DRAIN_FIX: &str =
    "drain scheduler telemetry before counters reach u64::MAX";

/// Reason a priority scheduler should drain telemetry into a launch request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PriorityDrainReason {
    /// No priority telemetry is pending.
    None,
    /// Non-empty priority telemetry should be propagated to the policy.
    PendingTelemetry,
    /// The requeue counter is inside the configured drain headroom.
    RequeueCounterNearLimit,
    /// The aged-promotion counter is inside the configured drain headroom.
    AgedPromotionCounterNearLimit,
    /// The requeue counter is exhausted.
    RequeueCounterExhausted,
    /// The aged-promotion counter is exhausted.
    AgedPromotionCounterExhausted,
}

impl PriorityDrainReason {
    /// Stable label for tests, reports, and scheduler diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PendingTelemetry => "pending_telemetry",
            Self::RequeueCounterNearLimit => "requeue_counter_near_limit",
            Self::AgedPromotionCounterNearLimit => "aged_promotion_counter_near_limit",
            Self::RequeueCounterExhausted => "requeue_counter_exhausted",
            Self::AgedPromotionCounterExhausted => "aged_promotion_counter_exhausted",
        }
    }
}

/// Structured drain recommendation for priority scheduler counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PriorityDrainRecommendation {
    /// True when the scheduler should drain telemetry before accepting more work.
    pub should_drain: bool,
    /// Concrete reason for the recommendation.
    pub reason: PriorityDrainReason,
    /// Requeue counter value included for propagation into launch telemetry.
    pub requeue_count: u64,
    /// Aged-promotion counter value included for propagation into launch telemetry.
    pub aged_promotions: u64,
    /// Largest priority age observed for any queued slot.
    pub max_priority_age: u32,
    /// Remaining requeue counter increments before exact overflow.
    pub requeue_counter_headroom: u64,
    /// Remaining aged-promotion counter increments before exact overflow.
    pub aged_promotion_counter_headroom: u64,
    /// Stable operator fix string to surface with drain diagnostics.
    pub fix: &'static str,
}

impl PriorityRequeueAccounting {
    /// Return a structured drain recommendation for scheduler telemetry.
    #[must_use]
    pub fn drain_recommendation(self) -> PriorityDrainRecommendation {
        let requeue_counter_headroom = u64::MAX.saturating_sub(self.requeue_count);
        let aged_promotion_counter_headroom = u64::MAX.saturating_sub(self.aged_promotions);
        let reason = if self.requeue_count == u64::MAX {
            PriorityDrainReason::RequeueCounterExhausted
        } else if self.aged_promotions == u64::MAX {
            PriorityDrainReason::AgedPromotionCounterExhausted
        } else if requeue_counter_headroom <= PRIORITY_COUNTER_DRAIN_HEADROOM {
            PriorityDrainReason::RequeueCounterNearLimit
        } else if aged_promotion_counter_headroom <= PRIORITY_COUNTER_DRAIN_HEADROOM {
            PriorityDrainReason::AgedPromotionCounterNearLimit
        } else if self.requeue_count != 0 || self.aged_promotions != 0 || self.max_priority_age != 0
        {
            PriorityDrainReason::PendingTelemetry
        } else {
            PriorityDrainReason::None
        };
        PriorityDrainRecommendation {
            should_drain: reason != PriorityDrainReason::None,
            reason,
            requeue_count: self.requeue_count,
            aged_promotions: self.aged_promotions,
            max_priority_age: self.max_priority_age,
            requeue_counter_headroom,
            aged_promotion_counter_headroom,
            fix: PRIORITY_COUNTER_DRAIN_FIX,
        }
    }

    /// Record one requeue event.
    pub fn record_requeue(&mut self, age_ticks: u32) {
        self.requeue_count = self.requeue_count.saturating_add(1);
        self.max_priority_age = self.max_priority_age.max(age_ticks);
    }

    /// Record one requeue event with exact overflow reporting.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the requeue counter would overflow.
    pub fn try_record_requeue(&mut self, age_ticks: u32) -> Result<(), BackendError> {
        self.requeue_count = self.requeue_count.checked_add(1).ok_or_else(|| {
            BackendError::new(
                "megakernel priority requeue_count overflowed u64. Fix: drain scheduler telemetry before counters reach u64::MAX.",
            )
        })?;
        self.max_priority_age = self.max_priority_age.max(age_ticks);
        Ok(())
    }

    /// Record one priority-aging promotion.
    pub fn record_aged_promotion(&mut self, age_ticks: u32) {
        self.aged_promotions = self.aged_promotions.saturating_add(1);
        self.max_priority_age = self.max_priority_age.max(age_ticks);
    }

    /// Record one priority-aging promotion with exact overflow reporting.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the aged-promotion counter would overflow.
    pub fn try_record_aged_promotion(&mut self, age_ticks: u32) -> Result<(), BackendError> {
        self.aged_promotions = self.aged_promotions.checked_add(1).ok_or_else(|| {
            BackendError::new(
                "megakernel aged_promotions overflowed u64. Fix: drain scheduler telemetry before counters reach u64::MAX.",
            )
        })?;
        self.max_priority_age = self.max_priority_age.max(age_ticks);
        Ok(())
    }
}

/// Diffuse priority signals across a set of priority-class siblings
/// via sheaf diffusion (P-RUNTIME-3). Higher-priority siblings pull
/// neighbors toward higher priority; lower-priority siblings drag
/// down. After a few diffusion steps, each item's priority reflects
/// both its own age and its neighborhood pressure  -  letting requeue
/// decisions be group-aware without hand-rolling a propagation pass.
///
/// `priority_stalks` is the per-item priority value (caller's choice
/// of scale; higher = more urgent). `restriction_diag` is the
/// per-item transmission coefficient (1.0 = freely shares priority,
/// 0.0 = isolated). `damping` controls the diffusion rate in [0, 1].
///
/// Returns the post-diffusion priority vector, same shape as input.
///
/// # Errors
///
/// Returns [`BackendError`] when host staging cannot be reserved for the
/// priority vector.
pub fn try_diffuse_priority_across_siblings(
    priority_stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    iterations: u32,
) -> Result<Vec<f64>, BackendError> {
    let mut current = Vec::new();
    let mut next = Vec::new();
    try_diffuse_priority_across_siblings_into(
        priority_stalks,
        restriction_diag,
        damping,
        iterations,
        &mut current,
        &mut next,
    )?;
    Ok(current)
}

/// Diffuse priority signals into caller-owned storage.
///
/// # Errors
///
/// Returns [`BackendError`] when host staging cannot be reserved for the
/// priority vector.
pub fn try_diffuse_priority_across_siblings_into(
    priority_stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    iterations: u32,
    out: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> Result<(), BackendError> {
    out.clear();
    reserve_target_capacity(out, priority_stalks.len(), "priority diffusion output")?;
    out.extend_from_slice(priority_stalks);
    scratch.clear();
    if priority_stalks.len() != restriction_diag.len() {
        return Ok(());
    }
    for _ in 0..iterations {
        diffuse_step_into(out, restriction_diag, damping, scratch)?;
        std::mem::swap(out, scratch);
    }
    Ok(())
}

/// Single policy surface for megakernel launch sizing and telemetry-driven routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentLaunchPolicy {
    /// Sizing policy for worker counts and grid geometry.
    pub sizing: super::planner::ResidentSizingPolicy,
    /// Minimum capacity for sparse-hit results.
    pub min_hit_capacity: u32,
    /// Multiplier for expected hits to determine capacity.
    pub hit_capacity_multiplier: u32,
    /// Number of waves that define a saturated queue.
    pub saturated_waves: u32,
    /// Threshold for promoting hot opcodes to JIT.
    pub hot_opcode_threshold: u32,
    /// Threshold for promoting hot windows to JIT.
    pub hot_window_threshold: u32,
    /// Queue length threshold to prefer JIT over interpreter.
    pub jit_queue_len_threshold: u32,
    /// Priority age threshold to trigger aging promotions.
    pub priority_age_threshold: u32,
    /// Frontier density at or below this value uses sparse expansion.
    pub sparse_frontier_threshold_bps: u16,
    /// Frontier density at or above this value uses dense propagation.
    pub dense_frontier_threshold_bps: u16,
    /// Memory pressure at or above this value uses the memory-constrained path.
    pub memory_pressure_threshold_bps: u16,
    /// Minimum graph edge count before dense hot work is eligible for fusion.
    pub fusion_edge_threshold: u32,
    /// Conservative resident scratch bytes needed per sparse-hit entry.
    pub scratch_bytes_per_hit: u32,
}

impl Default for ResidentLaunchPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

const FRONTIER_TOPOLOGY_HYSTERESIS_BPS: u16 = 250;
const MEMORY_TOPOLOGY_HYSTERESIS_BPS: u16 = 250;

impl ResidentLaunchPolicy {
    /// Standard launch policy used by VYRE megakernel dispatchers.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            sizing: super::planner::ResidentSizingPolicy::standard(),
            min_hit_capacity: 1024,
            hit_capacity_multiplier: 2,
            saturated_waves: 4,
            hot_opcode_threshold: 8,
            hot_window_threshold: 4,
            jit_queue_len_threshold: 4096,
            priority_age_threshold: 32,
            sparse_frontier_threshold_bps: 500,
            dense_frontier_threshold_bps: 4_000,
            memory_pressure_threshold_bps: 8_500,
            fusion_edge_threshold: 65_536,
            scratch_bytes_per_hit: 16,
        }
    }

    /// Return launch recommendation cache telemetry for the current thread.
    #[must_use]
    pub fn launch_cache_stats() -> ResidentLaunchCacheStats {
        cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| cache.borrow().stats())
    }

    /// Clear launch recommendation cache entries and counters for this thread.
    pub fn reset_launch_cache_for_thread() {
        cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| cache.borrow_mut().clear());
    }

    /// Recommend geometry, hit capacity, and interpreter/JIT route.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when required adapter limits are zero or derived
    /// launch values cannot fit the u32 ring protocol.
    pub fn recommend(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<ResidentLaunchRecommendation, BackendError> {
        self.recommend_inner(request, None)
    }

    /// Recommend a launch and emit topology evidence for parity benches.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the underlying recommendation cannot be
    /// built from the request or adapter limits.
    pub fn recommend_with_topology_evidence(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<(ResidentLaunchRecommendation, ResidentTopologyEvidence), BackendError> {
        let (effective_request, recommendation) = self.recommend_with_effective_request(request)?;
        let evidence = self.topology_evidence_for(effective_request, recommendation);
        Ok((recommendation, evidence))
    }

    /// Recommend a launch and emit hot opcode/window promotion evidence.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when the underlying recommendation cannot be
    /// built from the request or adapter limits.
    pub fn recommend_with_promotion_evidence(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<(ResidentLaunchRecommendation, ResidentPromotionEvidence), BackendError> {
        let (effective_request, recommendation) = self.recommend_with_effective_request(request)?;
        let evidence = self.promotion_evidence_for(effective_request, recommendation);
        Ok((recommendation, evidence))
    }

    /// Recommend a launch while preserving the previous topology inside a
    /// narrow hysteresis band.
    ///
    /// Resident device graphs and long-running dataflow streams should use this
    /// entry point when they can track the last successful topology. It prevents
    /// borderline frontier-density or memory-pressure telemetry from repeatedly
    /// switching kernel variants, invalidating launch plans, and disturbing
    /// cache locality at scale.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when required adapter limits are zero or derived
    /// launch values cannot fit the u32 ring protocol.
    pub fn recommend_with_previous_topology(
        &self,
        request: ResidentLaunchRequest,
        previous_topology: ResidentQueueTopology,
    ) -> Result<ResidentLaunchRecommendation, BackendError> {
        self.recommend_inner(request, Some(previous_topology))
    }

    fn recommend_inner(
        &self,
        request: ResidentLaunchRequest,
        previous_topology: Option<ResidentQueueTopology>,
    ) -> Result<ResidentLaunchRecommendation, BackendError> {
        let cache_key = cache::LaunchRecommendationCacheKey {
            policy: *self,
            request,
        };
        if previous_topology.is_none() {
            if let Some(cached) =
                cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| cache.borrow_mut().get(&cache_key))
            {
                return Ok(cached);
            }
        }

        let effective_request = self.infer_missing_scale_signals(request)?;
        let promote_hot_opcodes = effective_request.hot_opcode_count >= self.hot_opcode_threshold;
        let promote_hot_windows = effective_request.hot_window_count >= self.hot_window_threshold;
        let raw_topology =
            self.dispatch_topology_for(effective_request, promote_hot_opcodes, promote_hot_windows);
        let topology = self.stabilize_topology(
            raw_topology,
            effective_request,
            previous_topology,
            promote_hot_opcodes,
            promote_hot_windows,
        );
        let scheduled_request = self.apply_topology_worker_policy(effective_request, topology)?;
        let grid = self.sizing.calculate_optimal_grid(
            ResidentGridRequest::new(
                scheduled_request.queue_len,
                scheduled_request.requested_worker_groups,
            ),
            ResidentGridLimits::new(
                scheduled_request.max_workgroup_size_x,
                scheduled_request.max_compute_workgroups_per_dimension,
                scheduled_request.max_compute_invocations_per_workgroup,
            ),
        )?;
        let geometry = grid.geometry;
        let worker_groups = grid.worker_groups;
        let lanes = u64::from(geometry.dispatch_grid[0])
            .checked_mul(u64::from(geometry.workgroup_size_x))
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel launch lane count overflowed u64. Fix: reduce dispatch grid or workgroup size.",
                )
            })?;
        let pressure = classify_pressure(
            effective_request.queue_len,
            lanes,
            effective_request.requeue_count,
            self,
        )?;
        let hit_capacity = self.hit_capacity_for(effective_request)?;
        let estimated_peak_device_bytes =
            self.estimated_peak_device_bytes(effective_request, hit_capacity)?;
        if effective_request.device_memory_budget_bytes != 0
            && estimated_peak_device_bytes > effective_request.device_memory_budget_bytes
        {
            return Err(BackendError::DeviceOutOfMemory {
                requested: estimated_peak_device_bytes,
                available: effective_request.device_memory_budget_bytes,
            });
        }
        let execution_mode = if effective_request.queue_len >= self.jit_queue_len_threshold
            || promote_hot_opcodes
            || promote_hot_windows
            || topology == ResidentQueueTopology::FusedDense
        {
            ResidentExecutionMode::Jit
        } else {
            ResidentExecutionMode::Interpreter
        };
        let age_priority_work = effective_request.requeue_count > 0
            || effective_request.max_priority_age >= self.priority_age_threshold;

        let recommendation = ResidentLaunchRecommendation {
            geometry,
            worker_groups,
            hit_capacity,
            pressure,
            execution_mode,
            topology,
            promote_hot_opcodes,
            promote_hot_windows,
            age_priority_work,
            estimated_peak_device_bytes,
            device_memory_budget_bytes: effective_request.device_memory_budget_bytes,
        };
        if previous_topology.is_none() {
            cache::LAUNCH_RECOMMENDATION_CACHE.with(|cache| {
                cache.borrow_mut().insert(cache_key, recommendation);
            });
        }
        Ok(recommendation)
    }

    fn recommend_with_effective_request(
        &self,
        request: ResidentLaunchRequest,
    ) -> Result<(ResidentLaunchRequest, ResidentLaunchRecommendation), BackendError> {
        let effective_request = self.infer_missing_scale_signals(request)?;
        let recommendation = self.recommend(effective_request)?;
        Ok((effective_request, recommendation))
    }

    fn topology_evidence_for(
        &self,
        request: ResidentLaunchRequest,
        recommendation: ResidentLaunchRecommendation,
    ) -> ResidentTopologyEvidence {
        ResidentTopologyEvidence {
            schema_version: TOPOLOGY_EVIDENCE_SCHEMA_VERSION,
            queue_pressure: recommendation.pressure,
            frontier_density_bps: request.frontier_density_bps,
            semiring_frontier_density_bps: request.frontier_density_bps,
            selected_topology: recommendation.topology,
            graphblas_switch_class: Self::graphblas_switch_class_for(recommendation.topology),
            resident_device_bytes: request.resident_device_bytes,
            estimated_peak_device_bytes: recommendation.estimated_peak_device_bytes,
            output_parity_required: true,
        }
    }

    fn promotion_evidence_for(
        &self,
        request: ResidentLaunchRequest,
        recommendation: ResidentLaunchRecommendation,
    ) -> ResidentPromotionEvidence {
        ResidentPromotionEvidence {
            schema_version: HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION,
            queue_len: request.queue_len,
            jit_queue_len_threshold: self.jit_queue_len_threshold,
            hot_opcode_count: request.hot_opcode_count,
            hot_opcode_threshold: self.hot_opcode_threshold,
            hot_window_count: request.hot_window_count,
            hot_window_threshold: self.hot_window_threshold,
            execution_mode: recommendation.execution_mode,
            promote_hot_opcodes: recommendation.promote_hot_opcodes,
            promote_hot_windows: recommendation.promote_hot_windows,
            promotion_route: Self::promotion_route_for(recommendation),
            fused_descriptor_window_required: recommendation.promote_hot_windows,
            output_parity_required: true,
        }
    }

    fn promotion_route_for(recommendation: ResidentLaunchRecommendation) -> ResidentPromotionRoute {
        if recommendation.execution_mode == ResidentExecutionMode::Interpreter {
            return ResidentPromotionRoute::Interpreter;
        }
        match (
            recommendation.promote_hot_opcodes,
            recommendation.promote_hot_windows,
        ) {
            (true, true) => ResidentPromotionRoute::OpcodeAndWindowJit,
            (true, false) => ResidentPromotionRoute::OpcodeJit,
            (false, true) => ResidentPromotionRoute::WindowJit,
            (false, false) => ResidentPromotionRoute::QueueJit,
        }
    }

    fn graphblas_switch_class_for(topology: ResidentQueueTopology) -> ResidentGraphBlasSwitchClass {
        match topology {
            ResidentQueueTopology::Empty => ResidentGraphBlasSwitchClass::Empty,
            ResidentQueueTopology::SparseFrontier => ResidentGraphBlasSwitchClass::Sparse,
            ResidentQueueTopology::HybridFrontier => ResidentGraphBlasSwitchClass::Hybrid,
            ResidentQueueTopology::DenseFrontier | ResidentQueueTopology::FusedDense => {
                ResidentGraphBlasSwitchClass::Dense
            }
            ResidentQueueTopology::MemoryConstrained => {
                ResidentGraphBlasSwitchClass::MemoryConstrained
            }
        }
    }

    fn hit_capacity_for(&self, request: ResidentLaunchRequest) -> Result<u32, BackendError> {
        if request.requested_hit_capacity != 0 {
            return Ok(request.requested_hit_capacity);
        }
        let expected_hits = request.expected_hits_per_item.max(1);
        let multiplier = if request.memory_pressure_bps >= self.memory_pressure_threshold_bps {
            1
        } else {
            self.hit_capacity_multiplier
        };
        let derived = request
            .queue_len
            .checked_mul(expected_hits)
            .and_then(|value| value.checked_mul(multiplier))
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel sparse-hit capacity overflowed u32. Fix: lower queue length, expected_hits_per_item, or hit_capacity_multiplier.",
                )
            })?;
        Ok(derived.max(self.min_hit_capacity))
    }

    fn estimated_peak_device_bytes(
        &self,
        request: ResidentLaunchRequest,
        hit_capacity: u32,
    ) -> Result<u64, BackendError> {
        let scratch_bytes = u64::from(hit_capacity)
            .checked_mul(u64::from(self.scratch_bytes_per_hit))
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel scratch byte estimate overflowed u64. Fix: lower hit capacity or scratch_bytes_per_hit.",
                )
            })?;
        request
            .resident_device_bytes
            .checked_add(scratch_bytes)
            .ok_or_else(|| {
                BackendError::new(
                    "megakernel peak resident byte estimate overflowed u64. Fix: reduce resident buffers or scratch capacity.",
                )
            })
    }

    fn infer_missing_scale_signals(
        &self,
        mut request: ResidentLaunchRequest,
    ) -> Result<ResidentLaunchRequest, BackendError> {
        if request.frontier_density_bps == 0
            && request.queue_len != 0
            && request.graph_node_count != 0
        {
            let active_nodes = u64::from(request.queue_len.min(request.graph_node_count));
            let density = active_nodes
                .checked_mul(10_000)
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel frontier-density numerator overflowed u64. Fix: shard the resident graph before launch.",
                    )
                })?
                .checked_div(u64::from(request.graph_node_count))
                .unwrap_or(0)
                .clamp(1, 10_000);
            request.frontier_density_bps = u16::try_from(density).map_err(|error| {
                BackendError::new(format!(
                    "megakernel frontier density cannot fit u16: {error}. Fix: clamp density before ABI encoding."
                ))
            })?;
        }
        if request.memory_pressure_bps == 0
            && request.device_memory_budget_bytes != 0
            && request.resident_device_bytes != 0
        {
            let pressure = (u128::from(request.resident_device_bytes)
                .checked_mul(10_000)
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel memory-pressure numerator overflowed u128. Fix: reduce resident device bytes before launch.",
                    )
                })?
                / u128::from(request.device_memory_budget_bytes))
            .min(10_000);
            request.memory_pressure_bps = u16::try_from(pressure).map_err(|error| {
                BackendError::new(format!(
                    "megakernel memory pressure cannot fit u16: {error}. Fix: clamp pressure before ABI encoding."
                ))
            })?;
        }
        Ok(request)
    }

    fn apply_topology_worker_policy(
        &self,
        mut request: ResidentLaunchRequest,
        topology: ResidentQueueTopology,
    ) -> Result<ResidentLaunchRequest, BackendError> {
        if topology == ResidentQueueTopology::MemoryConstrained
            && request.memory_pressure_bps != 0
            && request.requested_worker_groups > 1
        {
            let pressure_span = u32::from(
                10_000_u16
                    .checked_sub(self.memory_pressure_threshold_bps)
                    .ok_or_else(|| {
                        BackendError::new(
                            "megakernel memory-pressure threshold exceeds 10000 bps. Fix: configure threshold in basis points.",
                        )
                    })?,
            )
            .max(1);
            let over_threshold = u32::from(
                request
                    .memory_pressure_bps
                    .saturating_sub(self.memory_pressure_threshold_bps),
            )
            .min(pressure_span);
            let shed_bps = 2_500_u32
                .checked_add(
                    over_threshold
                        .checked_mul(2_500)
                        .ok_or_else(|| {
                            BackendError::new(
                                "megakernel memory-pressure worker shed overflowed u32. Fix: lower pressure telemetry before launch.",
                            )
                        })?
                        / pressure_span,
                )
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel memory-pressure worker shed overflowed u32. Fix: lower pressure telemetry before launch.",
                    )
                })?;
            let keep_bps = 10_000_u32.checked_sub(shed_bps).ok_or_else(|| {
                BackendError::new(
                    "megakernel memory-pressure worker keep ratio underflowed. Fix: keep shed_bps within 0..=10000.",
                )
            })?;
            let scaled = u64::from(request.requested_worker_groups)
                .checked_mul(u64::from(keep_bps))
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel memory-constrained worker count overflowed u64. Fix: reduce requested worker groups.",
                    )
                })?
                / 10_000;
            request.requested_worker_groups = u32::try_from(scaled)
                .map_err(|error| {
                    BackendError::new(format!(
                        "megakernel memory-constrained worker count cannot fit u32: {error}. Fix: reduce requested worker groups."
                    ))
                })?
                .max(1);
        }
        if topology == ResidentQueueTopology::SparseFrontier
            && request.graph_node_count != 0
            && request.frontier_density_bps != 0
            && request.requested_worker_groups > 1
        {
            let sparse_span = u32::from(self.sparse_frontier_threshold_bps).max(1);
            let density = u32::from(request.frontier_density_bps).clamp(1, sparse_span);
            let scaled = u64::from(request.requested_worker_groups)
                .checked_mul(u64::from(density))
                .ok_or_else(|| {
                    BackendError::new(
                        "megakernel sparse-frontier worker count overflowed u64. Fix: reduce requested worker groups.",
                    )
                })?
                / u64::from(sparse_span);
            let warp_floor = request.requested_worker_groups.min(32);
            request.requested_worker_groups = u32::try_from(scaled)
                .map_err(|error| {
                    BackendError::new(format!(
                        "megakernel sparse-frontier worker count cannot fit u32: {error}. Fix: reduce requested worker groups."
                    ))
                })?
                .max(warp_floor)
                .min(request.requested_worker_groups);
        }
        Ok(request)
    }

    fn dispatch_topology_for(
        &self,
        request: ResidentLaunchRequest,
        promote_hot_opcodes: bool,
        promote_hot_windows: bool,
    ) -> ResidentQueueTopology {
        if request.queue_len == 0 {
            return ResidentQueueTopology::Empty;
        }
        if request.memory_pressure_bps >= self.memory_pressure_threshold_bps {
            return ResidentQueueTopology::MemoryConstrained;
        }
        if request.frontier_density_bps <= self.sparse_frontier_threshold_bps {
            return ResidentQueueTopology::SparseFrontier;
        }
        let dense = request.frontier_density_bps >= self.dense_frontier_threshold_bps;
        let graph_is_large =
            request.graph_node_count > 0 && request.graph_edge_count >= self.fusion_edge_threshold;
        if dense && graph_is_large && (promote_hot_opcodes || promote_hot_windows) {
            return ResidentQueueTopology::FusedDense;
        }
        if dense {
            return ResidentQueueTopology::DenseFrontier;
        }
        ResidentQueueTopology::HybridFrontier
    }

    fn stabilize_topology(
        &self,
        raw_topology: ResidentQueueTopology,
        request: ResidentLaunchRequest,
        previous_topology: Option<ResidentQueueTopology>,
        promote_hot_opcodes: bool,
        promote_hot_windows: bool,
    ) -> ResidentQueueTopology {
        if raw_topology == ResidentQueueTopology::Empty {
            return raw_topology;
        }
        if raw_topology == ResidentQueueTopology::MemoryConstrained {
            return raw_topology;
        }
        let Some(previous_topology) = previous_topology else {
            return raw_topology;
        };
        if previous_topology == ResidentQueueTopology::MemoryConstrained
            && request.memory_pressure_bps
                >= hysteresis_sub(
                    self.memory_pressure_threshold_bps,
                    MEMORY_TOPOLOGY_HYSTERESIS_BPS,
                )
        {
            return ResidentQueueTopology::MemoryConstrained;
        }

        match previous_topology {
            ResidentQueueTopology::SparseFrontier
                if raw_topology != ResidentQueueTopology::SparseFrontier
                    && request.frontier_density_bps
                        <= hysteresis_add(
                            self.sparse_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        ) =>
            {
                ResidentQueueTopology::SparseFrontier
            }
            ResidentQueueTopology::HybridFrontier
                if raw_topology == ResidentQueueTopology::SparseFrontier
                    && request.frontier_density_bps
                        >= hysteresis_sub(
                            self.sparse_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        ) =>
            {
                ResidentQueueTopology::HybridFrontier
            }
            ResidentQueueTopology::HybridFrontier
                if matches!(
                    raw_topology,
                    ResidentQueueTopology::DenseFrontier | ResidentQueueTopology::FusedDense
                ) && request.frontier_density_bps
                    <= hysteresis_add(
                        self.dense_frontier_threshold_bps,
                        FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                    ) =>
            {
                ResidentQueueTopology::HybridFrontier
            }
            ResidentQueueTopology::DenseFrontier
                if raw_topology == ResidentQueueTopology::HybridFrontier
                    && request.frontier_density_bps
                        >= hysteresis_sub(
                            self.dense_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        ) =>
            {
                ResidentQueueTopology::DenseFrontier
            }
            ResidentQueueTopology::FusedDense
                if raw_topology == ResidentQueueTopology::HybridFrontier
                    && request.frontier_density_bps
                        >= hysteresis_sub(
                            self.dense_frontier_threshold_bps,
                            FRONTIER_TOPOLOGY_HYSTERESIS_BPS,
                        )
                    && request.graph_edge_count >= self.fusion_edge_threshold
                    && (promote_hot_opcodes || promote_hot_windows) =>
            {
                ResidentQueueTopology::FusedDense
            }
            _ => raw_topology,
        }
    }

    /// Select the best `hit_capacity_multiplier` from a candidate set.
    ///
    /// `candidate_multipliers` are the multipliers to try; `costs[i]`
    /// is the observed dispatch latency (or any minimization metric)
    /// when `candidate_multipliers[i]` was used. Lower cost wins; the
    /// minimum observed cost selects the multiplier.
    ///
    /// Returns the chosen multiplier. If `candidate_multipliers` is
    /// empty, returns the policy's existing `hit_capacity_multiplier`.
    ///
    #[must_use]
    pub fn autotune_hit_capacity_multiplier(
        &self,
        candidate_multipliers: &[u32],
        costs: &[f64],
    ) -> u32 {
        if candidate_multipliers.is_empty() || costs.is_empty() {
            return self.hit_capacity_multiplier;
        }
        let n = candidate_multipliers.len().min(costs.len());
        best_cost_index(&costs[..n])
            .and_then(|chosen| candidate_multipliers.get(chosen).copied())
            .unwrap_or(self.hit_capacity_multiplier)
    }

    /// Select the best workgroup-size from a candidate set.
    ///
    /// `candidate_sizes[i]` is paired
    /// with `costs[i]` (lower is better). Returns the chosen size or
    /// the policy's `sizing.default_workgroup_size_x()` fallback.
    #[must_use]
    pub fn autotune_workgroup_size(
        &self,
        candidate_sizes: &[u32],
        costs: &[f64],
        current_size: u32,
    ) -> u32 {
        if candidate_sizes.is_empty() || costs.is_empty() {
            return current_size;
        }
        let n = candidate_sizes.len().min(costs.len());
        best_cost_index(&costs[..n])
            .and_then(|chosen| candidate_sizes.get(chosen).copied())
            .unwrap_or(current_size)
    }

    /// Compute the next-step parameter delta for a continuous autotune
    /// knob using a Fisher-preconditioned natural-gradient step.
    ///
    /// `m_inv_sqrt`: inverse-square-root of the Fisher block (n×n
    /// row-major). Passing an identity matrix reduces the natural
    /// gradient to plain gradient descent.
    ///
    /// `grad`: plain gradient ∂latency/∂param (length n).
    ///
    /// Returns the parameter delta `-lr · M_inv_sqrt · grad`.
    ///
    /// P-DRIVER-8: every continuous autotune knob (workgroup size,
    /// hit-capacity, fixpoint iteration count, …) should follow the
    /// natural-gradient direction by default  -  Fisher-preconditioned
    /// descent converges 5-10× faster than plain gradient on the
    /// elongated-valley latency surfaces typical of GPU autotuning.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when host staging cannot be reserved for the
    /// natural-gradient vector.
    pub fn try_natural_gradient_autotune_step(
        m_inv_sqrt: &[f64],
        grad: &[f64],
        n: u32,
        learning_rate: f64,
    ) -> Result<Vec<f64>, BackendError> {
        let mut out = Vec::new();
        Self::try_natural_gradient_autotune_step_into(
            m_inv_sqrt,
            grad,
            n,
            learning_rate,
            &mut out,
        )?;
        Ok(out)
    }

    /// Compute the natural-gradient autotune step into caller-owned storage.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when host staging cannot be reserved for the
    /// natural-gradient vector.
    pub fn try_natural_gradient_autotune_step_into(
        m_inv_sqrt: &[f64],
        grad: &[f64],
        n: u32,
        learning_rate: f64,
        out: &mut Vec<f64>,
    ) -> Result<(), BackendError> {
        let n = u32_to_usize_checked(n, "natural-gradient dimension")?;
        out.clear();
        let Some(required_matrix_len) = n.checked_mul(n) else {
            return Ok(());
        };
        if m_inv_sqrt.len() < required_matrix_len || grad.len() < n {
            return Ok(());
        }
        reserve_target_capacity(out, n, "natural-gradient output")?;
        out.resize(n, 0.0);
        for row in 0..n {
            let mut acc = 0.0;
            for col in 0..n {
                acc += m_inv_sqrt[row * n + col] * grad[col];
            }
            out[row] = -learning_rate * acc;
        }
        Ok(())
    }
}

fn diffuse_step_into(
    stalks: &[f64],
    restriction_diag: &[f64],
    damping: f64,
    out: &mut Vec<f64>,
) -> Result<(), BackendError> {
    out.clear();
    reserve_target_capacity(out, stalks.len(), "priority diffusion scratch")?;
    out.resize(stalks.len(), 0.0);
    for ((slot, &stalk), &restriction) in out
        .iter_mut()
        .zip(stalks.iter())
        .zip(restriction_diag.iter())
    {
        *slot = stalk - damping * restriction * stalk;
    }
    Ok(())
}

fn reserve_target_capacity<T>(
    out: &mut Vec<T>,
    target_capacity: usize,
    label: &'static str,
) -> Result<(), BackendError> {
    try_reserve_vec_capacity(out, target_capacity).map_err(|source| {
        BackendError::new(format!(
            "megakernel {label} reservation failed for {target_capacity} element(s): {source}. Fix: shard the policy input before launch-policy math."
        ))
    })
}

/// Index of the lowest cost, or `None` when no cost was measured.
///
/// The empty case is in the return type rather than in an assertion, because an
/// assertion compiled out of the shipped binary leaves an unchecked index at the
/// only point where the caller has nothing to select from.
fn best_cost_index(costs: &[f64]) -> Option<usize> {
    let (first, rest) = costs.split_first()?;
    let mut best = 0;
    let mut best_cost = *first;
    for (index, &cost) in rest.iter().enumerate() {
        if cost.total_cmp(&best_cost).is_lt() {
            best = index + 1;
            best_cost = cost;
        }
    }
    Some(best)
}

fn u32_to_usize_checked(value: u32, label: &'static str) -> Result<usize, BackendError> {
    usize::try_from(value).map_err(|error| {
        BackendError::new(format!(
            "{label} cannot fit usize: {error}. Fix: shard the autotune surface."
        ))
    })
}

fn hysteresis_add(value: u16, hysteresis: u16) -> u16 {
    value.saturating_add(hysteresis)
}

fn hysteresis_sub(value: u16, hysteresis: u16) -> u16 {
    value.saturating_sub(hysteresis)
}

fn classify_pressure(
    queue_len: u32,
    lanes: u64,
    requeue_count: u64,
    policy: &ResidentLaunchPolicy,
) -> Result<ResidentQueuePressure, BackendError> {
    if queue_len == 0 {
        return Ok(ResidentQueuePressure::Empty);
    }
    let lanes = lanes.max(1);
    let queue_len = u64::from(queue_len);
    let saturated_lanes = lanes
        .checked_mul(u64::from(policy.saturated_waves))
        .ok_or_else(|| {
            BackendError::new(
                "megakernel pressure wave threshold overflowed u64. Fix: reduce worker lanes or saturated_waves.",
            )
        })?;
    if requeue_count > 0 || queue_len >= saturated_lanes {
        Ok(ResidentQueuePressure::Saturated)
    } else if queue_len >= lanes {
        Ok(ResidentQueuePressure::Balanced)
    } else {
        Ok(ResidentQueuePressure::Light)
    }
}

// Inline: covers the private `cache` module and its `pub(super)`
// `LaunchRecommendationCache` and `LaunchRecommendationCacheKey`, which no
// integration test can reach.
#[cfg(test)]
mod tests {
    use super::cache::{LaunchRecommendationCache, LaunchRecommendationCacheKey};
    use super::*;

    mod cache_contracts {
        use super::*;

        #[test]
        fn launch_cache_update_does_not_duplicate_entries() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest::direct(128, 64, 256);
            let key = LaunchRecommendationCacheKey { policy, request };
            let rec = policy
                .recommend(request)
                .expect("Fix: policy should accept non-zero adapter limits");
            let mut cache = LaunchRecommendationCache::default();

            cache.insert(key, rec);
            cache.insert(key, rec);

            assert_eq!(cache.len(), 1);
        }

        #[test]

        fn launch_cache_get_promotes_hot_key_before_eviction() {
            let policy = ResidentLaunchPolicy::standard();
            let hot_request = ResidentLaunchRequest::direct(1, 64, 256);
            let hot_key = LaunchRecommendationCacheKey {
                policy,
                request: hot_request,
            };
            let hot_rec = policy
                .recommend(hot_request)
                .expect("Fix: policy should accept non-zero adapter limits");
            let mut cache = LaunchRecommendationCache::default();

            cache.insert(hot_key, hot_rec);
            for queue_len in 2..=128 {
                let request = ResidentLaunchRequest::direct(queue_len, 64, 256);
                let rec = policy
                    .recommend(request)
                    .expect("Fix: policy should accept non-zero adapter limits");
                cache.insert(LaunchRecommendationCacheKey { policy, request }, rec);
            }
            assert!(cache.get(&hot_key).is_some());
            assert_eq!(cache.hits, 1);
            assert_eq!(cache.misses, 0);

            let cold_request = ResidentLaunchRequest::direct(129, 64, 256);
            let cold_rec = policy
                .recommend(cold_request)
                .expect("Fix: policy should accept non-zero adapter limits");
            cache.insert(
                LaunchRecommendationCacheKey {
                    policy,
                    request: cold_request,
                },
                cold_rec,
            );

            assert!(cache.get(&hot_key).is_some());
            assert_eq!(cache.hits, 2);
            assert_eq!(cache.len(), 128);
        }

        #[test]
        fn launch_cache_records_misses_without_mutating_capacity() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest::direct(128, 64, 256);
            let missing = LaunchRecommendationCacheKey { policy, request };
            let mut cache = LaunchRecommendationCache::default();

            assert!(cache.get(&missing).is_none());

            assert_eq!(cache.hits, 0);
            assert_eq!(cache.misses, 1);
            assert_eq!(cache.len(), 0);
        }

        #[test]
        fn launch_policy_exposes_thread_local_cache_stats() {
            ResidentLaunchPolicy::reset_launch_cache_for_thread();
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest::direct(512, 64, 256);

            let initial = ResidentLaunchPolicy::launch_cache_stats();
            assert_eq!(initial.entries, 0);
            assert_eq!(initial.hits, 0);
            assert_eq!(initial.misses, 0);

            let first = policy
                .recommend(request)
                .expect("Fix: valid policy request must recommend");
            let after_miss = ResidentLaunchPolicy::launch_cache_stats();
            assert_eq!(after_miss.entries, 1);
            assert_eq!(after_miss.hits, 0);
            assert_eq!(after_miss.misses, 1);

            let second = policy
                .recommend(request)
                .expect("Fix: cached policy request must recommend");
            let after_hit = ResidentLaunchPolicy::launch_cache_stats();
            assert_eq!(first, second);
            assert_eq!(after_hit.entries, 1);
            assert_eq!(after_hit.hits, 1);
            assert_eq!(after_hit.misses, 1);

            ResidentLaunchPolicy::reset_launch_cache_for_thread();
        }
    }

    mod hysteresis_contracts {
        use super::*;

        #[test]
        fn stable_recommendation_holds_sparse_topology_inside_frontier_hysteresis() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest {
                queue_len: 8_192,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                graph_node_count: 100_000,
                graph_edge_count: 250_000,
                frontier_density_bps: policy.sparse_frontier_threshold_bps + 125,
                ..ResidentLaunchRequest::direct(8_192, 128, 256)
            };
            let stateless = policy
                .recommend(request)
                .expect("Fix: stateless launch recommendation should accept valid adapter limits");
            let stable = policy
                .recommend_with_previous_topology(request, ResidentQueueTopology::SparseFrontier)
                .expect("Fix: stable launch recommendation should accept valid adapter limits");

            assert_eq!(stateless.topology, ResidentQueueTopology::HybridFrontier);
            assert_eq!(stable.topology, ResidentQueueTopology::SparseFrontier);
        }

        #[test]
        fn stable_recommendation_releases_sparse_topology_outside_frontier_hysteresis() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend_with_previous_topology(
                    ResidentLaunchRequest {
                        queue_len: 8_192,
                        requested_worker_groups: 128,
                        max_workgroup_size_x: 256,
                        graph_node_count: 100_000,
                        graph_edge_count: 250_000,
                        frontier_density_bps: policy.sparse_frontier_threshold_bps + 300,
                        ..ResidentLaunchRequest::direct(8_192, 128, 256)
                    },
                    ResidentQueueTopology::SparseFrontier,
                )
                .expect("Fix: stable launch recommendation should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
        }

        #[test]
        fn stable_recommendation_holds_hybrid_topology_inside_sparse_hysteresis() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend_with_previous_topology(
                    ResidentLaunchRequest {
                        queue_len: 8_192,
                        requested_worker_groups: 128,
                        max_workgroup_size_x: 256,
                        graph_node_count: 100_000,
                        graph_edge_count: 250_000,
                        frontier_density_bps: policy.sparse_frontier_threshold_bps - 125,
                        ..ResidentLaunchRequest::direct(8_192, 128, 256)
                    },
                    ResidentQueueTopology::HybridFrontier,
                )
                .expect("Fix: stable launch recommendation should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
        }

        #[test]
        fn stable_recommendation_holds_hybrid_topology_inside_dense_hysteresis() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend_with_previous_topology(
                    ResidentLaunchRequest {
                        queue_len: 16_384,
                        requested_worker_groups: 128,
                        max_workgroup_size_x: 256,
                        graph_node_count: 16_384,
                        graph_edge_count: 250_000,
                        frontier_density_bps: policy.dense_frontier_threshold_bps + 125,
                        ..ResidentLaunchRequest::direct(16_384, 128, 256)
                    },
                    ResidentQueueTopology::HybridFrontier,
                )
                .expect("Fix: stable launch recommendation should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
        }

        #[test]
        fn stable_recommendation_holds_dense_topology_inside_frontier_hysteresis() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest {
                queue_len: 16_384,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                graph_node_count: 16_384,
                graph_edge_count: 250_000,
                frontier_density_bps: policy.dense_frontier_threshold_bps - 125,
                ..ResidentLaunchRequest::direct(16_384, 128, 256)
            };
            let stateless = policy
                .recommend(request)
                .expect("Fix: stateless launch recommendation should accept valid adapter limits");
            let stable = policy
                .recommend_with_previous_topology(request, ResidentQueueTopology::DenseFrontier)
                .expect("Fix: stable launch recommendation should accept valid adapter limits");

            assert_eq!(stateless.topology, ResidentQueueTopology::HybridFrontier);
            assert_eq!(stable.topology, ResidentQueueTopology::DenseFrontier);
        }

        #[test]
        fn stable_recommendation_preserves_fused_dense_when_hot_graph_stays_near_dense() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend_with_previous_topology(
                    ResidentLaunchRequest {
                        queue_len: 131_072,
                        requested_worker_groups: 256,
                        max_workgroup_size_x: 256,
                        graph_node_count: 32_768,
                        graph_edge_count: 500_000,
                        frontier_density_bps: policy.dense_frontier_threshold_bps - 125,
                        hot_window_count: policy.hot_window_threshold,
                        ..ResidentLaunchRequest::direct(131_072, 256, 256)
                    },
                    ResidentQueueTopology::FusedDense,
                )
                .expect("Fix: stable fused dense recommendation should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
            assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
        }

        #[test]
        fn stable_recommendation_holds_memory_constrained_topology_inside_pressure_hysteresis() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest {
                queue_len: 16_384,
                requested_worker_groups: 128,
                max_workgroup_size_x: 256,
                graph_node_count: 16_384,
                graph_edge_count: 250_000,
                frontier_density_bps: 9_000,
                memory_pressure_bps: policy.memory_pressure_threshold_bps - 125,
                ..ResidentLaunchRequest::direct(16_384, 128, 256)
            };
            let stateless = policy
                .recommend(request)
                .expect("Fix: stateless launch recommendation should accept valid adapter limits");
            let stable = policy
                .recommend_with_previous_topology(request, ResidentQueueTopology::MemoryConstrained)
                .expect("Fix: stable launch recommendation should accept valid adapter limits");

            assert_eq!(stateless.topology, ResidentQueueTopology::DenseFrontier);
            assert_eq!(stable.topology, ResidentQueueTopology::MemoryConstrained);
            assert!(
                stable.worker_groups < stateless.worker_groups,
                "stable memory-constrained topology must preserve worker shedding near pressure threshold"
            );
        }
    }

    mod priority_diffusion_contracts {
        use super::*;

        #[test]
        fn priority_accounting_reports_structured_drain_before_overflow() {
            let accounting = PriorityRequeueAccounting {
                requeue_count: u64::MAX - 8,
                aged_promotions: 3,
                max_priority_age: 64,
            };
            let recommendation = accounting.drain_recommendation();

            assert!(recommendation.should_drain);
            assert_eq!(
                recommendation.reason,
                PriorityDrainReason::RequeueCounterNearLimit
            );
            assert_eq!(recommendation.requeue_count, u64::MAX - 8);
            assert_eq!(recommendation.aged_promotions, 3);
            assert_eq!(recommendation.max_priority_age, 64);
            assert_eq!(recommendation.requeue_counter_headroom, 8);
            assert_eq!(recommendation.aged_promotion_counter_headroom, u64::MAX - 3);
            assert_eq!(recommendation.fix, PRIORITY_COUNTER_DRAIN_FIX);
        }

        #[test]
        fn priority_accounting_reports_no_drain_for_empty_counters() {
            let recommendation = PriorityRequeueAccounting::default().drain_recommendation();

            assert!(!recommendation.should_drain);
            assert_eq!(recommendation.reason, PriorityDrainReason::None);
            assert_eq!(recommendation.requeue_count, 0);
            assert_eq!(recommendation.aged_promotions, 0);
            assert_eq!(recommendation.max_priority_age, 0);
            assert_eq!(recommendation.requeue_counter_headroom, u64::MAX);
            assert_eq!(recommendation.aged_promotion_counter_headroom, u64::MAX);
            assert_eq!(recommendation.fix, PRIORITY_COUNTER_DRAIN_FIX);
        }

        #[test]
        fn diffuse_priority_mismatched_restrictions_preserve_input_shape() {
            let input = [3.0, 1.0, 2.0];
            let restrictions = [1.0, 0.5];
            let mut out = Vec::with_capacity(input.len());
            let mut scratch = Vec::with_capacity(input.len());

            try_diffuse_priority_across_siblings_into(
                &input,
                &restrictions,
                0.5,
                4,
                &mut out,
                &mut scratch,
            )
            .expect("Fix: diffusion staging must succeed for three siblings");

            assert_eq!(out, input);
            assert!(scratch.is_empty());
            assert_eq!(out.capacity(), input.len());
        }

        #[test]
        fn diffuse_priority_reuses_exact_scratch_capacity() {
            let input = [4.0, 2.0, 1.0];
            let restrictions = [1.0, 1.0, 1.0];
            let mut out = Vec::with_capacity(input.len());
            let mut scratch = Vec::with_capacity(input.len());
            let out_ptr = out.as_ptr();
            let scratch_ptr = scratch.as_ptr();

            try_diffuse_priority_across_siblings_into(
                &input,
                &restrictions,
                0.25,
                2,
                &mut out,
                &mut scratch,
            )
            .expect("Fix: diffusion staging must succeed for three siblings");

            assert_eq!(out.len(), input.len());
            assert_eq!(scratch.len(), input.len());
            assert_eq!(out.capacity(), input.len());
            assert_eq!(scratch.capacity(), input.len());
            assert_eq!(out.as_ptr(), out_ptr);
            assert_eq!(scratch.as_ptr(), scratch_ptr);
        }
    }

    mod recommendation_contracts {
        use super::*;

        #[test]
        fn policy_recommends_padded_geometry_and_hit_capacity() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 300,
                    requested_worker_groups: 64,
                    max_workgroup_size_x: 256,
                    requested_hit_capacity: 0,
                    expected_hits_per_item: 3,
                    ..ResidentLaunchRequest::direct(300, 64, 256)
                })
                .expect("Fix: policy should accept non-zero adapter limits");
            assert_eq!(rec.geometry.workgroup_size_x, 64);
            assert_eq!(rec.geometry.slot_count, 320);
            assert_eq!(rec.geometry.dispatch_grid, [5, 1, 1]);
            assert_eq!(rec.hit_capacity, 1800);
            assert_eq!(rec.estimated_peak_device_bytes, 28_800);
            assert_eq!(rec.device_memory_budget_bytes, 0);
            assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
        }

        #[test]
        fn telemetry_pressure_selects_jit_and_priority_aging() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 8192,
                    requested_worker_groups: 64,
                    max_workgroup_size_x: 256,
                    hot_opcode_count: 8,
                    requeue_count: 1,
                    max_priority_age: 64,
                    ..ResidentLaunchRequest::direct(8192, 64, 256)
                })
                .expect("Fix: policy should accept non-zero adapter limits");
            assert_eq!(rec.pressure, ResidentQueuePressure::Saturated);
            assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
            assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
            assert!(rec.promote_hot_opcodes);
            assert!(rec.age_priority_work);
        }

        #[test]
        fn dense_large_hot_graph_selects_fused_dense_topology() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 131_072,
                    requested_worker_groups: 256,
                    max_workgroup_size_x: 256,
                    graph_node_count: 32_768,
                    graph_edge_count: 500_000,
                    frontier_density_bps: 7_500,
                    hot_window_count: policy.hot_window_threshold,
                    ..ResidentLaunchRequest::direct(131_072, 256, 256)
                })
                .expect("Fix: fused dense topology should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
            assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
        }

        #[test]
        fn topology_evidence_reports_graphblas_switch_inputs_and_parity_contract() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest {
                queue_len: 131_072,
                requested_worker_groups: 256,
                max_workgroup_size_x: 256,
                graph_node_count: 32_768,
                graph_edge_count: 500_000,
                frontier_density_bps: 7_500,
                hot_window_count: policy.hot_window_threshold,
                resident_device_bytes: 64 * 1024 * 1024,
                ..ResidentLaunchRequest::direct(131_072, 256, 256)
            };
            let (rec, evidence) = policy
                .recommend_with_topology_evidence(request)
                .expect("Fix: topology evidence should be emitted for valid launch telemetry");

            assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
            assert_eq!(evidence.schema_version, TOPOLOGY_EVIDENCE_SCHEMA_VERSION);
            assert_eq!(evidence.selected_topology, rec.topology);
            assert_eq!(evidence.queue_pressure, rec.pressure);
            assert_eq!(evidence.frontier_density_bps, 7_500);
            assert_eq!(evidence.semiring_frontier_density_bps, 7_500);
            assert_eq!(
                evidence.graphblas_switch_class,
                ResidentGraphBlasSwitchClass::Dense
            );
            assert_eq!(evidence.resident_device_bytes, 64 * 1024 * 1024);
            assert_eq!(
                evidence.estimated_peak_device_bytes,
                rec.estimated_peak_device_bytes
            );
            assert!(evidence.output_parity_required);
            assert!(evidence.is_complete());
        }

        #[test]
        fn promotion_evidence_reports_fused_window_lowerer_contract() {
            let policy = ResidentLaunchPolicy::standard();
            let request = ResidentLaunchRequest {
                queue_len: 1024,
                requested_worker_groups: 64,
                max_workgroup_size_x: 256,
                hot_window_count: policy.hot_window_threshold,
                ..ResidentLaunchRequest::direct(1024, 64, 256)
            };
            let (rec, evidence) = policy
                .recommend_with_promotion_evidence(request)
                .expect("Fix: promotion evidence should be emitted for valid hot-window telemetry");

            assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
            assert!(rec.promote_hot_windows);
            assert_eq!(
                evidence.schema_version,
                HOT_WINDOW_PROMOTION_EVIDENCE_SCHEMA_VERSION
            );
            assert_eq!(evidence.queue_len, 1024);
            assert_eq!(evidence.hot_window_count, policy.hot_window_threshold);
            assert_eq!(evidence.hot_window_threshold, policy.hot_window_threshold);
            assert_eq!(evidence.hot_opcode_count, 0);
            assert_eq!(evidence.hot_opcode_threshold, policy.hot_opcode_threshold);
            assert_eq!(evidence.execution_mode, ResidentExecutionMode::Jit);
            assert_eq!(evidence.promotion_route, ResidentPromotionRoute::WindowJit);
            assert!(evidence.promote_hot_windows);
            assert!(!evidence.promote_hot_opcodes);
            assert!(evidence.fused_descriptor_window_required);
            assert!(evidence.output_parity_required);
            assert!(evidence.is_complete());
        }

        #[test]
        fn high_memory_pressure_overrides_dense_frontier() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 16_384,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    graph_node_count: 16_384,
                    graph_edge_count: 250_000,
                    frontier_density_bps: 9_000,
                    memory_pressure_bps: policy.memory_pressure_threshold_bps,
                    ..ResidentLaunchRequest::direct(16_384, 128, 256)
                })
                .expect("Fix: memory-constrained topology should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::MemoryConstrained);
            assert!(
                rec.worker_groups < 128,
                "memory-constrained topology must lower worker-group pressure, got {}",
                rec.worker_groups
            );
            assert_eq!(
                rec.hit_capacity, 16_384,
                "memory-constrained topology must avoid the normal sparse-hit over-allocation multiplier"
            );
        }

        #[test]
        fn explicit_hit_capacity_survives_memory_constrained_worker_shedding() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 16_384,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    requested_hit_capacity: 65_536,
                    memory_pressure_bps: 10_000,
                    ..ResidentLaunchRequest::direct(16_384, 128, 256)
                })
                .expect(
                    "Fix: memory-constrained explicit-capacity launch should accept valid adapter limits",
                );

            assert_eq!(rec.topology, ResidentQueueTopology::MemoryConstrained);
            assert_eq!(rec.hit_capacity, 65_536);
            assert_eq!(rec.worker_groups, 64);
        }

        #[test]
        fn device_memory_budget_rejects_oversized_hit_plan_before_allocation() {
            let policy = ResidentLaunchPolicy::standard();
            let err = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 1024,
                    requested_worker_groups: 64,
                    max_workgroup_size_x: 256,
                    expected_hits_per_item: 4,
                    resident_device_bytes: 1024,
                    device_memory_budget_bytes: 64 * 1024,
                    ..ResidentLaunchRequest::direct(1024, 64, 256)
                })
                .expect_err("Fix: launch policy must reject plans that exceed explicit device budget");

            match err {
                vyre_driver::BackendError::DeviceOutOfMemory {
                    requested,
                    available,
                } => {
                    assert_eq!(requested, 132_096);
                    assert_eq!(available, 64 * 1024);
                }
                other => panic!("expected DeviceOutOfMemory for budget overflow, got {other:?}"),
            }
        }

        #[test]
        fn device_memory_budget_infers_pressure_without_manual_bps() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 1024,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    resident_device_bytes: 900_000,
                    device_memory_budget_bytes: 1_000_000,
                    ..ResidentLaunchRequest::direct(1024, 128, 256)
                })
                .expect("Fix: budget-aware policy should accept launches under the byte budget");

            assert_eq!(rec.topology, ResidentQueueTopology::MemoryConstrained);
            assert!(
                rec.worker_groups < 128,
                "inferred memory pressure must shed worker groups before launch"
            );
            assert_eq!(rec.estimated_peak_device_bytes, 916_384);
            assert_eq!(rec.device_memory_budget_bytes, 1_000_000);
        }

        #[test]
        fn dense_frontier_without_hot_fusion_stays_dense() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 16_384,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    graph_node_count: 16_384,
                    graph_edge_count: 250_000,
                    frontier_density_bps: policy.dense_frontier_threshold_bps,
                    ..ResidentLaunchRequest::direct(16_384, 128, 256)
                })
                .expect("Fix: dense topology should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::DenseFrontier);
        }

        #[test]
        fn mid_density_frontier_selects_hybrid_topology() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 8192,
                    requested_worker_groups: 128,
                    max_workgroup_size_x: 256,
                    graph_node_count: 8192,
                    graph_edge_count: 32_768,
                    frontier_density_bps: policy.sparse_frontier_threshold_bps + 1,
                    ..ResidentLaunchRequest::direct(8192, 128, 256)
                })
                .expect("Fix: hybrid topology should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::HybridFrontier);
        }

        #[test]
        fn missing_frontier_telemetry_infers_density_from_queue_and_graph_scale() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 90_000,
                    requested_worker_groups: 256,
                    max_workgroup_size_x: 256,
                    graph_node_count: 100_000,
                    graph_edge_count: 750_000,
                    hot_opcode_count: policy.hot_opcode_threshold,
                    frontier_density_bps: 0,
                    ..ResidentLaunchRequest::direct(90_000, 256, 256)
                })
                .expect("Fix: inferred-density topology should accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::FusedDense);
            assert_eq!(rec.execution_mode, ResidentExecutionMode::Jit);
        }

        #[test]
        fn sparse_frontier_density_sheds_worker_pressure_without_losing_warp_floor() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 100_000,
                    requested_worker_groups: 256,
                    max_workgroup_size_x: 256,
                    graph_node_count: 1_000_000,
                    graph_edge_count: 4_000_000,
                    frontier_density_bps: 100,
                    ..ResidentLaunchRequest::direct(100_000, 256, 256)
                })
                .expect("Fix: sparse density worker shedding must accept valid adapter limits");

            assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
            assert_eq!(rec.worker_groups, 51);
            assert_eq!(rec.geometry.workgroup_size_x, 51);
            assert_eq!(rec.geometry.dispatch_grid, [51, 1, 1]);
        }

        #[test]
        fn sparse_frontier_worker_shedding_preserves_warp_floor_for_tiny_density() {
            let policy = ResidentLaunchPolicy::standard();
            let rec = policy
                .recommend(ResidentLaunchRequest {
                    queue_len: 1_000,
                    requested_worker_groups: 256,
                    max_workgroup_size_x: 256,
                    graph_node_count: 1_000_000,
                    graph_edge_count: 4_000_000,
                    frontier_density_bps: 1,
                    ..ResidentLaunchRequest::direct(1_000, 256, 256)
                })
                .expect("Fix: sparse density worker shedding must retain a useful GPU width");

            assert_eq!(rec.topology, ResidentQueueTopology::SparseFrontier);
            assert_eq!(rec.worker_groups, 32);
            assert_eq!(rec.geometry.workgroup_size_x, 32);
        }
    }

    // Inline: `best_cost_index` is crate-private and the public knobs refuse an
    // empty candidate set before reaching it, so no integration test can hand it
    // the empty slice its own contract has to answer for.
    mod autotune_selection_contracts {
        use super::*;

        /// No measured cost selects nothing.
        ///
        /// The empty case used to be a `debug_assert` in front of `costs[0]`,
        /// which is absent from a release build, so the shipped binary indexed
        /// an empty slice.
        #[test]
        fn no_measured_cost_selects_nothing() {
            assert_eq!(best_cost_index(&[]), None);
        }

        /// The lowest cost wins, and the first of a tie keeps the selection stable.
        #[test]
        fn the_lowest_cost_wins_and_a_tie_keeps_the_earlier_candidate() {
            assert_eq!(best_cost_index(&[3.0]), Some(0));
            assert_eq!(best_cost_index(&[3.0, 1.0, 2.0]), Some(1));
            assert_eq!(best_cost_index(&[1.0, 5.0, 1.0]), Some(0));
            assert_eq!(best_cost_index(&[5.0, 4.0, 3.0, 2.0]), Some(3));
        }

        /// Every position is reachable, so the scan reports the index it scanned.
        ///
        /// The scan skips the first cost and counts from the rest, so an index
        /// that is off by one selects the neighbour of the cheapest candidate at
        /// every position except the first, which is exactly the case a single
        /// example misses.
        #[test]
        fn the_reported_index_is_the_position_of_the_lowest_cost() {
            let width = 6;
            for cheapest in 0..width {
                let costs: Vec<f64> = (0..width)
                    .map(|index| if index == cheapest { 1.0 } else { 9.0 })
                    .collect();
                assert_eq!(
                    best_cost_index(&costs),
                    Some(cheapest),
                    "Fix: the lowest cost at {cheapest} of {width} selected the wrong candidate"
                );
            }
        }

        /// A cost that is not a number never beats a measured one.
        #[test]
        fn an_unmeasurable_cost_never_wins() {
            assert_eq!(best_cost_index(&[f64::NAN, 2.0]), Some(1));
            assert_eq!(best_cost_index(&[2.0, f64::NAN]), Some(0));
            assert_eq!(best_cost_index(&[f64::NAN, f64::NAN]), Some(0));
        }

        /// The public knobs return the candidate that the cheapest cost sits against.
        #[test]
        fn the_autotune_knobs_return_the_candidate_paired_with_the_lowest_cost() {
            let policy = ResidentLaunchPolicy::standard();
            assert_eq!(
                policy.autotune_workgroup_size(&[64, 128, 256], &[3.0, 1.0, 2.0], 32),
                128
            );
            assert_eq!(
                policy.autotune_hit_capacity_multiplier(&[2, 4, 8], &[5.0, 4.0, 1.0]),
                8
            );
        }

        /// A knob with nothing measured keeps the value it was given.
        #[test]
        fn the_autotune_knobs_keep_the_current_value_when_nothing_was_measured() {
            let policy = ResidentLaunchPolicy::standard();
            assert_eq!(policy.autotune_workgroup_size(&[64, 128], &[], 32), 32);
            assert_eq!(policy.autotune_workgroup_size(&[], &[1.0], 32), 32);
            assert_eq!(
                policy.autotune_hit_capacity_multiplier(&[2, 4], &[]),
                policy.hit_capacity_multiplier
            );
            assert_eq!(
                policy.autotune_hit_capacity_multiplier(&[], &[1.0]),
                policy.hit_capacity_multiplier
            );
        }

        /// More candidates than costs selects only among the costs that exist.
        #[test]
        fn a_candidate_without_a_cost_is_not_selected() {
            let policy = ResidentLaunchPolicy::standard();
            assert_eq!(
                policy.autotune_workgroup_size(&[64, 128, 256], &[2.0, 1.0], 32),
                128
            );
        }
    }
}
