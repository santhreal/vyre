//! Inputs and outputs of a resident launch decision: pressure, execution mode,
//! topology, promotion evidence, request, and recommendation.

use super::super::planner::ResidentLaunchGeometry;

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
