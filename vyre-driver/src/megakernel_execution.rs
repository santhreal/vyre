//! Backend-neutral execution planning for persistent megakernel waves.
//!
//! Backends can feed telemetry and device budgets into this module to choose a
//! sparse, dense, hybrid, or fused execution topology before allocating device
//! scratch. The policy is deterministic, allocation-free, and validates byte
//! pressure before a backend reaches an API-specific allocation path.
//!
//! One rule used to live only in one backend's copy of this policy: a `FusedWave`
//! runs dependency-ordered waves inside a single launch, so it needs a barrier
//! across every resident block, and a device without one cannot run the plan at
//! all. The neutral policy did not know that, so for the same wave it answered
//! `FusedWave` where that fork answered a per-launch topology, and any
//! backend that had not written the check itself would have been handed an
//! unlaunchable plan. The check is a property of the device, not of one backend, so it
//! is [`crate::megakernel_execution::MegakernelDeviceCapabilities`] here and every backend inherits it.

pub use vyre_megakernel::{
    select_frontier_topology, select_frontier_topology_stable, FrontierExecutionSample,
    FrontierGraphShape, FrontierMemoryBudget, FrontierTopology, FrontierTopologyDecision,
};

/// Device capabilities that constrain which wave topologies are launchable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MegakernelDeviceCapabilities {
    /// Whether every resident block can synchronize inside one launch.
    pub supports_device_wide_barrier: bool,
}

impl MegakernelDeviceCapabilities {
    /// Device that can host a fused wave.
    pub const FUSION_CAPABLE: Self = Self {
        supports_device_wide_barrier: true,
    };
    /// Device that must keep every wave in its own launch.
    pub const FUSION_INCAPABLE: Self = Self {
        supports_device_wide_barrier: false,
    };

    /// Fusion pressure this device can act on.
    ///
    /// Without a device-wide barrier the fused plan is unlaunchable, so the
    /// measured pressure toward it is zero however high the caller observed it.
    #[must_use]
    pub fn admissible_fusion_pressure(self, fusion_pressure: f64) -> f64 {
        if self.supports_device_wide_barrier {
            fusion_pressure
        } else {
            0.0
        }
    }
}

/// Per-candidate telemetry used to bias megakernel fusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MegakernelExecutionSample {
    /// Observed candidate dispatch cost in nanoseconds.
    pub dispatch_cost_ns: f64,
    /// Observed active-frontier density in `[0, 1]`.
    pub frontier_density: f64,
    /// Observed final readback byte volume.
    pub readback_bytes: u64,
}

/// Static graph shape used by topology selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MegakernelGraphShape {
    /// Logical graph node count.
    pub node_count: u64,
    /// Logical graph edge count.
    pub edge_count: u64,
}

/// Device memory envelope for a candidate megakernel plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MegakernelMemoryBudget {
    /// Estimated resident plus transient bytes required by the candidate plan.
    pub required_bytes: u64,
    /// Caller-approved device-memory budget for the plan.
    pub budget_bytes: u64,
}

/// Detailed megakernel memory plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MegakernelMemoryPlan {
    /// Graph-layout bytes retained on device.
    pub graph_bytes: u64,
    /// Frontier-state bytes retained on device.
    pub frontier_bytes: u64,
    /// Temporary scratch bytes required by the selected topology.
    pub scratch_bytes: u64,
    /// Final compact output/readback bytes.
    pub output_bytes: u64,
    /// Total peak bytes required by the plan.
    pub required_bytes: u64,
    /// Caller-approved byte budget.
    pub budget_bytes: u64,
    /// Required/budget pressure in basis points.
    pub memory_pressure_bps: u32,
}

/// Complete megakernel execution plan selected from runtime telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MegakernelExecutionPlan {
    /// Final topology after memory-budget validation.
    pub topology: FrontierTopology,
    /// Memory plan for the final topology.
    pub memory: MegakernelMemoryPlan,
    /// Whether the planner downgraded a denser/fused topology to sparse to fit
    /// the explicit memory budget.
    pub downgraded_to_sparse: bool,
}

/// Memory planning failure for megakernel execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MegakernelMemoryError {
    /// A byte-count multiplication or addition overflowed.
    ByteCountOverflow {
        /// Field being computed when overflow happened.
        field: &'static str,
    },
    /// The candidate plan exceeds the caller-approved device-memory budget.
    OverBudget {
        /// Selected topology.
        topology: FrontierTopology,
        /// Required peak bytes.
        required_bytes: u64,
        /// Caller-approved budget bytes.
        budget_bytes: u64,
        /// Graph node count.
        node_count: u64,
        /// Graph edge count.
        edge_count: u64,
    },
    /// An observed telemetry fact lies outside the domain its type declares.
    InvalidSample {
        /// Observed fact that lies outside its declared domain.
        field: &'static str,
    },
}

impl std::fmt::Display for MegakernelMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ByteCountOverflow { field } => write!(
                f,
                "megakernel memory planner overflowed while computing {field}. Fix: shard the graph or lower the candidate topology before planning device residency."
            ),
            Self::OverBudget {
                topology,
                required_bytes,
                budget_bytes,
                node_count,
                edge_count,
            } => write!(
                f,
                "megakernel {topology:?} plan requires {required_bytes} bytes but budget allows {budget_bytes} bytes for graph nodes={node_count} edges={edge_count}. Fix: choose a sparse topology, reduce fusion pressure, shard the graph, or raise the explicit device-memory budget."
            ),
            Self::InvalidSample { field } => write!(
                f,
                "megakernel execution sample states an unrepresentable {field}. Fix: supply measured finite telemetry inside the domain the sample type declares."
            ),
        }
    }
}

impl std::error::Error for MegakernelMemoryError {}

/// Resident bytes a graph layout occupies before any wave state.
///
/// # Errors
///
/// Returns [`MegakernelMemoryError::ByteCountOverflow`] when the node or edge
/// layout does not fit `u64`.
pub fn megakernel_resident_graph_bytes(
    graph: MegakernelGraphShape,
    bytes_per_node: u64,
    bytes_per_edge: u64,
) -> Result<u64, MegakernelMemoryError> {
    let node_bytes = checked_mul(graph.node_count, bytes_per_node, "node layout bytes")?;
    let edge_bytes = checked_mul(graph.edge_count, bytes_per_edge, "edge layout bytes")?;
    checked_add(node_bytes, edge_bytes, "graph layout bytes")
}

/// The byte accounting for one megakernel wave.
///
/// The resident layout and the approved cap always travel together, from the
/// caller that measures them through every planning hop. They are one value
/// because a positional list of six `u64` counts transposes silently at a call
/// site and named fields cannot.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MegakernelByteLayout {
    /// Resident bytes per graph node.
    pub bytes_per_node: u64,
    /// Resident bytes per graph edge.
    pub bytes_per_edge: u64,
    /// Frontier-state bytes for the wave.
    pub frontier_bytes: u64,
    /// Base scratch bytes before the topology multiplier.
    pub scratch_bytes: u64,
    /// Final compact output bytes.
    pub output_bytes: u64,
    /// Caller-approved device-memory budget.
    pub budget_bytes: u64,
}

/// Resident bytes a wave needs before any topology scratch multiplier applies.
///
/// Topology selection needs a required-bytes figure before a topology exists.
/// This prices the graph, the frontier state, the unmultiplied scratch and the
/// compact output, so obtaining that figure names no topology.
///
/// # Errors
///
/// Returns [`MegakernelMemoryError::ByteCountOverflow`] when byte accounting
/// overflows.
pub fn megakernel_base_required_bytes(
    graph: MegakernelGraphShape,
    bytes: MegakernelByteLayout,
) -> Result<u64, MegakernelMemoryError> {
    let graph_bytes =
        megakernel_resident_graph_bytes(graph, bytes.bytes_per_node, bytes.bytes_per_edge)?;
    let without_output = checked_add(
        graph_bytes,
        bytes.frontier_bytes,
        "graph plus frontier bytes",
    )?;
    let without_output = checked_add(without_output, bytes.scratch_bytes, "scratch bytes")?;
    checked_add(without_output, bytes.output_bytes, "output bytes")
}

/// Compute and validate a megakernel device-memory plan.
pub fn plan_megakernel_memory_budget(
    topology: FrontierTopology,
    graph: MegakernelGraphShape,
    bytes: MegakernelByteLayout,
) -> Result<MegakernelMemoryPlan, MegakernelMemoryError> {
    let graph_bytes =
        megakernel_resident_graph_bytes(graph, bytes.bytes_per_node, bytes.bytes_per_edge)?;
    let topology_scratch_bytes = topology_scratch_bytes(topology, bytes.scratch_bytes)?;
    let required_without_output = checked_add(
        graph_bytes,
        bytes.frontier_bytes,
        "graph plus frontier bytes",
    )?;
    let required_without_output = checked_add(
        required_without_output,
        topology_scratch_bytes,
        "scratch bytes",
    )?;
    let required_bytes = checked_add(required_without_output, bytes.output_bytes, "output bytes")?;
    if required_bytes > bytes.budget_bytes {
        return Err(MegakernelMemoryError::OverBudget {
            topology,
            required_bytes,
            budget_bytes: bytes.budget_bytes,
            node_count: graph.node_count,
            edge_count: graph.edge_count,
        });
    }
    Ok(MegakernelMemoryPlan {
        graph_bytes,
        frontier_bytes: bytes.frontier_bytes,
        scratch_bytes: topology_scratch_bytes,
        output_bytes: bytes.output_bytes,
        required_bytes,
        budget_bytes: bytes.budget_bytes,
        memory_pressure_bps: pressure_bps(required_bytes, bytes.budget_bytes),
    })
}

/// Select a megakernel topology and validate its device-memory plan.
///
/// Telemetry is checked before selection because selection saturates a fact it
/// cannot price, which would rank a measurement defect as a measured extreme.
///
/// # Errors
///
/// Returns [`MegakernelMemoryError::InvalidSample`] when an observed fact lies
/// outside its declared domain, [`MegakernelMemoryError::ByteCountOverflow`]
/// when byte accounting overflows, and [`MegakernelMemoryError::OverBudget`]
/// when no topology fits the approved budget.
pub fn plan_megakernel_execution(
    sample: MegakernelExecutionSample,
    graph: MegakernelGraphShape,
    bytes: MegakernelByteLayout,
    launch_overhead_ns: f64,
    fusion_pressure: f64,
    capabilities: MegakernelDeviceCapabilities,
) -> Result<MegakernelExecutionPlan, MegakernelMemoryError> {
    let frontier_sample = FrontierExecutionSample {
        dispatch_cost_ns: sample.dispatch_cost_ns,
        frontier_density: sample.frontier_density,
        readback_bytes: sample.readback_bytes,
    };
    if let Some(field) = frontier_sample.unrepresentable_fact() {
        return Err(MegakernelMemoryError::InvalidSample { field });
    }
    let base_required_bytes = megakernel_base_required_bytes(graph, bytes)?;
    let decision = select_frontier_topology(
        frontier_sample,
        FrontierGraphShape {
            node_count: graph.node_count,
            edge_count: graph.edge_count,
        },
        FrontierMemoryBudget {
            required_bytes: base_required_bytes,
            budget_bytes: bytes.budget_bytes,
        },
        launch_overhead_ns,
        fusion_pressure,
        capabilities.supports_device_wide_barrier,
    );
    match plan_megakernel_memory_budget(decision.topology, graph, bytes) {
        Ok(memory) => Ok(MegakernelExecutionPlan {
            topology: decision.topology,
            memory,
            downgraded_to_sparse: false,
        }),
        Err(MegakernelMemoryError::OverBudget { .. }) if !decision.topology.is_baseline() => {
            let baseline = decision.topology.fallback_baseline();
            Ok(MegakernelExecutionPlan {
                memory: plan_megakernel_memory_budget(baseline, graph, bytes)?,
                topology: baseline,
                downgraded_to_sparse: true,
            })
        }
        Err(error) => Err(error),
    }
}

/// Every input one candidate wave needs to reach an execution plan.
///
/// This is the argument list of [`plan_megakernel_execution`] as one value so a
/// backend can memoize the decision without restating it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MegakernelExecutionRequest {
    /// Runtime telemetry for the candidate wave.
    pub sample: MegakernelExecutionSample,
    /// Static graph shape.
    pub graph: MegakernelGraphShape,
    /// Byte accounting for the wave, including the approved cap.
    pub bytes: MegakernelByteLayout,
    /// Per-launch overhead observed for this device.
    pub launch_overhead_ns: f64,
    /// Caller-measured pressure toward fusing adjacent waves.
    pub fusion_pressure: f64,
    /// Capabilities of the device that will run the wave.
    pub capabilities: MegakernelDeviceCapabilities,
}

/// Source of memory-validated megakernel execution plans.
///
/// The decision itself is [`plan_megakernel_execution`]. A backend implements
/// this trait only to put a device-local cache in front of that decision, never
/// to make a different one.
pub trait MegakernelExecutionPlanner {
    /// Plan one candidate wave.
    ///
    /// # Errors
    ///
    /// Returns [`MegakernelMemoryError`] when the request overflows byte
    /// accounting or cannot fit the approved budget.
    fn plan_execution(
        &mut self,
        request: MegakernelExecutionRequest,
    ) -> Result<MegakernelExecutionPlan, MegakernelMemoryError>;
}

/// The neutral policy with no memoization, for backends without a plan cache.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NeutralMegakernelExecutionPlanner;

impl MegakernelExecutionPlanner for NeutralMegakernelExecutionPlanner {
    fn plan_execution(
        &mut self,
        request: MegakernelExecutionRequest,
    ) -> Result<MegakernelExecutionPlan, MegakernelMemoryError> {
        plan_megakernel_execution(
            request.sample,
            request.graph,
            request.bytes,
            request.launch_overhead_ns,
            request.fusion_pressure,
            request.capabilities,
        )
    }
}

fn pressure_bps(numerator: u64, denominator: u64) -> u32 {
    let clamped = pressure_bps_u64(numerator, denominator).min(10_000);
    match u32::try_from(clamped) {
        Ok(value) => value,
        Err(error) => {
            tracing::error!(
                "megakernel pressure conversion failed after clamping value {clamped}: {error}. Fix: inspect ratio/clamp invariants before topology selection."
            );
            10_000
        }
    }
}

fn pressure_bps_u64(numerator: u64, denominator: u64) -> u64 {
    crate::numeric::ratio_basis_points_u64_wide(
        numerator,
        denominator,
        if numerator == 0 { 0 } else { u64::MAX },
        "megakernel scheduler pressure",
        "megakernel execution",
    )
}

fn topology_scratch_bytes(
    topology: FrontierTopology,
    base_scratch_bytes: u64,
) -> Result<u64, MegakernelMemoryError> {
    match topology {
        FrontierTopology::WarpSparseFrontier => Ok(base_scratch_bytes.max(32)),
        FrontierTopology::SparseFrontier => Ok(base_scratch_bytes),
        FrontierTopology::BlockDenseFrontier => checked_mul(
            base_scratch_bytes.max(1024),
            2,
            "block dense topology scratch bytes",
        ),
        FrontierTopology::DenseFrontier => {
            checked_mul(base_scratch_bytes, 2, "dense topology scratch bytes")
        }
        FrontierTopology::HybridFrontier => {
            checked_mul(base_scratch_bytes, 3, "hybrid topology scratch bytes")
        }
        FrontierTopology::FusedWave => {
            checked_mul(base_scratch_bytes, 4, "fused topology scratch bytes")
        }
    }
}

fn checked_add(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, MegakernelMemoryError> {
    lhs.checked_add(rhs)
        .ok_or(MegakernelMemoryError::ByteCountOverflow { field })
}

fn checked_mul(lhs: u64, rhs: u64, field: &'static str) -> Result<u64, MegakernelMemoryError> {
    lhs.checked_mul(rhs)
        .ok_or(MegakernelMemoryError::ByteCountOverflow { field })
}
