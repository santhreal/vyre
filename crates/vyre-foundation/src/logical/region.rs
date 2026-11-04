//! Versioned region descriptors, extents, contracts, and error types.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::{GraphNodeId, GraphValueId};
use crate::logical_partition::LogicalPartitionFacts;
use crate::numeric::NumericContract;

/// Current logical algorithm schema and identity version.
pub const LOGICAL_ALGORITHM_VERSION: u16 = 4;

/// One validated logical extent.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LogicalExtent {
    /// Compile-time extent.
    Static(u64),
    /// Dynamic extent read from a typed graph value contract and resolved by the compile request.
    GraphValue {
        /// Graph value whose contract declares this dimension.
        value: u32,
        /// Zero-based dimension within the graph value.
        axis: u32,
        /// Symbol name as declared by the graph value contract.
        symbol: String,
        /// Value bound by the compile request.
        bound: u64,
    },
}

impl LogicalExtent {
    pub(super) fn bound(&self) -> u64 {
        match self {
            Self::Static(value) | Self::GraphValue { bound: value, .. } => *value,
        }
    }
}

/// Semantic parallelism of one region before physical mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LogicalRegionKind {
    /// Independent points in an iteration domain.
    Parallel,
    /// Ordered points with a loop-carried dependence.
    Sequential,
    /// Associative combination over one or more axes.
    Reduction,
    /// State retained from one submission to the next.
    RetainedState,
    /// Segmented mapping over independent contiguous or ragged partitions.
    SegmentedMap,
    /// Associative prefix or suffix scans.
    Scan,
    /// Tiled stateful recurrence and recurrent sequence state.
    RecurrentState,
    /// Sliding, stenciled, or tiled windowed operations with boundary halos.
    Window,
    /// Dynamic or ragged extents per segment or batch.
    RaggedExtent,
    /// Combining or joining partial results across distributed partitions.
    PartialResultJoin,
}

impl LogicalRegionKind {
    /// Exhaustive roster of all logical region kinds.
    pub const ALL: [Self; 10] = [
        Self::Parallel,
        Self::Sequential,
        Self::Reduction,
        Self::RetainedState,
        Self::SegmentedMap,
        Self::Scan,
        Self::RecurrentState,
        Self::Window,
        Self::RaggedExtent,
        Self::PartialResultJoin,
    ];

    /// Whether this region performs an associative combination.
    #[must_use]
    pub fn is_reduction_or_scan(self) -> bool {
        matches!(self, Self::Reduction | Self::Scan | Self::PartialResultJoin)
    }

    /// Whether this region carries loop or step state.
    #[must_use]
    pub fn is_stateful(self) -> bool {
        matches!(
            self,
            Self::Sequential | Self::RetainedState | Self::RecurrentState
        )
    }
}

/// Combination operator for reductions, scans, and partial joins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LogicalCombineOp {
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Bitwise OR.
    BitOr,
    /// Bitwise AND.
    BitAnd,
    /// Bitwise XOR.
    BitXor,
}

/// Direction for scan regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum ScanDirection {
    /// Inclusive forward prefix scan.
    InclusiveForward,
    /// Exclusive forward prefix scan.
    ExclusiveForward,
    /// Inclusive backward suffix scan.
    InclusiveBackward,
    /// Exclusive backward suffix scan.
    ExclusiveBackward,
}

/// Window configuration for stenciled or sliding window regions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct WindowDescriptor {
    /// Window size in elements per axis.
    pub window_shape: Vec<u64>,
    /// Stride in elements per axis.
    pub strides: Vec<u64>,
    /// Dilation in elements per axis.
    pub dilations: Vec<u64>,
    /// Halo / padding size before and after each axis.
    pub halo_padding: Vec<(u64, u64)>,
}

/// Segment configuration for segmented map and ragged extent regions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct SegmentDescriptor {
    /// Number of independent segments.
    pub segment_count: u64,
    /// Maximum elements per segment.
    pub max_segment_len: u64,
    /// Whether segment lengths are uniform or ragged.
    pub is_ragged: bool,
}

/// Recurrence configuration for recurrent state regions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct RecurrenceDescriptor {
    /// Sequence length / recurrence steps.
    pub sequence_steps: u64,
    /// State elements per step.
    pub state_elements: u64,
    /// Unroll / tile factor for recurrence lowering.
    pub tile_factor: u32,
}

/// Partial result join configuration across distributed partitions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct PartialResultJoinDescriptor {
    /// Axis partitioned across splits.
    pub split_axis: u32,
    /// Number of partial result partitions.
    pub partition_count: u32,
    /// Combination operator used to merge partial results.
    pub combine_op: LogicalCombineOp,
}

/// Bounded scratch contract for a logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct ScratchContract {
    /// Maximum workgroup/shared memory bytes required (never replicated per lane).
    pub workgroup_scratch_bytes: u64,
    /// Maximum partition/global scratch bytes required.
    pub partition_scratch_bytes: u64,
    /// Whether scratch memory is reusable across non-overlapping phases.
    pub reusable: bool,
}

/// Monotone progress and termination contract for a logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct ProgressContract {
    /// Exact upper bound on loop/step iterations.
    pub max_iterations: u64,
    /// Monotone progress metric ensuring finite loop execution.
    pub monotone_progress: bool,
    /// Guaranteed termination without livelock or deadlock.
    pub guaranteed_termination: bool,
}

/// Generic ordering and synchronization contract for a logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct OrderingContract {
    /// Whether execution points require sequential causal ordering.
    pub causal_ordering: bool,
    /// Required memory synchronization scope across partitions.
    pub sync_scope: OrderingSyncScope,
}

/// Memory and execution synchronization scope across partitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum OrderingSyncScope {
    /// No cross-point synchronization needed (fully parallel).
    None,
    /// Workgroup / threadblock scoped synchronization.
    Workgroup,
    /// Queue / device-wide synchronization.
    DeviceQueue,
    /// Retained state epoch synchronization.
    RetainedEpoch,
}

/// Logical index projection, independent of lanes and workgroups.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalIndexMap {
    /// Axis names in declaration order.
    pub axes: Vec<String>,
    /// Row-major strides in elements.
    pub row_major_strides: Vec<u64>,
}

/// Tensor storage layout attached to one logical domain.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalLayout {
    /// Logical axes in increasing physical storage order.
    pub storage_order: Vec<u32>,
    /// Element strides in logical-axis order.
    pub strides: Vec<u64>,
    /// Whether every axis is densely row-major.
    pub contiguous: bool,
}

/// Closed alias facts for one logical region.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalAliasFacts {
    /// Retained outputs paired with the prior value whose storage they replace.
    pub retained_successors: Vec<(u32, u32)>,
    /// Graph values updated in place by a read-write input binding.
    pub in_place_values: Vec<u32>,
    /// Input values are pairwise distinct.
    pub inputs_disjoint: bool,
    /// Output values are pairwise distinct.
    pub outputs_disjoint: bool,
}

/// Kind of schedule-free dependence between logical regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum LogicalDependenceKind {
    /// A producer writes values consumed by this region.
    Flow,
    /// This region advances state produced by an earlier submission.
    RetainedState,
}

/// One explicit dependence on a preceding logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalDependence {
    /// Preceding graph node.
    pub predecessor: GraphNodeId,
    /// Graph values that induce the dependence.
    pub values: Vec<u32>,
    /// Exact packed bytes of the values that induce the dependence.
    pub bytes: u64,
    /// Dependence semantics.
    pub kind: LogicalDependenceKind,
}

/// Closed read/write and synchronization effects for one logical region.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize)]
pub struct LogicalEffects {
    /// Graph values read by this region.
    pub reads: Vec<u32>,
    /// Graph values written by this region.
    pub writes: Vec<u32>,
    /// Whether this region updates retained state.
    pub retained_state: bool,
    /// Whether the region contains atomic memory effects.
    pub atomics: bool,
    /// Whether the region contains an ordering or collective synchronization effect.
    pub synchronizes: bool,
}

/// One validated schedule-free algorithm region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LogicalRegion {
    /// Source graph node.
    pub node: GraphNodeId,
    /// Stable semantic node name.
    pub name: String,
    /// Logical execution kind.
    pub kind: LogicalRegionKind,
    /// Versioned domain extents.
    pub extents: Vec<LogicalExtent>,
    /// Logical index projection.
    pub index_map: LogicalIndexMap,
    /// Tensor storage layout.
    pub layout: LogicalLayout,
    /// Logical axes combined by a reduction region.
    pub reduction_axes: Vec<u32>,
    /// Closed alias facts derived from graph ports.
    pub aliases: LogicalAliasFacts,
    /// Explicit dependencies on earlier regions.
    pub dependencies: Vec<LogicalDependence>,
    /// Closed effects derived from graph ports and executable semantics.
    pub effects: LogicalEffects,
    /// Closed statement of how this region may be distributed.
    pub partition: LogicalPartitionFacts,
    /// Exact packed bytes of the values this region writes.
    pub written_bytes: u64,
    /// Exact upper bound on logical points.
    pub max_points: u64,
    /// Numeric contract derived from the formats and combines this region states.
    pub numeric: NumericContract,
    /// Optional segment descriptor for segmented maps and ragged extents.
    pub segment: Option<SegmentDescriptor>,
    /// Optional window descriptor for stenciled or sliding window regions.
    pub window: Option<WindowDescriptor>,
    /// Optional recurrence descriptor for recurrent state regions.
    pub recurrence: Option<RecurrenceDescriptor>,
    /// Optional partial result join descriptor.
    pub partial_join: Option<PartialResultJoinDescriptor>,
    /// Validated scratch memory bounds.
    pub scratch: ScratchContract,
    /// Validated progress and termination invariants.
    pub progress: ProgressContract,
    /// Validated causal ordering and synchronization scopes.
    pub ordering: OrderingContract,
}

/// Logical-stage validation failure.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LogicalProgramError {
    /// Whole-program topology is invalid.
    #[error("logical graph rejected invalid topology: {0}")]
    Graph(String),
    /// A symbolic extent has no request binding.
    #[error("logical graph is missing symbolic extent `{0}`")]
    MissingSymbol(String),
    /// A request binding does not occur in the graph.
    #[error("logical graph has unexpected symbolic extent `{0}`")]
    UnexpectedSymbol(String),
    /// Product of logical extents cannot fit the bounded u64 domain.
    #[error("logical region for graph node {0:?} overflows its point bound")]
    ExtentOverflow(GraphNodeId),
    /// A zero extent has not been resolved to a schedulable positive bound.
    #[error(
        "logical region for graph node {node:?} has unresolved extent at graph value {value:?} axis {axis}"
    )]
    UnresolvedExtent {
        /// Region containing the unresolved extent.
        node: GraphNodeId,
        /// Graph value that declares the extent.
        value: GraphValueId,
        /// Axis within the graph value.
        axis: u32,
    },
    /// A node has a symbolic shape but no graph value that can define it.
    #[error("logical region for graph node {0:?} has no typed domain value")]
    MissingDomain(GraphNodeId),
    /// A graph value referenced by a logical domain does not exist.
    #[error("logical domain references missing graph value {0:?}")]
    MissingDomainValue(GraphValueId),
    /// A logical domain rank cannot be represented by the versioned wire contract.
    #[error("logical region for graph node {0:?} exceeds the u32 axis range")]
    DomainRankOverflow(GraphNodeId),
    /// A domain dependence points to the same or a later graph node.
    #[error("logical region for graph node {node:?} has cyclic dependence on {predecessor:?}")]
    CyclicDomain {
        /// Region containing the invalid dependence.
        node: GraphNodeId,
        /// Non-preceding dependency.
        predecessor: GraphNodeId,
    },
    /// Graph ports do not establish pairwise-disjoint logical values.
    #[error("logical region for graph node {0:?} has incompatible alias declarations")]
    IncompatibleAliases(GraphNodeId),
    /// A whole-program graph cannot produce schedule-free canonical bytes.
    #[error("logical graph is not canonical: {0}")]
    CanonicalGraph(String),
    /// Logical identity serialization failed.
    #[error("logical identity serialization failed: {0}")]
    Identity(String),
    /// A semantic exchange payload cannot be sized.
    #[error("logical exchange is not sizable: {0}")]
    Exchange(String),
    /// A region states a numeric contract that cannot be priced or composed.
    #[error("logical region for graph node {node:?} has no numeric budget: {reason}")]
    Numeric {
        /// Region whose contract was refused.
        node: GraphNodeId,
        /// Exact refusal.
        reason: String,
    },
}
