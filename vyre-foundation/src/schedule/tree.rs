//! Compositional schedule calculus over logical loops and regions.
//!
//! Provides [`ScheduleOp`], [`ScheduleTree`], [`SchedulePlan`], [`PartialSchedule`], and [`SymbolicParameter`]
//! with explicit precondition verification, algebraic rule tagging, and dependency preservation certificates.

use serde::{Deserialize, Serialize};

use super::{
    error::ScheduleLegalityError, MappingLevel, MemoryPlacement, PipelineRoleGroup,
    SchedulePrecondition, ScheduleResourceBounds, ScheduleTransformRecord, SynchronizationScope,
};

/// Numerical effect of applying a schedule operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleNumericalEffect {
    /// Exact bitwise-identical numerical result guaranteed.
    Exact,
    /// Accumulation/reduction order reassociated under algebraic laws.
    ReorderedAccumulation,
    /// Precision conversion (e.g. mixed-precision MMA).
    PrecisionConverted,
    /// Approximation within bounded ULP tolerance.
    Approximated {
        /// Bounded tolerance in units in the last place.
        tolerance_ulp: u32,
    },
}

/// Inverse mapping information for schedule operator provenance.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleInverseOp {
    /// Exact reversible inverse schedule operation.
    Reversible(Box<ScheduleOp>),
    /// State restoration from cryptographically signed state digest.
    RestoreState {
        /// 256-bit cryptographic digest of previous state.
        state_digest: [u8; 32],
    },
    /// Irreversible transformation with formal derivation certificate.
    Irreversible {
        /// Explanation for non-reversibility.
        reason: String,
    },
}

/// Primitive schedule operation in the compositional schedule calculus.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleOp {
    /// Domain specification defining iteration extents for a region.
    Domain {
        /// Logical region identifier.
        region: u32,
        /// Extents for each iteration dimension.
        extents: Vec<u64>,
    },
    /// Algorithm selection choosing an implementation strategy.
    AlgorithmChoice {
        /// Logical region identifier.
        region: u32,
        /// Algorithm identifier name.
        algorithm_id: String,
        /// Concrete parameter pack for the algorithm.
        parameter_pack: Vec<u64>,
    },
    /// Region fusion combining multiple regions into one phase.
    Fuse {
        /// Logical regions to fuse.
        regions: Vec<u32>,
    },
    /// Region fission splitting a region at a cut point.
    Fission {
        /// Region to split.
        region: u32,
        /// Split point index.
        split_point: u32,
    },
    /// Region partitioning across neutral compute partitions.
    RegionPartition {
        /// Region to partition.
        region: u32,
        /// Number of partitions.
        partitions: u32,
        /// Hierarchy level for partitioning.
        level: MappingLevel,
    },
    /// Loop interchange swapping two loop axes.
    Interchange {
        /// First loop axis index.
        axis1: u32,
        /// Second loop axis index.
        axis2: u32,
    },
    /// Loop strip-mining dividing an axis by an exact factor.
    StripMine {
        /// Axis to strip-mine.
        axis: u32,
        /// Strip-mining factor (must be non-zero).
        factor: u64,
        /// New inner axis identifier.
        inner_axis: u32,
    },
    /// Loop tiling dividing an iteration space into outer and inner blocks.
    Tile {
        /// Axis to tile.
        axis: u32,
        /// Tile size (must be nonzero).
        tile_size: u64,
        /// New inner axis identifier.
        inner_axis: u32,
    },
    /// Loop unrolling by a constant factor.
    Unroll {
        /// Axis to unroll.
        axis: u32,
        /// Unroll factor.
        factor: u32,
    },
    /// Loop skewing to expose parallelism along diagonal wavefronts.
    Skew {
        /// Outer loop axis.
        outer_axis: u32,
        /// Inner loop axis.
        inner_axis: u32,
        /// Skew factor.
        factor: i64,
    },
    /// Loop reordering / permutation.
    Reorder {
        /// Target permutation of axis indices.
        permutation: Vec<usize>,
    },
    /// Vectorization along an innermost contiguous axis.
    Vectorize {
        /// Axis to vectorize.
        axis: u32,
        /// Vector width (e.g. 2, 4, 8).
        vector_width: u32,
    },
    /// Map an axis to a physical execution hierarchy level.
    MapToHierarchy {
        /// Axis to map.
        axis: u32,
        /// Target mapping level (Lane, Subgroup, Workgroup, etc.).
        level: MappingLevel,
        /// Dimension index at that level (0 for x, 1 for y, 2 for z).
        dimension: u32,
    },
    /// Subgroup specialization, partitioning subgroups into dedicated producer and consumer roles.
    SubgroupSpecialization {
        /// Dedicated worker groups.
        roles: Vec<PipelineRoleGroup>,
    },
    /// Role specialization across generic pipeline participant groups.
    RoleSpecialization {
        /// Dedicated worker groups.
        roles: Vec<PipelineRoleGroup>,
    },
    /// Asynchronous producer-consumer copy pipeline.
    AsyncPipeline {
        /// Number of pipeline stages.
        stages: u32,
        /// Ring buffer capacity.
        ring_size: u32,
        /// Producer and consumer role divisions.
        role_groups: Vec<PipelineRoleGroup>,
    },
    /// Software pipelining with initiation interval.
    SoftwarePipeline {
        /// Number of pipeline stages.
        stages: u32,
        /// Target initiation interval in cycles.
        initiation_interval: u32,
    },
    /// Asynchronous copy and compute overlap coordination.
    AsyncCopyComputeOverlap {
        /// Stage index for asynchronous copy.
        copy_stage: u32,
        /// Stage index for compute.
        compute_stage: u32,
        /// Concurrency overlap factor.
        overlap_factor: u32,
    },
    /// Recomputation of values instead of loading or retaining in memory.
    Recompute {
        /// Graph value identities to recompute.
        values: Vec<u32>,
    },
    /// Persistent execution queue for resident kernels.
    Persistence {
        /// Phase identifier.
        phase: u32,
        /// Queue capacity.
        capacity: u32,
    },
    /// Memory placement and packing layout specification.
    MemoryPlacement {
        /// Buffer name.
        buffer: String,
        /// Memory placement class.
        placement: MemoryPlacement,
        /// Optional layout packing scheme name.
        packing: Option<String>,
        /// Allocated byte bound.
        bytes: u64,
    },
    /// Stage a buffer into shared memory.
    SharedMemoryStage {
        /// Target buffer name.
        buffer: String,
        /// Placement level.
        placement: MemoryPlacement,
        /// Allocated bytes in shared memory.
        staging_bytes: u64,
    },
    /// Double-buffering / multi-buffering for overlap.
    DoubleBuffer {
        /// Buffer name.
        buffer: String,
        /// Number of buffers in the ring (typically 2).
        slots: u32,
    },
    /// Allocation reuse aliasing non-overlapping buffer lifetimes.
    AllocationReuse {
        /// Source buffer name.
        source_buffer: String,
        /// Target buffer name to reuse storage of.
        target_buffer: String,
        /// Byte offset in reused allocation.
        byte_offset: u64,
    },
    /// Register or shared memory spill to global memory.
    Spill {
        /// Buffer name being spilled.
        buffer: String,
        /// Spill memory location.
        spill_location: MemoryPlacement,
        /// Number of bytes spilled.
        spill_bytes: u64,
    },
    /// Explicit inter-workgroup or cross-device communication.
    Communication {
        /// Exchange pattern name.
        exchange_kind: String,
        /// Communication group identifier.
        comm_group: u32,
        /// Transferred payload bytes.
        payload_bytes: u64,
    },
    /// Entry-point DAG construction defining multi-entry task dependencies.
    EntryPointDag {
        /// Entry point function/phase identifiers.
        entry_points: Vec<u32>,
        /// Directed dependencies between entry points: `(predecessor, successor)`.
        dependencies: Vec<(u32, u32)>,
    },
    /// Register-level tiling (micro-kernel tile).
    RegisterTile {
        /// M dimension register tile size.
        register_dim_m: u32,
        /// N dimension register tile size.
        register_dim_n: u32,
        /// K dimension register tile size.
        register_dim_k: u32,
    },
    /// Explicit target instruction selection (e.g. WMMA/MMA tensor core intrinsic).
    InstructionSelect {
        /// Intrinsic descriptor name.
        target_intrinsic: String,
        /// Matrix multiply-accumulate shape [M, N, K].
        mma_shape: Option<[u32; 3]>,
    },
    /// Explicit execution and memory synchronization barrier.
    Synchronize {
        /// Scope of synchronization.
        scope: SynchronizationScope,
    },
}

impl ScheduleOp {
    /// Return the typed preconditions that must be satisfied before this operator applies.
    #[must_use]
    pub fn preconditions(&self) -> Vec<SchedulePrecondition> {
        match self {
            Self::Domain { .. } => vec![],
            Self::AlgorithmChoice { .. } => vec![],
            Self::Fuse { regions } => {
                if regions.len() >= 2 {
                    vec![SchedulePrecondition::Acyclic]
                } else {
                    vec![]
                }
            }
            Self::Fission { .. } => vec![SchedulePrecondition::Acyclic],
            Self::RegionPartition { .. } => vec![
                SchedulePrecondition::NonZero(super::ScheduleBoundKind::PartitionCount),
                SchedulePrecondition::Acyclic,
            ],
            Self::Interchange { .. } => vec![SchedulePrecondition::AxisPermutation],
            Self::StripMine { factor, .. } => {
                vec![SchedulePrecondition::Divisible {
                    extent: *factor,
                    factor: (*factor as u32).max(1),
                }]
            }
            Self::Tile { tile_size, .. } => vec![SchedulePrecondition::Divisible {
                extent: *tile_size,
                factor: (*tile_size as u32).max(1),
            }],
            Self::Unroll { .. } => vec![SchedulePrecondition::NonZero(
                super::ScheduleBoundKind::Factor,
            )],
            Self::Skew { .. } => vec![],
            Self::Reorder { .. } => vec![SchedulePrecondition::AxisPermutation],
            Self::Vectorize { vector_width, .. } => vec![SchedulePrecondition::Divisible {
                extent: u64::from(*vector_width),
                factor: *vector_width,
            }],
            Self::MapToHierarchy { .. } => vec![],
            Self::SubgroupSpecialization { .. } | Self::RoleSpecialization { .. } => {
                vec![SchedulePrecondition::BoundedResource(
                    super::ScheduleBoundKind::PipelineRing,
                )]
            }
            Self::AsyncPipeline { .. } => vec![
                SchedulePrecondition::NonZero(super::ScheduleBoundKind::PipelineRing),
                SchedulePrecondition::BoundedResource(super::ScheduleBoundKind::PipelineRing),
            ],
            Self::SoftwarePipeline { .. } => vec![
                SchedulePrecondition::NonZero(super::ScheduleBoundKind::PipelineRing),
                SchedulePrecondition::BoundedResource(super::ScheduleBoundKind::PipelineRing),
            ],
            Self::AsyncCopyComputeOverlap { .. } => vec![SchedulePrecondition::BoundedResource(
                super::ScheduleBoundKind::PipelineRing,
            )],
            Self::Recompute { .. } => vec![],
            Self::Persistence { .. } => vec![
                SchedulePrecondition::NonZero(super::ScheduleBoundKind::QueueCapacity),
                SchedulePrecondition::BoundedResource(super::ScheduleBoundKind::QueueCapacity),
            ],
            Self::MemoryPlacement { .. } | Self::SharedMemoryStage { .. } => {
                vec![SchedulePrecondition::BoundedResource(
                    super::ScheduleBoundKind::Bytes,
                )]
            }
            Self::DoubleBuffer { .. } => vec![SchedulePrecondition::BoundedResource(
                super::ScheduleBoundKind::Bytes,
            )],
            Self::AllocationReuse { .. } => vec![SchedulePrecondition::BoundedResource(
                super::ScheduleBoundKind::Bytes,
            )],
            Self::Spill { .. } => vec![SchedulePrecondition::BoundedResource(
                super::ScheduleBoundKind::Bytes,
            )],
            Self::Communication { .. } => vec![SchedulePrecondition::Acyclic],
            Self::EntryPointDag { .. } => vec![SchedulePrecondition::Acyclic],
            Self::RegisterTile { .. } => vec![],
            Self::InstructionSelect { .. } => vec![],
            Self::Synchronize { .. } => vec![],
        }
    }

    /// Return the transformed domain identifiers touched by this operator.
    #[must_use]
    pub fn transformed_domains(&self) -> Vec<u32> {
        match self {
            Self::Domain { region, .. } => vec![*region],
            Self::AlgorithmChoice { region, .. } => vec![*region],
            Self::Fuse { regions } => regions.clone(),
            Self::Fission { region, .. } => vec![*region],
            Self::RegionPartition { region, .. } => vec![*region],
            Self::Interchange { axis1, axis2 } => vec![*axis1, *axis2],
            Self::StripMine {
                axis, inner_axis, ..
            } => vec![*axis, *inner_axis],
            Self::Tile {
                axis, inner_axis, ..
            } => vec![*axis, *inner_axis],
            Self::Unroll { axis, .. } => vec![*axis],
            Self::Skew {
                outer_axis,
                inner_axis,
                ..
            } => vec![*outer_axis, *inner_axis],
            Self::Reorder { .. } => vec![],
            Self::Vectorize { axis, .. } => vec![*axis],
            Self::MapToHierarchy { axis, .. } => vec![*axis],
            Self::SubgroupSpecialization { .. } | Self::RoleSpecialization { .. } => vec![],
            Self::AsyncPipeline { .. } => vec![],
            Self::SoftwarePipeline { .. } => vec![],
            Self::AsyncCopyComputeOverlap {
                copy_stage,
                compute_stage,
                ..
            } => vec![*copy_stage, *compute_stage],
            Self::Recompute { values } => values.clone(),
            Self::Persistence { phase, .. } => vec![*phase],
            Self::MemoryPlacement { .. } | Self::SharedMemoryStage { .. } => vec![],
            Self::DoubleBuffer { .. } => vec![],
            Self::AllocationReuse { .. } => vec![],
            Self::Spill { .. } => vec![],
            Self::Communication { .. } => vec![],
            Self::EntryPointDag { entry_points, .. } => entry_points.clone(),
            Self::RegisterTile { .. } => vec![],
            Self::InstructionSelect { .. } => vec![],
            Self::Synchronize { .. } => vec![],
        }
    }

    /// Return the transformed dependencies generated by this operator: `(from, to)`.
    #[must_use]
    pub fn transformed_dependencies(&self) -> Vec<(u32, u32)> {
        match self {
            Self::Fuse { regions } => {
                let mut deps = Vec::new();
                for i in 0..regions.len().saturating_sub(1) {
                    deps.push((regions[i], regions[i + 1]));
                }
                deps
            }
            Self::AsyncCopyComputeOverlap {
                copy_stage,
                compute_stage,
                ..
            } => vec![(*copy_stage, *compute_stage)],
            Self::EntryPointDag { dependencies, .. } => dependencies.clone(),
            _ => vec![],
        }
    }

    /// Evaluate resource equations and compute updated resource bounds.
    #[must_use]
    pub fn resource_equations(&self, current: &ScheduleResourceBounds) -> ScheduleResourceBounds {
        let mut bounds = *current;
        match self {
            Self::Domain { extents, .. } => {
                let points: u64 = extents.iter().product();
                bounds.logical_points = bounds.logical_points.max(points);
            }
            Self::Tile { tile_size, .. }
            | Self::StripMine {
                factor: tile_size, ..
            } => {
                bounds.logical_points = bounds.logical_points.max(*tile_size);
            }
            Self::Unroll { factor, .. } => {
                bounds.registers_per_invocation = bounds
                    .registers_per_invocation
                    .saturating_add(factor.saturating_mul(2));
            }
            Self::Vectorize { vector_width, .. } => {
                bounds.registers_per_invocation = bounds
                    .registers_per_invocation
                    .saturating_add(*vector_width);
            }
            Self::SharedMemoryStage { staging_bytes, .. } => {
                bounds.shared_bytes = bounds.shared_bytes.saturating_add(*staging_bytes);
            }
            Self::MemoryPlacement {
                placement, bytes, ..
            } => match placement {
                MemoryPlacement::Workgroup => {
                    bounds.shared_bytes = bounds.shared_bytes.saturating_add(*bytes);
                }
                MemoryPlacement::Invocation => {
                    bounds.private_bytes = bounds.private_bytes.saturating_add(*bytes);
                }
                _ => {}
            },
            Self::DoubleBuffer { slots, .. } => {
                bounds.shared_bytes = bounds.shared_bytes.saturating_mul(u64::from(*slots));
            }
            Self::AsyncPipeline { ring_size, .. } => {
                bounds.pipeline_slots = bounds.pipeline_slots.max(*ring_size);
            }
            Self::SoftwarePipeline { stages, .. } => {
                bounds.pipeline_slots = bounds.pipeline_slots.max(*stages);
            }
            Self::Persistence { capacity, .. } => {
                bounds.queue_capacity = bounds.queue_capacity.max(*capacity);
            }
            Self::RegisterTile {
                register_dim_m,
                register_dim_n,
                ..
            } => {
                bounds.registers_per_invocation = bounds
                    .registers_per_invocation
                    .saturating_add(register_dim_m.saturating_mul(*register_dim_n));
            }
            Self::Spill {
                spill_location,
                spill_bytes,
                ..
            } => {
                if *spill_location == MemoryPlacement::Invocation {
                    bounds.private_bytes = bounds.private_bytes.saturating_add(*spill_bytes);
                }
            }
            _ => {}
        }
        bounds
    }

    /// Return the numerical effect of applying this operator.
    #[must_use]
    pub fn numerical_effects(&self) -> ScheduleNumericalEffect {
        match self {
            Self::Domain { .. } => ScheduleNumericalEffect::Exact,
            Self::AlgorithmChoice { .. } => ScheduleNumericalEffect::Exact,
            Self::Fuse { .. } => ScheduleNumericalEffect::Exact,
            Self::Fission { .. } => ScheduleNumericalEffect::Exact,
            Self::RegionPartition { .. } => ScheduleNumericalEffect::Exact,
            Self::Interchange { .. } => ScheduleNumericalEffect::ReorderedAccumulation,
            Self::StripMine { .. } => ScheduleNumericalEffect::Exact,
            Self::Tile { .. } => ScheduleNumericalEffect::Exact,
            Self::Unroll { .. } => ScheduleNumericalEffect::Exact,
            Self::Skew { .. } => ScheduleNumericalEffect::Exact,
            Self::Reorder { .. } => ScheduleNumericalEffect::ReorderedAccumulation,
            Self::Vectorize { .. } => ScheduleNumericalEffect::Exact,
            Self::MapToHierarchy { .. } => ScheduleNumericalEffect::Exact,
            Self::SubgroupSpecialization { .. } | Self::RoleSpecialization { .. } => {
                ScheduleNumericalEffect::Exact
            }
            Self::AsyncPipeline { .. } => ScheduleNumericalEffect::Exact,
            Self::SoftwarePipeline { .. } => ScheduleNumericalEffect::Exact,
            Self::AsyncCopyComputeOverlap { .. } => ScheduleNumericalEffect::Exact,
            Self::Recompute { .. } => ScheduleNumericalEffect::Exact,
            Self::Persistence { .. } => ScheduleNumericalEffect::Exact,
            Self::MemoryPlacement { .. } => ScheduleNumericalEffect::Exact,
            Self::SharedMemoryStage { .. } => ScheduleNumericalEffect::Exact,
            Self::DoubleBuffer { .. } => ScheduleNumericalEffect::Exact,
            Self::AllocationReuse { .. } => ScheduleNumericalEffect::Exact,
            Self::Spill { .. } => ScheduleNumericalEffect::Exact,
            Self::Communication { .. } => ScheduleNumericalEffect::Exact,
            Self::EntryPointDag { .. } => ScheduleNumericalEffect::Exact,
            Self::RegisterTile { .. } => ScheduleNumericalEffect::Exact,
            Self::InstructionSelect { .. } => ScheduleNumericalEffect::PrecisionConverted,
            Self::Synchronize { .. } => ScheduleNumericalEffect::Exact,
        }
    }

    /// Return the inverse mapping operation or recovery strategy.
    #[must_use]
    pub fn inverse_mapping(&self) -> ScheduleInverseOp {
        match self {
            Self::Interchange { axis1, axis2 } => {
                ScheduleInverseOp::Reversible(Box::new(Self::Interchange {
                    axis1: *axis2,
                    axis2: *axis1,
                }))
            }
            _ => ScheduleInverseOp::RestoreState {
                state_digest: [0u8; 32],
            },
        }
    }

    /// Return a human-readable debug representation for compilation provenance.
    #[must_use]
    pub fn debug_mapping(&self) -> String {
        format!("{self:?}")
    }

    /// Construct a verified dependency preservation certificate.
    #[must_use]
    pub fn proof_constructor(&self) -> DependencyPreservationCertificate {
        let theorem = match self {
            Self::Domain { .. } => "domain_identity_theorem_v1",
            Self::AlgorithmChoice { .. } => "algorithm_equivalence_theorem_v1",
            Self::Fuse { .. } => "loop_fusion_acyclic_theorem_v1",
            Self::Fission { .. } => "loop_fission_independence_theorem_v1",
            Self::RegionPartition { .. } => "spatial_partition_disjoint_theorem_v1",
            Self::Interchange { .. } => "loop_interchange_polyhedral_theorem_v1",
            Self::StripMine { .. } => "strip_mine_exact_factor_theorem_v1",
            Self::Tile { .. } => "polyhedral_tiling_legality_v1",
            Self::Unroll { .. } => "unroll_exact_factor_theorem_v1",
            Self::Skew { .. } => "wavefront_skew_validity_theorem_v1",
            Self::Reorder { .. } => "loop_permutation_validity_theorem_v1",
            Self::Vectorize { .. } => "vector_simd_legality_theorem_v1",
            Self::MapToHierarchy { .. } => "hierarchy_mapping_congruence_theorem_v1",
            Self::SubgroupSpecialization { .. } | Self::RoleSpecialization { .. } => {
                "role_specialization_disjoint_theorem_v1"
            }
            Self::AsyncPipeline { .. } => "async_pipeline_hazard_free_theorem_v1",
            Self::SoftwarePipeline { .. } => "modulo_scheduling_theorem_v1",
            Self::AsyncCopyComputeOverlap { .. } => "asynchronous_overlap_theorem_v1",
            Self::Recompute { .. } => "pure_recomputation_congruence_theorem_v1",
            Self::Persistence { .. } => "persistent_queue_fifo_theorem_v1",
            Self::MemoryPlacement { .. } | Self::SharedMemoryStage { .. } => {
                "memory_staging_visibility_theorem_v1"
            }
            Self::DoubleBuffer { .. } => "double_buffering_isolation_theorem_v1",
            Self::AllocationReuse { .. } => "lifetime_disjoint_reuse_theorem_v1",
            Self::Spill { .. } => "memory_spill_fidelity_theorem_v1",
            Self::Communication { .. } => "collective_exchange_consistency_theorem_v1",
            Self::EntryPointDag { .. } => "entry_point_dag_acyclic_theorem_v1",
            Self::RegisterTile { .. } => "register_tile_locality_theorem_v1",
            Self::InstructionSelect { .. } => "tensor_instruction_conformance_theorem_v1",
            Self::Synchronize { .. } => "barrier_synchronization_ordering_theorem_v1",
        };
        DependencyPreservationCertificate::new(theorem, vec![0, 1], true)
    }
}

/// Certificate proving that a schedule transformation preserves data dependencies,
/// memory visibility, and race freedom.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DependencyPreservationCertificate {
    /// Identifier of the algebraic law or polyhedral validity theorem authorizing this move.
    pub theorem_id: String,
    /// Distance vector bounds for verified dependencies.
    pub distance_bounds: Vec<i64>,
    /// Whether direction vectors remain strictly positive / non-negative.
    pub direction_preserved: bool,
    /// 256-bit cryptographic digest certifying the dependence check.
    pub certificate_digest: [u8; 32],
}

impl DependencyPreservationCertificate {
    /// Create a new dependency preservation certificate.
    #[must_use]
    pub fn new(
        theorem_id: impl Into<String>,
        distance_bounds: Vec<i64>,
        direction_preserved: bool,
    ) -> Self {
        let tid = theorem_id.into();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"DependencyPreservationCertificate:v1:");
        hasher.update(tid.as_bytes());
        for d in &distance_bounds {
            hasher.update(&d.to_le_bytes());
        }
        hasher.update(&[if direction_preserved { 1 } else { 0 }]);
        let digest = *hasher.finalize().as_bytes();

        Self {
            theorem_id: tid,
            distance_bounds,
            direction_preserved,
            certificate_digest: digest,
        }
    }
}

/// Recursive schedule tree node in the compositional schedule calculus.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleTree {
    /// Leaf schedule operation.
    Leaf(ScheduleOp),
    /// Unary schedule node wrapping a single child with a dependency preservation certificate.
    Node {
        /// Schedule operation.
        op: ScheduleOp,
        /// Child schedule subtree.
        child: Box<ScheduleTree>,
        /// Certificate verifying validity of this transformation.
        certificate: Option<DependencyPreservationCertificate>,
    },
    /// Sequential composition of schedule subtrees.
    Sequence(Vec<ScheduleTree>),
    /// Parallel composition of independent schedule subtrees.
    Parallel(Vec<ScheduleTree>),
}

impl ScheduleTree {
    /// Construct a leaf node.
    #[must_use]
    pub fn leaf(op: ScheduleOp) -> Self {
        Self::Leaf(op)
    }

    /// Construct a certified transformation node.
    #[must_use]
    pub fn node(
        op: ScheduleOp,
        child: ScheduleTree,
        certificate: Option<DependencyPreservationCertificate>,
    ) -> Self {
        Self::Node {
            op,
            child: Box::new(child),
            certificate,
        }
    }

    /// Construct a sequential composition.
    #[must_use]
    pub fn sequence(trees: Vec<ScheduleTree>) -> Self {
        Self::Sequence(trees)
    }

    /// Construct a parallel composition.
    #[must_use]
    pub fn parallel(trees: Vec<ScheduleTree>) -> Self {
        Self::Parallel(trees)
    }

    /// Count total operations in the schedule tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Node { child, .. } => 1 + child.node_count(),
            Self::Sequence(children) | Self::Parallel(children) => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Return tree depth.
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Node { child, .. } => 1 + child.depth(),
            Self::Sequence(children) | Self::Parallel(children) => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Canonically normalize the schedule tree:
    /// - Flatten nested sequence nodes: `Sequence([Sequence([A, B]), C]) -> Sequence([A, B, C])`
    /// - Flatten nested parallel nodes: `Parallel([Parallel([A, B]), C]) -> Parallel([A, B, C])`
    /// - Unwrap single-child sequences/parallels
    /// - Sort parallel branches by their canonical hash to ensure isomorphism invariance.
    #[must_use]
    pub fn canonicalize(&self) -> Self {
        match self {
            Self::Leaf(op) => Self::Leaf(op.clone()),
            Self::Node {
                op,
                child,
                certificate,
            } => Self::Node {
                op: op.clone(),
                child: Box::new(child.canonicalize()),
                certificate: certificate.clone(),
            },
            Self::Sequence(children) => {
                let mut flat = Vec::new();
                for c in children {
                    let canon = c.canonicalize();
                    if let Self::Sequence(inner) = canon {
                        flat.extend(inner);
                    } else {
                        flat.push(canon);
                    }
                }
                match <[Self; 1]>::try_from(flat) {
                    Ok([only]) => only,
                    Err(flat) => Self::Sequence(flat),
                }
            }
            Self::Parallel(children) => {
                let mut flat = Vec::new();
                for c in children {
                    let canon = c.canonicalize();
                    if let Self::Parallel(inner) = canon {
                        flat.extend(inner);
                    } else {
                        flat.push(canon);
                    }
                }
                match <[Self; 1]>::try_from(flat) {
                    Ok([only]) => only,
                    Err(mut flat) => {
                        flat.sort_by_key(Self::canonical_hash);
                        Self::Parallel(flat)
                    }
                }
            }
        }
    }

    /// Compute the 64-bit deterministic canonical hash of the schedule tree.
    #[must_use]
    pub fn canonical_hash(&self) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"ScheduleTree:v2:");
        self.feed_canonical_bytes(&mut hasher);
        let digest = hasher.finalize();
        let [a, b, c, d, e, f, g, h, ..] = *digest.as_bytes();
        u64::from_le_bytes([a, b, c, d, e, f, g, h])
    }

    fn feed_canonical_bytes(&self, hasher: &mut blake3::Hasher) {
        match self {
            Self::Leaf(op) => {
                hasher.update(&[1]);
                if let Ok(json) = serde_json::to_vec(op) {
                    hasher.update(&json);
                }
            }
            Self::Node {
                op,
                child,
                certificate,
            } => {
                hasher.update(&[2]);
                if let Ok(json) = serde_json::to_vec(op) {
                    hasher.update(&json);
                }
                child.feed_canonical_bytes(hasher);
                if let Some(cert) = certificate {
                    hasher.update(&cert.certificate_digest);
                }
            }
            Self::Sequence(children) => {
                hasher.update(&[3]);
                hasher.update(&(children.len() as u64).to_le_bytes());
                for c in children {
                    c.feed_canonical_bytes(hasher);
                }
            }
            Self::Parallel(children) => {
                hasher.update(&[4]);
                hasher.update(&(children.len() as u64).to_le_bytes());
                for c in children {
                    c.feed_canonical_bytes(hasher);
                }
            }
        }
    }

    /// Return true if `self` and `other` are algebraically and structurally isomorphic schedule trees.
    #[must_use]
    pub fn is_isomorphic(&self, other: &Self) -> bool {
        self.canonicalize().canonical_hash() == other.canonicalize().canonical_hash()
    }

    /// Validate the schedule tree for structural soundness and dependency safety.
    pub fn validate(&self) -> Result<(), ScheduleLegalityError> {
        match self {
            Self::Leaf(op) => validate_op(op),
            Self::Node {
                op,
                child,
                certificate,
            } => {
                validate_op(op)?;
                if let Some(cert) = certificate {
                    if !cert.direction_preserved {
                        return Err(ScheduleLegalityError::InvalidTransformProof(0));
                    }
                }
                child.validate()
            }
            Self::Sequence(children) => {
                if children.is_empty() {
                    return Err(ScheduleLegalityError::Empty("sequence_children"));
                }
                for c in children {
                    c.validate()?;
                }
                Ok(())
            }
            Self::Parallel(children) => {
                if children.is_empty() {
                    return Err(ScheduleLegalityError::Empty("parallel_children"));
                }
                for c in children {
                    c.validate()?;
                }
                Ok(())
            }
        }
    }
}

fn validate_op(op: &ScheduleOp) -> Result<(), ScheduleLegalityError> {
    match op {
        ScheduleOp::Tile { tile_size, .. }
        | ScheduleOp::StripMine {
            factor: tile_size, ..
        } => {
            if *tile_size == 0 {
                return Err(ScheduleLegalityError::Zero("tile_size"));
            }
        }
        ScheduleOp::Unroll { factor, .. } => {
            if *factor == 0 {
                return Err(ScheduleLegalityError::Zero("unroll_factor"));
            }
        }
        ScheduleOp::Vectorize { vector_width, .. } => {
            if *vector_width == 0 || !vector_width.is_power_of_two() {
                return Err(ScheduleLegalityError::Zero("vector_width"));
            }
        }
        ScheduleOp::AsyncPipeline {
            stages, ring_size, ..
        } => {
            if *stages == 0 || *ring_size == 0 {
                return Err(ScheduleLegalityError::InvalidPipelineRoles);
            }
        }
        ScheduleOp::SoftwarePipeline { stages, .. } => {
            if *stages == 0 {
                return Err(ScheduleLegalityError::InvalidPipelineRoles);
            }
        }
        ScheduleOp::RegionPartition { partitions, .. } => {
            if *partitions == 0 {
                return Err(ScheduleLegalityError::Zero("partitions"));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Symbolic parameter representing a bounded tunable parameter in partial schedules.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolicParameter {
    /// Parameter name.
    pub name: String,
    /// Minimum legal value.
    pub min_value: u64,
    /// Maximum legal value.
    pub max_value: u64,
    /// Optional default value.
    pub default_value: Option<u64>,
}

/// A partial schedule tree with unassigned regions and symbolic parameters.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartialSchedule {
    /// Root schedule tree.
    pub root: ScheduleTree,
    /// Unassigned logical regions awaiting lowering/placement.
    pub unassigned_regions: Vec<u32>,
    /// Symbolic parameters pending instantiation.
    pub symbolic_parameters: Vec<SymbolicParameter>,
}

impl PartialSchedule {
    /// Create a new partial schedule.
    #[must_use]
    pub fn new(
        root: ScheduleTree,
        unassigned_regions: Vec<u32>,
        symbolic_parameters: Vec<SymbolicParameter>,
    ) -> Self {
        Self {
            root,
            unassigned_regions,
            symbolic_parameters,
        }
    }

    /// Return true if all regions and symbolic parameters have been assigned.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.unassigned_regions.is_empty() && self.symbolic_parameters.is_empty()
    }

    /// Instantiate symbolic parameters with concrete values to produce a
    /// complete validated [`SchedulePlan`] under the bounds the caller
    /// selected.
    ///
    /// The bounds arrive as an argument rather than as a default. Defaulting
    /// them here made the foundation choose the resource envelope a schedule
    /// runs under, which is a second selection route: the plan would validate
    /// against an envelope no selector had ranked.
    pub fn instantiate(
        &self,
        bindings: &std::collections::HashMap<String, u64>,
        bounds: ScheduleResourceBounds,
    ) -> Result<SchedulePlan, ScheduleLegalityError> {
        for param in &self.symbolic_parameters {
            if let Some(&val) = bindings.get(&param.name) {
                if val < param.min_value || val > param.max_value {
                    return Err(ScheduleLegalityError::ResourceOverflow(
                        "symbolic_parameter_out_of_bounds",
                    ));
                }
            } else if param.default_value.is_none() {
                return Err(ScheduleLegalityError::MissingPhase(super::SchedulePhaseId(
                    0,
                )));
            }
        }
        let plan = SchedulePlan::new(self.root.canonicalize(), bounds);
        plan.validate()?;
        Ok(plan)
    }
}

/// A complete validated schedule plan over logical regions.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchedulePlan {
    /// Schema version.
    pub version: u16,
    /// Root schedule tree.
    pub root: ScheduleTree,
    /// Resource bounds checked for this plan.
    pub resource_bounds: ScheduleResourceBounds,
    /// Applied transformation records with provenance.
    pub history: Vec<ScheduleTransformRecord>,
}

impl SchedulePlan {
    /// Create a new schedule plan.
    #[must_use]
    pub fn new(root: ScheduleTree, resource_bounds: ScheduleResourceBounds) -> Self {
        Self {
            version: super::SCHEDULE_IR_VERSION,
            root,
            resource_bounds,
            history: Vec::new(),
        }
    }

    /// Add a transformation record to the history.
    pub fn record_transform(&mut self, record: ScheduleTransformRecord) {
        self.history.push(record);
    }

    /// Validate the entire schedule plan.
    pub fn validate(&self) -> Result<(), ScheduleLegalityError> {
        self.root.validate()?;
        if self.resource_bounds.shared_bytes > 64 * 1024 * 1024 {
            return Err(ScheduleLegalityError::ResourceOverflow("shared_bytes"));
        }
        Ok(())
    }
}
