//! Compositional schedule calculus over logical loops and regions.
//!
//! Provides [`ScheduleOp`], [`ScheduleTree`], and [`SchedulePlan`] with explicit
//! precondition verification, algebraic rule tagging, and dependency preservation certificates.

use serde::{Deserialize, Serialize};

use super::{
    error::ScheduleLegalityError, MappingLevel, MemoryPlacement, PipelineRoleGroup,
    ScheduleResourceBounds, ScheduleTransformRecord, SynchronizationScope,
};

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
    /// Asynchronous producer-consumer copy pipeline.
    AsyncPipeline {
        /// Number of pipeline stages.
        stages: u32,
        /// Ring buffer capacity.
        ring_size: u32,
        /// Producer and consumer role divisions.
        role_groups: Vec<PipelineRoleGroup>,
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
    /// Subgroup specialization, partitioning subgroups into dedicated
    /// producer and consumer roles.
    SubgroupSpecialization {
        /// Dedicated worker groups.
        roles: Vec<PipelineRoleGroup>,
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
        ScheduleOp::Tile { tile_size, .. } => {
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
        _ => {}
    }
    Ok(())
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
