//! Type-safe compositional schedule calculus.
//!
//! Provides static type guarantees preventing invalid schedule compositions at compile time:
//! hierarchy mapping order, memory scope legality, and nonzero dimension bounds.

use std::num::{NonZeroU32, NonZeroU64};

use super::{
    tree::{DependencyPreservationCertificate, ScheduleOp, ScheduleTree},
    MappingLevel, MemoryPlacement, PipelineRoleGroup, SynchronizationScope,
};

/// Marker trait representing a valid execution hierarchy level in schedule construction.
pub trait ScheduleScope: Clone + Copy + std::fmt::Debug + PartialEq + Eq + std::hash::Hash {}

/// Device-wide whole-grid execution scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceScope;
impl ScheduleScope for DeviceScope {}

/// Workgroup (thread-block) cooperative execution scope with shared memory access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorkgroupScope;
impl ScheduleScope for WorkgroupScope {}

/// Subgroup SIMD cooperative execution scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupScope;
impl ScheduleScope for SubgroupScope {}

/// Individual thread execution scope with private register access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ThreadScope;
impl ScheduleScope for ThreadScope {}

/// Innermost loop body eligible for unrolling and SIMD vectorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InnermostLoopScope;
impl ScheduleScope for InnermostLoopScope {}

/// A type-safe schedule stage enforcing valid operator nesting and scope at compile time.
#[derive(Clone, Debug)]
pub struct TypedScheduleStage<S: ScheduleScope> {
    /// Scope marker type.
    pub scope: S,
    /// Logical region identifier.
    pub region: u32,
    /// Accumulated schedule operations.
    pub ops: Vec<ScheduleOp>,
    /// Optional dependency preservation certificate.
    pub certificate: Option<DependencyPreservationCertificate>,
}

impl TypedScheduleStage<DeviceScope> {
    /// Create a new device-level schedule stage for a logical region.
    #[must_use]
    pub fn new(region: u32) -> Self {
        Self {
            scope: DeviceScope,
            region,
            ops: vec![ScheduleOp::Domain {
                region,
                extents: vec![1],
            }],
            certificate: None,
        }
    }

    /// Partition regions across compute units.
    #[must_use]
    pub fn region_partition(mut self, partitions: NonZeroU32) -> Self {
        self.ops.push(ScheduleOp::RegionPartition {
            region: self.region,
            partitions: partitions.get(),
            level: MappingLevel::ComputeUnitPartition,
        });
        self
    }

    /// Map an outer axis to workgroup hierarchy level, transitioning into WorkgroupScope.
    #[must_use]
    pub fn map_to_workgroup(
        mut self,
        axis: u32,
        dimension: u32,
    ) -> TypedScheduleStage<WorkgroupScope> {
        self.ops.push(ScheduleOp::MapToHierarchy {
            axis,
            level: MappingLevel::Workgroup,
            dimension,
        });
        TypedScheduleStage {
            scope: WorkgroupScope,
            region: self.region,
            ops: self.ops,
            certificate: self.certificate,
        }
    }

    /// Synchronize all workgroups at device scope.
    #[must_use]
    pub fn synchronize_device(mut self) -> Self {
        self.ops.push(ScheduleOp::Synchronize {
            scope: SynchronizationScope::Device,
        });
        self
    }
}

impl TypedScheduleStage<WorkgroupScope> {
    /// Tile an axis into outer and inner blocks using a non-zero tile size.
    #[must_use]
    pub fn tile(mut self, axis: u32, tile_size: NonZeroU64, inner_axis: u32) -> Self {
        self.ops.push(ScheduleOp::Tile {
            axis,
            tile_size: tile_size.get(),
            inner_axis,
        });
        self
    }

    /// Stage a buffer into workgroup shared memory.
    #[must_use]
    pub fn stage_shared_memory(mut self, buffer: impl Into<String>, staging_bytes: u64) -> Self {
        self.ops.push(ScheduleOp::SharedMemoryStage {
            buffer: buffer.into(),
            placement: MemoryPlacement::Workgroup,
            staging_bytes,
        });
        self
    }

    /// Synchronize invocations within the workgroup.
    #[must_use]
    pub fn synchronize_workgroup(mut self) -> Self {
        self.ops.push(ScheduleOp::Synchronize {
            scope: SynchronizationScope::Workgroup,
        });
        self
    }

    /// Map an axis to subgroup level, transitioning into SubgroupScope.
    #[must_use]
    pub fn map_to_subgroup(
        mut self,
        axis: u32,
        dimension: u32,
    ) -> TypedScheduleStage<SubgroupScope> {
        self.ops.push(ScheduleOp::MapToHierarchy {
            axis,
            level: MappingLevel::Subgroup,
            dimension,
        });
        TypedScheduleStage {
            scope: SubgroupScope,
            region: self.region,
            ops: self.ops,
            certificate: self.certificate,
        }
    }
}

impl TypedScheduleStage<SubgroupScope> {
    /// Specialize subgroups into dedicated producer/consumer roles.
    #[must_use]
    pub fn specialize_roles(mut self, roles: Vec<PipelineRoleGroup>) -> Self {
        self.ops.push(ScheduleOp::SubgroupSpecialization { roles });
        self
    }

    /// Map an axis to lane/thread level, transitioning into ThreadScope.
    #[must_use]
    pub fn map_to_thread(mut self, axis: u32, dimension: u32) -> TypedScheduleStage<ThreadScope> {
        self.ops.push(ScheduleOp::MapToHierarchy {
            axis,
            level: MappingLevel::Lane,
            dimension,
        });
        TypedScheduleStage {
            scope: ThreadScope,
            region: self.region,
            ops: self.ops,
            certificate: self.certificate,
        }
    }
}

impl TypedScheduleStage<ThreadScope> {
    /// Place micro-kernel register tiles in thread-private memory.
    #[must_use]
    pub fn register_tile(mut self, m: u32, n: u32, k: u32) -> Self {
        self.ops.push(ScheduleOp::RegisterTile {
            register_dim_m: m,
            register_dim_n: n,
            register_dim_k: k,
        });
        self
    }

    /// Enter the innermost loop body.
    #[must_use]
    pub fn enter_innermost_loop(self) -> TypedScheduleStage<InnermostLoopScope> {
        TypedScheduleStage {
            scope: InnermostLoopScope,
            region: self.region,
            ops: self.ops,
            certificate: self.certificate,
        }
    }
}

impl TypedScheduleStage<InnermostLoopScope> {
    /// Unroll the loop by a non-zero constant factor.
    #[must_use]
    pub fn unroll(mut self, axis: u32, factor: NonZeroU32) -> Self {
        self.ops.push(ScheduleOp::Unroll {
            axis,
            factor: factor.get(),
        });
        self
    }

    /// Vectorize the loop with a non-zero SIMD lane width.
    #[must_use]
    pub fn vectorize(mut self, axis: u32, width: NonZeroU32) -> Self {
        self.ops.push(ScheduleOp::Vectorize {
            axis,
            vector_width: width.get(),
        });
        self
    }

    /// Select a target tensor instruction.
    #[must_use]
    pub fn instruction_select(
        mut self,
        intrinsic: impl Into<String>,
        mma_shape: Option<[u32; 3]>,
    ) -> Self {
        self.ops.push(ScheduleOp::InstructionSelect {
            target_intrinsic: intrinsic.into(),
            mma_shape,
        });
        self
    }

    /// Build the validated schedule tree.
    #[must_use]
    pub fn build(self) -> ScheduleTree {
        let trees = self
            .ops
            .into_iter()
            .map(ScheduleTree::Leaf)
            .collect::<Vec<_>>();
        match <[ScheduleTree; 1]>::try_from(trees) {
            Ok([only]) => only,
            Err(trees) => ScheduleTree::Sequence(trees),
        }
    }
}
