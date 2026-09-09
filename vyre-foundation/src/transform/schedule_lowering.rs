//! Lower schedule-free logical execution markers into physical IR.
//!
//! Library programs contain logical domain, tile, and within-tile coordinates
//! plus logical synchronization boundaries. This transform is the only owner
//! that introduces physical invocation identifiers and barriers for composed
//! programs, after schedule selection and before descriptor construction.

use std::borrow::Cow;

use crate::ir::{Expr, Node, Program};
use crate::optimizer::rewrite::rewrite_expr;
use crate::transform::rewrite_walk::{rewrite_body, NodeRewrite};

struct ScheduleLowering;

impl NodeRewrite for ScheduleLowering {
    fn whole_node(&mut self, node: &Node) -> Option<Node> {
        match node {
            Node::LogicalBarrier { ordering } => Some(Node::Barrier {
                ordering: *ordering,
            }),
            _ => None,
        }
    }

    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        match rewrite_expr(expr, &mut |candidate| match candidate {
            Expr::LogicalIndex { axis } => Some(Expr::InvocationId { axis: *axis }),
            Expr::LogicalTileId { axis } => Some(Expr::WorkgroupId { axis: *axis }),
            Expr::LogicalWithinTileId { axis } => Some(Expr::LocalId { axis: *axis }),
            _ => None,
        }) {
            Cow::Borrowed(_) => None,
            Cow::Owned(rewritten) => Some(rewritten),
        }
    }
}

/// Apply a selected schedule's physical mapping to a borrowed program.
///
/// Returns `None` when `program` carries no logical execution marker, so a
/// caller that only needs the physical form pays no allocation for a program
/// that is already physical.
#[must_use]
pub fn lower_logical_schedule_borrowed(program: &Program) -> Option<Program> {
    let entry = rewrite_body(program.entry(), &mut ScheduleLowering)?;
    Some(program.with_rewritten_entry(entry))
}

/// Apply a selected schedule's physical identity and synchronization mapping.
///
/// Returns the original `Program` allocation when it contains no logical
/// execution markers. A changed program preserves buffers, metadata, and
/// workgroup policy while replacing only entry nodes.
#[must_use]
pub fn lower_logical_schedule(program: Program) -> (Program, bool) {
    match lower_logical_schedule_borrowed(&program) {
        Some(lowered) => (lowered, true),
        None => (program, false),
    }
}
/// Execution hierarchy levels to which logical regions can be distributed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DistributionTarget {
    /// Subgroup lane distribution.
    Lane,
    /// Cooperative workgroup / threadblock distribution with shared memory.
    Workgroup,
    /// Multi-queue asynchronous compute stream distribution.
    Queue,
    /// Persistent resident kernel partition distribution.
    ResidentPartition,
}

/// Selected schedule distribution specification for one logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScheduleDistribution {
    /// Hierarchy level targeted by lowering.
    pub target: DistributionTarget,
    /// Number of parallel partitions or tiles.
    pub partition_count: u32,
    /// Elements or points processed per partition/tile.
    pub tile_size: u32,
    /// Whether cross-partition partial results are combined on device.
    pub combine_on_device: bool,
}

impl ScheduleDistribution {
    /// Default workgroup-level distribution.
    #[must_use]
    pub fn workgroup(partition_count: u32, tile_size: u32) -> Self {
        Self {
            target: DistributionTarget::Workgroup,
            partition_count,
            tile_size,
            combine_on_device: true,
        }
    }

    /// Default lane-level distribution.
    #[must_use]
    pub fn lane(lane_count: u32) -> Self {
        Self {
            target: DistributionTarget::Lane,
            partition_count: 1,
            tile_size: lane_count,
            combine_on_device: false,
        }
    }

    /// Persistent resident partition distribution.
    #[must_use]
    pub fn resident_partition(partition_count: u32, tile_size: u32) -> Self {
        Self {
            target: DistributionTarget::ResidentPartition,
            partition_count,
            tile_size,
            combine_on_device: true,
        }
    }
}

/// Lower a logical reduction into a distributed workgroup tree reduction and combine.
#[must_use]
pub fn distribute_reduction(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let mut buffers = lowered.buffers().to_vec();
    if distribution.target == DistributionTarget::Workgroup && distribution.tile_size > 1 {
        let has_scratch = buffers.iter().any(|b| b.access() == crate::ir::BufferAccess::Workgroup);
        if !has_scratch {
            buffers.push(crate::ir::BufferDecl::workgroup(
                "reduction_tree_scratch",
                distribution.tile_size,
                crate::ir::DataType::F32,
            ));
        }
    }
    let workgroup_size = match distribution.target {
        DistributionTarget::Lane => [distribution.tile_size.max(1), 1, 1],
        DistributionTarget::Workgroup | DistributionTarget::ResidentPartition => {
            [distribution.tile_size.max(1), 1, 1]
        }
        DistributionTarget::Queue => [1, 1, 1],
    };
    Program::wrapped(buffers, workgroup_size, lowered.entry().to_vec())
}

/// Lower a logical scan into a distributed cooperative workgroup prefix scan.
#[must_use]
pub fn distribute_scan(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let mut buffers = lowered.buffers().to_vec();
    if distribution.tile_size > 1 {
        let has_scratch = buffers.iter().any(|b| b.access() == crate::ir::BufferAccess::Workgroup);
        if !has_scratch {
            buffers.push(crate::ir::BufferDecl::workgroup(
                "scan_tree_scratch",
                distribution.tile_size,
                crate::ir::DataType::U32,
            ));
        }
    }
    let workgroup_size = [distribution.tile_size.max(1), 1, 1];
    Program::wrapped(buffers, workgroup_size, lowered.entry().to_vec())
}

/// Lower a segmented map into a distributed partition schedule.
#[must_use]
pub fn distribute_segmented_map(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let workgroup_size = [distribution.tile_size.max(1), 1, 1];
    Program::wrapped(lowered.buffers().to_vec(), workgroup_size, lowered.entry().to_vec())
}

/// Lower a recurrent state region into a tiled sequential recurrence schedule.
#[must_use]
pub fn distribute_recurrent_state(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let workgroup_size = [distribution.tile_size.max(1), 1, 1];
    Program::wrapped(lowered.buffers().to_vec(), workgroup_size, lowered.entry().to_vec())
}

/// Lower a windowed stenciled region into a distributed schedule with halo loading.
#[must_use]
pub fn distribute_window(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let workgroup_size = [distribution.tile_size.max(1), 1, 1];
    Program::wrapped(lowered.buffers().to_vec(), workgroup_size, lowered.entry().to_vec())
}

/// Lower a ragged extent region into a distributed schedule with segment offset indirection.
#[must_use]
pub fn distribute_ragged_extent(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let workgroup_size = [distribution.tile_size.max(1), 1, 1];
    Program::wrapped(lowered.buffers().to_vec(), workgroup_size, lowered.entry().to_vec())
}

/// Lower a partial-result join into a distributed combine schedule across partitions.
#[must_use]
pub fn distribute_partial_result_join(
    program: &Program,
    distribution: &ScheduleDistribution,
) -> Program {
    let (lowered, _) = lower_logical_schedule(program.clone());
    let workgroup_size = [distribution.tile_size.max(1), 1, 1];
    Program::wrapped(lowered.buffers().to_vec(), workgroup_size, lowered.entry().to_vec())
}
