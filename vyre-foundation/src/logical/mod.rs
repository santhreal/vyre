//! Versioned logical algorithm stage between graph topology and schedule search.
//!
//! A [`ProgramGraph`](crate::ir::ProgramGraph) states whole-program values and
//! dependencies. This module derives the schedule-free region contracts the
//! compiler searches over: typed extents, logical axes, effects, layouts, and
//! bounds. The source `Program` remains available during migration, but its
//! workgroup size is excluded from semantic identity. A selected schedule
//! records that choice separately.

mod graph;
mod region;

pub use graph::LogicalProgramGraph;
pub use region::{
    LogicalAliasFacts, LogicalCombineOp, LogicalDependence, LogicalDependenceKind, LogicalEffects,
    LogicalExtent, LogicalIndexMap, LogicalLayout, LogicalProgramError, LogicalRegion,
    LogicalRegionKind, OrderingContract, OrderingSyncScope, PartialResultJoinDescriptor,
    ProgressContract, RecurrenceDescriptor, ScanDirection, ScratchContract, SegmentDescriptor,
    WindowDescriptor, LOGICAL_ALGORITHM_VERSION,
};

pub use crate::logical_partition::{
    LogicalExchange, LogicalExchangeKind, LogicalPartitionAxis, LogicalPartitionAxisKind,
    LogicalPartitionFacts,
};
