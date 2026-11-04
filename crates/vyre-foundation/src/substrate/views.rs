//! Stage-specific read-only views for the five compiler levels.
//!
//! Enforces immutability of shared IR and prevents smuggling lower-level objects
//! (e.g. physical kernel descriptors or target payloads) into upper-level queries.

use core::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ir::{GraphNodeId, GraphValueId, ProgramGraph, ProgramGraphNode, ProgramGraphValue};
use crate::logical::{LogicalProgramGraph, LogicalRegion};

/// The five architectural compiler levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CompilerLevelStage {
    /// Level 1: Whole-program graph of connected operations and value contracts.
    WholeProgramGraph = 1,
    /// Level 2: Logical region and algorithm IR with extents and effect contracts.
    LogicalRegion = 2,
    /// Level 3: Selected schedule IR with tiling, fusion, and launch geometry.
    SelectedSchedule = 3,
    /// Level 4: Lowered physical kernel IR with register allocation and descriptors.
    PhysicalKernel = 4,
    /// Level 5: Target payload bytecode, shader text, and entry points.
    TargetPayload = 5,
}

impl CompilerLevelStage {
    /// Human-readable level name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::WholeProgramGraph => "Level 1: Whole-Program Graph",
            Self::LogicalRegion => "Level 2: Logical Region IR",
            Self::SelectedSchedule => "Level 3: Selected Schedule IR",
            Self::PhysicalKernel => "Level 4: Physical Kernel IR",
            Self::TargetPayload => "Level 5: Target Payload",
        }
    }

    /// Numeric level index (1 to 5).
    #[must_use]
    pub const fn level_number(self) -> u8 {
        self as u8
    }
}

impl fmt::Display for CompilerLevelStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Error raised when an upper-level query attempts to access or smuggle a lower-level construct.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum LevelAccessError {
    /// Smuggling violation: caller at an upper level attempted to access a lower level.
    #[error("Level boundary violation: {caller} attempted to access {target}; upper-level passes cannot depend on lower-level physical/target state")]
    SmugglingViolation {
        /// Level of the caller or pass.
        caller: CompilerLevelStage,
        /// Level of the target data being accessed.
        target: CompilerLevelStage,
    },
}

/// Enforce that a caller at `caller_level` can only query state at or above its owning abstraction level.
pub fn enforce_level_boundary(
    caller_level: CompilerLevelStage,
    target_level: CompilerLevelStage,
) -> Result<(), LevelAccessError> {
    // An upper-level pass (e.g. Level 1 or 2) must NEVER access lower-level objects (Level 3, 4, 5).
    if (caller_level as u8) < (target_level as u8) {
        return Err(LevelAccessError::SmugglingViolation {
            caller: caller_level,
            target: target_level,
        });
    }
    Ok(())
}

/// Read-only view over Level 1 whole-program graph.
///
/// Immutability guarantee: this type exposes only read-only methods and provides no
/// mutable access to the underlying `ProgramGraph`.
#[derive(Debug, Clone, Copy)]
pub struct WholeProgramGraphView<'a> {
    graph: &'a ProgramGraph,
}

impl<'a> WholeProgramGraphView<'a> {
    /// Create a new read-only view for a program graph.
    #[must_use]
    pub fn new(graph: &'a ProgramGraph) -> Self {
        Self { graph }
    }

    /// Access all nodes in the graph.
    #[must_use]
    pub fn nodes(&self) -> &'a [ProgramGraphNode] {
        self.graph.nodes()
    }

    /// Access all connected values in the graph.
    #[must_use]
    pub fn values(&self) -> &'a [ProgramGraphValue] {
        self.graph.values()
    }

    /// Access a specific node by its canonical identifier.
    #[must_use]
    pub fn node(&self, id: GraphNodeId) -> Option<&'a ProgramGraphNode> {
        self.graph.nodes().get(id.0 as usize)
    }

    /// Access a specific value by its canonical identifier.
    #[must_use]
    pub fn value(&self, id: GraphValueId) -> Option<&'a ProgramGraphValue> {
        self.graph.values().get(id.0 as usize)
    }

    /// Total number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Total number of values in the graph.
    #[must_use]
    pub fn value_count(&self) -> usize {
        self.graph.values().len()
    }

    /// Level stage classification.
    #[must_use]
    pub const fn level(&self) -> CompilerLevelStage {
        CompilerLevelStage::WholeProgramGraph
    }
}

/// Read-only view over Level 2 logical algorithm regions.
///
/// Immutability guarantee: provides read-only inspection of logical regions,
/// extents, and contracts without permitting direct mutation.
#[derive(Debug, Clone, Copy)]
pub struct LogicalRegionView<'a> {
    logical_graph: &'a LogicalProgramGraph<'a>,
}

impl<'a> LogicalRegionView<'a> {
    /// Create a new read-only view for a logical program graph.
    #[must_use]
    pub fn new(logical_graph: &'a LogicalProgramGraph<'a>) -> Self {
        Self { logical_graph }
    }

    /// Access logical regions in the graph.
    #[must_use]
    pub fn regions(&self) -> &'a [LogicalRegion] {
        self.logical_graph.regions()
    }

    /// Total number of logical regions.
    #[must_use]
    pub fn region_count(&self) -> usize {
        self.logical_graph.regions().len()
    }

    /// Level stage classification.
    #[must_use]
    pub const fn level(&self) -> CompilerLevelStage {
        CompilerLevelStage::LogicalRegion
    }
}

/// Read-only view over Level 3 selected schedule IR.
#[derive(Debug, Clone)]
pub struct SelectedScheduleView<'a> {
    /// Serialized or structured schedule identifier.
    pub schedule_digest: [u8; 32],
    /// Fusion group count.
    pub fusion_groups: usize,
    /// Total barrier count.
    pub barrier_count: usize,
    /// Launch geometry: workgroup shape `[x, y, z]`.
    pub workgroup_shape: [u32; 3],
    /// Lifetime marker for the borrowed schedule plan.
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> SelectedScheduleView<'a> {
    /// Create a new read-only selected schedule view.
    #[must_use]
    pub fn new(
        schedule_digest: [u8; 32],
        fusion_groups: usize,
        barrier_count: usize,
        workgroup_shape: [u32; 3],
    ) -> Self {
        Self {
            schedule_digest,
            fusion_groups,
            barrier_count,
            workgroup_shape,
            _marker: core::marker::PhantomData,
        }
    }

    /// Level stage classification.
    #[must_use]
    pub const fn level(&self) -> CompilerLevelStage {
        CompilerLevelStage::SelectedSchedule
    }
}

/// Read-only view over Level 4 physical kernel descriptors.
#[derive(Debug, Clone)]
pub struct PhysicalKernelView<'a> {
    /// Entry point symbol name.
    pub entry_point: String,
    /// Workgroup size configuration `[x, y, z]`.
    pub workgroup_size: [u32; 3],
    /// Required registers per invocation.
    pub registers_required: u32,
    /// Bound storage and uniform buffer indices.
    pub binding_indices: Vec<u32>,
    /// Lifetime marker.
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a> PhysicalKernelView<'a> {
    /// Create a new read-only physical kernel view.
    #[must_use]
    pub fn new(
        entry_point: impl Into<String>,
        workgroup_size: [u32; 3],
        registers_required: u32,
        binding_indices: Vec<u32>,
    ) -> Self {
        Self {
            entry_point: entry_point.into(),
            workgroup_size,
            registers_required,
            binding_indices,
            _marker: core::marker::PhantomData,
        }
    }

    /// Level stage classification.
    #[must_use]
    pub const fn level(&self) -> CompilerLevelStage {
        CompilerLevelStage::PhysicalKernel
    }
}

/// Read-only view over Level 5 target payload.
#[derive(Debug, Clone)]
pub struct TargetPayloadView<'a> {
    /// Format identity supplied by the target materializer.
    pub format_name: String,
    /// Format version number.
    pub format_version: u16,
    /// Emitted target bytecode or text bytes.
    pub payload_bytes: &'a [u8],
    /// Entry point names in the payload.
    pub entry_points: Vec<String>,
}

impl<'a> TargetPayloadView<'a> {
    /// Create a new read-only target payload view.
    #[must_use]
    pub fn new(
        format_name: impl Into<String>,
        format_version: u16,
        payload_bytes: &'a [u8],
        entry_points: Vec<String>,
    ) -> Self {
        Self {
            format_name: format_name.into(),
            format_version,
            payload_bytes,
            entry_points,
        }
    }

    /// Level stage classification.
    #[must_use]
    pub const fn level(&self) -> CompilerLevelStage {
        CompilerLevelStage::TargetPayload
    }
}
