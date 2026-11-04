//! Schedule tiles for region-level fusion.
//!
//! Schedule tiles divide multi-dimensional logical iteration spaces into
//! hierarchy-aware execution chunks (registers, shared memory, workgroups).
//! Fusing across schedule tiles enables register reuse, shared-memory staging,
//! and software pipelining.

use serde::{Deserialize, Serialize};

use crate::ir_inner::model::tile::Residency;

/// Hardware residency selected for a schedule tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileResidency {
    /// Private to one thread invocation (registers).
    Register,
    /// Shared memory within one cooperative workgroup.
    WorkgroupShared,
    /// Distributed across the invocations of a subgroup.
    Subgroup,
    /// Global memory buffer tile.
    Global,
}

impl From<Residency> for TileResidency {
    fn from(residency: Residency) -> Self {
        match residency {
            Residency::Register => Self::Register,
            Residency::Workgroup => Self::WorkgroupShared,
            Residency::Subgroup => Self::Subgroup,
            Residency::Global => Self::Global,
        }
    }
}

/// A multi-dimensional schedule tile configuration.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScheduleTile {
    /// Static tile dimensions along logical axes [tile_x, tile_y, ...].
    pub dimensions: Vec<u32>,
    /// Vector width along the innermost contiguous dimension.
    pub vector_width: u32,
    /// Target hardware residency for the tile.
    pub residency: TileResidency,
    /// Number of elements in one tile.
    pub capacity: u64,
}

impl ScheduleTile {
    /// Construct a new schedule tile description.
    #[must_use]
    pub fn new(dimensions: Vec<u32>, vector_width: u32, residency: TileResidency) -> Self {
        let capacity = if dimensions.is_empty() {
            1
        } else {
            dimensions.iter().map(|&d| u64::from(d.max(1))).product()
        };
        Self {
            dimensions,
            vector_width: vector_width.max(1),
            residency,
            capacity,
        }
    }

    /// Construct a register-level 1D tile.
    #[must_use]
    pub fn register_1d(size: u32) -> Self {
        Self::new(vec![size], 1, TileResidency::Register)
    }

    /// Construct a workgroup shared memory tile.
    #[must_use]
    pub fn shared_tile(dimensions: Vec<u32>) -> Self {
        Self::new(dimensions, 1, TileResidency::WorkgroupShared)
    }
}

/// Pipelining plan across schedule tile iterations.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TilePipeliningPlan {
    /// Number of ring buffer slots allocated for asynchronous staging.
    pub ring_slots: u32,
    /// Number of pipeline stages overlapped across iterations.
    pub stages: u32,
    /// Whether double-buffering is active for shared-memory tiles.
    pub double_buffered: bool,
}

impl Default for TilePipeliningPlan {
    fn default() -> Self {
        Self {
            ring_slots: 1,
            stages: 1,
            double_buffered: false,
        }
    }
}

impl TilePipeliningPlan {
    /// Construct a 2-stage double-buffered pipeline plan.
    #[must_use]
    pub fn double_buffered() -> Self {
        Self {
            ring_slots: 2,
            stages: 2,
            double_buffered: true,
        }
    }
}
