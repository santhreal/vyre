//! Logical region iteration spaces and region-level fusion planning.
//!
//! A fusion decision is made over [`LogicalRegion`]s and their iteration spaces,
//! not over finished kernels glued end to end. This module defines the iteration
//! space representation, domain compatibility analysis, and region fusion planning.

use serde::Serialize;

use crate::logical::{LogicalExtent, LogicalRegion, LogicalRegionKind};

/// Multi-dimensional iteration space for a logical region.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct IterationSpace {
    /// Extents along each logical iteration axis.
    pub extents: Vec<u64>,
    /// Semantic kind of the iteration domain.
    pub kind: LogicalRegionKind,
    /// Number of dimensions in the iteration space.
    pub rank: usize,
    /// Whether all extents are statically known and fixed.
    pub is_static: bool,
}

impl IterationSpace {
    /// Construct an iteration space from extents and region kind.
    #[must_use]
    pub fn new(extents: Vec<u64>, kind: LogicalRegionKind) -> Self {
        let rank = extents.len();
        Self {
            extents,
            kind,
            rank,
            is_static: true,
        }
    }

    /// Derive the iteration space from a [`LogicalRegion`].
    #[must_use]
    pub fn from_region(region: &LogicalRegion) -> Self {
        let extents: Vec<u64> = region
            .extents
            .iter()
            .map(|e| match e {
                LogicalExtent::Static(val) => *val,
                LogicalExtent::GraphValue { bound, .. } => *bound,
            })
            .collect();
        Self::new(extents, region.kind)
    }

    /// Total number of logical points in the iteration space.
    #[must_use]
    pub fn total_points(&self) -> u64 {
        if self.extents.is_empty() {
            1
        } else {
            self.extents.iter().copied().product()
        }
    }
    /// Check whether two iteration spaces are point-to-point congruent (same rank and extents).
    #[must_use]
    pub fn is_congruent_with(&self, other: &Self) -> bool {
        self.extents == other.extents
    }

    /// Check whether this iteration space can be partitioned into tiles of shape `tile_shape`.
    #[must_use]
    pub fn is_tile_compatible(&self, tile_shape: &[u32]) -> bool {
        if tile_shape.len() > self.extents.len() {
            return false;
        }
        tile_shape
            .iter()
            .zip(&self.extents)
            .all(|(&tile_dim, &extent)| {
                tile_dim > 0 && (extent == 0 || extent >= u64::from(tile_dim))
            })
    }
}

/// Relationship between two logical regions proposed for fusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum RegionRelation {
    /// Exact 1:1 pointwise correspondence across the entire iteration domain.
    PointwiseMatch,
    /// Compatible iteration spaces that align on common tile boundaries.
    TileCompatible,
    /// Producer provides an intermediate tensor or matrix tile consumed by a reduction region.
    ReductionConsumer,
    /// Producer provides values accessed via a sliding window / stencil neighborhood.
    StencilWindow,
    /// Producer and consumer have independent iteration spaces with no direct dataflow.
    IndependentFrontier,
    /// Iteration spaces have conflicting or non-alignable dimensions that cannot be fused in one launch.
    Incompatible,
}

/// Planner that analyzes logical regions and their iteration spaces to produce fusion decisions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionFusionPlanner;

impl RegionFusionPlanner {
    /// Classify the iteration space relationship between a producer and consumer region.
    #[must_use]
    pub fn classify_relation(producer: &LogicalRegion, consumer: &LogicalRegion) -> RegionRelation {
        let prod_space = IterationSpace::from_region(producer);
        let cons_space = IterationSpace::from_region(consumer);

        if prod_space.is_congruent_with(&cons_space)
            && producer.index_map == consumer.index_map
            && producer.kind == LogicalRegionKind::Parallel
            && consumer.kind == LogicalRegionKind::Parallel
        {
            return RegionRelation::PointwiseMatch;
        }

        if consumer.kind.is_reduction_or_scan() {
            return RegionRelation::ReductionConsumer;
        }

        if consumer.window.is_some() || consumer.kind == LogicalRegionKind::Window {
            return RegionRelation::StencilWindow;
        }

        if prod_space.rank == cons_space.rank
            && prod_space
                .extents
                .iter()
                .zip(&cons_space.extents)
                .all(|(&p, &c)| p == c || (p > 0 && c > 0 && (p % c == 0 || c % p == 0)))
        {
            return RegionRelation::TileCompatible;
        }

        RegionRelation::Incompatible
    }
}
