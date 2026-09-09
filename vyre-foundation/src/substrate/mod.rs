//! Compiler data substrate, arenas, interners, and query engine (Backlog Row 89).
//!
//! Provides immutable hash-consed arenas, stable typed IDs, canonical interners
//! for strings, types, constants, and layouts, stage-specific read-only views for
//! all five compiler levels, versioned caching, and a single deterministic query engine.

mod arenas;
mod cache;
mod ids;
mod pass_contracts;
mod query;
mod views;

pub use arenas::{
    CanonicalConst, CanonicalLayout, CanonicalType, ConstInterner, ExprArena, LayoutInterner,
    NodeArena, RegionArena, StringInterner, SubstrateArena, TypeInterner,
};
pub use cache::{
    StaleCacheError, VersionedCacheEntry, VersionedCacheKey, SUBSTRATE_CACHE_SCHEMA_VERSION,
};
pub use ids::{
    ArtifactId, ExprId, InternedConstId, InternedLayoutId, InternedStringId, InternedTypeId,
    NodeId, PhysicalKernelId, RegionId, Revision, ScheduleNodeId,
};
pub use pass_contracts::{derive_registered_pass_descriptors, PassDescriptor, TransformOutcome};
pub use query::{
    CancellationToken, MemoryAccounting, Query, QueryEngine, QueryError, QueryKey, QueryOutput,
};
pub use views::{
    enforce_level_boundary, CompilerLevelStage, LevelAccessError, LogicalRegionView,
    PhysicalKernelView, SelectedScheduleView, TargetPayloadView, WholeProgramGraphView,
};
