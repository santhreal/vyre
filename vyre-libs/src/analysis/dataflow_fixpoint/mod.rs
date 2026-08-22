//! Region-graph dataflow fixpoint via semiring_gemm.
//!
//! Treats vyre's Region tree adjacency as a sparse boolean matrix
//! and computes reachability / liveness / dominance / constant-prop
//! via `semiring_gemm` iterations under different semirings:
//!
//! | Analysis | Semiring | Combine | Accumulate |
//! |---|---|---|---|
//! | Reachability | BoolOr | AND | OR |
//! | Liveness | BoolOr (reverse direction) | AND | OR |
//! | Reaching defs | Lineage | OR (zero-absorbing) | OR |
//! | Constant prop | Lineage | OR | OR |
//! | Min-cost path | MinPlus | + (sat) | min |
//!
//! Same primitive, same Program, four different IR analyses.
//! Demonstrates the recursion thesis directly.
//!
//! `dense_matrix` owns the shared adjacency shape checks,
//! `fixpoint_comparison` the three-engine reachability comparison,
//! `delta_maintenance` the incremental relation update, `scc_decomposition` the
//! strongly-connected-component driver, and `gpu_dispatch` every
//! dispatcher-backed wrapper. The host closures are the foundation substrate's
//! own, re-exported here so this module's documented paths keep resolving;
//!
//! The semiring these analyses select is `vyre_spec::Semiring`, published by
//! `math::semiring_gemm` because that is the composition it parameterizes. This
//! module names it without republishing it, so the type has one path here.

use vyre_foundation::pass_substrate::semiring_closure::Semiring;

#[cfg(test)]
mod delta_maintenance;
#[cfg(test)]
mod dense_matrix;
#[cfg(test)]
mod fixpoint_comparison;
mod gpu_dispatch;
#[cfg(test)]
mod reference_gemm;
mod scc_decomposition;

pub use gpu_dispatch::{
    forward_backward_bitsets_for_pivot_via, lineage_closure_via, reachability_closure_via,
    reachability_closure_via_into, reachability_closure_via_with_scratch_into,
    scc_components_via_substrate_via, scc_components_via_substrate_with_scratch_into,
    scc_components_via_substrate_with_scratch_via, semiring_gemm_via, semiring_gemm_via_bool_or,
    semiring_gemm_via_into, semiring_gemm_via_lineage, semiring_gemm_via_min_plus,
    semiring_gemm_via_with_scratch_into, shortest_path_closure_via,
};

/// Caller-owned dispatch scratch for repeated semiring-GEMM GPU calls.
#[derive(Debug, Default)]
pub struct SemiringGemmGpuScratch {
    pub(super) inputs: Vec<Vec<u8>>,
}

/// Caller-owned scratch for GPU-backed SCC composition over reachability closure.
#[derive(Debug, Default)]
pub struct SccComponentsGpuScratch {
    pub(super) fwd_closure: Vec<u32>,
    pub(super) bwd_closure: Vec<u32>,
    pub(super) fwd_next: Vec<u32>,
    pub(super) bwd_next: Vec<u32>,
    pub(super) transpose: Vec<u32>,
    pub(super) forward: Vec<u32>,
    pub(super) backward: Vec<u32>,
    pub(super) semiring: SemiringGemmGpuScratch,
    pub(super) inputs: Vec<Vec<u8>>,
}

/// Telemetry emitted by one static-analysis fixpoint formulation.
#[cfg(test)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct FixpointEngineTelemetry {
    /// Stable engine id.
    pub(crate) engine_id: &'static str,
    /// Fixpoint iterations or frontier layers evaluated.
    pub(crate) iterations: u32,
    /// Estimated host bytes touched while producing the closure.
    pub(crate) bytes_touched: u64,
    /// Average active-frontier density in basis points.
    pub(crate) frontier_density_bps: u32,
    /// Measured active CPU time for the comparison implementation.
    pub(crate) active_time_ns: u128,
}

/// Reachability output plus telemetry for one formulation.
#[cfg(test)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct FixpointEngineReport {
    /// Engine telemetry.
    pub(crate) telemetry: FixpointEngineTelemetry,
    /// Dense `n*n` boolean reachability matrix, row-major.
    pub(crate) reachability: Vec<u32>,
}

/// Side-by-side reachability comparison for static-analysis fixpoints.
#[cfg(test)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct StaticAnalysisFixpointComparison {
    /// Number of graph nodes.
    pub(crate) node_count: u32,
    /// Maximum iterations supplied to each formulation.
    pub(crate) max_iterations: u32,
    /// Vyre dense semiring-GEMM closure report.
    pub(crate) vyre_semiring: FixpointEngineReport,
    /// external-engine CSR frontier closure report.
    pub(crate) external_frontier: FixpointEngineReport,
    /// GraphBLAS-style sparse boolean frontier closure report.
    pub(crate) graphblas_sparse: FixpointEngineReport,
    /// Whether all three closures are byte-identical.
    pub(crate) exact_reachability_sets: bool,
}

/// One directed relation tuple insertion or deletion.
#[cfg(test)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct DeltaRelationChange {
    /// Source node.
    pub(crate) source: u32,
    /// Target node.
    pub(crate) target: u32,
}

/// Insertion/deletion batch for a boolean dataflow relation.
#[cfg(test)]
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub(crate) struct DeltaRelationBatch {
    /// Tuples inserted into the relation.
    pub(crate) insertions: Vec<DeltaRelationChange>,
    /// Tuples deleted from the relation.
    pub(crate) deletions: Vec<DeltaRelationChange>,
}

/// Delta-maintained reachability evidence compared against full recompute.
#[cfg(test)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct DeltaDataflowEvidence {
    /// Number of graph nodes.
    pub(crate) node_count: u32,
    /// Inserted tuple count.
    pub(crate) inserted_tuple_count: u32,
    /// Deleted tuple count.
    pub(crate) deleted_tuple_count: u32,
    /// Reachability tuples that changed after applying the batch.
    pub(crate) changed_tuple_count: u32,
    /// Tuples recomputed by the delta path.
    pub(crate) recomputed_tuple_count: u32,
    /// Delta fixpoint passes or full-recompute iterations.
    pub(crate) iterations: u32,
    /// Measured active time for the delta path.
    pub(crate) elapsed_active_time_ns: u128,
    /// Whether delta-maintained output matched full recompute.
    pub(crate) exact_result_parity: bool,
}

/// Delta-maintained closure plus full-recompute comparator output.
#[cfg(test)]
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct DeltaDataflowReport {
    /// Evidence row.
    pub(crate) evidence: DeltaDataflowEvidence,
    /// Closure produced by the delta-maintained relation path.
    pub(crate) delta_closure: Vec<u32>,
    /// Closure produced by full recompute after applying the batch.
    pub(crate) full_recompute_closure: Vec<u32>,
}
