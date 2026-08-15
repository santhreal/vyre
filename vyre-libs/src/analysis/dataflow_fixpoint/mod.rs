//! Region-graph dataflow fixpoint via #1 semiring_gemm (#26 substrate).
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
//! Same primitive (#1), same Program, four different IR analyses.
//! Demonstrates the recursion thesis directly.
//!
//! `dense_matrix` owns the shared adjacency shape checks,
//! `fixpoint_comparison` the three-engine reachability comparison,
//! `delta_maintenance` the incremental relation update, `scc_decomposition` the
//! strongly-connected-component driver, and `gpu_dispatch` every
//! dispatcher-backed wrapper. The host closures are the foundation substrate's
//! own, re-exported here so this module's documented paths keep resolving;
//! `reference_gemm` adds the call counter and nothing else. The public types
//! stay declared here so their rendered documentation paths do not move.

pub use vyre_foundation::pass_substrate::semiring_closure::Semiring;

mod delta_maintenance;
mod dense_matrix;
mod fixpoint_comparison;
mod gpu_dispatch;
mod reference_gemm;
mod scc_decomposition;

pub use delta_maintenance::compare_delta_maintained_reachability;
pub use fixpoint_comparison::compare_static_analysis_reachability_fixpoints;
pub use gpu_dispatch::{
    forward_backward_bitsets_for_pivot_via, lineage_closure_via, reachability_closure_via,
    reachability_closure_via_into, reachability_closure_via_with_scratch_into,
    scc_components_via_substrate_via, scc_components_via_substrate_with_scratch_into,
    scc_components_via_substrate_with_scratch_via, semiring_gemm_via, semiring_gemm_via_bool_or,
    semiring_gemm_via_into, semiring_gemm_via_lineage, semiring_gemm_via_min_plus,
    semiring_gemm_via_with_scratch_into, shortest_path_closure_via,
};
pub use reference_gemm::{reference_semiring_gemm, reference_semiring_gemm_into};
#[cfg(any(test, feature = "cpu-parity"))]
pub use scc_decomposition::{
    forward_backward_bitsets_for_pivot, forward_backward_bitsets_for_pivot_into,
    reference_scc_components_via_substrate_into, scc_components_via_substrate,
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
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FixpointEngineTelemetry {
    /// Stable engine id.
    pub engine_id: &'static str,
    /// Fixpoint iterations or frontier layers evaluated.
    pub iterations: u32,
    /// Estimated host bytes touched while producing the closure.
    pub bytes_touched: u64,
    /// Average active-frontier density in basis points.
    pub frontier_density_bps: u32,
    /// Measured active CPU time for the comparison implementation.
    pub active_time_ns: u128,
}

/// Reachability output plus telemetry for one formulation.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FixpointEngineReport {
    /// Engine telemetry.
    pub telemetry: FixpointEngineTelemetry,
    /// Dense `n*n` boolean reachability matrix, row-major.
    pub reachability: Vec<u32>,
}

/// Side-by-side reachability comparison for static-analysis fixpoints.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StaticAnalysisFixpointComparison {
    /// Number of graph nodes.
    pub node_count: u32,
    /// Maximum iterations supplied to each formulation.
    pub max_iterations: u32,
    /// Vyre dense semiring-GEMM closure report.
    pub vyre_semiring: FixpointEngineReport,
    /// external-engine CSR frontier closure report.
    pub external_frontier: FixpointEngineReport,
    /// GraphBLAS-style sparse boolean frontier closure report.
    pub graphblas_sparse: FixpointEngineReport,
    /// Whether all three closures are byte-identical.
    pub exact_reachability_sets: bool,
}

/// One directed relation tuple insertion or deletion.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct DeltaRelationChange {
    /// Source node.
    pub source: u32,
    /// Target node.
    pub target: u32,
}

/// Insertion/deletion batch for a boolean dataflow relation.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct DeltaRelationBatch {
    /// Tuples inserted into the relation.
    pub insertions: Vec<DeltaRelationChange>,
    /// Tuples deleted from the relation.
    pub deletions: Vec<DeltaRelationChange>,
}

/// Delta-maintained reachability evidence compared against full recompute.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeltaDataflowEvidence {
    /// Number of graph nodes.
    pub node_count: u32,
    /// Inserted tuple count.
    pub inserted_tuple_count: u32,
    /// Deleted tuple count.
    pub deleted_tuple_count: u32,
    /// Reachability tuples that changed after applying the batch.
    pub changed_tuple_count: u32,
    /// Tuples recomputed by the delta path.
    pub recomputed_tuple_count: u32,
    /// Delta fixpoint passes or full-recompute iterations.
    pub iterations: u32,
    /// Measured active time for the delta path.
    pub elapsed_active_time_ns: u128,
    /// Whether delta-maintained output matched full recompute.
    pub exact_result_parity: bool,
}

/// Delta-maintained closure plus full-recompute comparator output.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeltaDataflowReport {
    /// Evidence row.
    pub evidence: DeltaDataflowEvidence,
    /// Closure produced by the delta-maintained relation path.
    pub delta_closure: Vec<u32>,
    /// Closure produced by full recompute after applying the batch.
    pub full_recompute_closure: Vec<u32>,
}

/// Reusable buffers for SCC/dataflow closure queries.
#[derive(Debug, Default)]
#[cfg(any(test, feature = "cpu-parity"))]
pub struct DataflowFixpointScratch {
    pub(super) fwd_closure: Vec<u32>,
    pub(super) bwd_closure: Vec<u32>,
    pub(super) transpose: Vec<u32>,
    pub(super) forward: Vec<u32>,
    pub(super) backward: Vec<u32>,
    pub(super) next_components: Vec<u32>,
}
