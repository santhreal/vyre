//! The CPU pass math the optimizer reasons with: pass-ordering validity,
//! region-graph closures, fusion ordering, and the composition checks a rewrite
//! has to satisfy. Every module here answers a question about passes or region
//! adjacency and knows nothing about a backend.

/// Optimizer pass-ordering validity via causal adjustment-set analysis.
pub mod adjustment_set_pass_dependency;
/// Functorial composition helpers for optimizer pass rows.
pub mod functorial_pass_composition;
/// Multigrid-style smoothing step used by matroid fusion relaxations.
pub mod multigrid_matroid_solver;
/// Polyhedral and affine fusion queries over pass dependency graphs.
pub mod polyhedral_fusion;
/// Region-graph reachability, lineage and shortest-path closures over a
/// semiring, plus the lattice joins a fixpoint iteration needs.
pub mod semiring_closure;
/// String-diagram composition checks for IR rewrite arrows.
pub mod string_diagram_ir_rewrite;
/// Tensor-network contraction ordering for fusion planning.
pub mod tensor_network_fusion_order;
