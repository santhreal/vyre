//! Topological-data-analysis primitives.
//!
//! Persistent homology + simplicial-complex operations. Composes
//! with `vyre-primitives::math` and `vyre-primitives::graph`.

/// Vietoris-Rips filtration boundary-matrix construction.
pub mod vietoris_rips;

/// Simplicial neural network message-passing step. Triangle-
/// level boundary-operator message aggregation.
pub mod simplicial;
