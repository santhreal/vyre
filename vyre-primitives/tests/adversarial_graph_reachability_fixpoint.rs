//! Adversarial contract tests for graph reachability, fixpoint, and
//! traversal invariants.
//!
//! Coverage: reachable, toposort, scc_decompose, path_reconstruct,
//! tensor_scc, csr_forward_or_changed, dominator_frontier, and
//! fixpoint convergence semantics. GPU acquisition: none  -  every
//! assertion uses CPU reference oracles.
//!
//! Implementation lives in two `include!`-d chunks under `__split/`.
#![cfg(feature = "graph")]
#![cfg(feature = "fixpoint")]
#![cfg(feature = "math")]

include!("__split/adversarial_graph_reachability_fixpoint_chunk1.rs");
include!("__split/adversarial_graph_reachability_fixpoint_chunk2.rs");
