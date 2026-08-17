//! Adversarial contract tests for graph reachability, fixpoint, and
//! traversal invariants.
//!
//! Coverage: reachable, toposort, scc_decompose, path_reconstruct,
//! tensor_scc, csr_forward_or_changed, dominator_frontier, and
//! fixpoint convergence semantics. GPU acquisition: none  -  every
//! assertion uses CPU reference oracles.
//!
//! Implementation lives in two `include!`-d chunks under `contract_cases/`.
#![cfg(feature = "graph")]
#![cfg(feature = "fixpoint")]
#![cfg(feature = "math")]

use std::collections::HashSet;

use vyre_libs::fixpoint::bitset_fixpoint::*;
use vyre_libs::graph::csr_forward_or_changed::cpu_ref as csr_cpu_ref;
use vyre_libs::graph::dominator_frontier::cpu_ref as dom_cpu_ref;
use vyre_libs::graph::path_reconstruct::cpu_ref as path_cpu_ref;
use vyre_libs::graph::reachable::{reachable, reachable_program};
use vyre_libs::graph::scc_decompose::cpu_ref as scc_cpu_ref;
use vyre_libs::graph::toposort::{toposort, ToposortError};
use vyre_libs::math::tensor_scc::{cpu_ref as tensor_scc_cpu_ref, tensor_scc_fixpoint};

// ---------------------------------------------------------------------------
// Reachable  -  directed reachability
// ---------------------------------------------------------------------------

fn hs(items: &[u32]) -> HashSet<u32> {
    items.iter().copied().collect()
}

mod csr_dominator_contracts;
mod fixpoint_contracts;
mod reachable_contracts;
mod scc_path_tensor_contracts;
mod toposort_contracts;
