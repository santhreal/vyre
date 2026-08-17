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

use vyre_libs::graph::reachable::{reachable_program, UnknownNode};
use vyre_libs::graph::toposort::ToposortError;
use vyre_libs::math::tensor_scc::tensor_scc_fixpoint;
use vyre_reference::composition_witness::{
    bitset_difference_flag_witness as reference_eval,
    bitset_warm_start_witness as reference_eval_warm_start,
    csr_forward_or_changed_witness as csr_cpu_ref, dominator_frontier_witness as dom_cpu_ref,
    path_reconstruct_witness,
};
use vyre_reference::composition_witness::{
    reachable_witness, scc_decompose_witness as scc_cpu_ref,
    tensor_scc_witness as tensor_scc_cpu_ref, toposort_witness,
};

fn toposort(node_count: u32, edges: &[(u32, u32)]) -> Result<Vec<u32>, ToposortError> {
    for (edge, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(ToposortError::UnknownNode { edge, node: from });
        }
        if to >= node_count {
            return Err(ToposortError::UnknownNode { edge, node: to });
        }
    }
    toposort_witness(node_count, edges).map_err(|err| {
        if let Some(rest) = err.strip_prefix("Cycle detected involving node ") {
            if let Ok(node) = rest.parse::<u32>() {
                return ToposortError::Cycle { node };
            }
        }
        ToposortError::InconsistentState { message: err }
    })
}

fn reachable(
    node_count: u32,
    edges: &[(u32, u32)],
    sources: &[u32],
) -> Result<HashSet<u32>, UnknownNode> {
    for (index, &(from, to)) in edges.iter().enumerate() {
        if from >= node_count {
            return Err(UnknownNode {
                index,
                node: from,
                node_count,
            });
        }
        if to >= node_count {
            return Err(UnknownNode {
                index,
                node: to,
                node_count,
            });
        }
    }
    reachable_witness(node_count, edges, sources).map_err(|_| UnknownNode {
        index: 0,
        node: 0,
        node_count,
    })
}

// ---------------------------------------------------------------------------
// Reachable  -  directed reachability
// ---------------------------------------------------------------------------

fn hs(items: &[u32]) -> HashSet<u32> {
    items.iter().copied().collect()
}

fn path_cpu_ref(predecessor: &[u32], target: u32, max_depth: u32, output: &mut Vec<u32>) -> u32 {
    let (path, length) = path_reconstruct_witness(predecessor, target, max_depth);
    output.clear();
    output.extend_from_slice(&path);
    length
}

mod csr_dominator_contracts;
mod fixpoint_contracts;
mod reachable_contracts;
mod scc_path_tensor_contracts;
mod toposort_contracts;
