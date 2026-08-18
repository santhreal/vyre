//! Failure-oriented adversarial integration tests for graph primitives.
//!
//! Coverage: csr_forward_traverse, csr_backward_traverse, toposort,
//! scc_decompose, path_reconstruct  -  hostile boundaries, empty graphs,
//! edge-kind diversity (M8), malformed CSR, cross-word bitsets.
#![cfg(feature = "graph")]

#[path = "../wire_words/mod.rs"]
mod wire_words;
use wire_words::toposort;

use vyre_libs::bitset::bitset_words;
use vyre_libs::graph::csr_frontier_queue::{
    frontier_word_block_prefix_to_queue_parallel, frontier_word_counts_scan_pass_a,
    frontier_words_to_queue_parallel,
};
use vyre_libs::graph::toposort::ToposortError;
use vyre_reference::composition_witness::{
    csr_backward_traverse_witness as bwd_cpu_ref, csr_forward_traverse_witness as fwd_cpu_ref,
};
use vyre_reference::composition_witness::{
    frontier_to_queue_witness as frontier_to_queue_cpu, path_reconstruct_witness,
    scc_decompose_witness as scc_cpu_ref,
};

use vyre_reference::value::Value;

fn path_cpu_ref(predecessor: &[u32], target: u32, max_depth: u32, output: &mut Vec<u32>) -> u32 {
    let (path, length) = path_reconstruct_witness(predecessor, target, max_depth);
    output.clear();
    output.extend_from_slice(&path);
    length
}

mod backward_traverse_contracts;
mod forward_traverse_contracts;
mod frontier_queue_contracts;
mod path_reconstruct_contracts;
mod scc_contracts;
mod toposort_contracts;
