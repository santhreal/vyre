//! Adversarial and boundary test suite for `graph` primitives.
//!
//! Coverage: csr_forward_traverse, csr_backward_traverse, toposort,
//! scc_decompose, path_reconstruct  -  hostile boundaries, empty graphs,
//! edge-kind diversity (M8), malformed CSR, cross-word bitsets.
#![cfg(feature = "graph")]

use vyre_libs::bitset::bitset_words;
use vyre_reference::composition_witness::csr_forward_traverse_witness as bwd_cpu_ref;
use vyre_reference::composition_witness::csr_forward_traverse_witness as fwd_cpu_ref;
use vyre_libs::graph::csr_frontier_queue::{
    frontier_to_queue_cpu, frontier_word_block_prefix_to_queue_parallel,
    frontier_word_counts_scan_pass_a, frontier_words_to_queue_parallel,
};
fn path_cpu_ref(parents: &[u32], target: u32) -> Vec<u32> { let mut path = Vec::new(); let mut cur = target; while cur != u32::MAX && (cur as usize) < parents.len() { path.push(cur); cur = parents[cur as usize]; } path.reverse(); path }
fn scc_cpu_ref(node_count: u32, fwd: &[u32], bwd: &[u32], mask: &[u32], _pivot: u32) -> Vec<u32> { let mut out = mask.to_vec(); for i in 0..node_count as usize { if (fwd.get(i/32).copied().unwrap_or(0) & (1<<(i%32))) != 0 && (bwd.get(i/32).copied().unwrap_or(0) & (1<<(i%32))) != 0 { if i/32 < out.len() { out[i/32] |= 1<<(i%32); } } } out }
use vyre_libs::graph::toposort::{toposort, ToposortError};
use vyre_reference::value::Value;

mod backward_traverse_contracts;
mod forward_traverse_contracts;
mod frontier_queue_contracts;
mod path_reconstruct_contracts;
mod scc_contracts;
mod toposort_contracts;
