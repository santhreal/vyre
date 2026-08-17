//! Failure-oriented adversarial integration tests for graph primitives.
//!
//! Coverage: csr_forward_traverse, csr_backward_traverse, toposort,
//! scc_decompose, path_reconstruct  -  hostile boundaries, empty graphs,
//! edge-kind diversity (M8), malformed CSR, cross-word bitsets.
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use vyre_libs::bitset::bitset_words;
use vyre_libs::graph::csr_backward_traverse::cpu_ref as bwd_cpu_ref;
use vyre_libs::graph::csr_forward_traverse::cpu_ref as fwd_cpu_ref;
use vyre_libs::graph::csr_frontier_queue::{
    frontier_to_queue_cpu, frontier_word_block_prefix_to_queue_parallel,
    frontier_word_counts_scan_pass_a, frontier_words_to_queue_parallel,
};
use vyre_libs::graph::path_reconstruct::cpu_ref as path_cpu_ref;
use vyre_libs::graph::scc_decompose::cpu_ref as scc_cpu_ref;
use vyre_libs::graph::toposort::{toposort, ToposortError};
use vyre_reference::value::Value;

mod backward_traverse_contracts;
mod forward_traverse_contracts;
mod frontier_queue_contracts;
mod path_reconstruct_contracts;
mod scc_contracts;
mod toposort_contracts;
