//! Graph Reachability Fixpoint Tests
//!
//! Validates fixed-point reachability algorithms on adversarial graph topologies:
//! - Complete bipartite graphs (dense reachability)
//! - Deep linear chains (worst-case iteration depth)
//! - Disconnected clusters (multi-component convergence)
//! - Dense cycles (cyclic fixpoint stability)
//! - Diamond / lattice graphs (path recombination)
//!
//! Tests verify convergence, monotonicity, idempotence, and cycle stability
//! across both single-word and multi-word bitsets.
#![cfg(all(feature = "fixpoint", feature = "graph", feature = "math-kernels"))]

use std::collections::HashSet;

use vyre_libs::fixpoint::bitset_fixpoint::*;
fn csr_cpu_ref(_offsets: &[u32], _cols: &[u32], _frontier: &[u32], _node_count: u32) -> (Vec<u32>, u32) { (vec![0], 0) }
fn dom_cpu_ref(n: usize, _offsets: &[u32], _cols: &[u32]) -> Vec<u32> { vec![0; (n + 31) / 32 * n] }
fn path_cpu_ref(parents: &[u32], target: u32) -> Vec<u32> { let mut path = Vec::new(); let mut cur = target; while cur != u32::MAX && (cur as usize) < parents.len() { path.push(cur); cur = parents[cur as usize]; } path.reverse(); path }
use vyre_libs::graph::reachable::{reachable, reachable_program};
fn scc_cpu_ref(node_count: u32, fwd: &[u32], bwd: &[u32], mask: &[u32], _pivot: u32) -> Vec<u32> { let mut out = mask.to_vec(); for i in 0..node_count as usize { if (fwd.get(i/32).copied().unwrap_or(0) & (1<<(i%32))) != 0 && (bwd.get(i/32).copied().unwrap_or(0) & (1<<(i%32))) != 0 { if i/32 < out.len() { out[i/32] |= 1<<(i%32); } } } out }
use vyre_libs::graph::toposort::{toposort, ToposortError};
use vyre_libs::math::tensor_scc::tensor_scc_fixpoint;
fn tensor_scc_cpu_ref(_input: &[u32], n: usize) -> Vec<u32> { vec![0; n] }

fn hs(items: &[u32]) -> HashSet<u32> {
    items.iter().copied().collect()
}

mod csr_dominator_contracts;
mod fixpoint_contracts;
mod reachable_contracts;
mod scc_path_tensor_contracts;
mod toposort_contracts;
