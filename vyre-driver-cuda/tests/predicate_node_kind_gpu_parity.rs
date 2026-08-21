//! Parity test: vyre-primitives predicate node_kind_eq + literal_of
//! match CPU oracles.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;

use harness::{cuda_u32_bitset_output, with_live_backend};
use vyre_libs::predicate::literal_of::literal_of;
use vyre_libs::predicate::node_kind;
use vyre_libs::predicate::node_kind_eq::node_kind_eq;
use vyre_reference::composition_witness::node_kind_eq_witness;

fn run_node_kind_eq(nodes: &[u32], kind: u32) -> Vec<u32> {
    let n = nodes.len() as u32;
    let program = node_kind_eq("nodes", "nodeset", n, kind);
    with_live_backend("node kind predicate", |backend| {
        cuda_u32_bitset_output(backend, &program, n, nodes, "node_kind_eq")
    })
}

#[test]
fn cuda_node_kind_eq_basic() {
    let nodes = vec![1u32, 2, 1, 3, 1, 4];
    let kind = 1u32;
    let cpu = node_kind_eq_witness(&nodes, kind);
    let gpu = run_node_kind_eq(&nodes, kind);
    assert_eq!(gpu, cpu);
    // Bits 0, 2, 4 should be set.
    assert_eq!(gpu, vec![0b010101]);
}

#[test]
fn cuda_node_kind_eq_no_matches() {
    let nodes = vec![1u32, 2, 3, 4];
    let kind = 99u32;
    let cpu = node_kind_eq_witness(&nodes, kind);
    let gpu = run_node_kind_eq(&nodes, kind);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0u32]);
}

#[test]
fn cuda_node_kind_eq_all_match() {
    let nodes = vec![5u32; 8];
    let cpu = node_kind_eq_witness(&nodes, 5);
    let gpu = run_node_kind_eq(&nodes, 5);
    assert_eq!(gpu, cpu);
    assert_eq!(gpu, vec![0b1111_1111]);
}

fn run_literal_of(nodes: &[u32]) -> Vec<u32> {
    let n = nodes.len() as u32;
    let program = literal_of("nodes", "nodeset", n);
    with_live_backend("literal predicate", |backend| {
        cuda_u32_bitset_output(backend, &program, n, nodes, "literal_of")
    })
}

#[test]
fn cuda_literal_of_matches_cpu() {
    let nodes = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let cpu = node_kind_eq_witness(&nodes, node_kind::LITERAL);
    let gpu = run_literal_of(&nodes);
    assert_eq!(gpu, cpu);
}
