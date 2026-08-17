//! Generated contract matrix for CSR topological-sort witnesses.
//!
//! Exercises allocating and caller-owned storage paths over thousands of DAG
//! shapes and validates every result against the production CSR invariant.

#![forbid(unsafe_code)]

use vyre_libs::graph::toposort::validate_toposort_csr_order;
use vyre_reference::composition_witness::{
    toposort_csr_with_scratch_into_witness, toposort_csr_witness,
};

#[derive(Debug, Default, Clone)]
struct ToposortCsrScratch {
    indegree: Vec<u32>,
    queue: Vec<u32>,
}

impl ToposortCsrScratch {
    fn new() -> Self {
        Self::default()
    }
}

fn toposort_csr(node_count: u32, offsets: &[u32], targets: &[u32]) -> Result<Vec<u32>, String> {
    toposort_csr_witness(node_count, offsets, targets)
}

fn toposort_csr_into_with_scratch(
    node_count: u32,
    offsets: &[u32],
    targets: &[u32],
    order: &mut Vec<u32>,
    scratch: &mut ToposortCsrScratch,
) -> Result<(), String> {
    toposort_csr_with_scratch_into_witness(
        node_count,
        offsets,
        targets,
        order,
        &mut scratch.indegree,
        &mut scratch.queue,
    )
}

#[test]
fn toposort_csr_allocating_and_scratch_paths_satisfy_generated_dags() {
    let mut order = Vec::new();
    let mut scratch = ToposortCsrScratch::new();

    for case in 0..8192usize {
        let (node_count, offsets, targets) = generated_dag_csr(case as u64 ^ 0x7E57_1D00);
        let actual = toposort_csr(node_count, &offsets, &targets)
            .expect("Fix: generated lower-triangular CSR graph must be a valid DAG.");
        validate_toposort_csr_order(node_count, &offsets, &targets, &actual)
            .expect("Fix: allocating topological order must satisfy the CSR contract.");

        toposort_csr_into_with_scratch(node_count, &offsets, &targets, &mut order, &mut scratch)
            .expect("Fix: scratch-backed oracle must accept every generated valid DAG.");
        assert_eq!(
            order, actual,
            "Fix: allocating and scratch paths must agree for generated case {case}."
        );
        validate_toposort_csr_order(node_count, &offsets, &targets, &order)
            .expect("Fix: scratch-backed topological order must satisfy the CSR contract.");
    }
}

fn generated_dag_csr(seed: u64) -> (u32, Vec<u32>, Vec<u32>) {
    let mut rng = seed;
    let node_count = 1 + (rng as u32 % 96);
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut offsets = Vec::with_capacity(node_count as usize + 1);
    let mut targets = Vec::new();
    offsets.push(0);
    for src in 0..node_count {
        let max_dst = node_count.saturating_sub(src + 1);
        let degree = if max_dst == 0 {
            0
        } else {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng as u32 % (max_dst.min(5) + 1)
        };
        for _ in 0..degree {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let dst = src + 1 + (rng as u32 % max_dst);
            targets.push(dst);
        }
        offsets.push(targets.len() as u32);
    }
    (node_count, offsets, targets)
}
