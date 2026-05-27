//! Parity test: GPU IFDS exploded supergraph builder matches CPU oracle.

#![cfg(test)]

mod common;

use common::{live_dispatcher, CudaOptimizerDispatcher};
use vyre_primitives::graph::exploded::build_cpu_reference;
use vyre_self_substrate::exploded::{build_ifds_csr_via, reference_canonicalize_csr_within_rows};

fn assert_csr_equiv(cpu: &(Vec<u32>, Vec<u32>), gpu: &(Vec<u32>, Vec<u32>), label: &str) {
    let (cpu_row, cpu_col) = reference_canonicalize_csr_within_rows(&cpu.0, &cpu.1);
    let (gpu_row, gpu_col) = (gpu.0.clone(), gpu.1.clone());
    assert_eq!(cpu_row, gpu_row, "{label}: row_ptr divergence");
    assert_eq!(cpu_col, gpu_col, "{label}: col_idx divergence");
}

#[test]
fn cuda_ifds_intra_only_two_procs() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let intra = vec![(0, 0, 1), (1, 0, 1)];
    let cpu = build_cpu_reference(2, 2, 2, &intra, &[], &[], &[]);
    let gpu = build_ifds_csr_via(&dispatcher, 2, 2, 2, &intra, &[], &[], &[]).expect("dispatch");
    assert_csr_equiv(&cpu, &gpu, "intra-only two-procs");
}

#[test]
fn cuda_ifds_intra_with_kill_suppresses() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let intra = vec![(0, 0, 1)];
    let kill = vec![(0, 0, 1)];
    let cpu = build_cpu_reference(1, 2, 2, &intra, &[], &[], &kill);
    let gpu = build_ifds_csr_via(&dispatcher, 1, 2, 2, &intra, &[], &[], &kill).expect("dispatch");
    assert_csr_equiv(&cpu, &gpu, "kill suppresses fact");
}

#[test]
fn cuda_ifds_intra_with_gen_injects() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let intra = vec![(0, 0, 1)];
    let flow_gen = vec![(0, 0, 1)];
    let cpu = build_cpu_reference(1, 2, 2, &intra, &[], &flow_gen, &[]);
    let gpu =
        build_ifds_csr_via(&dispatcher, 1, 2, 2, &intra, &[], &flow_gen, &[]).expect("dispatch");
    assert_csr_equiv(&cpu, &gpu, "gen injects fact");
}

#[test]
fn cuda_ifds_inter_only_propagates_every_fact() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let inter = vec![(0, 0, 1, 1)];
    let cpu = build_cpu_reference(2, 2, 2, &[], &inter, &[], &[]);
    let gpu = build_ifds_csr_via(&dispatcher, 2, 2, 2, &[], &inter, &[], &[]).expect("dispatch");
    assert_csr_equiv(&cpu, &gpu, "inter-only every fact");
}

#[test]
fn cuda_ifds_combined_intra_inter_gen_kill() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let intra = vec![(0, 0, 1), (1, 0, 1)];
    let inter = vec![(0, 1, 1, 0)];
    let flow_gen = vec![(0, 0, 1)];
    let kill = vec![(1, 0, 0)];
    let cpu = build_cpu_reference(2, 2, 2, &intra, &inter, &flow_gen, &kill);
    let gpu = build_ifds_csr_via(&dispatcher, 2, 2, 2, &intra, &inter, &flow_gen, &kill)
        .expect("dispatch");
    assert_csr_equiv(&cpu, &gpu, "combined intra/inter/gen/kill");
}

#[test]
fn cuda_ifds_empty_dimensions_returns_singleton_row_ptr() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let gpu = build_ifds_csr_via(&dispatcher, 0, 0, 0, &[], &[], &[], &[]).expect("dispatch");
    assert_eq!(gpu.0, vec![0u32]);
    assert!(gpu.1.is_empty());
}

#[test]
fn cuda_ifds_larger_chain_three_procs_four_blocks_three_facts() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    // Chain CFG inside each proc: 0->1->2->3.
    let mut intra = Vec::new();
    for p in 0..3 {
        for b in 0..3 {
            intra.push((p, b, b + 1));
        }
    }
    // Inter: each proc calls the next on its last block.
    let inter = vec![(0, 3, 1, 0), (1, 3, 2, 0)];
    // GEN at (0, 0) injects fact 1 and fact 2 into the chain.
    let flow_gen = vec![(0, 0, 1), (0, 0, 2)];
    // KILL fact 1 at (1, 1).
    let kill = vec![(1, 1, 1)];
    let cpu = build_cpu_reference(3, 4, 3, &intra, &inter, &flow_gen, &kill);
    let gpu = build_ifds_csr_via(&dispatcher, 3, 4, 3, &intra, &inter, &flow_gen, &kill)
        .expect("dispatch");
    assert_csr_equiv(&cpu, &gpu, "three-proc chain combined");
}
