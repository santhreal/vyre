//! Parity test: GPU IFDS exploded supergraph builder matches CPU oracle.

#![cfg(all(test, feature = "device-tests"))]

mod harness;

use harness::with_cuda_optimizer_dispatcher;
use vyre_libs::graph::dispatch::exploded::build_ifds_csr_via;
use vyre_reference::composition_witness::exploded_ifds_csr_witness;

fn assert_csr_equiv(cpu: &(Vec<u32>, Vec<u32>), gpu: &(Vec<u32>, Vec<u32>), label: &str) {
    let mut cpu_col = cpu.1.clone();
    for window in cpu.0.windows(2) {
        let start = window[0] as usize;
        let end = window[1] as usize;
        if start <= end && end <= cpu_col.len() {
            cpu_col[start..end].sort_unstable();
        }
    }
    let (cpu_row, gpu_row, gpu_col) = (&cpu.0, &gpu.0, &gpu.1);
    assert_eq!(cpu_row, gpu_row, "{label}: row_ptr divergence");
    assert_eq!(&cpu_col, gpu_col, "{label}: col_idx divergence");
}

fn assert_ifds_matches_reference(
    label: &str,
    procs: u32,
    blocks: u32,
    facts: u32,
    intra: &[(u32, u32, u32)],
    inter: &[(u32, u32, u32, u32)],
    flow_gen: &[(u32, u32, u32)],
    kill: &[(u32, u32, u32)],
) {
    let cpu = exploded_ifds_csr_witness(procs, blocks, facts, intra, inter, flow_gen, kill);
    with_cuda_optimizer_dispatcher(label, |dispatcher, policy| {
        let gpu = build_ifds_csr_via(
            dispatcher, policy, procs, blocks, facts, intra, inter, flow_gen, kill,
        )
        .expect("dispatch");
        assert_csr_equiv(&cpu, &gpu, label);
    });
}

#[test]
fn cuda_ifds_intra_only_two_procs() {
    let intra = vec![(0, 0, 1), (1, 0, 1)];
    assert_ifds_matches_reference("intra-only two-procs", 2, 2, 2, &intra, &[], &[], &[]);
}

#[test]
fn cuda_ifds_intra_with_kill_suppresses() {
    let intra = vec![(0, 0, 1)];
    let kill = vec![(0, 0, 1)];
    assert_ifds_matches_reference("kill suppresses fact", 1, 2, 2, &intra, &[], &[], &kill);
}

#[test]
fn cuda_ifds_intra_with_gen_injects() {
    let intra = vec![(0, 0, 1)];
    let flow_gen = vec![(0, 0, 1)];
    assert_ifds_matches_reference("gen injects fact", 1, 2, 2, &intra, &[], &flow_gen, &[]);
}

#[test]
fn cuda_ifds_inter_only_propagates_every_fact() {
    let inter = vec![(0, 0, 1, 1)];
    assert_ifds_matches_reference("inter-only every fact", 2, 2, 2, &[], &inter, &[], &[]);
}

#[test]
fn cuda_ifds_combined_intra_inter_gen_kill() {
    let intra = vec![(0, 0, 1), (1, 0, 1)];
    let inter = vec![(0, 1, 1, 0)];
    let flow_gen = vec![(0, 0, 1)];
    let kill = vec![(1, 0, 0)];
    assert_ifds_matches_reference(
        "combined intra/inter/gen/kill",
        2,
        2,
        2,
        &intra,
        &inter,
        &flow_gen,
        &kill,
    );
}

#[test]
fn cuda_ifds_empty_dimensions_returns_singleton_row_ptr() {
    with_cuda_optimizer_dispatcher("empty IFDS dimensions", |dispatcher, policy| {
        let gpu =
            build_ifds_csr_via(dispatcher, policy, 0, 0, 0, &[], &[], &[], &[]).expect("dispatch");
        assert_eq!(gpu.0, vec![0u32]);
        assert!(gpu.1.is_empty());
    });
}

#[test]
fn cuda_ifds_larger_chain_three_procs_four_blocks_three_facts() {
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
    assert_ifds_matches_reference(
        "three-proc chain combined",
        3,
        4,
        3,
        &intra,
        &inter,
        &flow_gen,
        &kill,
    );
}

#[test]
fn cuda_ifds_large_serial_grid_matches_cpu() {
    let procs = 8;
    let blocks = 9;
    let facts = 5;
    let mut intra = Vec::new();
    for p in 0..procs {
        for b in 0..(blocks - 1) {
            intra.push((p, b, b + 1));
        }
    }
    let mut inter = Vec::new();
    for p in 0..(procs - 1) {
        inter.push((p, blocks - 1, p + 1, 0));
    }
    let flow_gen = (0..procs).map(|p| (p, 0, (p % facts))).collect::<Vec<_>>();
    let kill = (0..procs)
        .map(|p| (p, blocks / 2, ((p + 1) % facts)))
        .collect::<Vec<_>>();
    assert_ifds_matches_reference(
        "large serial-grid IFDS",
        procs,
        blocks,
        facts,
        &intra,
        &inter,
        &flow_gen,
        &kill,
    );
}
