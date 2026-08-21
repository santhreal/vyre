//! Parity test: GPU bellman_tn_order matches the reference oracle.

#![cfg(feature = "device-tests")]
#![cfg(test)]

mod harness;

use harness::with_cuda_optimizer_dispatcher;
use vyre_libs::solvers::bellman_tn_order::bellman_tn_order_via;
fn reference_bellman_shortest_path(
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist_init: &[u32],
    nodes: u32,
    max_iters: u32,
) -> (Vec<u32>, u32) {
    let mut dist = dist_init.to_vec();
    let mut iters = 0;
    for _ in 0..max_iters {
        let mut changed = false;
        for i in 0..src.len() {
            let u = src[i] as usize;
            let v = dst[i] as usize;
            let w = weight[i];
            if u < nodes as usize && v < nodes as usize && dist[u] != u32::MAX {
                let new_dist = dist[u].saturating_add(w);
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    changed = true;
                }
            }
        }
        iters += 1;
        if !changed {
            break;
        }
    }
    (dist, iters)
}

fn assert_bellman_tn_order_matches_reference(
    label: &str,
    src: &[u32],
    dst: &[u32],
    weight: &[u32],
    dist_init: &[u32],
    nodes: u32,
    max_iters: u32,
) -> Vec<u32> {
    let (reference, _) =
        reference_bellman_shortest_path(src, dst, weight, dist_init, nodes, max_iters);
    with_cuda_optimizer_dispatcher(label, |dispatcher| {
        let gpu = bellman_tn_order_via(dispatcher, src, dst, weight, dist_init, nodes, max_iters)
            .expect("dispatch");
        assert_eq!(gpu, reference, "{label}: distance divergence");
        gpu
    })
}

#[test]
fn cuda_bellman_tn_order_matches_reference_chain() {
    // 0 -> 1 (10), 1 -> 2 (20), 2 -> 3 (30), 0 -> 3 (100).
    let src = vec![0u32, 1, 2, 0];
    let dst = vec![1u32, 2, 3, 3];
    let weight = vec![10u32, 20, 30, 100];
    let dist_init = vec![0u32, u32::MAX, u32::MAX, u32::MAX];
    let gpu = assert_bellman_tn_order_matches_reference(
        "weighted chain",
        &src,
        &dst,
        &weight,
        &dist_init,
        4,
        10,
    );
    assert_eq!(gpu, vec![0, 10, 30, 60]);
}

#[test]
fn cuda_bellman_tn_order_chain_4_tensors() {
    let src = vec![0u32, 0, 0, 1, 2, 3];
    let dst = vec![1u32, 2, 3, 4, 4, 4];
    let weight = vec![100u32, 200, 300, 50, 40, 10];
    let mut dist_init = vec![u32::MAX; 5];
    dist_init[0] = 0;
    let gpu = assert_bellman_tn_order_matches_reference(
        "four-tensor chain",
        &src,
        &dst,
        &weight,
        &dist_init,
        5,
        10,
    );
    assert_eq!(gpu[4], 150);
}

#[test]
fn cuda_bellman_tn_order_disconnected_stays_max() {
    let src = vec![0u32];
    let dst = vec![1u32];
    let weight = vec![5u32];
    // Node 2 is isolated.
    let mut dist_init = vec![u32::MAX; 3];
    dist_init[0] = 0;
    let gpu = assert_bellman_tn_order_matches_reference(
        "disconnected graph",
        &src,
        &dst,
        &weight,
        &dist_init,
        3,
        8,
    );
    assert_eq!(gpu[2], u32::MAX);
}
