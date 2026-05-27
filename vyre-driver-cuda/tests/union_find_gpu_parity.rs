//! Parity test: GPU union-find produces the same connected-component
//! partition as the reference. Compares roots after path
//! compression since GPU's atomic-CAS unions can produce different
//! intermediate parent links but must agree on which nodes share a
//! root.

#![cfg(test)]

mod common;

use common::{live_dispatcher, CudaOptimizerDispatcher};
use vyre_self_substrate::union_find_emit::{
    canonicalize_parent_to_roots, reference_union_find_alias, union_find_alias_via,
};

fn assert_same_partition(a: &[u32], b: &[u32]) {
    let n = a.len();
    assert_eq!(n, b.len());
    let ra = canonicalize_parent_to_roots(a);
    let rb = canonicalize_parent_to_roots(b);
    for i in 0..n {
        for j in (i + 1)..n {
            let same_a = ra[i] == ra[j];
            let same_b = rb[i] == rb[j];
            assert_eq!(
                same_a, same_b,
                "partition mismatch at ({i}, {j}): gpu_roots={ra:?} reference_roots={rb:?}"
            );
        }
    }
}

#[test]
fn cuda_union_find_disconnected_pairs() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let parent_init: Vec<u32> = (0..6).collect();
    let edge_a = vec![0u32, 2, 4];
    let edge_b = vec![1u32, 3, 5];
    let reference = reference_union_find_alias(&parent_init, &edge_a, &edge_b);
    let gpu = union_find_alias_via(&dispatcher, &parent_init, &edge_a, &edge_b).expect("dispatch");
    assert_same_partition(&reference, &gpu);
}

#[test]
fn cuda_union_find_chain() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let parent_init: Vec<u32> = (0..5).collect();
    let edge_a = vec![0u32, 1, 2, 3];
    let edge_b = vec![1u32, 2, 3, 4];
    let reference = reference_union_find_alias(&parent_init, &edge_a, &edge_b);
    let gpu = union_find_alias_via(&dispatcher, &parent_init, &edge_a, &edge_b).expect("dispatch");
    assert_same_partition(&reference, &gpu);
    let reference_roots = canonicalize_parent_to_roots(&reference);
    for i in 1..5 {
        assert_eq!(reference_roots[0], reference_roots[i]);
    }
}

#[test]
fn cuda_union_find_two_disjoint_components() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let parent_init: Vec<u32> = (0..6).collect();
    let edge_a = vec![0u32, 1, 3, 4];
    let edge_b = vec![1u32, 2, 4, 5];
    let reference = reference_union_find_alias(&parent_init, &edge_a, &edge_b);
    let gpu = union_find_alias_via(&dispatcher, &parent_init, &edge_a, &edge_b).expect("dispatch");
    assert_same_partition(&reference, &gpu);
    let roots = canonicalize_parent_to_roots(&gpu);
    assert_eq!(roots[0], roots[1]);
    assert_eq!(roots[0], roots[2]);
    assert_eq!(roots[3], roots[4]);
    assert_eq!(roots[3], roots[5]);
    assert_ne!(roots[0], roots[3]);
}

#[test]
fn cuda_union_find_no_edges_keeps_singletons() {
    let backend = live_dispatcher();
    let dispatcher = CudaOptimizerDispatcher { backend: &backend };
    let parent_init: Vec<u32> = (0..4).collect();
    let reference = reference_union_find_alias(&parent_init, &[], &[]);
    let gpu = union_find_alias_via(&dispatcher, &parent_init, &[], &[]).expect("dispatch");
    assert_eq!(reference, parent_init);
    let roots = canonicalize_parent_to_roots(&gpu);
    for i in 0..4u32 {
        assert_eq!(roots[i as usize], i);
    }
}
