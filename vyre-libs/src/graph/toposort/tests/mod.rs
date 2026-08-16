//! Contracts for both topological-sort oracles and the emitted program.

use super::csr::{
    toposort_csr, toposort_csr_into, toposort_csr_into_with_scratch,
    validate_toposort_csr_inputs, validate_toposort_csr_order, ToposortCsrLayout,
    ToposortCsrScratch,
};
use super::edge_list::toposort;
use super::error::{ToposortCsrError, ToposortError};
use super::program::toposort_program;


#[test]
fn empty_graph_sorts_to_empty() {
    assert_eq!(toposort(0, &[]), Ok(Vec::new()));
}

#[test]
fn no_edges_returns_all_nodes() {
    let got = toposort(3, &[])
        .expect("Fix: no-cycle case; restore this invariant before continuing.");
    assert_eq!(got.len(), 3);
    let mut sorted = got.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, vec![0, 1, 2]);
}

#[test]
fn linear_chain_respects_order() {
    // 0 depends on 1 depends on 2 → sort places 2 before 1 before 0.
    let got = toposort(3, &[(0, 1), (1, 2)])
        .expect("Fix: linear chain is acyclic; restore this invariant before continuing.");
    let pos = |v: u32| got.iter().position(|&x| x == v).unwrap();
    assert!(pos(2) < pos(1));
    assert!(pos(1) < pos(0));
}

#[test]
fn cycle_is_rejected() {
    let err = toposort(2, &[(0, 1), (1, 0)]).expect_err("2-cycle must be detected");
    assert!(matches!(err, ToposortError::Cycle { .. }));
}

#[test]
fn cycle_diagnostic_names_node_on_cycle_not_downstream() {
    // AUDIT_2026-04-24 F-TS-03: graph where node 0 depends on
    // the cycle {1 → 2 → 3 → 1} but is not on it. Prior code
    // reported the first `indeg > 0` node (node 0  -  downstream of
    // the cycle), which was misleading because 0 itself is not on
    // any cycle. Diagnostic must name a node actually on a cycle.
    let err = toposort(4, &[(0, 1), (1, 2), (2, 3), (3, 1)])
        .expect_err("3-cycle with downstream consumer must be detected");
    match err {
        ToposortError::Cycle { node } => {
            assert!(
                matches!(node, 1..=3),
                "cycle node {node} must be on the cycle {{1,2,3}}, not the downstream node 0"
            );
        }
        other => panic!("expected Cycle error, got {other:?}"),
    }
}

#[test]
fn unknown_node_surfaces_edge_index() {
    let err = toposort(2, &[(0, 5)]).expect_err("node 5 is out of range");
    match err {
        ToposortError::UnknownNode { edge, node } => {
            assert_eq!(edge, 0);
            assert_eq!(node, 5);
        }
        _ => panic!("expected UnknownNode"),
    }
}

#[test]
fn diamond_graph_sorts() {
    // 0 depends on 1 and 2; both depend on 3.
    let got = toposort(4, &[(0, 1), (0, 2), (1, 3), (2, 3)])
        .expect("Fix: diamond is acyclic; restore this invariant before continuing.");
    let pos = |v: u32| got.iter().position(|&x| x == v).unwrap();
    assert!(pos(3) < pos(1));
    assert!(pos(3) < pos(2));
    assert!(pos(1) < pos(0));
    assert!(pos(2) < pos(0));
}

#[test]
fn emitted_program_has_expected_buffers_and_workgroup_size() {
    let p = toposort_program(4, "offsets", "targets", "indeg", "queue", "order");
    assert_eq!(p.workgroup_size, [1, 1, 1]);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert_eq!(names, vec!["offsets", "targets", "indeg", "queue", "order"]);
    assert_eq!(p.buffers[0].count(), 5); // node_count + 1
    assert_eq!(p.buffers[2].count(), 4); // node_count
    assert_eq!(p.buffers[3].count(), 4); // node_count
    assert_eq!(p.buffers[4].count(), 4); // node_count
}

#[test]
fn csr_reference_sorts_prerequisites_before_dependents() {
    let order = toposort_csr(3, &[0, 2, 3, 3], &[1, 2, 2]).unwrap();
    let pos = |v: u32| order.iter().position(|&x| x == v).unwrap();
    assert!(pos(0) < pos(1));
    assert!(pos(0) < pos(2));
    assert!(pos(1) < pos(2));
}

#[test]
fn csr_reference_reuses_output_storage() {
    let mut order = Vec::with_capacity(8);
    toposort_csr_into(3, &[0, 1, 2, 2], &[1, 2], &mut order).unwrap();
    let capacity = order.capacity();
    assert_eq!(order.len(), 3);

    toposort_csr_into(2, &[0, 1, 1], &[1], &mut order).unwrap();
    assert_eq!(order.capacity(), capacity);
    assert_eq!(order.len(), 2);
}

#[test]
fn csr_reference_with_scratch_reuses_storage_and_clears_stale_state() {
    let mut order = Vec::with_capacity(8);
    order.extend_from_slice(&[99, 98, 97]);
    let mut queue = Vec::with_capacity(8);
    queue.extend_from_slice(&[6, 5, 4]);
    let mut scratch = ToposortCsrScratch {
        indeg: vec![7; 8],
        queue,
    };
    let order_capacity = order.capacity();
    let indeg_capacity = scratch.indeg.capacity();
    let queue_capacity = scratch.queue.capacity();

    toposort_csr_into_with_scratch(4, &[0, 2, 3, 3, 3], &[1, 2, 3], &mut order, &mut scratch)
        .expect("Fix: valid DAG must sort while reusing caller-owned scratch.");

    validate_toposort_csr_order(4, &[0, 2, 3, 3, 3], &[1, 2, 3], &order)
        .expect("Fix: scratch-backed topological order must satisfy the CSR contract.");
    assert_eq!(order.capacity(), order_capacity);
    assert_eq!(scratch.indeg.capacity(), indeg_capacity);
    assert_eq!(scratch.queue.capacity(), queue_capacity);
    assert_eq!(
        scratch.indeg,
        vec![0, 0, 0, 0],
        "Fix: scratch-backed traversal must not retain stale indegree counts."
    );
    assert!(
        scratch.queue.is_empty(),
        "Fix: scratch-backed traversal must consume stale and live queue entries."
    );

    toposort_csr_into_with_scratch(2, &[0, 1, 1], &[1], &mut order, &mut scratch)
        .expect("Fix: second smaller DAG must reuse the same workspace.");
    validate_toposort_csr_order(2, &[0, 1, 1], &[1], &order)
        .expect("Fix: reused workspace must not leak prior graph state.");
    assert_eq!(order.capacity(), order_capacity);
    assert_eq!(scratch.indeg.capacity(), indeg_capacity);
    assert_eq!(scratch.queue.capacity(), queue_capacity);
    assert_eq!(scratch.indeg, vec![0, 0]);
    assert!(scratch.queue.is_empty());
}

#[test]
fn csr_reference_with_scratch_validates_before_mutating_reused_storage() {
    let mut order = vec![9, 8, 7];
    let mut scratch = ToposortCsrScratch {
        indeg: vec![1, 2],
        queue: vec![3],
    };
    let err = toposort_csr_into_with_scratch(2, &[0, 2, 1], &[1], &mut order, &mut scratch)
        .expect_err("Fix: malformed CSR offsets must be rejected.");

    assert!(matches!(err, ToposortCsrError::BadCsr { .. }));
    assert_eq!(
        order,
        vec![9, 8, 7],
        "Fix: validation failures must not clobber reusable output storage."
    );
    assert_eq!(
        scratch.indeg,
        vec![1, 2],
        "Fix: validation failures must not clear reusable indegree scratch."
    );
    assert_eq!(
        scratch.queue,
        vec![3],
        "Fix: validation failures must not clear reusable queue scratch."
    );
}

#[test]
fn generated_csr_reference_with_scratch_matches_allocating_reference() {
    let mut order = Vec::new();
    let mut scratch = ToposortCsrScratch::new();

    for case in 0..2048usize {
        let n = case % 17;
        let mut offsets = Vec::with_capacity(n + 1);
        let mut targets = Vec::new();
        offsets.push(0);
        for src in 0..n {
            for dst in src + 1..n {
                let mixed = case
                    .wrapping_mul(31)
                    .wrapping_add(src.wrapping_mul(17))
                    .wrapping_add(dst.wrapping_mul(13));
                if mixed % 5 == 0 || (case % 11 == 0 && dst == src + 1) {
                    targets.push(dst as u32);
                }
            }
            offsets.push(targets.len() as u32);
        }

        let expected = toposort_csr(n as u32, &offsets, &targets)
            .expect("Fix: generated lower-triangular CSR graph must be a valid DAG.");
        toposort_csr_into_with_scratch(n as u32, &offsets, &targets, &mut order, &mut scratch)
            .expect("Fix: scratch-backed oracle must accept every generated valid DAG.");
        assert_eq!(
            order, expected,
            "Fix: scratch-backed oracle diverged from allocating oracle at generated case {case}."
        );
    }
}

#[test]
fn csr_validation_rejects_bad_shape() {
    let err = validate_toposort_csr_inputs(2, &[0, 2, 1], &[1]).unwrap_err();
    assert!(matches!(err, ToposortCsrError::BadCsr { .. }));
}

#[test]
fn csr_validation_returns_dispatch_layout() {
    assert_eq!(
        validate_toposort_csr_inputs(3, &[0, 2, 3, 3], &[1, 2, 2]).unwrap(),
        ToposortCsrLayout {
            node_count: 3,
            node_words: 3,
            offset_words: 4,
            target_words: 3,
        }
    );
    assert_eq!(
        validate_toposort_csr_inputs(0, &[0], &[]).unwrap(),
        ToposortCsrLayout {
            node_count: 0,
            node_words: 0,
            offset_words: 1,
            target_words: 0,
        }
    );
}

#[test]
fn csr_order_validation_rejects_duplicate_backend_output() {
    let err = validate_toposort_csr_order(3, &[0, 1, 2, 2], &[1, 2], &[0, 1, 1]).unwrap_err();
    assert!(matches!(err, ToposortCsrError::BadOrder { .. }));
}

#[test]
fn csr_order_validation_rejects_dependency_inversion() {
    let err = validate_toposort_csr_order(2, &[0, 1, 1], &[1], &[1, 0]).unwrap_err();
    assert!(matches!(err, ToposortCsrError::BadOrder { .. }));
}

// ------------------------------------------------------------------
// Adversarial fixtures  -  empty/single/disconnected/self-loop/max-size.
// ------------------------------------------------------------------

#[test]
fn single_node_no_edges() {
    assert_eq!(toposort(1, &[]), Ok(vec![0]));
}

#[test]
fn self_loops_only_rejected() {
    // Every node has a self-loop  -  each is a 1-cycle.
    let err = toposort(3, &[(0, 0), (1, 1), (2, 2)]).expect_err("self-loops are cycles");
    assert!(matches!(err, ToposortError::Cycle { .. }));
}

#[test]
fn disconnected_components_sorts_all() {
    // Component A: 0 depends on 1. Component B: 2 depends on 3.
    let got = toposort(4, &[(0, 1), (2, 3)]).unwrap();
    assert_eq!(got.len(), 4);
    let pos = |v: u32| got.iter().position(|&x| x == v).unwrap();
    assert!(pos(1) < pos(0), "1 must come before 0");
    assert!(pos(3) < pos(2), "3 must come before 2");
}

#[test]
fn max_node_count_min_edges() {
    // 1000 nodes, one chain edge 0→1.
    let got = toposort(1000, &[(0, 1)]).unwrap();
    assert_eq!(got.len(), 1000);
    let pos = |v: u32| got.iter().position(|&x| x == v).unwrap();
    assert!(pos(1) < pos(0), "1 must come before 0");
}

#[test]
fn cycle_on_large_graph_diagnostic_is_on_cycle() {
    // 100 nodes in a line, back-edge creating cycle 50→51→…→99→50.
    let mut edges: Vec<(u32, u32)> = (0..99).map(|i| (i, i + 1)).collect();
    edges.push((99, 50));
    let err = toposort(100, &edges).expect_err("cycle must be detected");
    match err {
        ToposortError::Cycle { node } => {
            assert!(
                (50..=99).contains(&node),
                "cycle node {node} must be on the back-edge cycle"
            );
        }
        other => panic!("expected Cycle, got {other:?}"),
    }
}
