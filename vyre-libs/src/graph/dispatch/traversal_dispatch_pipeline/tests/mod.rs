use vyre_foundation::ir::{Node, Program};
use vyre_primitives::graph::{
    adaptive_traverse::{adaptive_dense_step, cpu_dense_step, should_use_dense},
    csr_forward_or_changed::{
        cpu_ref_closure_into, csr_forward_or_changed_body, csr_forward_or_changed_body_prefixed,
        csr_forward_or_changed_child, csr_forward_or_changed_child_prefixed,
        csr_forward_or_changed_parallel, csr_forward_or_changed_parallel_batch,
        csr_forward_or_changed_parallel_batch_global,
        csr_forward_or_changed_parallel_batch_global_slot,
        try_csr_forward_or_changed_parallel_batch,
        try_csr_forward_or_changed_parallel_batch_global_slot,
    },
    csr_frontier_degree_sum::{
        csr_frontier_degree_sum, csr_frontier_degree_sum_cpu, try_csr_frontier_degree_sum_cpu,
    },
    persistent_bfs_step::{
        persistent_bfs_step, persistent_bfs_step_body, persistent_bfs_step_body_prefixed,
        persistent_bfs_step_child, persistent_bfs_step_child_prefixed,
        persistent_bfs_step_child_prefixed_with_active,
    },
    program_graph::ProgramGraphShape,
};

fn shape() -> ProgramGraphShape {
    ProgramGraphShape::new(4, 4)
}

const PARENT: &str = "vyre-libs::graph::dispatch::traversal_dispatch_pipeline";

const CSR_FORWARD: &str = "vyre-primitives::graph::csr_forward_or_changed";
const PERSISTENT_STEP: &str = "vyre-primitives::graph::persistent_bfs_step";

fn region_generator(node: &Node) -> &str {
    let Node::Region { generator, .. } = node else {
        panic!("Fix: graph traversal child builder must emit a Region.");
    };
    generator.as_str()
}

fn program_generator(program: &Program) -> &str {
    let Some(Node::Region { generator, .. }) = program.entry.first() else {
        panic!("Fix: graph traversal Program must start with a Region.");
    };
    generator.as_str()
}

fn dense_adj(edges: &[(u32, u32)], node_count: u32) -> Vec<u32> {
    let words = vyre_primitives::bitset::bitset_words(node_count) as usize;
    let mut rows = vec![0; node_count as usize * words];
    for &(src, dst) in edges {
        rows[dst as usize * words + src as usize / 32] |= 1 << (src % 32);
    }
    rows
}

#[test]
fn dispatch_programs_emit_expected_graph_primitives() {
    let cases: [(&str, Program, &str); 7] = [
        (
            "adaptive_dense_step",
            adaptive_dense_step("fin", "fout", "adj", 4),
            "vyre-primitives::graph::adaptive_traverse_dense",
        ),
        (
            "csr_forward_or_changed_parallel",
            csr_forward_or_changed_parallel(shape(), "frontier", "changed", 1),
            CSR_FORWARD,
        ),
        (
            "csr_forward_or_changed_parallel_batch",
            csr_forward_or_changed_parallel_batch(shape(), "frontier", "changed", 1, 2),
            CSR_FORWARD,
        ),
        (
            "csr_forward_or_changed_parallel_batch_global",
            csr_forward_or_changed_parallel_batch_global(shape(), "frontier", "changed", 1, 2),
            CSR_FORWARD,
        ),
        (
            "csr_forward_or_changed_parallel_batch_global_slot",
            csr_forward_or_changed_parallel_batch_global_slot(
                shape(), "frontier", "changed", 1, 2, 0, 1,
            ),
            CSR_FORWARD,
        ),
        (
            "csr_frontier_degree_sum",
            csr_frontier_degree_sum(shape()),
            "vyre-primitives::graph::csr_frontier_degree_sum",
        ),
        (
            "persistent_bfs_step",
            persistent_bfs_step(shape(), "frontier", "changed", 1),
            PERSISTENT_STEP,
        ),
    ];

    for (builder, program, expected) in cases {
        assert_eq!(
            program_generator(&program),
            expected,
            "Fix: {builder} must stamp `{expected}` on the Program it returns."
        );
    }
}

#[test]
fn child_regions_preserve_parent_context() {
    let cases: [(&str, Node, &str); 5] = [
        (
            "csr_forward_or_changed_child",
            csr_forward_or_changed_child(PARENT, shape(), "frontier", "changed", 1),
            CSR_FORWARD,
        ),
        (
            "csr_forward_or_changed_child_prefixed",
            csr_forward_or_changed_child_prefixed(PARENT, shape(), "frontier", "changed", 1, "csr"),
            CSR_FORWARD,
        ),
        (
            "persistent_bfs_step_child",
            persistent_bfs_step_child(PARENT, shape(), "frontier", "changed", "scratch", 1),
            PERSISTENT_STEP,
        ),
        (
            "persistent_bfs_step_child_prefixed",
            persistent_bfs_step_child_prefixed(
                PARENT, shape(), "frontier", "changed", "scratch", 1, "step",
            ),
            PERSISTENT_STEP,
        ),
        (
            "persistent_bfs_step_child_prefixed_with_active",
            persistent_bfs_step_child_prefixed_with_active(
                PARENT,
                shape(),
                "frontier",
                "changed",
                "scratch",
                "active",
                1,
                "active_step",
            ),
            PERSISTENT_STEP,
        ),
    ];

    for (builder, node, expected) in cases {
        assert_eq!(
            region_generator(&node),
            expected,
            "Fix: {builder} must stamp `{expected}` on the child Region it returns."
        );
    }
}

#[test]
fn body_builders_emit_composable_ir() {
    let cases: [(&str, Vec<Node>); 4] = [
        (
            "csr_forward_or_changed_body",
            csr_forward_or_changed_body(shape(), "frontier", "changed", 1),
        ),
        (
            "csr_forward_or_changed_body_prefixed",
            csr_forward_or_changed_body_prefixed(shape(), "frontier", "changed", 1, "csr"),
        ),
        (
            "persistent_bfs_step_body",
            persistent_bfs_step_body(shape(), "frontier", "changed", "scratch", 1),
        ),
        (
            "persistent_bfs_step_body_prefixed",
            persistent_bfs_step_body_prefixed(shape(), "frontier", "changed", "scratch", 1, "step"),
        ),
    ];

    for (builder, body) in cases {
        assert!(
            !body.is_empty(),
            "Fix: {builder} must emit a composable body, not an empty node list."
        );
    }
}

#[test]
fn checked_batch_builders_reject_invalid_dimensions() {
    assert!(
        try_csr_forward_or_changed_parallel_batch(shape(), "frontier", "changed", 1, 0).is_err(),
        "Fix: try_csr_forward_or_changed_parallel_batch must reject query_count == 0 with a diagnostic instead of panicking."
    );
    assert!(
        try_csr_forward_or_changed_parallel_batch_global_slot(
            shape(), "frontier", "changed", 1, 1, 2, 2
        )
        .is_err(),
        "Fix: try_csr_forward_or_changed_parallel_batch_global_slot must reject an out-of-range changed_slot with a diagnostic instead of panicking."
    );
}

#[test]
fn cpu_oracles_answer_the_traversal_cases_dispatch_assumes() {
    assert!(!should_use_dense(&[0x7f], 32));
    assert!(should_use_dense(&[0xff], 32));

    let dense = dense_adj(&[(0, 1), (2, 3)], 8);
    assert_eq!(cpu_dense_step(&[1], &dense, 8), vec![0b10]);

    let frontier = [0b101];
    let edge_offsets = [0, 3, 7, 9, 9, 12];
    assert_eq!(csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5), 5);
    assert_eq!(
        try_csr_frontier_degree_sum_cpu(&frontier, &edge_offsets, 5).unwrap(),
        5
    );

    let mut current = Vec::new();
    let mut next = Vec::new();
    cpu_ref_closure_into(
        3,
        &[0, 1, 2, 2],
        &[1, 2],
        &[1, 1],
        &[0b001],
        1,
        4,
        &mut current,
        &mut next,
    );
    assert_eq!(current, vec![0b111]);
}
