use vyre_foundation::ir::{Node, Program};
use crate::graph::program_graph::ProgramGraphShape;
use crate::graph::{
    adjustment_set::{backdoor_descendants_check, backdoor_descendants_check_cpu},
    chebyshev_filter::{chebyshev_filter, chebyshev_filter_cpu_into, try_chebyshev_filter},
    csr_backward_or_changed::csr_backward_or_changed_parallel,
    csr_backward_traverse::csr_backward_traverse,
    csr_frontier_queue::{
        csr_queue_forward_traverse, csr_queue_forward_traverse_cpu, frontier_to_queue,
        frontier_to_queue_cpu, try_csr_queue_forward_traverse_cpu, try_frontier_to_queue_cpu,
    },
    do_calculus::{
        do_intervention_delete_incoming, do_intervention_delete_incoming_cpu,
        do_rule2_reverse_incoming, do_rule2_reverse_incoming_cpu, do_rule3_subgraph,
        do_rule3_subgraph_cpu, try_do_intervention_delete_incoming,
        try_do_intervention_delete_incoming_cpu, try_do_rule2_reverse_incoming,
        try_do_rule2_reverse_incoming_cpu, try_do_rule3_subgraph_cpu,
    },
    dominator_frontier::{dominator_frontier, validate_csr_shape},
    exploded::{
        build_ifds_csr_program, decode_node, ifds_node_count_checked, max_ifds_col_count,
        validate_ifds_csr_layout,
    },
    functorial::{functor_apply, functor_apply_cpu},
    knowledge_compile::{
        ddnnf_evaluate, ddnnf_evaluate_cpu, try_ddnnf_evaluate, try_ddnnf_evaluate_cpu, AND_NODE,
        LITERAL_TRUE,
    },
    matroid::{
        matroid_exchange_bfs_step, matroid_exchange_bfs_step_cpu, try_matroid_exchange_bfs_step,
    },
    path_reconstruct::{batched_path_reconstruct, cpu_ref_batched},
    persistent_bfs::{persistent_bfs_batch, try_persistent_bfs_batch},
    reachable::reachable_program,
    sheaf::{sheaf_diffusion_step, try_sheaf_diffusion_step},
    string_diagram::{monoidal_compose, monoidal_compose_cpu, try_monoidal_compose},
    sum_product_circuit::{
        sum_product_evaluate, sum_product_evaluate_cpu, try_sum_product_evaluate, KIND_LEAF,
        KIND_PRODUCT, KIND_SUM,
    },
    tensor_flow_forward::tensor_flow_forward,
    toposort::toposort_csr,
    union_find::{find_root_body, union_find_program, union_roots_body},
};
use crate::math::tensor_scc::{cpu_ref as tensor_scc_cpu_ref, tensor_scc_fixpoint};

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-8 * (1.0 + a.abs() + b.abs())
}

fn approx_eq_f32(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5 * (1.0 + a.abs() + b.abs())
}

fn program_generator(program: &Program) -> &str {
    let Some(Node::Region { generator, .. }) = program.entry.first() else {
        panic!("Fix: structural graph Program must start with a Region.");
    };
    generator.as_str()
}

#[test]
fn program_builders_emit_expected_structural_primitives() {
    let shape = ProgramGraphShape::new(4, 3);
    assert_eq!(
        program_generator(&sum_product_evaluate(
            "k", "off", "cnt", "ch", "w", "leaf", "out", 3, 2
        )),
        "vyre-primitives::graph::sum_product_evaluate"
    );
    assert_eq!(
        program_generator(&matroid_exchange_bfs_step(
            "fin", "adj", "vis", "fout", "changed", 3
        )),
        "vyre-primitives::graph::matroid_exchange_bfs_step"
    );
    assert_eq!(
        program_generator(&monoidal_compose("f", "g", "out", 2, 2, 2)),
        "vyre-primitives::graph::monoidal_compose"
    );
    assert_eq!(
        program_generator(&tensor_flow_forward(shape, "tin", "tout", 2, 2, 1)),
        "vyre-primitives::graph::tensor_flow_forward"
    );
    assert_eq!(
        program_generator(&functor_apply("src", "map", "dst", 3)),
        "vyre-primitives::graph::functor_apply"
    );
    assert_eq!(
        program_generator(&persistent_bfs_batch(
            shape,
            "fin",
            "fout",
            "changed",
            "converged",
            2,
            1,
            2
        )),
        "vyre-primitives::graph::persistent_bfs_batch"
    );
    assert_eq!(
        program_generator(&dominator_frontier(4, 4, 4, "seed", "out")),
        "vyre-primitives::graph::dominator_frontier"
    );
    assert_eq!(
        program_generator(&ddnnf_evaluate(
            "kind", "var", "off", "cnt", "ch", "assign", "out", 3, 2, 2
        )),
        "vyre-primitives::graph::ddnnf_evaluate"
    );
    assert_eq!(
        program_generator(&frontier_to_queue("frontier", "queue", "len", 4, 4)),
        "vyre-primitives::graph::frontier_to_queue"
    );
    assert_eq!(
        program_generator(&csr_queue_forward_traverse(
            "queue", "len", "off", "target", "kind", "out", 4, 3, 4, 1
        )),
        "vyre-primitives::graph::csr_queue_forward_traverse"
    );
    assert_eq!(
        program_generator(&csr_backward_traverse(shape, "fin", "fout", 1)),
        "vyre-primitives::graph::csr_backward_traverse"
    );
    assert_eq!(
        program_generator(&csr_backward_or_changed_parallel(
            shape, "frontier", "changed", 1
        )),
        "vyre-primitives::graph::csr_backward_or_changed"
    );
    assert_eq!(
        program_generator(&chebyshev_filter("l", "x", "c", "y", "scratch", 2, 1)),
        "vyre-primitives::graph::chebyshev_filter"
    );
    assert_eq!(
        program_generator(&sheaf_diffusion_step("s", "r", "d", "out", 2, 2)),
        "vyre-primitives::graph::sheaf_diffusion_step"
    );
    assert_eq!(
        program_generator(&backdoor_descendants_check("z", "d", "v", 4)),
        "vyre-primitives::graph::backdoor_descendants_check"
    );
    assert_eq!(
        program_generator(&do_intervention_delete_incoming("a", "m", "out", 2)),
        "vyre-primitives::graph::do_intervention_delete_incoming"
    );
    assert_eq!(
        program_generator(&do_rule2_reverse_incoming("a", "m", "out", 2)),
        "vyre-primitives::graph::do_rule2_reverse_incoming"
    );
    assert_eq!(
        program_generator(&do_rule3_subgraph(
            "a", "m", "reduced", "kept", "kept_len", 2
        )),
        "vyre-primitives::graph::do_rule3_subgraph"
    );
    assert_eq!(
        program_generator(&tensor_scc_fixpoint("rows", "seed", "group", "out", 4, 8)),
        "vyre-primitives::math::tensor_scc"
    );
}

#[test]
fn composed_programs_and_bodies_are_non_empty() {
    let reach = reachable_program(4, 3, "sources", "reach", 2);
    assert!(!reach.buffers().is_empty());
    assert!(!reach.entry().is_empty());

    let ifds = build_ifds_csr_program(1, 2, 2, 1, 0, 1, 0, 4);
    assert_eq!(
        program_generator(&ifds),
        "vyre-primitives::graph::exploded_build_ifds_csr"
    );

    let batched = batched_path_reconstruct(3, 4);
    assert_eq!(
        program_generator(&batched),
        "vyre-primitives::graph::batched_path_reconstruct"
    );

    assert!(!find_root_body("parent", "id", "root", "scratch", 4).is_empty());
    assert!(!union_roots_body("parent", "a", "b", "edge", 4).is_empty());
    assert_eq!(
        program_generator(&union_find_program("parent", "a", "b", 4, 2)),
        "vyre-primitives::graph::union_find"
    );
}

#[test]
fn checked_builders_reject_bad_shapes_without_panicking() {
    assert!(
        try_sum_product_evaluate("k", "o", "c", "ch", "w", "l", "out", 0, 0)
            .unwrap_err()
            .contains("n_nodes > 0")
    );
    assert!(try_matroid_exchange_bfs_step("f", "a", "v", "o", "c", 0)
        .unwrap_err()
        .contains("n > 0"));
    assert!(try_monoidal_compose("f", "g", "o", 0, 1, 1)
        .unwrap_err()
        .contains("a, b, c > 0"));
    assert!(try_persistent_bfs_batch(
        ProgramGraphShape::new(u32::MAX, 3),
        "i",
        "o",
        "c",
        "cv",
        u32::MAX,
        1,
        1,
    )
    .unwrap_err()
    .contains("frontier words overflow"));
    assert!(
        try_ddnnf_evaluate("k", "v", "o", "c", "ch", "a", "out", 0, 0, 1)
            .unwrap_err()
            .contains("n_nodes > 0")
    );
    assert!(try_chebyshev_filter("l", "x", "c", "y", "s", 0, 1)
        .unwrap_err()
        .contains("n > 0"));
    assert!(try_sheaf_diffusion_step("s", "r", "d", "o", 0, 1)
        .unwrap_err()
        .contains("n_nodes > 0"));
    assert!(try_do_intervention_delete_incoming("a", "m", "o", 0)
        .unwrap_err()
        .contains("n > 0"));
    assert!(try_do_rule2_reverse_incoming("a", "m", "o", 0)
        .unwrap_err()
        .contains("n > 0"));
}

#[test]
fn cpu_references_cover_logic_and_category_contracts() {
    let sp = sum_product_evaluate_cpu(
        &[KIND_LEAF, KIND_LEAF, KIND_SUM, KIND_PRODUCT],
        &[0, 0, 0, 0],
        &[0, 0, 2, 2],
        &[0, 1, 0, 1],
        &[0.25, 0.75, 0.0, 0.0],
        &[0.6, 0.4, 0.0, 0.0],
        &[0, 1, 2, 3],
    );
    assert!(approx_eq(sp[2], 0.45));
    assert!(approx_eq(sp[3], 0.24));

    assert_eq!(
        matroid_exchange_bfs_step_cpu(&[1, 0, 0], &[0, 1, 0, 0, 0, 0, 0, 0, 0], &[0, 0, 0], 3),
        (vec![0, 1, 0], true)
    );
    assert_eq!(
        monoidal_compose_cpu(&[1.0, 2.0, 3.0, 4.0], &[1.0, 0.0, 0.0, 1.0], 2, 2, 2),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        functor_apply_cpu(&[10, 20, 30], &[2, 0, 1], 3),
        vec![20, 30, 10]
    );
    assert_eq!(
        ddnnf_evaluate_cpu(
            &[(LITERAL_TRUE, 0, 0), (LITERAL_TRUE, 0, 0), (AND_NODE, 0, 2)],
            &[0, 1, 0],
            &[0, 1],
            &[u32::MAX, u32::MAX],
            &[0, 1, 2],
        )[2],
        1
    );
    assert_eq!(
        try_ddnnf_evaluate_cpu(&[(LITERAL_TRUE, 0, 0)], &[0], &[], &[1], &[0]).unwrap(),
        vec![1]
    );
}

#[test]
fn cpu_references_cover_resident_traversal_and_numeric_graphs() {
    assert_eq!(frontier_to_queue_cpu(&[0b10111], 5, 3), (vec![0, 1, 2], 4));
    assert!(
        try_frontier_to_queue_cpu(&[0b10111], 64, 3)
            .unwrap_err()
            .contains("frontier_in.len() == bitset_words(node_count)"),
        "frontier-to-queue CPU oracle must reject a frontier whose width is not bitset_words(node_count)"
    );
    assert_eq!(
        csr_queue_forward_traverse_cpu(&[0, 1], 2, &[0, 2, 3, 3, 3], &[1, 2, 3], &[1, 2, 1], 4, 1,),
        vec![0b1010]
    );
    assert!(
        try_csr_queue_forward_traverse_cpu(&[0], 1, &[0, 1, 1], &[4], &[1], 2, 1)
            .unwrap_err()
            .contains("outside node_count"),
        "queue-driven CSR traversal CPU oracle must reject an edge target outside node_count"
    );

    let mut paths = Vec::new();
    let mut lens = Vec::new();
    cpu_ref_batched(&[0, 0, 1, 2], &[3, 0, 2], 4, &mut paths, &mut lens);
    assert_eq!(lens, vec![4, 1, 3]);
    assert_eq!(&paths[0..4], &[3, 2, 1, 0]);

    let mut out = Vec::new();
    let mut t_prev = Vec::new();
    let mut t_curr = Vec::new();
    let mut t_next = Vec::new();
    chebyshev_filter_cpu_into(
        &[0.5, 0.0, 0.0, 0.5],
        &[1.0, 1.0],
        &[0.0, 0.0, 1.0],
        2,
        2,
        &mut out,
        &mut t_prev,
        &mut t_curr,
        &mut t_next,
    );
    assert!(approx_eq_f32(out[0], -0.5));
    assert!(approx_eq_f32(out[1], -0.5));

    assert!(backdoor_descendants_check_cpu(&[0, 1], &[0, 1]));
    assert_eq!(
        tensor_scc_cpu_ref(&[0b0010, 0b0100, 0b0001], 0b0001, 0b0111, 8),
        0b0111
    );
}

#[test]
fn validation_helpers_cover_ifds_dominance_toposort_and_causal_contracts() {
    assert_eq!(ifds_node_count_checked(2, 3, 4), Some(24));
    assert_eq!(max_ifds_col_count(2, 1, 1, 4), Some(14));
    let layout = validate_ifds_csr_layout(1, 2, 2, 1, 0, 1).unwrap();
    assert_eq!(layout.total_nodes, 4);
    assert_eq!(decode_node((1 << 20) | (2 << 10) | 3), (1, 2, 3));

    assert_eq!(
        validate_csr_shape("test", 3, &[0, 1, 1, 1], &[1]).unwrap(),
        1
    );
    assert_eq!(
        toposort_csr(3, &[0, 1, 2, 2], &[1, 2]).unwrap(),
        vec![0, 1, 2]
    );

    assert_eq!(
        do_intervention_delete_incoming_cpu(&[1, 2, 3, 4], &[1, 0], 2),
        vec![0, 2, 0, 4]
    );
    assert!(
        try_do_intervention_delete_incoming_cpu(&[1], &[1], 2)
            .unwrap_err()
            .contains("adjacency.len() == n*n"),
        "do-intervention CPU oracle must reject an adjacency buffer that is not n*n"
    );
    assert_eq!(
        do_rule2_reverse_incoming_cpu(&[0, 1, 0, 0], &[0, 1], 2),
        vec![0, 0, 1, 0]
    );
    assert!(
        try_do_rule2_reverse_incoming_cpu(&[1], &[1], 2)
            .unwrap_err()
            .contains("adjacency.len() == n*n"),
        "Rule-2 reversal CPU oracle must reject an adjacency buffer that is not n*n"
    );
    assert_eq!(
        do_rule3_subgraph_cpu(&[0, 1, 1, 0], &[1, 0], 2),
        (vec![0], vec![0])
    );
    assert!(
        try_do_rule3_subgraph_cpu(&[1], &[1, 0], 2)
            .unwrap_err()
            .contains("adjacency.len() == n*n"),
        "Rule-3 subgraph CPU oracle must reject an adjacency buffer that is not n*n"
    );
}
