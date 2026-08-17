use super::*;
use vyre_reference::composition_witness::{
    dominator_idoms_witness as cooper_harvey_kennedy_idoms, dominator_idoms_witness as cpu_ref,
    dominator_sets_idoms_witness as lengauer_tarjan_idoms, idoms_to_dominator_sets_witness,
};

fn try_lengauer_tarjan_idoms(
    node_count: u32,
    root: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<Option<u32>>, String> {
    Ok(lengauer_tarjan_idoms(node_count, root, edges))
}

fn try_cpu_ref(
    node_count: u32,
    root: u32,
    edges: &[(u32, u32)],
) -> Result<Vec<Option<u32>>, String> {
    Ok(cpu_ref(node_count, root, edges))
}

fn try_idoms_to_dominator_sets(
    idoms: &[Option<u32>],
    node_count: u32,
) -> Result<Vec<Vec<u32>>, String> {
    Ok(idoms_to_dominator_sets_witness(idoms, node_count))
}

#[test]
fn program_builds_without_panic() {
    let p = dominator_tree_program(4, 4, 4, "idom");
    assert_eq!(p.workgroup_size, [1, 1, 1]);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert!(names.contains(&"idom"));
    assert!(names.contains(&"dt_depth"));
}

/// The fixpoint exposes its three phases as child regions, and the two that
/// answer a question of their own name the operations that answer it.
///
/// The generator strings come from the phase modules, not from literals here,
/// so inlining a phase back into the fixpoint fails this test rather than
/// silently shrinking what a composition reader can see. That a composed phase
/// is also a live registration is `tests/dominator_tree_composition.rs`, which
/// needs the registry feature this unit test does not.
#[test]
fn program_exposes_chk_phases_as_child_regions() {
    use vyre_foundation::composition::is_anonymous_generator;
    use vyre_foundation::ir::Node;

    fn collect(nodes: &[Node], generators: &mut Vec<String>) {
        for node in nodes {
            match node {
                Node::Region {
                    generator,
                    source_region,
                    body,
                } => {
                    if source_region.is_some() {
                        generators.push(generator.as_str().to_string());
                    }
                    collect(body.as_ref(), generators);
                }
                Node::Block(children) => collect(children, generators),
                Node::If {
                    then, otherwise, ..
                } => {
                    collect(then, generators);
                    collect(otherwise, generators);
                }
                Node::Loop { body, .. } => collect(body, generators),
                _ => {}
            }
        }
    }

    let p = dominator_tree_program(4, 4, 4, "idom");
    let mut generators = Vec::new();
    collect(p.entry(), &mut generators);

    for phase in [
        super::depth::OP_ID,
        super::intersect_step::OP_ID,
        super::program::INIT_PHASE_GENERATOR,
    ] {
        assert!(
            generators.iter().any(|g| g == phase),
            "Fix: dominator_tree must expose CHK initialization, depth recompute, and predecessor intersection as child regions so composition printing can see real phase boundaries. Missing `{phase}` in {generators:?}"
        );
    }
    assert!(
        is_anonymous_generator(super::program::INIT_PHASE_GENERATOR),
        "Fix: clearing the forest answers no question a second caller could ask, so `{}` must carry an anonymous-generator prefix",
        super::program::INIT_PHASE_GENERATOR
    );
    for phase in [super::depth::OP_ID, super::intersect_step::OP_ID] {
        assert!(
            !is_anonymous_generator(phase),
            "Fix: `{phase}` is an operation of its own, so it must not carry an anonymous-generator prefix"
        );
    }
}

#[test]
fn checked_builder_rejects_u32_max_node_count() {
    let err = try_dominator_tree_program(u32::MAX, 0, 0, "idom").unwrap_err();
    assert!(err.contains("u32::MAX collides with IDOM_NONE"));
}

#[test]
fn legacy_builder_returns_inert_trap_on_u32_max() {
    let p = dominator_tree_program(u32::MAX, 0, 0, "idom");
    assert_eq!(p.workgroup_size, [1, 1, 1]);
    assert_eq!(p.buffers.len(), 6);
    let names: Vec<&str> = p.buffers.iter().map(|b| b.name()).collect();
    assert!(names.contains(&"idom"));
    use vyre_foundation::ir::Node;
    assert!(
            matches!(
                p.entry.first(),
                Some(Node::Region { body, .. }) if body.len() == 1
            ),
            "Fix: invalid dominator_tree shape must compile to an inert early-return trap, not a full kernel."
        );
}

#[test]
fn empty_graph_returns_empty() {
    let idoms = cpu_ref(0, 0, &[]);
    assert!(idoms.is_empty());
}

#[test]
fn single_node_self_idom() {
    let idoms = cpu_ref(1, 0, &[]);
    assert_eq!(idoms, vec![Some(0)]);
}

#[test]
fn linear_chain_idoms() {
    // 0 -> 1 -> 2 -> 3
    let idoms = cpu_ref(4, 0, &[(0, 1), (1, 2), (2, 3)]);
    assert_eq!(idoms[0], Some(0));
    assert_eq!(idoms[1], Some(0));
    assert_eq!(idoms[2], Some(1));
    assert_eq!(idoms[3], Some(2));
}

#[test]
fn diamond_idoms() {
    // 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
    let idoms = cpu_ref(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    assert_eq!(idoms[0], Some(0));
    assert_eq!(idoms[1], Some(0));
    assert_eq!(idoms[2], Some(0));
    assert_eq!(idoms[3], Some(0));
}

#[test]
fn while_loop_idoms() {
    // 0 -> 1, 1 -> 2, 2 -> 1, 1 -> 3
    let idoms = cpu_ref(4, 0, &[(0, 1), (1, 2), (2, 1), (1, 3)]);
    assert_eq!(idoms[0], Some(0));
    assert_eq!(idoms[1], Some(0));
    assert_eq!(idoms[2], Some(1));
    assert_eq!(idoms[3], Some(1));
}

#[test]
fn unreachable_nodes_are_none() {
    // 0 -> 1. 2 and 3 are disconnected.
    let idoms = cpu_ref(4, 0, &[(0, 1)]);
    assert_eq!(idoms[0], Some(0));
    assert_eq!(idoms[1], Some(0));
    assert_eq!(idoms[2], None);
    assert_eq!(idoms[3], None);
}

#[test]
fn lt_matches_chk_on_diamond() {
    let lt = lengauer_tarjan_idoms(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    let chk = cooper_harvey_kennedy_idoms(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    assert_eq!(lt, chk);
}

#[test]
fn lt_matches_chk_on_while_loop() {
    let edges = &[(0, 1), (1, 2), (2, 1), (1, 3)];
    let lt = lengauer_tarjan_idoms(4, 0, edges);
    let chk = cooper_harvey_kennedy_idoms(4, 0, edges);
    assert_eq!(lt, chk);
}

#[test]
fn generated_try_lt_matches_chk_on_small_graphs() {
    for case in 0..16384usize {
        let n = 1 + case % 10;
        let mut edges = Vec::new();
        for src in 0..n {
            for dst in 0..n {
                if src != dst && ((src * 17 + dst * 31 + case) % 11) < 3 {
                    edges.push((src as u32, dst as u32));
                }
            }
        }
        let lt = try_lengauer_tarjan_idoms(n as u32, 0, &edges)
                .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - generated dominator LT oracle should reserve and evaluate");
        let chk = cooper_harvey_kennedy_idoms(n as u32, 0, &edges);

        assert_eq!(lt, chk, "case {case}: LT and CHK idoms diverged");
    }
}

#[test]
fn repeated_reference_calls_do_not_retain_prior_graph_state() {
    let diamond = cpu_ref(4, 0, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
    assert_eq!(diamond, vec![Some(0), Some(0), Some(0), Some(0)]);

    let chain = cpu_ref(2, 0, &[(0, 1)]);
    assert_eq!(chain, vec![Some(0), Some(0)]);

    let invalid_root = cpu_ref(3, 5, &[(0, 1)]);
    assert_eq!(invalid_root, vec![None, None, None]);
}

#[test]
fn generated_idom_set_conversion_is_sorted_and_includes_self() {
    for case in 0..8192usize {
        let n = 1 + case % 32;
        let edges: Vec<(u32, u32)> = (1..n)
            .map(|node| ((node - 1) as u32, node as u32))
            .collect();
        let idoms = try_cpu_ref(n as u32, 0, &edges)
                .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - generated dominator CPU oracle should reserve and evaluate");
        let sets = try_idoms_to_dominator_sets(&idoms, n as u32)
                .expect("Fix: caller must pre-size buffers; use fallible reserve or return ResourceExhausted - generated dominator set conversion should reserve and evaluate");

        assert_eq!(sets.len(), n, "case {case}: one set per node");
        for (node, set) in sets.iter().enumerate() {
            assert!(
                set.windows(2).all(|pair| pair[0] < pair[1]),
                "case {case} node {node}: dominator set must be sorted and unique"
            );
            assert!(
                set.contains(&(node as u32)),
                "case {case} node {node}: dominator set must contain the node itself"
            );
        }
    }
}

#[test]
fn validation_rejects_bad_offsets() {
    let err = validate_dominator_tree_inputs(2, &[0, 1], &[0], &[0, 0, 0], &[]).unwrap_err();
    assert!(matches!(err, DominatorTreeError::BadOffsets { .. }));
}

#[test]
fn validation_rejects_oob_target() {
    let err = validate_dominator_tree_inputs(2, &[0, 1, 1], &[5], &[0, 0, 0], &[]).unwrap_err();
    assert!(matches!(
        err,
        DominatorTreeError::TargetOutOfRange { target: 5, .. }
    ));
}

#[test]
fn validation_rejects_non_monotonic_offsets() {
    let err = validate_dominator_tree_inputs(2, &[0, 2, 1], &[0, 0], &[0, 0, 0], &[]).unwrap_err();
    assert!(matches!(
        err,
        DominatorTreeError::NonMonotonicOffsets { .. }
    ));
}

#[test]
fn validation_returns_layout() {
    let layout =
        validate_dominator_tree_inputs(3, &[0, 1, 2, 2], &[1, 2], &[0, 0, 0, 0], &[]).unwrap();
    assert_eq!(layout.node_count, 3);
    assert_eq!(layout.edge_count, 2);
    assert_eq!(layout.pred_edge_count, 0);
}
