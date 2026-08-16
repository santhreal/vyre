//! The dominator-tree fixpoint is composed, and each phase it composes answers
//! its declared question.
//!
//! Cooper-Harvey-Kennedy is a repetition of two queries: recompute the depth of
//! the current idom forest, then relax every node against its predecessors.
//! Both used to be anonymous phase boundaries, which is a comment rather than a
//! contract: nothing said the bodies were separable, nothing could reuse them,
//! and Gate 1 read the whole fixpoint as one inline operation.
//!
//! What this file holds:
//!
//! - The composition contract. Every registered `dominator_tree_*` phase must
//!   appear as a child region of the fixpoint, and every child region of the
//!   fixpoint that is not one must carry an anonymous prefix. The phase set is
//!   read from the registry, so a phase added later and not composed fails here
//!   instead of passing unnoticed.
//! - The value contract of each phase against `vyre-reference`, pinned
//!   independently of the fixture the registration declares.
//! - The boundaries each phase has to survive mid-fixpoint: a forest still
//!   holding `IDOM_NONE`, a parent chain that cycles, and a node whose whole
//!   predecessor list is still unreached.
//!
//! What it does not cover: the fixpoint's own convergence. That is
//! `dominator_tree_pristine`'s Tier 5, which compares the whole program against
//! the Lengauer-Tarjan oracle.
#![cfg(feature = "graph")]

use std::collections::BTreeSet;

use vyre_foundation::composition::is_anonymous_generator;
use vyre_foundation::ir::{Node, Program};
use vyre_foundation::operation::OperationRegistry;
use vyre_foundation::visit::child_bodies;
use vyre_libs::graph::dominator_tree::{
    dominator_tree_depth, dominator_tree_intersect_step, dominator_tree_program, IDOM_NONE,
    OP_ID as DOMINATOR_TREE_OP_ID,
};
use vyre_reference::value::Value;

/// Prefix every phase of the fixpoint answers to.
const PHASE_PREFIX: &str = "vyre-libs::graph::dominator_tree_";

/// Generators of every child region in `nodes`, at any depth.
fn child_generators(nodes: &[Node], out: &mut BTreeSet<String>) {
    for node in nodes {
        if let Node::Region {
            generator,
            source_region,
            ..
        } = node
        {
            if source_region.is_some() {
                out.insert(generator.as_str().to_string());
            }
        }
        for body in child_bodies(node) {
            child_generators(body, out);
        }
    }
}

/// Every registered operation id, from the registry rather than a list.
fn registered_ids() -> BTreeSet<String> {
    OperationRegistry::global()
        .iter()
        .map(|op| op.id.to_string())
        .collect()
}

/// Run `program` over `inputs` and return every writable buffer as words.
fn reference_words(program: &Program, inputs: &[&[u32]]) -> Vec<Vec<u32>> {
    let values: Vec<Value> = inputs
        .iter()
        .map(|words| Value::from(vyre_primitives::wire::pack_u32_slice(words)))
        .collect();
    vyre_reference::reference_eval(program, &values)
        .expect("dominator-tree phase program must evaluate")
        .into_iter()
        .map(|value| {
            value
                .to_bytes()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("u32 chunk has four bytes")))
                .collect()
        })
        .collect()
}

#[test]
fn fixpoint_composes_every_registered_phase() {
    let registered = registered_ids();
    let phases: BTreeSet<String> = registered
        .iter()
        .filter(|id| id.starts_with(PHASE_PREFIX))
        .cloned()
        .collect();
    assert!(
        !phases.is_empty(),
        "Fix: the Cooper-Harvey-Kennedy fixpoint must be built from registered phase operations, \
         not inline bodies. No registered op id starts with `{PHASE_PREFIX}`."
    );

    let mut generators = BTreeSet::new();
    child_generators(
        dominator_tree_program(4, 4, 4, "idom").entry(),
        &mut generators,
    );

    let composed: BTreeSet<String> = generators
        .iter()
        .filter(|generator| registered.contains(*generator))
        .cloned()
        .collect();
    assert_eq!(
        composed, phases,
        "Fix: every registered dominator-tree phase must be a child region of \
         `{DOMINATOR_TREE_OP_ID}`, and the fixpoint must not name a phase it does not compose."
    );

    for generator in &generators {
        assert!(
            registered.contains(generator) || is_anonymous_generator(generator),
            "Fix: child region `{generator}` names neither a registered operation nor an \
             anonymous phase boundary. Register it, or give it an anonymous-generator prefix."
        );
    }
}

#[test]
fn depth_phase_witness_matches_reference() {
    let entry = OperationRegistry::global()
        .get("vyre-libs::graph::dominator_tree_depth")
        .expect("dominator_tree_depth is registered");
    let inputs = (entry.test_inputs.expect("declared test inputs"))();
    let declared = (entry.expected_output.expect("declared expected output"))();
    // Forest `0 <- 1 <- 2` with node 3 unreached.
    assert_eq!(
        declared,
        vec![vec![vyre_primitives::wire::pack_u32_slice(&[0, 1, 2, 0])]],
        "declared witness drift for dominator_tree_depth"
    );

    let build = entry.build.expect("neutral builder");
    for (case, (input_set, expected)) in inputs.iter().zip(declared.iter()).enumerate() {
        let outputs = vyre_reference::reference_eval(
            &build(),
            &input_set
                .iter()
                .cloned()
                .map(Value::from)
                .collect::<Vec<_>>(),
        )
        .expect("reference run for dominator_tree_depth")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
        assert_eq!(outputs, *expected, "CPU witness drift, case {case}");
    }
}

#[test]
fn depth_phase_bounds_a_cycling_parent_chain() {
    // Nodes 1 and 2 point at each other, which a forest mid-fixpoint can hold.
    // The walk is bounded by node_count, so it terminates with the depth it
    // reached rather than spinning, and the entry stays at 0.
    let outputs = reference_words(
        &dominator_tree_depth(4, "idom", "dt_depth"),
        &[&[0, 2, 1, IDOM_NONE], &[0; 4]],
    );
    assert_eq!(
        outputs,
        vec![vec![0, 4, 4, 0]],
        "Fix: the depth walk must stop after node_count steps on a parent cycle"
    );
}

#[test]
fn depth_phase_treats_an_unreached_forest_as_flat() {
    let outputs = reference_words(
        &dominator_tree_depth(4, "idom", "dt_depth"),
        &[&[0, IDOM_NONE, IDOM_NONE, IDOM_NONE], &[9, 9, 9, 9]],
    );
    assert_eq!(
        outputs,
        vec![vec![0, 0, 0, 0]],
        "Fix: an unreached node has depth 0, and the phase must overwrite whatever the \
         buffer held rather than accumulate onto it"
    );
}

#[test]
fn intersect_phase_witness_matches_reference() {
    let entry = OperationRegistry::global()
        .get("vyre-libs::graph::dominator_tree_intersect_step")
        .expect("dominator_tree_intersect_step is registered");
    let inputs = (entry.test_inputs.expect("declared test inputs"))();
    let declared = (entry.expected_output.expect("declared expected output"))();
    // Diamond `0 -> {1, 2} -> 3`: node 3's two predecessors intersect at the
    // entry, and the sweep reports that it moved node 3.
    assert_eq!(
        declared,
        vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[0, 0, 0, 0]),
            vyre_primitives::wire::pack_u32_slice(&[1]),
        ]],
        "declared witness drift for dominator_tree_intersect_step"
    );

    let build = entry.build.expect("neutral builder");
    for (case, (input_set, expected)) in inputs.iter().zip(declared.iter()).enumerate() {
        let outputs = vyre_reference::reference_eval(
            &build(),
            &input_set
                .iter()
                .cloned()
                .map(Value::from)
                .collect::<Vec<_>>(),
        )
        .expect("reference run for dominator_tree_intersect_step")
        .into_iter()
        .map(|value| value.to_bytes())
        .collect::<Vec<_>>();
        assert_eq!(outputs, *expected, "CPU witness drift, case {case}");
    }
}

#[test]
fn intersect_phase_reports_no_movement_on_a_settled_forest() {
    // Same diamond, but node 3 already sits at the entry. Nothing moves, so the
    // fixpoint that composes this phase must be able to stop.
    let outputs = reference_words(
        &dominator_tree_intersect_step(4, 4, "idom", "dt_depth"),
        &[
            &[0, 0, 1, 2, 4],
            &[0, 0, 1, 2],
            &[0, 0, 0, 0],
            &[0, 1, 1, 1],
            &[0],
        ],
    );
    assert_eq!(
        outputs,
        vec![vec![0, 0, 0, 0], vec![0]],
        "Fix: a sweep that changes no parent must leave `changed` at 0"
    );
}

#[test]
fn intersect_phase_leaves_a_node_whose_predecessors_are_all_unreached() {
    // Node 1 is the only reachable node; nodes 2 and 3 form a component the
    // entry cannot reach, so node 3's predecessor 2 is still `IDOM_NONE`.
    // Leaving node 3 alone is what keeps an unreachable node unreachable.
    let outputs = reference_words(
        &dominator_tree_intersect_step(4, 4, "idom", "dt_depth"),
        &[
            &[0, 0, 1, 2, 3],
            &[0, 3, 2, 0],
            &[0, 0, IDOM_NONE, IDOM_NONE],
            &[0, 1, 0, 0],
            &[0],
        ],
    );
    assert_eq!(
        outputs,
        vec![vec![0, 0, IDOM_NONE, IDOM_NONE], vec![0]],
        "Fix: a node with no reached predecessor keeps IDOM_NONE and reports no movement"
    );
}
