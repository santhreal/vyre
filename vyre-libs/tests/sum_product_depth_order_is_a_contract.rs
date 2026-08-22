//! The depth-leveled sum-product evaluator refuses a depth array that is not a
//! topological order of the circuit it is given.
//!
//! # WHY
//!
//! `sum_product_evaluate_leveled` runs one wave per depth and relies on a
//! `GridSync` barrier between waves, so a node reads its children's committed
//! values only while every child sits at a strictly smaller depth. Fed an array
//! that does not describe the circuit, an internal node and its child land in
//! the same wave, and the two execution models answer differently rather than
//! wrongly: the reference interpreter steps lanes in order and observes the
//! child's new value, a device runs the lanes at once and observes the old one.
//!
//! That was reachable from the registry, not only from a hand-built program:
//! composing `quest_zero_fill` into this operation pipes an all-zero buffer into
//! `depths`, and the fused run produced 5.0 on the reference for a node the
//! device left at 0. No value assertion can bound that, because both answers are
//! correct for their own execution model. So the input is refused instead.
//!
//! # What these tests hold
//!
//! The illegal-depth space is enumerated from the circuit at run time rather
//! than listed: for every edge, the child is placed at its parent's depth and
//! then above it. A depth array is legal exactly while every edge decreases, so
//! those two families plus the collapsed all-zero array are the whole class, and
//! a circuit gaining a node or an edge widens the sweep without an edit here.
//!
//! What these do not hold: the trap says nothing about a `depths` array that is
//! a valid topological order of a DIFFERENT circuit but still decreasing on this
//! one. Such an array evaluates the circuit it was given, in an order that is
//! sound for it, which is the contract.
#![cfg(feature = "graph")]

use vyre_libs::graph::sum_product_circuit::{
    sum_product_depths, sum_product_evaluate_leveled, KIND_LEAF, KIND_PRODUCT, KIND_SUM,
};
use vyre_primitives::wire::pack_u32_slice as pack_u32;
use vyre_reference::value::Value;

/// The tag the guard traps with, as the reference reports it.
const TRAP_TAG: &str = "sum-product-depth-not-topological";

/// Flat CSR circuit: two leaves, a SUM over both, and a PRODUCT over that SUM
/// and a leaf, so the graph carries a genuine depth-2 edge.
struct Circuit {
    kinds: Vec<u32>,
    child_offsets: Vec<u32>,
    child_counts: Vec<u32>,
    children: Vec<u32>,
    weights: Vec<u32>,
    leaf_values: Vec<u32>,
}

impl Circuit {
    fn depth2() -> Self {
        // 0, 1: leaves. 2: SUM(0, 1). 3: PRODUCT(2, 0).
        Self {
            kinds: vec![KIND_LEAF, KIND_LEAF, KIND_SUM, KIND_PRODUCT],
            child_offsets: vec![0, 0, 0, 2],
            child_counts: vec![0, 0, 2, 2],
            children: vec![0, 1, 2, 0],
            weights: vec![1 << 16, 1 << 16, 1 << 16, 1 << 16],
            leaf_values: vec![2 << 16, 3 << 16, 0, 0],
        }
    }

    fn n_nodes(&self) -> u32 {
        u32::try_from(self.kinds.len()).expect("test circuit fits in u32")
    }

    /// Every `(parent, child)` edge the circuit declares.
    fn edges(&self) -> Vec<(usize, usize)> {
        (0..self.kinds.len())
            .flat_map(|parent| {
                let offset = self.child_offsets[parent] as usize;
                let count = self.child_counts[parent] as usize;
                self.children[offset..offset + count]
                    .iter()
                    .map(move |child| (parent, *child as usize))
            })
            .collect()
    }

    fn legal_depths(&self) -> (Vec<u32>, u32) {
        sum_product_depths(
            &self.child_offsets,
            &self.child_counts,
            &self.children,
            self.n_nodes(),
        )
        .expect("the test circuit is acyclic, so it has a topological depth assignment")
    }

    fn run(&self, depths: &[u32], max_depth: u32) -> Result<Vec<u32>, String> {
        let program = sum_product_evaluate_leveled(
            "depths",
            "kinds",
            "child_offsets",
            "child_counts",
            "children",
            "weights",
            "leaf_values",
            "out",
            self.n_nodes(),
            u32::try_from(self.children.len()).expect("edge count fits in u32"),
            max_depth,
        );
        let outputs = vyre_reference::reference_eval(
            &program,
            &[
                Value::from(pack_u32(depths)),
                Value::from(pack_u32(&self.kinds)),
                Value::from(pack_u32(&self.child_offsets)),
                Value::from(pack_u32(&self.child_counts)),
                Value::from(pack_u32(&self.children)),
                Value::from(pack_u32(&self.weights)),
                Value::from(pack_u32(&self.leaf_values)),
                Value::from(pack_u32(&vec![0u32; self.kinds.len()])),
            ],
        )
        .map_err(|error| error.to_string())?;
        Ok(outputs[0]
            .to_bytes()
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
            .collect())
    }
}

/// The contract's own case: a decreasing depth array evaluates the circuit.
///
/// SUM(2.0, 3.0) with unit weights is 5.0, and PRODUCT(5.0, 2.0) is 10.0. If the
/// guard fired here it would have made the operation unusable, which is the
/// failure mode a guard is most likely to have.
#[test]
fn a_topological_depth_array_evaluates_the_circuit() {
    let circuit = Circuit::depth2();
    let (depths, max_depth) = circuit.legal_depths();
    let out = circuit
        .run(&depths, max_depth)
        .expect("a topological depth array must not trap");
    assert_eq!(
        out,
        vec![2 << 16, 3 << 16, 5 << 16, 10 << 16],
        "the depth-2 circuit evaluates bottom-up in 16.16 fixed point"
    );
}

/// Every edge, placed at equal depth and then inverted, must trap.
///
/// The sweep is derived from the circuit, so an added node or edge is covered
/// without a new case here.
#[test]
fn a_child_that_does_not_sit_below_its_parent_traps() {
    let circuit = Circuit::depth2();
    let (legal, max_depth) = circuit.legal_depths();
    let mut judged = 0usize;

    for (parent, child) in circuit.edges() {
        for (label, child_depth) in [
            ("equal", legal[parent]),
            ("inverted", legal[parent].saturating_add(1)),
        ] {
            let mut depths = legal.clone();
            depths[child] = child_depth;
            if depths == legal {
                continue;
            }
            judged += 1;
            let wave_count = max_depth.max(child_depth.saturating_add(1));
            let error = circuit.run(&depths, wave_count).expect_err(&format!(
                "edge {parent} <- {child} at {label} depth must trap, not evaluate"
            ));
            assert!(
                error.contains(TRAP_TAG),
                "edge {parent} <- {child} at {label} depth trapped with the wrong reason: {error}"
            );
        }
    }

    assert!(
        judged > 0,
        "Fix: the circuit declared no edge, so this swept nothing."
    );
}

/// The case composition made reachable: every node collapsed to depth 0.
#[test]
fn a_collapsed_depth_array_traps() {
    let circuit = Circuit::depth2();
    let error = circuit
        .run(&vec![0; circuit.kinds.len()], 1)
        .expect_err("an all-zero depth array must trap");
    assert!(
        error.contains(TRAP_TAG),
        "an all-zero depth array trapped with the wrong reason: {error}"
    );
}

/// A leaf is exempt: it reads `leaf_values`, never `out`, so its child columns
/// are ignored and cannot race. Marking a leaf's children illegal must still
/// evaluate, or the guard would reject circuits whose CSR rows carry unused
/// entries.
#[test]
fn a_leaf_is_not_held_to_the_depth_order() {
    let mut circuit = Circuit::depth2();
    // Give leaf 1 a child pointing at the PRODUCT above it: illegal for an
    // internal node, irrelevant for a leaf.
    circuit.child_offsets[1] = 4;
    circuit.child_counts[1] = 1;
    circuit.children.push(3);
    circuit.weights.push(1 << 16);

    let depths = vec![0, 0, 1, 2];
    let out = circuit
        .run(&depths, 3)
        .expect("a leaf's child columns are not read, so they cannot trap");
    assert_eq!(
        out,
        vec![2 << 16, 3 << 16, 5 << 16, 10 << 16],
        "the circuit still evaluates with an unused child column on a leaf"
    );
}
