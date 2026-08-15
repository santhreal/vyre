//! One fixture per `Node` variant, so a traversal test can be written against
//! the whole enum instead of against the four variants its author remembered.
//!
//! # Why this exists in a shared crate
//!
//! `Node` is `#[non_exhaustive]`. Outside `vyre-foundation` no match over it is
//! exhaustive, so every traversal in every driver, runtime, and lowering crate
//! ends in a catch-all arm. A variant added tomorrow lands in that arm silently:
//! a recursive traversal that treats it as a leaf stops descending, and a
//! barrier, store, or early exit inside the new variant's body reads as ABSENT
//! rather than as unknown. Nothing fails to compile and no test goes red,
//! because every existing test builds its fixtures from the variants that
//! existed when it was written.
//!
//! The fix is a fixture set that is checked against the declaration site.
//! [`node_variant_samples`] must cover every name in
//! `vyre_foundation::ir::NODE_VARIANT_NAMES`, which the registry macro emits
//! from the enum body, and [`assert_covers_every_node_variant`] is the
//! assertion that says so. A new variant therefore turns every traversal suite
//! that uses these fixtures RED until somebody adds a fixture for it, and
//! adding the fixture forces a decision about what each traversal should do
//! with it.
//!
//! # What a traversal test gets
//!
//! - [`node_variant_samples`] enumerates every variant once, bare.
//! - [`node_body_slot_samples`] enumerates every *body slot* of every
//!   body-carrying variant with a marker node planted in it, and only in it, so
//!   a probe that misses one slot is distinguishable from one that misses the
//!   variant.
//! - [`node_operand_samples`] does the same for operand-carrying variants with
//!   a marker expression.

use std::sync::Arc;

use vyre_foundation::ir::model::expr::Ident;
use vyre_foundation::ir::{
    BufferDecl, CollectiveOp, CommGroup, DataType, Expr, Node, NodeExtension, Program,
    NODE_VARIANT_NAMES,
};
use vyre_foundation::transform::visit::node_shape;
use vyre_foundation::MemoryOrdering;

/// One `Node` fixture together with what was planted in it.
#[derive(Debug, Clone)]
pub struct NodeSample {
    /// Declared variant name, matching `vyre_foundation::ir::NODE_VARIANT_NAMES`.
    pub variant: &'static str,
    /// Which slot of the variant the marker was planted in.
    ///
    /// `Node::If` owns two node bodies and one operand, so it contributes three
    /// samples with slots `"then"`, `"otherwise"`, and `"cond"`. A sample with
    /// no marker planted carries `None`.
    pub slot: Option<&'static str>,
    /// The fixture.
    pub node: Node,
}

impl NodeSample {
    /// Human-readable identity for an assertion message.
    #[must_use]
    pub fn label(&self) -> String {
        match self.slot {
            Some(slot) => format!("Node::{}.{slot}", self.variant),
            None => format!("Node::{}", self.variant),
        }
    }
}

/// A statement-node extension payload with no reachable children.
///
/// The `Opaque` fixture needs a concrete payload, and core cannot look inside
/// one. That is the point of the variant, and it is why [`node_shape`] reports
/// it as opaque rather than as a leaf: an analysis must answer "unknown", not
/// "nothing here".
#[derive(Debug)]
struct FixtureExtension;

impl NodeExtension for FixtureExtension {
    fn extension_kind(&self) -> &'static str {
        "vyre.test_support.fixture_node"
    }

    fn debug_identity(&self) -> &str {
        "fixture-node"
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        [0x5a; 32]
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn sample(variant: &'static str, slot: Option<&'static str>, node: Node) -> NodeSample {
    NodeSample {
        variant,
        slot,
        node,
    }
}

/// Every declared `Node` variant exactly once, with no marker planted.
///
/// Body-carrying variants come back with empty bodies and operand-carrying
/// variants with placeholder operands, so this set is the one to use when the
/// question is "does this code handle the variant at all".
#[must_use]
pub fn node_variant_samples() -> Vec<NodeSample> {
    let mut out: Vec<NodeSample> = Vec::new();
    for mut candidate in body_slot_samples(&[])
        .into_iter()
        .chain(operand_samples(&Expr::u32(0)))
        .chain(inert_samples())
    {
        if out
            .iter()
            .any(|existing| existing.variant == candidate.variant)
        {
            continue;
        }
        candidate.slot = None;
        out.push(candidate);
    }
    out
}

/// Every body slot of every body-carrying variant, with `marker` planted in
/// that slot and nothing in the others.
///
/// A recursive traversal must find `marker` through every one of these. If it
/// misses one, the subtree under that slot is never visited.
#[must_use]
pub fn node_body_slot_samples(marker: &Node) -> Vec<NodeSample> {
    body_slot_samples(std::slice::from_ref(marker))
}

/// Every operand slot of every operand-carrying variant, with `marker` planted
/// in that slot alone.
#[must_use]
pub fn node_operand_samples(marker: &Expr) -> Vec<NodeSample> {
    operand_samples(marker)
}

/// A program whose only buffer is one `u32` output of four elements.
///
/// This is the smallest program a validation or optimization test can build
/// that still has somewhere to store a result. Every suite that spelled it
/// locally agreed on the shape, so the shape is stated once here.
#[must_use]
pub fn single_u32_output_program(nodes: Vec<Node>) -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(4)],
        [1, 1, 1],
        nodes,
    )
}

fn body_slot_samples(marker: &[Node]) -> Vec<NodeSample> {
    let body = marker.to_vec();
    vec![
        sample(
            "If",
            Some("then"),
            Node::If {
                cond: Expr::bool(true),
                then: body.clone(),
                otherwise: Vec::new(),
            },
        ),
        sample(
            "If",
            Some("otherwise"),
            Node::If {
                cond: Expr::bool(true),
                then: Vec::new(),
                otherwise: body.clone(),
            },
        ),
        sample(
            "Loop",
            Some("body"),
            Node::loop_for("fixture_i", Expr::u32(0), Expr::u32(1), body.clone()),
        ),
        sample("Block", Some("body"), Node::Block(body.clone())),
        sample(
            "Region",
            Some("body"),
            Node::Region {
                generator: Ident::from("vyre.test_support.fixture_region"),
                source_region: None,
                body: Arc::new(body),
            },
        ),
    ]
}

fn operand_samples(marker: &Expr) -> Vec<NodeSample> {
    vec![
        sample(
            "Let",
            Some("value"),
            Node::let_bind("fixture_v", marker.clone()),
        ),
        sample(
            "Assign",
            Some("value"),
            Node::assign("fixture_v", marker.clone()),
        ),
        sample(
            "Store",
            Some("value"),
            Node::store("fixture_buffer", Expr::u32(0), marker.clone()),
        ),
        sample(
            "If",
            Some("cond"),
            Node::If {
                cond: marker.clone(),
                then: Vec::new(),
                otherwise: Vec::new(),
            },
        ),
        sample(
            "Loop",
            Some("to"),
            Node::loop_for("fixture_i", Expr::u32(0), marker.clone(), Vec::new()),
        ),
        sample(
            "AsyncLoad",
            Some("offset"),
            Node::AsyncLoad {
                source: Ident::from("fixture_src"),
                destination: Ident::from("fixture_dst"),
                offset: Box::new(marker.clone()),
                size: Box::new(Expr::u32(4)),
                tag: Ident::from("fixture_tag"),
            },
        ),
        sample(
            "AsyncStore",
            Some("offset"),
            Node::AsyncStore {
                source: Ident::from("fixture_src"),
                destination: Ident::from("fixture_dst"),
                offset: Box::new(marker.clone()),
                size: Box::new(Expr::u32(4)),
                tag: Ident::from("fixture_tag"),
            },
        ),
        sample(
            "Trap",
            Some("address"),
            Node::trap(marker.clone(), "fixture_trap"),
        ),
    ]
}

fn inert_samples() -> Vec<NodeSample> {
    vec![
        sample("Return", None, Node::Return),
        sample(
            "Barrier",
            None,
            Node::barrier_with_ordering(MemoryOrdering::SeqCst),
        ),
        sample(
            "IndirectDispatch",
            None,
            Node::indirect_dispatch("fixture_counts", 0),
        ),
        sample("AsyncWait", None, Node::async_wait("fixture_tag")),
        sample("Resume", None, Node::resume("fixture_tag")),
        sample(
            "AllReduce",
            None,
            Node::AllReduce {
                buffer: Ident::from("fixture_buffer"),
                op: CollectiveOp::Sum,
                group: CommGroup(0),
            },
        ),
        sample(
            "AllGather",
            None,
            Node::AllGather {
                input: Ident::from("fixture_in"),
                output: Ident::from("fixture_out"),
                group: CommGroup(0),
            },
        ),
        sample(
            "ReduceScatter",
            None,
            Node::ReduceScatter {
                input: Ident::from("fixture_in"),
                output: Ident::from("fixture_out"),
                op: CollectiveOp::Sum,
                group: CommGroup(0),
            },
        ),
        sample(
            "Broadcast",
            None,
            Node::Broadcast {
                buffer: Ident::from("fixture_buffer"),
                root: 0,
                group: CommGroup(0),
            },
        ),
        sample("Opaque", None, Node::opaque(FixtureExtension)),
    ]
}

/// Panic unless `samples` names every declared `Node` variant.
///
/// This is the run-time half of the closure. The compile-time half is
/// [`node_shape`], whose match has no catch-all arm; together they mean a new
/// `Node` variant cannot reach a traversal without somebody recording what the
/// traversal should do with it.
///
/// # Panics
///
/// When a declared variant has no sample, or a sample names a variant that is
/// not declared.
pub fn assert_covers_every_node_variant(samples: &[NodeSample]) {
    let declared: std::collections::BTreeSet<&str> = NODE_VARIANT_NAMES.iter().copied().collect();
    let covered: std::collections::BTreeSet<&str> =
        samples.iter().map(|sample| sample.variant).collect();

    let missing: Vec<&str> = declared.difference(&covered).copied().collect();
    assert!(
        missing.is_empty(),
        "no Node fixture for declared variant(s) {missing:?}. Fix: add each one to \
         vyre_test_support::ir_variants, then decide for every traversal that consumes these \
         fixtures whether it must descend into the variant, treat it as unknown, or ignore it. \
         Skipping this leaves the variant handled by a catch-all arm nobody chose."
    );

    let undeclared: Vec<&str> = covered.difference(&declared).copied().collect();
    assert!(
        undeclared.is_empty(),
        "Node fixture(s) {undeclared:?} name variants that no longer exist. Fix: delete the \
         stale fixtures so the coverage assertion keeps meaning what it says."
    );
}

/// Panic unless every sample's planted slot agrees with [`node_shape`].
///
/// Keeps the fixtures honest: a body sample whose variant does not nest nodes,
/// or an operand sample whose variant owns no operands, would make a traversal
/// test pass for the wrong reason.
///
/// # Panics
///
/// When a sample's shape contradicts the slot it claims to have planted.
pub fn assert_samples_match_declared_shape(samples: &[NodeSample], expect_bodies: bool) {
    for sample in samples {
        let shape = node_shape(&sample.node);
        if expect_bodies {
            assert!(
                shape.nests_nodes,
                "{} carries a planted body but node_shape says it nests nothing",
                sample.label()
            );
        } else {
            assert!(
                shape.carries_operands,
                "{} carries a planted operand but node_shape says it owns none",
                sample.label()
            );
        }
    }
}
