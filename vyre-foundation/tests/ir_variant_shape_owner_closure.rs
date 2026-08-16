//! The class closed here: a second, hand-written enumeration of `Node`'s shape
//! that disagrees with the owning one, and a descent that treats a variant it
//! was never told about as a leaf.
//!
//! # What used to stand here
//!
//! `Node`'s shape was re-stated once per traversal direction and once per
//! consumer. `visit::node_map::map_body` listed `If`/`Loop`/`Block`/`Region`
//! and ended in `other => other`, so a body-bearing variant that list had not
//! been told about was handed back UNCHANGED and every pass composed on it
//! (`rematerialize_cheap_let`, the pass engine's constant propagation) reported
//! success while doing nothing inside that variant. `visit::bound_names` ended
//! in `_ => {}`, so a new binding form would have read as binding nothing and a
//! scope pass would have hoisted across a live rebinding.
//! `optimizer::cost::count_divergent_patterns` was checked against a
//! hand-copied twin walker in its own test module, which is not a check: two
//! copies of the same omission agree.
//!
//! # The property that replaced it
//!
//! Three owners, one per Rust reference mode, and nothing else may enumerate
//! the shape:
//!
//! - `visit::child_bodies` for a shared read,
//! - `visit::child_bodies_mut` for a move or an in-place mutation,
//! - `transform::rewrite_walk::rewrite_node` for a borrow-preserving rebuild.
//!
//! Each is an independent exhaustive match, so this suite holds them to each
//! other on arity, source order, and contents; a position added to one and
//! forgotten in another is visible here and nowhere else. The scalar namespace
//! has one owner, `visit::node_scalars`, held to `node_shape` and to
//! the rewriting walk the same way. The consumers that used to carry their own
//! list (`map_body`, `walk_nodes_mut`, and the cost fold behind
//! `CostCertificate::for_program`) are each required to reach a marker planted
//! in every body slot.
//!
//! # Why it fails by default on a new variant
//!
//! The member set is not written here. It comes from
//! `vyre_test_support::ir_variants`, whose fixtures are checked against
//! `vyre_foundation::ir::NODE_VARIANT_NAMES`, which the registry macro emits
//! from the enum body. Adding a `Node` variant turns this suite RED until
//! somebody adds a fixture, and adding the fixture forces a decision about
//! every owner and every consumer above.
//!
//! # The other way a second enumeration gets written
//!
//! Nothing above stops a crate from restating `Node` in a shape the three
//! owners cannot serve. `visit::NodeVisitor` is abstract-by-default: it
//! declares one hook per declared variant and gives almost none of them a
//! default, so implementing it to answer a question about ONE variant costs a
//! no-op body, with its full signature, for the other fifteen. That block of
//! stubs is a hand enumeration of `Node` living outside this crate, where
//! `Node` is `#[non_exhaustive]` and the compiler cannot tell its author that
//! it went stale. Four scanners in this workspace had copied it.
//!
//! So the implementors are enumerated from workspace source at run time and
//! held to a recorded list, and the trait's hooks are derived from
//! `NODE_VARIANT_NAMES`. A new implementor anywhere turns this RED, and so
//! does a new `Node` variant that no hook covers.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use vyre_foundation::ir::MemoryOrdering;
use vyre_foundation::ir::{
    BinOp, BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program, NODE_VARIANT_NAMES,
};
use vyre_foundation::optimizer::cost::CostCertificate;
use vyre_foundation::optimizer::{registered_pass_registrations, ProgramPass};
use vyre_foundation::transform::rewrite_walk::{self, NodeRewrite};
use vyre_foundation::visit::node_map::map_body;
use vyre_foundation::visit::{
    child_bodies, child_bodies_mut, for_each_expr, node_scalars, node_shape, node_tag,
    node_variadic_operands, walk_nodes_mut,
};
use vyre_test_support::ir_variants::{
    assert_covers_every_node_variant, assert_samples_match_declared_shape, node_body_slot_samples,
    node_operand_samples, node_variant_samples, NodeSample,
};

/// The node planted in one body slot at a time.
fn body_marker() -> Node {
    Node::barrier_with_ordering(MemoryOrdering::SeqCst)
}

/// The expression planted in one operand slot at a time.
fn operand_marker() -> Expr {
    Expr::var("vyre_shape_owner_marker")
}

/// Bare variants, plus one fixture per body slot and one per operand slot.
fn every_fixture() -> Vec<NodeSample> {
    let mut all = node_variant_samples();
    all.extend(node_body_slot_samples(&body_marker()));
    all.extend(node_operand_samples(&operand_marker()));
    all
}

/// A policy that changes nothing and does not descend, so what it records is
/// the positions of ONE node rather than of its whole subtree.
#[derive(Default)]
struct ObserveShallow {
    bodies: Vec<Vec<Node>>,
    operands: Vec<Expr>,
    bindings: Vec<Ident>,
    tags: Vec<Ident>,
}

impl NodeRewrite for ObserveShallow {
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        self.operands.push(expr.clone());
        None
    }

    fn binding(&mut self, name: &Ident) -> Option<Ident> {
        self.bindings.push(name.clone());
        None
    }

    fn tag(&mut self, name: &Ident) -> Option<Ident> {
        self.tags.push(name.clone());
        None
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        self.bodies.push(body.to_vec());
        None
    }
}

/// A policy that renames one identifier wherever a value position offers it.
struct RenameIdent {
    from: Ident,
    to: Ident,
}

impl NodeRewrite for RenameIdent {
    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        None
    }

    fn binding(&mut self, name: &Ident) -> Option<Ident> {
        (name == &self.from).then(|| self.to.clone())
    }
}

/// A policy that renames one identifier wherever a tag position offers it.
struct RenameTag {
    from: Ident,
    to: Ident,
}

impl NodeRewrite for RenameTag {
    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        None
    }

    fn tag(&mut self, name: &Ident) -> Option<Ident> {
        (name == &self.from).then(|| self.to.clone())
    }
}

fn program_of(nodes: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], nodes)
}

/// `if invocation_id.x == 0 { buf[0] = 1 }`: the one shape the divergence
/// dimension of the cost certificate scores.
fn divergent_branch() -> Node {
    Node::if_then(
        Expr::BinOp {
            op: BinOp::Eq,
            left: Box::new(Expr::gid_x()),
            right: Box::new(Expr::u32(0)),
        },
        vec![Node::store("buf", Expr::u32(0), Expr::u32(1))],
    )
}

/// The fixtures name every declared variant, and each one plants its marker in
/// a slot the declared shape says exists.
///
/// Without this the rest of the suite could pass by testing a subset: a
/// traversal that misses a variant no fixture covers is invisible.
#[test]
fn the_fixture_set_covers_every_declared_node_variant() {
    assert_covers_every_node_variant(&node_variant_samples());
    assert_samples_match_declared_shape(&node_body_slot_samples(&body_marker()), true);
    assert_samples_match_declared_shape(&node_operand_samples(&operand_marker()), false);
}

/// The three reference modes agree on which body slots a variant has, in which
/// order, holding which nodes.
///
/// `child_bodies` pads its answer to two groups so a caller can flatten
/// unconditionally; `child_bodies_mut` and the rewriting walk offer only the
/// real slots. The comparison is therefore arity and contents against the
/// rewriting walk, and flattened contents against `child_bodies`. A slot added
/// to one match and forgotten in another is a body a moving map never takes
/// out, which is the `other => other` defect wearing different clothes.
#[test]
fn every_reference_mode_reports_the_same_body_slots() {
    for sample in every_fixture() {
        let mut observed = ObserveShallow::default();
        rewrite_walk::rewrite_node(&sample.node, &mut observed);

        let mut owned = sample.node.clone();
        let moving: Vec<Vec<Node>> = child_bodies_mut(&mut owned)
            .into_iter()
            .map(|slot| slot.clone())
            .collect();

        assert_eq!(
            moving,
            observed.bodies,
            "Fix: child_bodies_mut and rewrite_node disagree about the body \
             slots of {}; a body one owner cannot reach is a body a pass \
             silently declines to rewrite",
            sample.label()
        );

        let reading: Vec<&Node> = child_bodies(&sample.node).into_iter().flatten().collect();
        let flattened: Vec<&Node> = moving.iter().flatten().collect();
        assert_eq!(
            flattened,
            reading,
            "Fix: child_bodies and child_bodies_mut disagree about the contents \
             of {}",
            sample.label()
        );

        assert_eq!(
            node_shape(&sample.node).nests_nodes,
            !moving.is_empty(),
            "Fix: node_shape and child_bodies_mut disagree about whether {} \
             nests statements",
            sample.label()
        );
    }
}

/// `map_body` offers the caller every body slot the moving owner reports, and
/// puts each result back in the slot it came from.
///
/// This is the defect itself: `map_body` used to end in `other => other`, so a
/// slot it did not name was never offered and the caller's transform never ran
/// inside it, with no error and no observable difference from a transform that
/// legitimately changed nothing.
#[test]
fn map_body_offers_and_restores_every_body_slot() {
    for sample in every_fixture() {
        let mut owned = sample.node.clone();
        let expected = child_bodies_mut(&mut owned).len();

        let mut offered = 0usize;
        let mapped = map_body(sample.node.clone(), &mut |body| {
            offered += 1;
            let mut body = body;
            body.push(Node::Return);
            body
        });

        assert_eq!(
            offered,
            expected,
            "Fix: map_body skipped a body slot of {}; a pass composed on it is \
             a silent no-op inside that slot",
            sample.label()
        );

        let mut mapped_owned = mapped;
        let after: Vec<Vec<Node>> = child_bodies_mut(&mut mapped_owned)
            .into_iter()
            .map(|slot| slot.clone())
            .collect();
        for (index, slot) in after.iter().enumerate() {
            assert_eq!(
                slot.last(),
                Some(&Node::Return),
                "Fix: map_body dropped the rewritten body of slot {index} of {}",
                sample.label()
            );
        }
    }
}

/// An identity map through `map_body` is an identity on the node.
///
/// Pairs with the test above: a `map_body` that offered every slot but wrote
/// the results back into the wrong ones would still count correctly.
#[test]
fn map_body_with_an_identity_transform_changes_nothing() {
    for sample in every_fixture() {
        let mapped = map_body(sample.node.clone(), &mut |body| body);
        assert_eq!(
            mapped,
            sample.node,
            "Fix: map_body altered {} while applying an identity transform",
            sample.label()
        );
    }
}

/// The in-place walk reaches a node planted in every body slot.
///
/// `walk_nodes_mut` drives every mutating analysis that rewrites nodes in
/// place. A slot it does not descend into is a subtree those analyses believe
/// is empty.
#[test]
fn walk_nodes_mut_reaches_every_body_slot() {
    let marker = body_marker();
    for sample in node_body_slot_samples(&marker) {
        let mut program = program_of(vec![sample.node.clone()]);
        let mut seen = 0usize;
        walk_nodes_mut(&mut program, |node| {
            if *node == marker {
                seen += 1;
            }
        });
        assert_eq!(
            seen,
            1,
            "Fix: walk_nodes_mut never reached the node planted in {}",
            sample.label()
        );
    }
}

/// The cost fold reaches a divergent branch planted in every body slot.
///
/// `CostCertificate` is the monotone-down post-condition gate: a dimension the
/// fold cannot see is a dimension a pass is free to raise. A fold that stops at
/// a variant it does not recognise reports a lower score than the program has,
/// and the gate then accepts the rewrite it exists to refuse.
#[test]
fn the_cost_fold_scores_a_divergent_branch_in_every_body_slot() {
    for sample in node_body_slot_samples(&divergent_branch()) {
        let certificate = CostCertificate::for_program(&program_of(vec![sample.node.clone()]));
        assert!(
            certificate.divergence_score >= 1,
            "Fix: the divergence fold never reached the branch planted in {}; \
             the cost gate cannot refuse a rewrite that raises a dimension it \
             cannot see",
            sample.label()
        );
    }
}

/// A body slot with nothing divergent in it scores zero, so the test above
/// cannot pass by counting everything.
#[test]
fn the_cost_fold_scores_nothing_for_a_non_divergent_branch() {
    let benign = Node::if_then(
        Expr::BinOp {
            op: BinOp::Lt,
            left: Box::new(Expr::var("a")),
            right: Box::new(Expr::var("b")),
        },
        vec![Node::store("buf", Expr::u32(0), Expr::u32(1))],
    );
    for sample in node_body_slot_samples(&benign) {
        let certificate = CostCertificate::for_program(&program_of(vec![sample.node.clone()]));
        assert_eq!(
            certificate.divergence_score,
            0,
            "Fix: the divergence fold counted a comparison that is not \
             invocation-id divergence in {}",
            sample.label()
        );
    }
}

/// `node_scalars` reports exactly the operand expressions the rewriting walk
/// offers, in the same order.
///
/// Two independent exhaustive matches over the same enum. An operand position
/// present in one and absent from the other is either an expression a scan
/// never examines or an expression a substitution never rewrites, and the two
/// halves of the IR then hold different values for the same variable.
#[test]
fn node_scalars_reports_every_operand_the_rewriting_walk_offers() {
    for sample in every_fixture() {
        let mut observed = ObserveShallow::default();
        rewrite_walk::rewrite_node(&sample.node, &mut observed);

        let mut scalars: Vec<Expr> = node_scalars(&sample.node)
            .operands
            .into_iter()
            .flatten()
            .cloned()
            .collect();
        scalars.extend(node_variadic_operands(&sample.node).iter().cloned());

        assert_eq!(
            scalars,
            observed.operands,
            "Fix: node_scalars and rewrite_node disagree about the operands of \
             {}",
            sample.label()
        );

        assert_eq!(
            node_shape(&sample.node).carries_operands,
            !scalars.is_empty(),
            "Fix: node_shape and node_scalars disagree about whether {} \
             carries operands",
            sample.label()
        );
    }
}

/// The name `node_scalars` reports as bound is the one the rewriting walk
/// offers to its VALUE hook, and never to its tag hook.
///
/// The scope passes read the binding through `node_scalars` and the renaming
/// passes write it through `rewrite_node`. If the two point at different
/// fields of a variant, a rename leaves a scope pass looking at the old name
/// and the pass extends a scope across a binding that is no longer there.
#[test]
fn the_reported_binding_is_the_ident_the_rewriting_walk_renames() {
    let renamed = Ident::new("vyre_shape_owner_renamed".into());
    for sample in every_fixture() {
        let Some((binding, name)) = node_scalars(&sample.node).binding else {
            continue;
        };

        let mut observed = ObserveShallow::default();
        rewrite_walk::rewrite_node(&sample.node, &mut observed);
        assert!(
            observed.bindings.contains(name),
            "Fix: {} binds `{name}` as {binding:?}, but the rewriting walk \
             never offers that identifier for renaming",
            sample.label()
        );
        assert!(
            !observed.tags.contains(name),
            "Fix: {} binds `{name}` as {binding:?} in the value namespace, and \
             the rewriting walk offers it to the tag hook. A value renamer \
             would then rename a transfer tag.",
            sample.label()
        );

        let mut rename = RenameIdent {
            from: name.clone(),
            to: renamed.clone(),
        };
        let rewritten =
            rewrite_walk::rewrite_node(&sample.node, &mut rename).unwrap_or_else(|| {
                panic!(
                    "Fix: renaming the binding of {} changed nothing",
                    sample.label()
                )
            });

        assert_eq!(
            node_scalars(&rewritten).binding.map(|(_, name)| name),
            Some(&renamed),
            "Fix: the rewriting walk and node_scalars name different fields of \
             {} as its binding",
            sample.label()
        );
    }
}

/// The tag `node_tag` reports is the one the rewriting walk offers to its TAG
/// hook, and never to its value hook.
///
/// WHY: a tag names an in-flight transfer, and `validate::async_pipeline`
/// pairs a start with the wait carrying the same tag. One hook used to be
/// offered both namespaces, so a pass renaming an induction variable renamed
/// any tag that spelled the same name and separated a start from its wait,
/// while `transform::inline` avoided that only by re-deriving which position
/// it had been called for. This fails for a tag-bearing variant wired to the
/// wrong hook, and for one `node_tag` has been told about that the walk has
/// not.
#[test]
fn the_reported_tag_is_the_ident_the_rewriting_walk_renames() {
    let renamed = Ident::new("vyre_shape_owner_retagged".into());
    for sample in every_fixture() {
        let Some(tag) = node_tag(&sample.node) else {
            continue;
        };

        let mut observed = ObserveShallow::default();
        rewrite_walk::rewrite_node(&sample.node, &mut observed);
        assert!(
            observed.tags.contains(tag),
            "Fix: {} carries tag `{tag}`, but the rewriting walk never offers \
             it for renaming",
            sample.label()
        );
        assert!(
            !observed.bindings.contains(tag),
            "Fix: {} carries `{tag}` as a stream tag, and the rewriting walk \
             offers it to the value hook. A pass renaming a variable would \
             then rename one end of a transfer pair.",
            sample.label()
        );

        let mut rename = RenameTag {
            from: tag.clone(),
            to: renamed.clone(),
        };
        let rewritten =
            rewrite_walk::rewrite_node(&sample.node, &mut rename).unwrap_or_else(|| {
                panic!("Fix: renaming the tag of {} changed nothing", sample.label())
            });

        assert_eq!(
            node_tag(&rewritten),
            Some(&renamed),
            "Fix: the rewriting walk and node_tag name different fields of {} \
             as its tag",
            sample.label()
        );
    }
}

/// The literal the propagating pass substitutes for [`propagation_marker`].
const PROPAGATED: u32 = 7;

/// The name the optimizer probe plants in one operand slot at a time.
///
/// It is a `Var` rather than the shared [`operand_marker`] because the property
/// under test is what the optimizer's expression rewrite REACHES, and a read of
/// a uniquely-bound literal is the cheapest thing every backend-neutral rewrite
/// is obliged to fold.
fn propagation_marker() -> Expr {
    Expr::var("vyre_optimizer_reach_marker")
}

/// The buffers the operand fixtures name, so a probe program is well formed.
fn fixture_buffers() -> Vec<BufferDecl> {
    ["fixture_buffer", "fixture_src", "fixture_dst"]
        .into_iter()
        .enumerate()
        .map(|(binding, name)| {
            let binding = u32::try_from(binding).expect("three fixture buffers fit in u32");
            BufferDecl::storage(name, binding, BufferAccess::ReadWrite, DataType::U32).with_count(4)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The hand-enumeration owner: `visit::NodeVisitor` and who implements it.
// ---------------------------------------------------------------------------

/// Every implementation of `visit::NodeVisitor` in this workspace, with why it
/// still writes the enum out by hand.
///
/// A type here restates one hook per `Node` variant. That is a second
/// enumeration of the enum, and outside `vyre-foundation` `Node` is
/// `#[non_exhaustive]`, so nothing tells its author when a variant is added:
/// the new variant simply never reaches whatever the visitor was counting.
///
/// A scan that wants one or two variants and descent for the rest wants
/// `visit::try_for_each_node`, which takes a closure and gets its
/// descent from `child_bodies`. Adding an implementation without adding it here
/// is the failure this list exists to force.
const RECORDED_NODE_VISITORS: &[(&str, &str)] = &[
    (
        "PreorderValidator",
        "The validation rule pipeline dispatches a different rule set per \
         variant, so the per-variant hook IS its work rather than boilerplate \
         around it. It lives in this crate, where the trait's hook set is a \
         compile error to leave incomplete.",
    ),
    (
        "CountingNodeVisitor",
        "The trait's own test. It exists to prove dispatch reaches every hook, \
         so it must implement every hook.",
    ),
    (
        "AsyncResumeRejector",
        "vyre-emit-naga: rejects two variants and ignores the rest. Wants \
         try_for_each_node; not yet routed.",
    ),
    (
        "TrapTagCollector",
        "vyre-emit-naga: collects one variant's tag and ignores the rest. \
         Wants try_for_each_node; not yet routed.",
    ),
    (
        "LocalSlots",
        "vyre-reference: collects declarations from three variants and ignores \
         the rest. Wants try_for_each_node; not yet routed.",
    ),
];

/// `Node` variants whose visitor hook is not `visit_<snake_case_variant>`, and
/// why.
///
/// Recorded rather than special-cased in the derivation, so the mapping is
/// visible and a new exception has to be written down.
const HOOK_NAME_EXCEPTIONS: &[(&str, &str, &str)] = &[
    (
        "AllReduce",
        "visit_collective",
        "The four collectives carry the same shape and dispatch to one hook.",
    ),
    ("AllGather", "visit_collective", "See AllReduce."),
    ("ReduceScatter", "visit_collective", "See AllReduce."),
    ("Broadcast", "visit_collective", "See AllReduce."),
    (
        "TileLoad",
        "visit_tile",
        "All six tile variants carry the same dispatch shape and fold into one hook.",
    ),
    ("TileStore", "visit_tile", "See TileLoad."),
    ("TileMatmul", "visit_tile", "See TileLoad."),
    ("TileReduce", "visit_tile", "See TileLoad."),
    ("TileElementwise", "visit_tile", "See TileLoad."),
    ("TileDecl", "visit_tile", "See TileLoad."),
    (
        "Opaque",
        "visit_opaque_node",
        "Named for what it hands over, a `&dyn NodeExtension`, not for the \
         variant: `visit_opaque` would read as an opaque expression.",
    ),
];

/// The hook name `visit::NodeVisitor` owes `variant`.
fn hook_for_variant(variant: &str) -> String {
    if let Some((_, hook, _)) = HOOK_NAME_EXCEPTIONS
        .iter()
        .find(|(name, _, _)| *name == variant)
    {
        return (*hook).to_string();
    }
    let mut hook = String::from("visit");
    for (index, ch) in variant.char_indices() {
        if ch.is_ascii_uppercase() && index > 0 {
            hook.push('_');
        } else if index == 0 {
            hook.push('_');
        }
        hook.push(ch.to_ascii_lowercase());
    }
    hook
}

/// The hook names the trait declares, read from its declaration.
fn declared_visitor_hooks() -> BTreeSet<String> {
    let source = read_workspace_file("vyre-foundation/src/visit/node_visitor.rs");
    let body = vyre_test_support::braced_body(&source, "pub trait NodeVisitor {").expect(
        "Fix: no `pub trait NodeVisitor` in vyre-foundation/src/visit/node_visitor.rs; this scan \
         is reading the wrong file",
    );
    body.match_indices("fn visit")
        .map(|(offset, _)| {
            let start = offset + "fn ".len();
            let end = body[start..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .map_or(body.len(), |len| start + len);
            body[start..end].to_string()
        })
        .collect()
}

/// The registered pass whose whole job is to substitute an expression at every
/// read site, instantiated from the registry.
///
/// Read by name from the inventory registry rather than constructed here, so a
/// pass that is renamed or unregistered fails this gate loudly instead of
/// leaving it asserting nothing.
fn literal_propagating_pass() -> Box<dyn ProgramPass> {
    const NAME: &str = "reaching_def_propagate";
    let registrations =
        registered_pass_registrations().expect("the registered pass graph must schedule");
    let registration = registrations
        .iter()
        .find(|registration| registration.metadata.name == NAME)
        .unwrap_or_else(|| {
            panic!(
                "no registered pass named `{NAME}`; this gate reaches the optimizer's \
                 expression rewrite through it, so a rename must be followed here"
            )
        });
    (registration.factory)()
}

/// True iff `program` still mentions [`propagation_marker`] anywhere.
fn marker_survives(program: &Program) -> bool {
    let Expr::Var(marker) = propagation_marker() else {
        unreachable!("the propagation marker is a Var by construction");
    };
    let mut found = false;
    for_each_expr(program.entry(), |expr| {
        if matches!(expr, Expr::Var(name) if *name == marker) {
            found = true;
        }
    });
    found
}

/// The optimizer's expression rewrite reaches every operand slot the rewriting
/// owner offers, measured through a registered pass rather than through the
/// walk itself.
///
/// `optimizer::rewrite` used to carry its own exhaustive enumeration of `Node`
/// beside `rewrite_walk::rewrite_node`, and the pair had diverged: the owner
/// descended into an async copy's `offset` and `size` and the optimizer copy did
/// not, so every pass routed through `rewrite_program` left those two
/// expression positions unrewritten. Nothing in this suite could see it, because
/// the optimizer path was outside the closure it checks.
///
/// The slot set is [`node_operand_samples`], derived from the variant registry,
/// so a `Node` variant that gains an operand fails
/// [`the_fixture_set_covers_every_declared_node_variant`] first and then fails
/// here until the optimizer reaches it.
#[test]
fn the_optimizer_expression_rewrite_reaches_every_operand_slot() {
    let pass = literal_propagating_pass();
    for sample in node_operand_samples(&propagation_marker()) {
        let Expr::Var(marker) = propagation_marker() else {
            unreachable!("the propagation marker is a Var by construction");
        };
        let program = Program::wrapped(
            fixture_buffers(),
            [1, 1, 1],
            vec![
                Node::let_bind(marker.as_ref(), Expr::u32(PROPAGATED)),
                sample.node.clone(),
            ],
        );
        assert!(
            marker_survives(&program),
            "{}: the probe program must contain the marker before the pass runs, \
             or this case proves nothing",
            sample.label()
        );
        let rewritten = pass
            .batch_apply(
                program,
                &vyre_foundation::optimizer::AdapterCaps::conservative(),
            )
            .program;
        assert!(
            !marker_survives(&rewritten),
            "{}: the optimizer's expression rewrite did not reach this operand slot, \
             so every pass routed through it silently leaves the position alone",
            sample.label()
        );
    }
}

/// The trait declares exactly one hook per declared `Node` variant.
///
/// This is the run-time half of the trait's contract. `dispatch_node` is
/// exhaustive, so a new variant is a compile error THERE, but the error is
/// solved just as easily by routing the variant into an existing hook as by
/// giving it one, and a variant folded into somebody else's hook is invisible
/// to every implementor. Here the fold has to be written into
/// `HOOK_NAME_EXCEPTIONS` with a reason.
#[test]
fn the_node_visitor_trait_declares_one_hook_per_declared_variant() {
    let declared: BTreeSet<String> = NODE_VARIANT_NAMES
        .iter()
        .map(|variant| hook_for_variant(variant))
        .collect();
    let hooks = declared_visitor_hooks();

    assert!(
        hooks.len() >= 10,
        "Fix: the NodeVisitor hook scan found only {} hooks; it is reading the wrong region, \
         not looking at a tiny trait",
        hooks.len()
    );

    let unhooked: Vec<&String> = declared.difference(&hooks).collect();
    assert!(
        unhooked.is_empty(),
        "Fix: visit::NodeVisitor declares no hook named {unhooked:?}. Either add the hook, or \
         record in HOOK_NAME_EXCEPTIONS which existing hook the variant folds into and why; a \
         variant folded in silently is one no implementor is asked about."
    );

    let orphan: Vec<&String> = hooks
        .difference(&declared)
        .filter(|hook| hook.as_str() != "visit_node")
        .collect();
    assert!(
        orphan.is_empty(),
        "Fix: visit::NodeVisitor declares {orphan:?}, which no declared Node variant reaches. \
         Delete the hook or record the variant it serves."
    );

    let stale: Vec<&&str> = HOOK_NAME_EXCEPTIONS
        .iter()
        .map(|(variant, _, _)| variant)
        .filter(|variant| !NODE_VARIANT_NAMES.contains(variant))
        .collect();
    assert!(
        stale.is_empty(),
        "Fix: HOOK_NAME_EXCEPTIONS names variants Node no longer declares: {stale:?}"
    );
}

/// The types implementing `visit::NodeVisitor` are exactly the recorded ones.
///
/// The set is read from workspace source on each run rather than listed twice,
/// so a fifth walker landing in any crate turns this RED. It is the only gate
/// that can: outside this crate an implementation compiles cleanly forever, and
/// the stub bodies it copies are indistinguishable from deliberate no-ops.
#[test]
fn every_node_visitor_implementation_in_the_workspace_is_recorded() {
    let hooks = declared_visitor_hooks();
    let found = scan_node_visitor_implementations(&hooks);

    assert!(
        found.len() >= 2,
        "Fix: the implementor scan found only {} NodeVisitor implementations; the trait's own \
         test and the validation pipeline are both in this crate, so a smaller answer means the \
         scan is broken",
        found.len()
    );

    let recorded: BTreeSet<&str> = RECORDED_NODE_VISITORS
        .iter()
        .map(|(name, _)| *name)
        .collect();
    let implemented: BTreeSet<&str> = found.keys().map(String::as_str).collect();

    let unrecorded: Vec<String> = implemented
        .difference(&recorded)
        .map(|name| format!("{name} ({})", found[*name].display()))
        .collect();
    assert!(
        unrecorded.is_empty(),
        "Fix: these types implement visit::NodeVisitor and are not recorded in \
         RECORDED_NODE_VISITORS: {unrecorded:?}. Each one restates the whole Node enum by hand, \
         and Node is #[non_exhaustive] outside vyre-foundation, so the copy goes stale in \
         silence. A scan that wants a variant or two and descent for the rest wants \
         visit::try_for_each_node instead; if the per-variant dispatch really is the \
         work, record it with that reason."
    );

    let departed: Vec<&&str> = recorded.difference(&implemented).collect();
    assert!(
        departed.is_empty(),
        "Fix: RECORDED_NODE_VISITORS names types that no longer implement visit::NodeVisitor: \
         {departed:?}. Delete the rows; a stale waiver hides the next real one."
    );

    let unreasoned: Vec<&&str> = RECORDED_NODE_VISITORS
        .iter()
        .filter(|(_, reason)| reason.trim().len() < 20)
        .map(|(name, _)| name)
        .collect();
    assert!(
        unreasoned.is_empty(),
        "Fix: record why {unreasoned:?} still enumerates Node by hand."
    );
}

/// Implementor type name to the file declaring it.
fn scan_node_visitor_implementations(hooks: &BTreeSet<String>) -> BTreeMap<String, PathBuf> {
    let root = vyre_test_support::monorepo::vyre_workspace_root();
    let mut found = BTreeMap::new();
    for path in rust_sources(&root) {
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (name, body) in impl_blocks_for_trait(&source, "NodeVisitor") {
            // Every hook is abstract, so an implementor states the whole enum.
            // A block that defines none of them is a trait of the same name in
            // another crate, not this one.
            let per_variant = hooks
                .iter()
                .filter(|hook| body.contains(&format!("fn {hook}(")))
                .count();
            if per_variant > 0 {
                found.insert(
                    name,
                    path.strip_prefix(&root).unwrap_or(&path).to_path_buf(),
                );
            }
        }
    }
    found
}

/// Every `.rs` file under `root`, skipping build output.
fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if path.is_dir() {
                if name != "target" && name != ".git" && !name.starts_with('.') {
                    stack.push(path);
                }
            } else if name.ends_with(".rs") {
                out.push(path);
            }
        }
    }
    out
}

/// `(implementing type, impl body)` for every `impl ... <trait_name> for ...`.
fn impl_blocks_for_trait<'a>(source: &'a str, trait_name: &str) -> Vec<(String, &'a str)> {
    let mut out = Vec::new();
    let marker = format!("{trait_name} for ");
    for (offset, _) in source.match_indices(&marker) {
        // A doc example is indented inside a `///` line; the declaration is not.
        let line_start = source[..offset].rfind('\n').map_or(0, |index| index + 1);
        if source[line_start..offset].trim_start().starts_with("//") {
            continue;
        }
        if !source[line_start..offset].trim_start().starts_with("impl") {
            continue;
        }
        let after = offset + marker.len();
        let end = source[after..]
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '_')
            .map_or(source.len(), |len| after + len);
        let name = source[after..end].to_string();
        let Some(open) = source[end..].find('{').map(|index| end + index) else {
            continue;
        };
        let mut depth = 0usize;
        let mut close = open;
        for (index, ch) in source[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = open + index;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push((name, &source[open..close]));
    }
    out
}

fn read_workspace_file(relative: &str) -> String {
    let path = vyre_test_support::monorepo::vyre_workspace_root().join(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("Fix: cannot read {path:?} for the walker scan: {err}"))
}
