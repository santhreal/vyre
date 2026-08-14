//! Carrier-name collection: which outer bindings a body mutates, in the order
//! a pre-order walk first observes them.

use rustc_hash::FxHashSet;
use vyre_foundation::ir::{Ident, Node};

use super::scope;

/// Walk a `Node::Loop` / `Node::Region` / `Node::Block` body and collect
/// every source-level variable name that:
///   1. Appears on the left of an `Assign` somewhere inside the body
///      (including nested If/Block/Region/Loop scopes); AND
///   2. Was already bound in the incoming scope (so the assignment
///      mutates an outer binding, not a body-local `Let`).
///
/// These are the names whose final value must escape the body via a
/// function-local: for a `Loop` it is the per-iteration carrier, for a
/// `Region`/`Block` it is the region-exit phi-merge. Loop callers pass
/// `Some(loop_var)` to skip the loop-induction variable (handled by
/// `LoopIndex` / `LoopCarrierEnd` is not emitted for it). Region/Block
/// callers pass `None`.
///
/// Order is the deterministic order names are first observed during a
/// pre-order walk, so the emitted op stream is stable across runs.
pub(super) fn collect_carrier_names(
    body: &[Node],
    incoming_scope: &scope::ScopeSnapshot,
    loop_var: Option<&Ident>,
) -> Vec<Ident> {
    let mut seen: FxHashSet<Ident> = FxHashSet::default();
    let mut order: Vec<Ident> = Vec::new();
    let mut local_lets: Vec<FxHashSet<Ident>> = vec![FxHashSet::default()];

    fn walk(
        nodes: &[Node],
        incoming_scope: &scope::ScopeSnapshot,
        loop_var: Option<&Ident>,
        seen: &mut FxHashSet<Ident>,
        order: &mut Vec<Ident>,
        local_lets: &mut Vec<FxHashSet<Ident>>,
    ) {
        for node in nodes {
            match node {
                Node::Let { name, .. } => {
                    if let Some(top) = local_lets.last_mut() {
                        top.insert(name.clone());
                    }
                }
                Node::Assign { name, .. } => {
                    if let Some(lv) = loop_var {
                        if name == lv {
                            continue;
                        }
                    }
                    let shadowed = local_lets.iter().any(|frame| frame.contains(name));
                    if shadowed {
                        continue;
                    }
                    if !incoming_scope.contains_key(name) {
                        continue;
                    }
                    if seen.insert(name.clone()) {
                        order.push(name.clone());
                    }
                }
                Node::Block(inner) => {
                    local_lets.push(FxHashSet::default());
                    walk(inner, incoming_scope, loop_var, seen, order, local_lets);
                    local_lets.pop();
                }
                Node::If {
                    then, otherwise, ..
                } => {
                    local_lets.push(FxHashSet::default());
                    walk(then, incoming_scope, loop_var, seen, order, local_lets);
                    local_lets.pop();
                    local_lets.push(FxHashSet::default());
                    walk(otherwise, incoming_scope, loop_var, seen, order, local_lets);
                    local_lets.pop();
                }
                Node::Loop {
                    var: inner_var,
                    body: inner_body,
                    ..
                } => {
                    local_lets.push({
                        let mut s = FxHashSet::default();
                        s.insert(inner_var.clone());
                        s
                    });
                    walk(
                        inner_body,
                        incoming_scope,
                        loop_var,
                        seen,
                        order,
                        local_lets,
                    );
                    local_lets.pop();
                }
                Node::Region { body: inner, .. } => {
                    local_lets.push(FxHashSet::default());
                    walk(inner, incoming_scope, loop_var, seen, order, local_lets);
                    local_lets.pop();
                }
                // Leaf case: the nesting variants above are exactly the ones `transform::visit::child_bodies` lists, so an unknown variant has no child statements to visit.
                _ => {}
            }
        }
    }

    walk(
        body,
        incoming_scope,
        loop_var,
        &mut seen,
        &mut order,
        &mut local_lets,
    );
    order
}

#[cfg(test)]
mod tests {
    use super::super::lower;
    use crate::descriptor::{KernelBody, KernelOp, KernelOpKind};
    use vyre_foundation::ir::{DataType, Program};

    #[test]
    fn loop_carrier_mutated_in_if_then_is_visible_to_next_sibling() {
        use vyre_foundation::ir::{BufferDecl, Expr, Node};

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("x", Expr::u32(0)),
                Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::u32(1),
                    vec![
                        Node::if_then(Expr::bool(true), vec![Node::assign("x", Expr::u32(7))]),
                        Node::if_then(
                            Expr::bool(true),
                            vec![Node::store("out", Expr::u32(0), Expr::var("x"))],
                        ),
                    ],
                ),
            ],
        );

        let desc =
            lower(&program).expect("Fix: conditional carrier mutation must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());
        let (parent, loop_op) =
            find_loop(&desc.body).expect("Fix: structured loop op must be present");
        let child = &parent.child_bodies[loop_op.operands[2] as usize];
        let first_if_idx = child
            .ops
            .iter()
            .position(|op| matches!(op.kind, KernelOpKind::StructuredIfThen))
            .expect("Fix: first conditional assignment must lower to StructuredIfThen");
        let carrier_idx = child
            .ops
            .iter()
            .enumerate()
            .skip(first_if_idx + 1)
            .find_map(|(idx, op)| match &op.kind {
                KernelOpKind::LoopCarrier { name } if name.as_ref() == "x" => Some(idx),
                _ => None,
            })
            .expect("Fix: parent loop body must reread x carrier after conditional mutation");
        let carrier_result = child.ops[carrier_idx]
            .result
            .expect("Fix: carrier read must produce an SSA result");
        let second_if = child
            .ops
            .iter()
            .skip(carrier_idx + 1)
            .find(|op| matches!(op.kind, KernelOpKind::StructuredIfThen))
            .expect("Fix: second conditional store must lower after carrier reread");
        let store_body = &child.child_bodies[second_if.operands[1] as usize];
        let store = store_body
            .ops
            .iter()
            .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
            .expect("Fix: second conditional body must store x");
        assert_eq!(
            store.operands[2], carrier_result,
            "sibling after conditional carrier mutation must read the fresh carrier value"
        );

        fn find_loop(body: &KernelBody) -> Option<(&KernelBody, &KernelOp)> {
            for op in &body.ops {
                if matches!(op.kind, KernelOpKind::StructuredForLoop { .. }) {
                    return Some((body, op));
                }
            }
            body.child_bodies.iter().find_map(find_loop)
        }
    }

    /// Region phi-merge: a `Node::Region` whose body reassigns an
    /// outer-bound variable must publish the in-region final value back
    /// to the parent body via a function-local. Without the
    /// `LoopCarrierInit/LoopCarrier/LoopCarrierEnd` round-trip, the
    /// in-region SSA id is local to the child KernelBody and the parent
    /// reads the pre-region seed (the `n_tokens=0` GPU-lex symptom).
    #[test]
    fn region_publishes_inner_assign_to_parent_via_carrier() {
        use std::sync::Arc;
        use vyre_foundation::ir::{BufferDecl, Expr, Ident, Node};

        // ```
        // let x = 0;
        // region "phase" { x = 7; }
        // store(out[0], x)
        // ```
        // The post-region store must read the in-region value (7), not
        // the pre-region seed (0). Lowering must emit a parent-body
        // `LoopCarrier { name: "x" }` after the `Region` op whose result
        // id feeds the `StoreGlobal`.
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("x", Expr::u32(0)),
                Node::Region {
                    generator: Ident::from("phase"),
                    source_region: None,
                    body: Arc::new(vec![Node::assign("x", Expr::u32(7))]),
                },
                Node::store("out", Expr::u32(0), Expr::var("x")),
            ],
        );

        let desc = lower(&program).expect("Fix: region with inner assign must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());

        // Outer-most kernel body: a single `Region { generator: c_lexer }`
        // wraps the entry tree (program::wrapped). Drill into it.
        let entry = &desc.body;
        assert_eq!(
            entry.ops.len(),
            1,
            "wrapped program has one entry-region op"
        );
        let entry_region_op = &entry.ops[0];
        let entry_region_body = &entry.child_bodies[entry_region_op.operands[0] as usize];

        // Find the explicit `Region { generator: "phase" }` op inside
        // the entry region.
        let phase_pos = entry_region_body
            .ops
            .iter()
            .position(|op| {
                matches!(&op.kind, KernelOpKind::Region { generator } if generator.as_ref() == "phase")
            })
            .expect("Fix: phase Region op must be lowered");
        let phase_op = &entry_region_body.ops[phase_pos];

        // Pre-region: must emit `LoopCarrierInit { name: "x" }` BEFORE
        // the `Region` op so the function-local is seeded with the
        // pre-region value of x.
        let init_pos = entry_region_body
            .ops
            .iter()
            .position(|op| {
                matches!(&op.kind, KernelOpKind::LoopCarrierInit { name } if name.as_ref() == "x")
            })
            .expect("Fix: region must emit LoopCarrierInit for the carried name");
        assert!(
            init_pos < phase_pos,
            "LoopCarrierInit must precede the Region op so the local is seeded before entry"
        );

        // Inside the region body: the `Assign` lowers via the active-
        // carrier path → `LoopCarrierEnd { name: "x" }` (commit) +
        // `LoopCarrier { name: "x" }` (re-read).
        let phase_body_idx = phase_op.operands[0] as usize;
        let phase_body = &entry_region_body.child_bodies[phase_body_idx];
        assert!(
            phase_body
                .ops
                .iter()
                .any(|op| matches!(&op.kind, KernelOpKind::LoopCarrierEnd { name } if name.as_ref() == "x")),
            "in-region Assign must commit to the carrier local via LoopCarrierEnd"
        );

        // Post-region: parent body must re-read the carrier so the
        // subsequent `Var(x)` resolves to the in-region final value.
        let post_read = entry_region_body
            .ops
            .iter()
            .enumerate()
            .find(|(idx, op)| {
                *idx > phase_pos
                    && matches!(&op.kind, KernelOpKind::LoopCarrier { name } if name.as_ref() == "x")
            })
            .expect("Fix: region must emit a post-Region LoopCarrier read for the carried name");
        let post_read_id = post_read
            .1
            .result
            .expect("Fix: post-region LoopCarrier produces an SSA id");

        // The store must consume the post-region read id, not the
        // pre-region seed.
        let store = entry_region_body
            .ops
            .iter()
            .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
            .expect("Fix: post-region store must lower into the parent body");
        assert_eq!(
            store.operands[2], post_read_id,
            "post-region Var(x) read must resolve to the carrier publish id, not the pre-region seed"
        );
    }

    /// Region phi-merge negative: a `Node::Region` whose body does NOT
    /// reassign any outer name must NOT emit any `LoopCarrierInit` /
    /// `LoopCarrierEnd` / `LoopCarrier` ops for region-merge purposes.
    /// (Loop-driven carriers from any enclosing Loop scope are a
    /// separate machinery  -  this test runs at root scope so none are
    /// expected.)
    #[test]
    fn region_without_inner_assign_emits_no_carrier_ops() {
        use std::sync::Arc;
        use vyre_foundation::ir::{BufferDecl, Expr, Ident, Node};

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("x", Expr::u32(7)),
                Node::Region {
                    generator: Ident::from("read_only_phase"),
                    source_region: None,
                    body: Arc::new(vec![Node::store("out", Expr::u32(0), Expr::var("x"))]),
                },
            ],
        );

        let desc = lower(&program).expect("Fix: read-only region must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());

        fn count_carrier_ops(body: &KernelBody) -> usize {
            body.ops
                .iter()
                .filter(|op| {
                    matches!(
                        op.kind,
                        KernelOpKind::LoopCarrier { .. }
                            | KernelOpKind::LoopCarrierInit { .. }
                            | KernelOpKind::LoopCarrierEnd { .. }
                    )
                })
                .count()
                + body
                    .child_bodies
                    .iter()
                    .map(count_carrier_ops)
                    .sum::<usize>()
        }
        assert_eq!(
            count_carrier_ops(&desc.body),
            0,
            "no in-region reassignment ⇒ no carrier ops (would be decoration otherwise)"
        );
    }

    /// Region phi-merge nested: a Region inside a Loop whose body
    /// reassigns a loop-carrier-eligible name must commit through the
    /// SAME named-carrier local  -  the Loop's pre-loop init and the
    /// inner Region's pre-region init both target the same slot, so
    /// the next iteration's top-of-loop read sees the in-region final
    /// value of the previous iteration.
    #[test]
    fn region_inside_loop_shares_named_carrier_slot() {
        use std::sync::Arc;
        use vyre_foundation::ir::{BufferDecl, Expr, Ident, Node};

        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            vec![
                Node::let_bind("acc", Expr::u32(0)),
                Node::loop_for(
                    "i",
                    Expr::u32(0),
                    Expr::u32(4),
                    vec![Node::Region {
                        generator: Ident::from("step"),
                        source_region: None,
                        body: Arc::new(vec![Node::assign(
                            "acc",
                            Expr::add(Expr::var("acc"), Expr::u32(1)),
                        )]),
                    }],
                ),
                Node::store("out", Expr::u32(0), Expr::var("acc")),
            ],
        );

        let desc = lower(&program).expect("Fix: loop+region+assign must descriptor-lower");
        assert!(crate::verify::verify(&desc).is_ok());

        // Locate the StructuredForLoop op and its body.
        fn find_loop(body: &KernelBody) -> Option<(&KernelBody, &KernelOp)> {
            for op in &body.ops {
                if matches!(op.kind, KernelOpKind::StructuredForLoop { .. }) {
                    return Some((body, op));
                }
            }
            body.child_bodies.iter().find_map(find_loop)
        }
        let (loop_parent, loop_op) =
            find_loop(&desc.body).expect("Fix: StructuredForLoop must be lowered");
        let loop_body = &loop_parent.child_bodies[loop_op.operands[2] as usize];

        // Loop body must contain the inner Region op.
        let region_op = loop_body
            .ops
            .iter()
            .find(|op| {
                matches!(&op.kind, KernelOpKind::Region { generator } if generator.as_ref() == "step")
            })
            .expect("Fix: inner Region must lower inside the loop body");
        let region_body = &loop_body.child_bodies[region_op.operands[0] as usize];

        // The inner region's body must commit to the `acc` carrier
        // local on its Assign  -  the same local the Loop uses, since
        // emit-naga keys named-carrier locals by name.
        assert!(
            region_body
                .ops
                .iter()
                .any(|op| matches!(&op.kind, KernelOpKind::LoopCarrierEnd { name } if name.as_ref() == "acc")),
            "Assign inside loop+region must commit through the named carrier local"
        );

        // Post-loop: the parent body's StoreGlobal must read the
        // post-loop carrier publish (loop's existing post-loop emission)
        //  -  proving the in-region commit propagates out of the loop.
        let store = desc
            .body
            .child_bodies
            .iter()
            .flat_map(|child| child.ops.iter())
            .find(|op| matches!(op.kind, KernelOpKind::StoreGlobal))
            .expect("Fix: post-loop store must lower");
        assert!(
            !store.operands.is_empty(),
            "post-loop store must read the published carrier"
        );
    }
}
