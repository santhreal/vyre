//! Shared legality and dependence analysis for the loop restructuring passes.
//!
//! `loop_fission` splits one loop body into two sibling loops and
//! `loop_fusion` merges two sibling loops into one. The transformations are
//! inverses, so both must answer the same three questions before rewriting:
//! which names a body binds, whether a name bound by one half is read by the
//! other, and whether a body holds an effect the touched-buffer summary cannot
//! see. Both also need the same variable-rename rewrite to re-key a body onto
//! a different induction variable. This module owns all four; neither pass
//! keeps a copy.
//!
//! Everything here is CPU-side and target-neutral: it reasons about IR shape
//! and name flow only.
//!
//! ## Drift resolved when the two copies were merged
//!
//! The passes' private copies had already diverged. Each divergence was
//! resolved toward the conservative side, so no transformation that was
//! previously refused becomes permitted:
//!
//! - **Bound names.** Fission's copy counted nested `Loop` induction
//!   variables, fusion's did not. Fission's superset wins:
//!   `loop_fission::freshen` needs a complete name-occupancy set to pick an
//!   unused variable, and for the dependence query the extra names only ever
//!   refuse more. A nested loop variable is out of scope in the sibling half
//!   of any program the validator accepts, so no accepted program changes
//!   verdict.
//! - **Variable rename.** Fusion's copy rewrote binding occurrences
//!   (`Let` / `Assign` targets) along with reads; fission's rewrote reads
//!   only, which leaves a `Let` of the renamed variable bound to a name the
//!   rewritten loop no longer mentions. Fusion's closed form wins. Both forms
//!   are reachable only through a body that binds its own loop variable,
//!   which the validator rejects as a shadowing binding.
//! - **Unsummarisable effects.** Fission's copy reported only `Expr::Opaque`,
//!   relying on its sibling-only barrier scan to catch opaque, trap, and
//!   resume *nodes*; that scan does not recurse, so a `Node::Opaque` nested
//!   inside an `If` or `Block` escaped both checks. Fusion's recursive copy
//!   wins and closes the hole. Fission keeps its own additional
//!   `has_barrier_like` gate on top of this module: refusing async,
//!   collective, and indirect-dispatch nodes is a fission-specific
//!   requirement, `verified-intentional` against fusion, whose disjointness
//!   proof already covers those nodes' buffer operands.

use super::substitution::expr_contains_opaque;
use super::{collect_var_reads, rename_var_in_expr};
use crate::ir::{Ident, Node};
use rustc_hash::FxHashSet;

/// Every name `nodes` binds, nested scopes included.
///
/// The per-variant answer is
/// [`node_bound_name`](crate::transform::visit::node_bound_name) and the descent
/// is [`for_each_node`](crate::transform::visit::for_each_node), both exhaustive.
/// The walk this replaces named its own variants and ended in `_ => {}`, so a
/// binding form it did not list read as binding nothing and
/// [`bindings_flow_across`] then let fusion or fission reorder statements across
/// a live binding.
pub(super) fn collect_bound_names(nodes: &[Node], out: &mut FxHashSet<Ident>) {
    crate::transform::visit::for_each_node(nodes, |node| {
        if let Some(name) = crate::transform::visit::node_bound_name(node) {
            out.insert(name.clone());
        }
    });
}

/// True iff a name bound by `binder` is read by `reader`, which makes moving
/// `reader` out of `binder`'s scope (fission) or into it (fusion) change name
/// resolution.
///
/// `induction` is the loop variable the two halves share. It is excluded
/// because each half sees it bound by its own loop header, so a reference to
/// it is not a cross-half flow. Excluding it from the bindings and excluding
/// it from the reads are the same set operation; both remove it from the
/// intersection.
pub(super) fn bindings_flow_across(binder: &[Node], reader: &[Node], induction: &Ident) -> bool {
    let mut bound: FxHashSet<Ident> = FxHashSet::default();
    collect_bound_names(binder, &mut bound);
    bound.remove(induction);
    if bound.is_empty() {
        return false;
    }
    let mut reads: FxHashSet<Ident> = FxHashSet::default();
    collect_var_reads(reader, &mut reads);
    !bound.is_disjoint(&reads)
}

/// True iff `nodes` holds an operation whose memory effect
/// [`super::collect_touched_buffers`] cannot summarise: an opaque extension
/// node or expression, or a trap / resume host handler.
///
/// `collect_touched_buffers` reports `Node::Opaque` and `Expr::Opaque` as
/// touching NO buffer and a `Trap` as touching only its explicit `address`
/// operand, but their real effect may read or write ANY buffer: an opaque
/// payload is extension-defined and a trap invokes an unknowable host handler
/// (see `effect_lattice`, which lifts all three to the `Diverging` lattice
/// top). Both passes reorder memory operations relative to each other, so a
/// hidden access could cross a sibling's writes and break a dependency the
/// disjointness proof never saw. Either side containing one blocks the
/// rewrite.
///
/// Async, collective, and indirect-dispatch nodes are deliberately absent:
/// their buffer operands ARE captured by `collect_touched_buffers`, so the
/// disjointness test already covers them. Fission additionally refuses them
/// through its own barrier gate because splitting a loop reorders them
/// against the surrounding work; fusion does not, and that difference is
/// intentional.
pub(super) fn unsummarisable_effect(nodes: &[Node]) -> bool {
    nodes.iter().any(node_unsummarisable_effect)
}

fn node_unsummarisable_effect(node: &Node) -> bool {
    match node {
        // Unknowable host or extension effect regardless of any operand.
        Node::Opaque(_) | Node::Trap { .. } | Node::Resume { .. } => true,
        Node::Let { value, .. } | Node::Assign { value, .. } => expr_contains_opaque(value),
        Node::Store { index, value, .. } => {
            expr_contains_opaque(index) || expr_contains_opaque(value)
        }
        Node::If {
            cond,
            then,
            otherwise,
        } => {
            expr_contains_opaque(cond)
                || unsummarisable_effect(then)
                || unsummarisable_effect(otherwise)
        }
        Node::Loop { from, to, body, .. } => {
            expr_contains_opaque(from) || expr_contains_opaque(to) || unsummarisable_effect(body)
        }
        Node::Block(body) => unsummarisable_effect(body),
        Node::Region { body, .. } => unsummarisable_effect(body),
        Node::AsyncLoad { offset, size, .. } | Node::AsyncStore { offset, size, .. } => {
            expr_contains_opaque(offset) || expr_contains_opaque(size)
        }
        // Buffer operands captured by `collect_touched_buffers`; no Expr
        // operand that could hide an opaque payload.
        Node::Barrier { .. }
        | Node::Return
        | Node::IndirectDispatch { .. }
        | Node::AsyncWait { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. } => false,
    }
}

/// Re-key `node` from induction variable `from` onto `to`, rewriting reads
/// and binding occurrences alike so the rewrite leaves no reference behind.
pub(super) fn rename_var_in_node(node: Node, from: &Ident, to: &Ident) -> Node {
    match node {
        Node::Let { name, value } => Node::Let {
            name: if name == *from { to.clone() } else { name },
            value: rename_var_in_expr(value, from, to),
        },
        Node::Assign { name, value } => Node::Assign {
            name: if name == *from { to.clone() } else { name },
            value: rename_var_in_expr(value, from, to),
        },
        Node::Store {
            buffer,
            index,
            value,
        } => Node::Store {
            buffer,
            index: rename_var_in_expr(index, from, to),
            value: rename_var_in_expr(value, from, to),
        },
        Node::If {
            cond,
            then,
            otherwise,
        } => Node::If {
            cond: rename_var_in_expr(cond, from, to),
            then: rename_var_in_body(then, from, to),
            otherwise: rename_var_in_body(otherwise, from, to),
        },
        Node::Loop {
            var,
            from: lo,
            to: hi,
            body,
        } => Node::Loop {
            var,
            from: rename_var_in_expr(lo, from, to),
            to: rename_var_in_expr(hi, from, to),
            body: rename_var_in_body(body, from, to),
        },
        Node::Block(body) => Node::Block(rename_var_in_body(body, from, to)),
        Node::Region {
            generator,
            source_region,
            body,
        } => {
            let body_vec = std::sync::Arc::try_unwrap(body).unwrap_or_else(|arc| (*arc).clone());
            Node::Region {
                generator,
                source_region,
                body: std::sync::Arc::new(rename_var_in_body(body_vec, from, to)),
            }
        }
        other => other,
    }
}

fn rename_var_in_body(body: Vec<Node>, from: &Ident, to: &Ident) -> Vec<Node> {
    body.into_iter()
        .map(|n| rename_var_in_node(n, from, to))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::test_fixtures::{loops_of, program};
    use super::*;
    use crate::ir::{DataType, Expr, ExprNode, Node, NodeExtension};
    use crate::optimizer::passes::loops::{loop_fission::LoopFission, loop_fusion::LoopFusion};

    fn sorted(nodes: &[Node]) -> Vec<String> {
        let mut out = FxHashSet::default();
        collect_bound_names(nodes, &mut out);
        let mut v: Vec<String> = out.iter().map(|n| n.as_str().to_string()).collect();
        v.sort();
        v
    }

    #[test]
    fn bound_names_cover_lets_assigns_loop_vars_and_nested_scopes() {
        let body = vec![
            Node::let_bind("v", Expr::u32(1)),
            Node::Assign {
                name: Ident::from("s"),
                value: Expr::u32(2),
            },
            Node::loop_for(
                "k",
                Expr::u32(0),
                Expr::u32(2),
                vec![Node::let_bind("inner", Expr::var("k"))],
            ),
            Node::If {
                cond: Expr::LitBool(true),
                then: vec![Node::let_bind("t", Expr::u32(3))],
                otherwise: vec![Node::Block(vec![Node::let_bind("e", Expr::u32(4))])],
            },
        ];
        assert_eq!(sorted(&body), ["e", "inner", "k", "s", "t", "v"]);
    }

    #[test]
    fn bindings_flow_across_ignores_the_shared_induction_variable() {
        let binder = vec![Node::let_bind("v", Expr::var("i"))];
        let reads_v = vec![Node::store("b", Expr::var("i"), Expr::var("v"))];
        let reads_i_only = vec![Node::store("b", Expr::var("i"), Expr::u32(1))];
        let i = Ident::from("i");

        assert!(bindings_flow_across(&binder, &reads_v, &i));
        assert!(!bindings_flow_across(&binder, &reads_i_only, &i));
        // The induction variable is bound by the binder half via its own
        // header, so a read of it never counts as a cross-half flow.
        let binds_i = vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(2),
            vec![Node::Return],
        )];
        assert!(!bindings_flow_across(&binds_i, &reads_i_only, &i));
    }

    #[test]
    fn rename_rewrites_reads_and_binding_occurrences() {
        let from = Ident::from("i");
        let to = Ident::from("z");
        assert_eq!(
            rename_var_in_node(Node::let_bind("i", Expr::var("i")), &from, &to),
            Node::let_bind("z", Expr::var("z"))
        );
        assert_eq!(
            rename_var_in_node(
                Node::Assign {
                    name: Ident::from("i"),
                    value: Expr::var("i"),
                },
                &from,
                &to
            ),
            Node::Assign {
                name: Ident::from("z"),
                value: Expr::var("z"),
            }
        );
        assert_eq!(
            rename_var_in_node(
                Node::Block(vec![Node::store("b", Expr::var("i"), Expr::var("keep"))]),
                &from,
                &to
            ),
            Node::Block(vec![Node::store("b", Expr::var("z"), Expr::var("keep"))])
        );
    }

    vyre_test_support::test_expr_extension! {
        OpaqueValue,
        kind: "test.legality.opaque_value",
        identity: "opaque_value",
        result_type: Some(DataType::U32),
        cse_safe: false,
        fingerprint: 21,
    }

    vyre_test_support::test_node_extension! {
        OpaqueStatement,
        kind: "test.legality.opaque_statement",
        identity: "opaque_statement",
        fingerprint: 22,
    }

    #[test]
    fn unsummarisable_effect_sees_through_nested_scopes() {
        assert!(!unsummarisable_effect(&[Node::store(
            "a",
            Expr::u32(0),
            Expr::u32(1)
        )]));
        // Nested opaque statement: the sibling-only barrier scan cannot see
        // this one, so the recursive check must.
        assert!(unsummarisable_effect(&[Node::Block(vec![Node::opaque(
            OpaqueStatement
        )])]));
        assert!(unsummarisable_effect(&[Node::If {
            cond: Expr::LitBool(true),
            then: vec![Node::store("a", Expr::u32(0), Expr::opaque(OpaqueValue))],
            otherwise: Vec::new(),
        }]));
        assert!(unsummarisable_effect(&[Node::loop_for(
            "k",
            Expr::u32(0),
            Expr::u32(2),
            vec![Node::Trap {
                address: Box::new(Expr::u32(0)),
                tag: Ident::from("test.legality.trap"),
            }],
        )]));
        // Async operands are summarised by the touched-buffer analysis.
        assert!(!unsummarisable_effect(&[Node::Barrier {
            ordering: crate::ir::MemoryOrdering::Relaxed,
        }]));
    }

    // ----------------------------------------------------------------
    // Cross-entry-point guard for the merged clone family. Both passes now
    // route their legality verdict and their induction-variable rewrite
    // through this module, so perturbing anything above must turn both of
    // these red. They pin the transformed IR, not just the `changed` flag,
    // so a rename defect is caught as well as a legality defect.
    // ----------------------------------------------------------------

    /// `body_a` / `body_b` halves both passes must judge the same way: fission
    /// sees them concatenated inside one loop, fusion sees them as two sibling
    /// loops. `legal` is the shared verdict.
    fn shared_fixtures() -> Vec<(&'static str, bool, Vec<Node>, Vec<Node>)> {
        vec![
            (
                "disjoint stores",
                true,
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
                vec![Node::store("b", Expr::var("i"), Expr::u32(2))],
            ),
            (
                "shared buffer",
                false,
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
                vec![Node::store("a", Expr::var("i"), Expr::u32(2))],
            ),
            (
                "name flows from first half to second",
                false,
                vec![Node::let_bind("v", Expr::var("i"))],
                vec![Node::store("b", Expr::var("i"), Expr::var("v"))],
            ),
            (
                "nested opaque statement in the first half",
                false,
                vec![Node::Block(vec![Node::opaque(OpaqueStatement)])],
                vec![Node::store("b", Expr::var("i"), Expr::u32(2))],
            ),
            (
                "opaque value nested in an If arm of the second half",
                false,
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
                vec![Node::If {
                    cond: Expr::LitBool(true),
                    then: vec![Node::store("b", Expr::var("i"), Expr::opaque(OpaqueValue))],
                    otherwise: Vec::new(),
                }],
            ),
            (
                "independent halves each binding a private name",
                true,
                vec![
                    Node::let_bind("va", Expr::var("i")),
                    Node::store("a", Expr::var("i"), Expr::var("va")),
                ],
                vec![
                    Node::let_bind("vb", Expr::var("i")),
                    Node::store("b", Expr::var("i"), Expr::var("vb")),
                ],
            ),
        ]
    }

    #[test]
    fn fission_follows_the_shared_legality_verdict() {
        for (label, legal, body_a, body_b) in shared_fixtures() {
            let mut body = body_a;
            body.extend(body_b);
            let entry = vec![Node::loop_for("i", Expr::u32(0), Expr::u32(8), body)];
            let result = LoopFission::transform(program(entry));
            assert_eq!(result.changed, legal, "fission verdict for `{label}`");
            let loops = loops_of(result.program.entry());
            assert_eq!(
                loops.len(),
                if legal { 2 } else { 1 },
                "fission loop count for `{label}`"
            );
        }
    }

    #[test]
    fn fusion_follows_the_shared_legality_verdict() {
        for (label, legal, body_a, body_b) in shared_fixtures() {
            let i = Ident::from("i");
            let j = Ident::from("j");
            let body_b = rename_var_in_body(body_b, &i, &j);
            let entry = vec![
                Node::loop_for("i", Expr::u32(0), Expr::u32(8), body_a),
                Node::loop_for("j", Expr::u32(0), Expr::u32(8), body_b),
            ];
            let result = LoopFusion::transform(program(entry));
            assert_eq!(result.changed, legal, "fusion verdict for `{label}`");
            let loops = loops_of(result.program.entry());
            assert_eq!(
                loops.len(),
                if legal { 1 } else { 2 },
                "fusion loop count for `{label}`"
            );
        }
    }

    #[test]
    fn fission_rekeys_the_split_off_half_onto_a_fresh_induction_variable() {
        let entry = vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(8),
            vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
        )];
        let result = LoopFission::transform(program(entry));
        assert!(result.changed);
        let binding = result.program;
        let loops = loops_of(binding.entry());
        let [first, second] = loops[..] else {
            panic!("Fix: fission must produce exactly two sibling loops");
        };
        let Node::Loop {
            var: var_a,
            body: body_a,
            ..
        } = first
        else {
            unreachable!("filtered to Loop above");
        };
        let Node::Loop {
            var: var_b,
            body: body_b,
            ..
        } = second
        else {
            unreachable!("filtered to Loop above");
        };
        assert_eq!(var_a.as_str(), "i");
        assert_ne!(var_a, var_b, "the split-off loop needs a fresh variable");
        assert_eq!(
            body_a,
            &vec![Node::store("a", Expr::var("i"), Expr::u32(1))]
        );
        assert_eq!(
            body_b,
            &vec![Node::store("b", Expr::var(var_b.as_str()), Expr::u32(2))],
            "the split-off half must read the fresh variable, not the original"
        );
    }

    #[test]
    fn fusion_rekeys_the_second_body_onto_the_surviving_induction_variable() {
        let entry = vec![
            Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("a", Expr::var("i"), Expr::u32(1))],
            ),
            Node::loop_for(
                "j",
                Expr::u32(0),
                Expr::u32(8),
                vec![Node::store("b", Expr::var("j"), Expr::u32(2))],
            ),
        ];
        let result = LoopFusion::transform(program(entry));
        assert!(result.changed);
        let binding = result.program;
        let loops = loops_of(binding.entry());
        assert_eq!(loops.len(), 1);
        let Node::Loop { var, body, .. } = loops[0] else {
            unreachable!("filtered to Loop above");
        };
        assert_eq!(var.as_str(), "i");
        assert_eq!(
            body,
            &vec![
                Node::store("a", Expr::var("i"), Expr::u32(1)),
                Node::store("b", Expr::var("i"), Expr::u32(2)),
            ],
            "the merged half must be re-keyed onto the surviving variable"
        );
    }
}
