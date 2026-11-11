//! Observable contract of the one scope walk every IR rewrite drives.
//!
//! WHY: four passes each carried their own loop over a scope, and each one
//! repeated two decisions the walk owner already makes: truncate the scope where
//! the encoder truncated it, and do not rebuild a node the policy left alone.
//! Three of the four discarded the second decision, so a pass that propagated
//! nothing still deep-copied every nested body it walked. These tests pin what a
//! caller can observe of both decisions, at every nesting the IR admits, with
//! the nesting set built at run time from `child_bodies` rather than listed here.
//!
//! Not covered: which positions of a node a rewrite must visit. That is IR
//! structure, owned by `vyre_foundation::transform::rewrite_walk`, whose match is
//! exhaustive with no catch-all so a new variant fails to compile there.

use std::sync::Arc;

use vyre_foundation::ir::{Expr, Node, Program};
use vyre_foundation::transform::const_prop::apply_const_prop;
use vyre_foundation::visit::child_bodies;

/// One way of nesting a body inside a node, named for the failure message.
struct Nesting {
    label: &'static str,
    wrap: fn(Vec<Node>) -> Node,
}

/// Every body-bearing wrapper, filtered by what `child_bodies` actually reports
/// a body for. A wrapper the IR stopped nesting drops out here instead of
/// asserting against a body no walk visits; a body-bearing variant the IR gains
/// makes `rewrite_node` fail to compile in its owner, which is where that
/// belongs.
fn nestings() -> Vec<Nesting> {
    let candidates = [
        Nesting {
            label: "Block",
            wrap: Node::Block,
        },
        Nesting {
            label: "If.then",
            wrap: |body| Node::if_then_else(Expr::LitBool(true), body, Vec::new()),
        },
        Nesting {
            label: "If.otherwise",
            wrap: |body| Node::if_then_else(Expr::LitBool(true), Vec::new(), body),
        },
        Nesting {
            label: "Loop",
            wrap: |body| Node::loop_for("i", Expr::u32(0), Expr::u32(4), body),
        },
        Nesting {
            label: "Region",
            wrap: |body| Node::Region {
                generator: "probe.region".into(),
                source_region: None,
                body: Arc::new(body),
            },
        },
    ];
    let probe = vec![Node::Return];
    candidates
        .into_iter()
        .filter(|nesting| {
            let node = (nesting.wrap)(probe.clone());
            child_bodies(&node)
                .iter()
                .any(|group| group.iter().any(|inner| inner == &Node::Return))
        })
        .collect()
}

/// A program whose entry is exactly `entry`, with no buffers.
fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], entry)
}

/// The `Arc` behind every `Region` body reachable from `entry`, in walk order.
fn region_bodies(entry: &[Node]) -> Vec<Arc<Vec<Node>>> {
    let mut found = Vec::new();
    let mut stack: Vec<&Node> = entry.iter().rev().collect();
    while let Some(node) = stack.pop() {
        if let Node::Region { body, .. } = node {
            found.push(Arc::clone(body));
        }
        for group in child_bodies(node) {
            stack.extend(group.iter().rev());
        }
    }
    found
}

#[test]
fn a_constant_reaches_every_nesting_the_ir_admits() {
    let nestings = nestings();
    assert!(
        nestings.len() >= 4,
        "the probe found {} body-bearing wrappers, which cannot be right for an IR with If, Loop, Block and Region",
        nestings.len()
    );
    for nesting in &nestings {
        let inner = vec![Node::let_bind("used", Expr::Var("outer".into()))];
        let rewritten = apply_const_prop(&program(vec![
            Node::let_bind("outer", Expr::u32(7)),
            (nesting.wrap)(inner),
        ]));

        let mut seen = Vec::new();
        let mut stack: Vec<&Node> = rewritten.entry().iter().rev().collect();
        while let Some(node) = stack.pop() {
            if let Node::Let { name, value } = node {
                if name.as_ref() == "used" {
                    seen.push(value.clone());
                }
            }
            for group in child_bodies(node) {
                stack.extend(group.iter().rev());
            }
        }
        assert_eq!(
            seen,
            vec![Expr::u32(7)],
            "const propagation did not reach the body of {}",
            nesting.label
        );
    }
}

#[test]
fn a_binding_made_inside_a_nesting_does_not_escape_it() {
    for nesting in &nestings() {
        let inner = vec![Node::let_bind("scoped", Expr::u32(3))];
        let rewritten = apply_const_prop(&program(vec![
            (nesting.wrap)(inner),
            Node::let_bind("after", Expr::Var("scoped".into())),
        ]));

        let after = rewritten
            .entry()
            .iter()
            .flat_map(|node| {
                let mut found = Vec::new();
                let mut stack: Vec<&Node> = vec![node];
                while let Some(node) = stack.pop() {
                    if let Node::Let { name, value } = node {
                        if name.as_ref() == "after" {
                            found.push(value.clone());
                        }
                    }
                    for group in child_bodies(node) {
                        stack.extend(group.iter().rev());
                    }
                }
                found
            })
            .collect::<Vec<_>>();
        assert_eq!(
            after,
            vec![Expr::Var("scoped".into())],
            "a binding made inside {} escaped it",
            nesting.label
        );
    }
}

#[test]
fn a_loop_induction_variable_shadows_an_enclosing_constant() {
    let rewritten = apply_const_prop(&program(vec![
        Node::let_bind("i", Expr::u32(9)),
        Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![Node::let_bind("read", Expr::Var("i".into()))],
        ),
    ]));

    let mut inside = Vec::new();
    let mut stack: Vec<&Node> = rewritten.entry().iter().rev().collect();
    while let Some(node) = stack.pop() {
        if let Node::Let { name, value } = node {
            if name.as_ref() == "read" {
                inside.push(value.clone());
            }
        }
        for group in child_bodies(node) {
            stack.extend(group.iter().rev());
        }
    }
    assert_eq!(
        inside,
        vec![Expr::Var("i".into())],
        "the loop induction variable was replaced by the enclosing constant"
    );
}

#[test]
fn nodes_after_a_return_are_dropped_in_every_nesting() {
    for nesting in &nestings() {
        let inner = vec![
            Node::Return,
            Node::let_bind("unreachable", Expr::u32(1)),
            Node::let_bind("also_unreachable", Expr::u32(2)),
        ];
        let rewritten = apply_const_prop(&program(vec![(nesting.wrap)(inner)]));

        let mut names = Vec::new();
        let mut stack: Vec<&Node> = rewritten.entry().iter().rev().collect();
        while let Some(node) = stack.pop() {
            if let Node::Let { name, .. } = node {
                names.push(name.to_string());
            }
            for group in child_bodies(node) {
                stack.extend(group.iter().rev());
            }
        }
        assert!(
            names.is_empty(),
            "{} kept {names:?} after its Return, which the encoder never gave an arena id",
            nesting.label
        );
    }
}

#[test]
fn a_pass_that_changes_nothing_does_not_rebuild_a_nested_region_body() {
    let inner = Arc::new(vec![Node::let_bind("only", Expr::Var("unbound".into()))]);
    let entry = vec![Node::Block(vec![Node::Region {
        generator: "probe.region".into(),
        source_region: None,
        body: Arc::clone(&inner),
    }])];

    let rewritten = apply_const_prop(&program(entry));

    let bodies = region_bodies(rewritten.entry());
    let nested = bodies
        .iter()
        .find(|body| body.as_slice() == inner.as_slice())
        .expect("the nested Region body must survive the pass");
    assert!(
        Arc::ptr_eq(nested, &inner),
        "const propagation rebuilt a nested Region body it changed nothing in, \
         which deep-copies the subtree on every pass that rewrites nothing"
    );
}

#[test]
fn a_pass_that_changes_something_still_rebuilds_only_what_changed() {
    let untouched = Arc::new(vec![Node::let_bind("only", Expr::Var("unbound".into()))]);
    let entry = vec![
        Node::let_bind("outer", Expr::u32(5)),
        Node::Block(vec![
            Node::Region {
                generator: "probe.region".into(),
                source_region: None,
                body: Arc::clone(&untouched),
            },
            Node::let_bind("changed", Expr::Var("outer".into())),
        ]),
    ];

    let rewritten = apply_const_prop(&program(entry));

    let mut changed = None;
    let mut stack: Vec<&Node> = rewritten.entry().iter().rev().collect();
    while let Some(node) = stack.pop() {
        if let Node::Let { name, value } = node {
            if name.as_ref() == "changed" {
                changed = Some(value.clone());
            }
        }
        for group in child_bodies(node) {
            stack.extend(group.iter().rev());
        }
    }
    assert_eq!(
        changed,
        Some(Expr::u32(5)),
        "the sibling that could be propagated was not"
    );

    let bodies = region_bodies(rewritten.entry());
    let nested = bodies
        .iter()
        .find(|body| body.as_slice() == untouched.as_slice())
        .expect("the untouched Region body must survive the pass");
    assert!(
        Arc::ptr_eq(nested, &untouched),
        "a change in one sibling rebuilt an untouched Region body in the same scope"
    );
}
