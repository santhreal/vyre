use super::{
    const_loop_empty, const_truth, eliminate_dead_lets, eliminate_unreachable_tracked,
    reachable_prefix, LiveSet,
};
use crate::ir::{Node, Program};
use crate::optimizer::PassResult;

fn slice_has_unreachable_work(nodes: &[Node]) -> bool {
    if reachable_prefix(nodes).len() < nodes.len() {
        return true;
    }
    for node in nodes {
        let hit = match node {
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                const_truth(cond).is_some()
                    || slice_has_unreachable_work(then)
                    || slice_has_unreachable_work(otherwise)
            }
            Node::Loop { from, to, body, .. } => {
                const_loop_empty(from, to) || slice_has_unreachable_work(body)
            }
            Node::Block(children) => children.is_empty() || slice_has_unreachable_work(children),
            Node::Region { body, .. } => slice_has_unreachable_work(body),
            Node::TileElementwise { body, .. } => slice_has_unreachable_work(body),
            _ => false,
        };
        if hit {
            return true;
        }
    }
    false
}

/// Remove unreachable statements and unused pure `let` bindings, preserving
/// `program` and its cached memos when nothing changes.
#[must_use]
#[inline]
pub(crate) fn dce_pass(program: Program) -> PassResult {
    if !program.stats().has_node_let() && !slice_has_unreachable_work(program.entry()) {
        return PassResult::unchanged(program);
    }
    let warm = program.clone();
    let r1 = eliminate_dead_lets(program.into_entry_vec(), LiveSet::new());
    let (n2, c2) = eliminate_unreachable_tracked(r1.nodes);
    if !r1.changed && !c2 {
        return PassResult::unchanged(warm);
    }
    let final_entry = if c2 {
        eliminate_dead_lets(n2, LiveSet::new()).nodes
    } else {
        n2
    };
    PassResult::changed(warm.with_rewritten_entry(final_entry))
}

/// Remove unreachable statements and unused pure `let` bindings.
#[must_use]
#[inline]
pub fn dce(program: Program) -> Program {
    dce_pass(program).program
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Expr, Node};
    use std::sync::Arc;

    #[test]
    fn dce_descends_into_region_bodies() {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![Node::Region {
                generator: "test".into(),
                source_region: None,
                body: Arc::new(vec![
                    Node::let_bind("dead", Expr::u32(1)),
                    Node::Return,
                    Node::let_bind("unreachable", Expr::u32(2)),
                ]),
            }],
        );

        let optimized = dce(program);
        let [Node::Region { body, .. }] = optimized.entry() else {
            panic!("Fix: DCE must preserve the Region wrapper while optimizing its body");
        };
        assert_eq!(body.as_slice(), &[Node::Return]);
    }

    #[test]
    fn dce_region_live_ins_propagate_to_outer_scope() {
        let program = Program::wrapped(
            Vec::new(),
            [1, 1, 1],
            vec![
                Node::let_bind("live", Expr::u32(7)),
                Node::Region {
                    generator: "test".into(),
                    source_region: None,
                    body: Arc::new(vec![Node::store("out", Expr::u32(0), Expr::var("live"))]),
                },
            ],
        );

        let optimized = dce(program);
        let [Node::Region { body, .. }] = optimized.entry() else {
            panic!("Fix: Program::wrapped must keep the root Region");
        };
        assert!(
            matches!(body.first(), Some(Node::Let { name, .. }) if name.as_str() == "live"),
            "Fix: variables read inside a Region must keep their outer definitions live: {:?}",
            body
        );
    }
}
