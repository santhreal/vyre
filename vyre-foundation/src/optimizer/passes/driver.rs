//! The decisions every structural pass in this crate used to restate.
//!
//! A pass answers two questions that have nothing to do with its rewrite rule:
//! when the scheduler should invoke it, and how its rule reaches a node buried
//! in a nested body. Eighteen `analyze_impl` bodies carried the same
//! stat-guard-then-scan shape, and sixteen transforms carried the same descent
//! together with the same `&mut bool` bookkeeping. Both are stated once here,
//! so a pass file holds only its rule.
//!
//! Three of those descents applied themselves to every child twice, once
//! through `map_children` and again through `map_body`, which is `2^depth`
//! visits for a nest of depth `depth` rather than one visit per node.
//!
//! The descent is borrow-preserving. The owned `map_entry` walk the passes used
//! has no way to report "no change", so each of them rebuilt the whole entry
//! tree on every run, including the runs that changed nothing.
//! [`rewrite_entry_nodes`] and [`rewrite_entry_bodies`] hand back the caller's
//! `Program` when no rule fired, which is the common case under the optimizer
//! fixpoint: a pass runs to completion once, then runs again to prove it has
//! converged.

use std::borrow::Cow;

use crate::ir::{Node, Program};
use crate::optimizer::{PassAnalysis, PassResult};
use crate::visit::{any_body, any_descendant, map_bodies_cow};

/// `RUN` when `program` carries every node kind in `required` and some node
/// under the entry satisfies `candidate`.
///
/// The kind masks come from `crate::ir::stats` and are an O(1) filter read off
/// the cached `ProgramStats` bitset: a pass that rewrites `If` need not walk a
/// program with no `If` in it at all. Every mask in `required` must be present,
/// so a pass whose rule needs two kinds passes both.
///
/// The walk itself is [`any_descendant`], whose nesting comes from
/// `child_bodies`. Use [`analyze_candidate_bodies`] instead when the candidate
/// is a relation between siblings rather than a property of one node.
#[must_use]
pub(crate) fn analyze_candidates(
    program: &Program,
    required: &[u32],
    candidate: &mut impl FnMut(&Node) -> bool,
) -> PassAnalysis {
    if !carries_every_kind(program, required) {
        return PassAnalysis::SKIP;
    }
    if program
        .entry()
        .iter()
        .any(|node| any_descendant(node, candidate))
    {
        PassAnalysis::RUN
    } else {
        PassAnalysis::SKIP
    }
}

/// `RUN` when `program` carries every node kind in `required` and `candidate`
/// holds for the entry body or any body nested under it.
///
/// The body form exists for a rule whose candidate is a relation between
/// adjacent siblings: two fusable loops are invisible from either loop alone,
/// so the scan has to see the enclosing sequence.
#[must_use]
pub(crate) fn analyze_candidate_bodies(
    program: &Program,
    required: &[u32],
    candidate: &mut impl FnMut(&[Node]) -> bool,
) -> PassAnalysis {
    if !carries_every_kind(program, required) {
        return PassAnalysis::SKIP;
    }
    if any_body(program.entry(), candidate) {
        PassAnalysis::RUN
    } else {
        PassAnalysis::SKIP
    }
}

/// True iff `program`'s cached kind bitset carries every mask in `required`.
///
/// The O(1) half of [`analyze_candidates`], separately callable by a pass whose
/// candidate predicate needs derived facts: deriving them walks the program, so
/// the kind filter has to come first.
#[must_use]
pub(crate) fn carries_every_kind(program: &Program, required: &[u32]) -> bool {
    let stats = program.stats();
    required.iter().all(|kind| stats.has_any_node_kind(*kind))
}

/// Apply `node_rule` bottom-up to every node under `program`'s entry.
///
/// `node_rule` sees a node whose own bodies have already been rewritten, and
/// returns `None` to keep it or `Some(replacement)` to replace it with zero or
/// more nodes.
#[must_use]
pub(crate) fn rewrite_entry_nodes<N>(program: Program, node_rule: &mut N) -> PassResult
where
    N: FnMut(&Node) -> Option<Vec<Node>>,
{
    rewrite_entry(program, node_rule, &mut |_: &[Node]| None)
}

/// Apply `body_rule` bottom-up to `program`'s entry body and every body nested
/// under it.
///
/// `body_rule` sees a body whose nodes have already been rewritten, and returns
/// `None` to keep it or `Some(replacement)` for the sequence that replaces it.
/// A rule whose decision is a relation between adjacent siblings needs this
/// form: dropping a store that a later sibling overwrites is not a property of
/// either store alone.
#[must_use]
pub(crate) fn rewrite_entry_bodies<B>(program: Program, body_rule: &mut B) -> PassResult
where
    B: FnMut(&[Node]) -> Option<Vec<Node>>,
{
    rewrite_entry(program, &mut |_: &Node| None, body_rule)
}

/// `body` without the nodes `drop_node` accepts, or `None` when it accepts
/// none of them.
///
/// The narrowest body rule there is, and the one two cleanup passes wrote out
/// by hand: drop a sibling on a per-node predicate, and report whether any went.
/// Pair it with [`rewrite_entry_bodies`].
#[must_use]
pub(crate) fn without_nodes(body: &[Node], drop_node: impl Fn(&Node) -> bool) -> Option<Vec<Node>> {
    if !body.iter().any(&drop_node) {
        return None;
    }
    Some(
        body.iter()
            .filter(|node| !drop_node(node))
            .cloned()
            .collect(),
    )
}

/// How many leading nodes the two arms of an `If` share, as judged pairwise by
/// `hoistable_pair`.
///
/// Two passes hoist a common prefix out of both arms and differ only in which
/// pairs they will move: one takes any identical observably-free `Let`, the
/// other only a `Let` of a read-only load whose name nothing else in the
/// enclosing body binds. The walk that turns a pairwise predicate into a prefix
/// length is the same decision in both, and is stated here.
#[must_use]
pub(crate) fn common_prefix_len(
    then: &[Node],
    otherwise: &[Node],
    hoistable_pair: impl Fn(&Node, &Node) -> bool,
) -> usize {
    then.iter()
        .zip(otherwise.iter())
        .take_while(|(t, o)| hoistable_pair(t, o))
        .count()
}

/// Apply both rules bottom-up, the node rule to each node and the body rule to
/// each sequence of them.
///
/// The driver owns the changed flag, so a rule cannot report a rewrite it did
/// not make, or make one it did not report.
///
/// A node replaced by several nodes splices flat at the entry level and stays
/// inside a `Node::Block` in a nested position. That is not a formatting
/// choice: `Block` is a scope, so a `let` in a replacement that came out of a
/// branch keeps the scope it was hoisted out of instead of becoming visible to
/// the branch's later siblings.
///
/// Nothing is allocated on the path where no rule fires. Each unchanged body
/// stays borrowed, so an unchanged node is the caller's node and an unchanged
/// program is the caller's `Program`, down to the `Arc` behind every
/// `Region::body`.
#[must_use]
pub(crate) fn rewrite_entry<N, B>(
    program: Program,
    node_rule: &mut N,
    body_rule: &mut B,
) -> PassResult
where
    N: FnMut(&Node) -> Option<Vec<Node>>,
    B: FnMut(&[Node]) -> Option<Vec<Node>>,
{
    let mut rules = Rules {
        node: node_rule,
        body: body_rule,
    };
    match rewrite_body(program.entry(), &mut rules, SPLICE_FLAT) {
        Cow::Borrowed(_) => PassResult {
            program,
            changed: false,
        },
        Cow::Owned(entry) => PassResult {
            program: program.with_rewritten_entry(entry),
            changed: true,
        },
    }
}

/// The pass's two rules, held together so the recursive descent takes one
/// parameter and monomorphizes once per pass rather than once per depth.
struct Rules<'rule, N, B> {
    node: &'rule mut N,
    body: &'rule mut B,
}

/// How a multi-node replacement lands in the body it replaces one node of.
type Placement = fn(&mut Vec<Node>, Vec<Node>);

/// Entry level: the replacement's nodes become siblings of the node they
/// replace, which is what a hoist out of the outermost body means.
const SPLICE_FLAT: Placement = Vec::extend;

/// Nested level: the replacement stays one node, so a binding it carries keeps
/// the scope of the body it came out of.
const KEEP_SCOPED: Placement = |out, replacement| out.push(one_node(replacement));

/// Rewrite every node of `nodes` bottom-up, then the resulting sequence.
fn rewrite_body<'a, N, B>(
    nodes: &'a [Node],
    rules: &mut Rules<'_, N, B>,
    place: Placement,
) -> Cow<'a, [Node]>
where
    N: FnMut(&Node) -> Option<Vec<Node>>,
    B: FnMut(&[Node]) -> Option<Vec<Node>>,
{
    let mut rewritten: Option<Vec<Node>> = None;
    for (index, node) in nodes.iter().enumerate() {
        let descended = descend(node, rules);
        match (rules.node)(&descended) {
            Some(replacement) => place(
                rewritten.get_or_insert_with(|| nodes[..index].to_vec()),
                replacement,
            ),
            None => match descended {
                Cow::Borrowed(_) if rewritten.is_none() => {}
                Cow::Borrowed(borrowed) => {
                    if let Some(out) = rewritten.as_mut() {
                        out.push(borrowed.clone());
                    }
                }
                Cow::Owned(owned) => rewritten
                    .get_or_insert_with(|| nodes[..index].to_vec())
                    .push(owned),
            },
        }
    }
    let rewritten = rewritten.map_or(Cow::Borrowed(nodes), Cow::Owned);
    match (rules.body)(&rewritten) {
        Some(replacement) => Cow::Owned(replacement),
        None => rewritten,
    }
}

/// Rewrite the bodies `node` nests, without applying the node rule to `node`.
///
/// Which body slots `node` has, and how it is rebuilt from them, is
/// `map_bodies_cow`'s decision. The driver states neither.
fn descend<'a, N, B>(node: &'a Node, rules: &mut Rules<'_, N, B>) -> Cow<'a, Node>
where
    N: FnMut(&Node) -> Option<Vec<Node>>,
    B: FnMut(&[Node]) -> Option<Vec<Node>>,
{
    map_bodies_cow(node, &mut |body| rewrite_body(body, rules, KEEP_SCOPED))
}

/// Collapse a replacement sequence into the one node a child position holds.
fn one_node(mut replacement: Vec<Node>) -> Node {
    if replacement.len() == 1 {
        replacement.pop().unwrap_or_else(|| Node::Block(Vec::new()))
    } else {
        Node::Block(replacement)
    }
}
