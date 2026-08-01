//! Barrier placement validation.
//!
//! Workgroup barriers in GPU shaders must only appear in uniform control
//! flow: every thread in the workgroup must reach the barrier or none
//! must reach it. This module checks that barrier nodes are not placed
//! inside divergent branches, catching a class of bugs that would
//! otherwise deadlock or produce undefined behavior on the GPU.

use crate::ir_inner::model::node::Node;
use crate::memory_model::MemoryOrdering;
use crate::validate::{err, ValidationError};

/// Ensure a barrier is not placed inside divergent control flow.
///
/// A barrier inside an `If` or `Loop` whose condition is not uniform
/// across the workgroup is illegal in vyre. This function appends a
/// validation error when `divergent` is `true`.
///
/// # Examples
///
/// `check_barrier` is `pub(crate)`; it's exercised indirectly through
/// [`crate::validate::validate::validate`] when a program contains a
/// `Node::Barrier { ordering: vyre::memory_model::MemoryOrdering::SeqCst }` inside a divergent `Node::If`. See the unit tests on
/// [`crate::validate::validate::validate`] for a runnable example.
///
/// # Errors
///
/// Appends a `ValidationError` with code `V010` when `divergent` is
/// `true`.
#[inline]
pub(crate) fn check_barrier(
    divergent: bool,
    ordering: MemoryOrdering,
    errors: &mut Vec<ValidationError>,
) {
    if divergent {
        errors.push(err(
            "V010: barrier may be reached by only part of a workgroup. Fix: move the barrier to uniform control flow."
                .to_string(),
        ));
    }
    if !ordering.is_valid_for_barrier() {
        errors.push(err(format!(
            "V043: barrier uses memory ordering `{ordering:?}`, but barriers must synchronize memory. Fix: use Acquire, Release, AcqRel, or SeqCst; use no barrier at all for Relaxed."
        )));
    }
}

/// Ensure a synchronizing loop does not let invocations leave the kernel
/// between its last barrier and its back edge.
///
/// # The failure mode, which looks like nothing
///
/// A loop body that ends with an early exit such as
/// `if flag[0] == 0 { Return }`, with no barrier after it, has an UNORDERED
/// access pair across the back edge whenever a later iteration writes the word
/// the exit read. The invocation that takes the back edge first can perform that
/// write while a sibling invocation of the same workgroup has not yet executed
/// the read. The sibling reads the new value, takes the `Return`, and leaves the
/// kernel while the rest keep iterating.
///
/// That is a PARTIAL EXIT, and the reason it needs a validator rule is that
/// nothing about it looks wrong at runtime:
///
/// - The invocations that left stop contributing to the loop body, so the data
///   they own freezes partway through and the dispatch returns a partially
///   computed result.
/// - Nothing hangs and nothing errors. A workgroup barrier does not count
///   invocations that have already returned, so the survivors sail through every
///   later barrier. The defect costs ANSWERS, never liveness.
/// - It needs no second workgroup and no host concurrency. Two invocations in
///   one workgroup are enough, so keeping a dispatch inside a single workgroup
///   does NOT make it safe.
///
/// This shipped in `fixpoint::persistent_fixpoint` and reached a consumer as
/// nondeterministic wrong output, with the loop's own pass counter BELOW its
/// budget because it had exited early rather than run out of iterations.
/// Removing only the guarding barrier reproduced it in 4 of 30 end-to-end runs;
/// restoring it gave 0 of 60.
///
/// # Why the trigger is a barrier and not a dataflow analysis
///
/// The obligation applies only to a COLLECTIVE loop, and the barrier-present
/// condition is load-bearing. NEVER drop it to make this rule fire on every
/// loop: that reads as extra strictness and is actually a regression. A loop
/// with no barrier in its body performs no cross-invocation communication, so it
/// has no race to order and an early exit in it is ordinary control flow.
/// Worse, the demand would be unsatisfiable: invocations of such a loop leave on
/// different iterations, so any barrier added to discharge this rule is one they
/// do not all reach, which [`check_barrier`] correctly refuses as V010. The rule
/// would then reject programs with no legal repair.
///
/// A barrier's presence is therefore the precise signal that invocations are
/// expected to stay in lockstep, which is exactly the expectation an unguarded
/// early exit breaks.
///
/// Know the symptom, because it does not look like this rule when it happens: if
/// that condition is ever dropped, unrelated builders start failing validation
/// with no repair available, since the barrier this rule would demand of them is
/// the one V010 refuses. The tempting response at that point is to delete the
/// whole rule rather than restore one condition, which is how a real race gets
/// readmitted while the change looks like a cleanup.
///
/// This is deliberately a structural rule rather than a proof that the exit
/// value is concurrently written. Deciding that needs buffer aliasing across the
/// whole body, and the remedy costs one barrier in a loop that already
/// synchronizes, so over-strictness here is cheap while a miss returns silently
/// wrong answers.
///
/// That trade has been paid for. This rule has caught four real programs in
/// this tree, all with the same shape (clear the flag, synchronize, step, exit
/// as the LAST node of the body): `persistent_fixpoint`, the
/// `wide_lineage_body` behind `scallop_join_wide`, the `single_word_lineage_body`
/// behind `scallop_join`, and the DCE fixpoint in `vyre-self-substrate`. The
/// first was root-caused from an intermittent wrong answer downstream, not from
/// a hang, which is what this rule's error message means by costing answers
/// rather than liveness (reported by `ExactnessRegression`, whose measurement of
/// the original was 4 wrong results in 30 runs with the barrier removed against
/// 0 in 60 with it restored).
///
/// The repair cost is not uniform, and it is worth knowing before you treat one
/// of these barriers as redundant. In `persistent_fixpoint` it was free, because
/// an existing consecutive barrier could be relocated into the slot. The other
/// three each pay one genuine extra barrier per iteration. They are load
/// bearing: removing one as an optimization reintroduces a silent wrong answer,
/// which is why each is commented at its call site.
///
/// The reach of that over-strictness is worth knowing exactly, because it is
/// wider than "lane-dependent exits". An UNCONDITIONAL return placed after the
/// body's last barrier is refused too, even though every invocation reaches it
/// together and none can be stranded mid iteration. The rule asks whether an
/// invocation can return after the last barrier, never whether the invocations
/// agree about it. That is intended today and is pinned by a test rather than
/// left to be rediscovered: see
/// `vyre-self-substrate/tests/dce_program_back_edge_contract.rs`.
///
/// A uniformity carve-out would be more precise, and the motivating case is
/// `vyre-self-substrate/src/optimizer/dce_program.rs`: its loop exit reads a
/// value the preceding barrier settles, so the exit is workgroup-uniform and
/// provably safe, yet it still has to carry a trailing barrier to satisfy this
/// rule. Deriving uniformity is a real analysis and is deferred, tracked as
/// `FINDING-V055-refuses-provably-uniform-loop-exits`. Whoever takes it should
/// know the bar: derive uniformity rather than assert it, keep refusing the
/// lane-dependent case, and prove the carve-out subsumes the workaround by
/// REMOVING that call-site barrier and finding its tests still green.
///
/// # Errors
///
/// Appends a `ValidationError` with code `V055` when `body` contains a barrier
/// and an invocation can return after the last one.
pub(crate) fn check_loop_back_edge(body: &[Node], errors: &mut Vec<ValidationError>) {
    // `Block` and `Region` execute unconditionally and in order, so they are
    // spliced into the enclosing sequence before any positional reasoning. This
    // is not cosmetic: wrapping a phase in a `Block` is established practice in
    // this tree (`persistent_fixpoint_grid` blocks each wave's transfer body so
    // its `let`s do not become duplicate siblings under V032), so without
    // splicing, both the trigger and the guard below would silently change
    // answers the day someone wrapped a phase that contains a barrier.
    let mut steps: Vec<&Node> = Vec::new();
    splice_straight_line(body, &mut steps);

    // TRIGGER: is this loop collective? Deep, and deliberately permissive: a
    // barrier reachable on ANY path, including inside a conditional or a nested
    // loop, means invocations are expected to stay in lockstep. Over-triggering
    // costs one barrier in a loop that already synchronizes, which is the trade
    // this rule is built on.
    if !steps.iter().any(|node| contains_barrier_anywhere(node)) {
        return;
    }
    // The LAST exit site is the only one that matters: a barrier after it also
    // orders every earlier exit against the back edge.
    let Some(last_exit) = steps.iter().rposition(|node| can_return(node)) else {
        return;
    };
    // GUARD: is the exit ordered against the back edge? Strict, and it must stay
    // strict. Only a barrier that is UNCONDITIONALLY executed after the exit
    // orders anything, which is why this counts `Node::Barrier` at spliced
    // straight-line depth and nothing else. A barrier inside an `If` orders
    // nothing for an invocation that skips the branch, and one inside a nested
    // `Loop` is not executed at all when its trip count is zero. Crediting
    // either would accept the exact race this rule exists to refuse, while
    // looking more thorough than the correct check.
    if steps[last_exit + 1..]
        .iter()
        .any(|node| matches!(node, Node::Barrier { .. }))
    {
        return;
    }
    errors.push(err(
        "V055: an invocation can return from a synchronizing loop body after its last barrier, \
         so the exit and the next iteration's writes are unordered across the back edge. One \
         invocation can take the back edge and write while a sibling has not yet reached the \
         exit; the sibling then leaves the kernel while the rest keep iterating, freezing the \
         data it owns partway through. Nothing hangs, because a barrier does not count \
         invocations that already returned, so this costs answers and not liveness, and one \
         workgroup is enough to hit it. Fix: put a barrier after the early exit, as the last \
         node of the loop body, so the exit is ordered against the back edge."
            .to_string(),
    ));
}

/// Splice `Block` and `Region` contents into the enclosing straight-line
/// sequence, leaving `If` and `Loop` as single opaque steps.
///
/// Both spliced kinds execute unconditionally and in source order, so flattening
/// them preserves the ordering relation the back-edge check reasons about. `If`
/// and `Loop` are NOT spliced: their contents are conditional, and the guard
/// below must never credit a conditional barrier as ordering.
fn splice_straight_line<'a>(nodes: &'a [Node], out: &mut Vec<&'a Node>) {
    for node in nodes {
        match node {
            Node::Block(inner) => splice_straight_line(inner, out),
            Node::Region { body, .. } => splice_straight_line(body, out),
            other => out.push(other),
        }
    }
}

/// True when a barrier is reachable anywhere under `node`, conditionally or not.
///
/// This is the TRIGGER depth and it is intentionally different from the guard's.
/// Do not unify the two: detecting that a loop is collective should be
/// permissive, while deciding that an exit is ordered must not be. Reading them
/// side by side invites merging them, and merging them in either direction is a
/// defect: permissive guarding accepts the race, strict triggering misses it.
fn contains_barrier_anywhere(node: &Node) -> bool {
    match node {
        Node::Barrier { .. } => true,
        Node::If {
            then, otherwise, ..
        } => {
            then.iter().any(contains_barrier_anywhere)
                || otherwise.iter().any(contains_barrier_anywhere)
        }
        Node::Loop { body, .. } => body.iter().any(contains_barrier_anywhere),
        Node::Block(nodes) => nodes.iter().any(contains_barrier_anywhere),
        Node::Region { body, .. } => body.iter().any(contains_barrier_anywhere),
        _ => false,
    }
}

/// True when executing `node` can leave the kernel.
///
/// `Node::Return` exits the whole invocation rather than the enclosing loop, so
/// a nested one still ends participation in the outer loop's barriers and is
/// counted here.
fn can_return(node: &Node) -> bool {
    match node {
        Node::Return => true,
        Node::If {
            then, otherwise, ..
        } => then.iter().any(can_return) || otherwise.iter().any(can_return),
        Node::Loop { body, .. } => body.iter().any(can_return),
        Node::Block(nodes) => nodes.iter().any(can_return),
        Node::Region { body, .. } => body.iter().any(can_return),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;

    #[test]
    fn divergent_barrier_emits_v010() {
        let mut errors = Vec::new();
        check_barrier(true, MemoryOrdering::SeqCst, &mut errors);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message().contains("V010"));
    }

    #[test]
    fn uniform_barrier_is_valid() {
        let mut errors = Vec::new();
        check_barrier(false, MemoryOrdering::SeqCst, &mut errors);
        assert!(errors.is_empty());
    }

    #[test]
    fn relaxed_barrier_is_rejected() {
        let mut errors = Vec::new();
        check_barrier(false, MemoryOrdering::Relaxed, &mut errors);
        assert!(errors.iter().any(|error| error.message().contains("V043")));
    }

    fn barrier() -> Node {
        Node::Barrier {
            ordering: MemoryOrdering::SeqCst,
        }
    }

    /// The exact pre-fix shape of `fixpoint::persistent_fixpoint`: a
    /// synchronizing loop whose last node is an early exit.
    #[test]
    fn exit_after_the_last_barrier_emits_v055() {
        let mut errors = Vec::new();
        check_loop_back_edge(
            &[
                barrier(),
                Node::If {
                    cond: Expr::bool(true),
                    then: vec![Node::Return],
                    otherwise: Vec::new(),
                },
            ],
            &mut errors,
        );
        assert!(
            errors.iter().any(|error| error.message().contains("V055")),
            "an early exit after the loop's last barrier must be refused"
        );
    }

    /// The post-fix shape: a barrier after the exit orders it against the back
    /// edge, which is the entire fix.
    #[test]
    fn a_barrier_after_the_exit_is_accepted() {
        let mut errors = Vec::new();
        check_loop_back_edge(
            &[
                barrier(),
                Node::If {
                    cond: Expr::bool(true),
                    then: vec![Node::Return],
                    otherwise: Vec::new(),
                },
                barrier(),
            ],
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "a barrier on the back edge discharges the obligation: {errors:?}"
        );
    }

    /// The rule MUST NOT fire on a loop that never synchronizes. Such a loop has
    /// no cross-invocation communication to order, its invocations legitimately
    /// leave on different iterations, and a barrier they do not all reach would
    /// itself be illegal. A false positive here would be unfixable.
    #[test]
    fn an_exit_in_a_loop_with_no_barrier_is_accepted() {
        let mut errors = Vec::new();
        check_loop_back_edge(
            &[Node::If {
                cond: Expr::bool(true),
                then: vec![Node::Return],
                otherwise: Vec::new(),
            }],
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "a loop with no barrier has no collective contract to break: {errors:?}"
        );
    }

    /// A `Return` exits the invocation, not the enclosing loop, so one nested
    /// inside an inner loop still ends participation in the outer loop's
    /// barriers and must be caught.
    #[test]
    fn a_nested_exit_after_the_last_barrier_emits_v055() {
        let mut errors = Vec::new();
        check_loop_back_edge(
            &[
                barrier(),
                Node::Loop {
                    var: "inner".into(),
                    from: Expr::u32(0),
                    to: Expr::u32(4),
                    body: vec![Node::If {
                        cond: Expr::bool(true),
                        then: vec![Node::Return],
                        otherwise: Vec::new(),
                    }],
                },
            ],
            &mut errors,
        );
        assert!(
            errors.iter().any(|error| error.message().contains("V055")),
            "a nested early exit still leaves the outer loop's barriers"
        );
    }
}

#[cfg(test)]
mod back_edge_depth_tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;

    fn barrier() -> Node {
        Node::Barrier {
            ordering: MemoryOrdering::SeqCst,
        }
    }

    fn exit_guard() -> Node {
        Node::If {
            cond: Expr::bool(true),
            then: vec![Node::Return],
            otherwise: Vec::new(),
        }
    }

    /// Locks out a MISSED defect: a loop whose barrier sits inside a `Node::Block`
    /// must still be recognized as collective.
    ///
    /// Wrapping a phase in a `Block` is established practice in this tree.
    /// `fixpoint::persistent_fixpoint_grid` blocks each wave's transfer body so
    /// its `let` bindings do not become duplicate siblings under V032. When the
    /// trigger only looked at top-level nodes, the day someone wrapped a phase
    /// containing the barrier this rule went silent on exactly the loop shape it
    /// was written for, and accepted an unguarded exit without a word.
    #[test]
    fn a_barrier_inside_a_block_still_makes_the_loop_collective() {
        let mut errors = Vec::new();
        check_loop_back_edge(&[Node::Block(vec![barrier()]), exit_guard()], &mut errors);
        assert!(
            errors.iter().any(|error| error.message().contains("V055")),
            "a barrier nested in a Block must still trigger the back-edge check"
        );
    }

    /// Locks out a FALSE POSITIVE: a correctly placed guarding barrier that
    /// happens to sit inside a `Block` after the exit does order the back edge,
    /// because a `Block` executes unconditionally and in order.
    ///
    /// A false refusal on a program that is already correct is what gets a rule
    /// deleted instead of fixed.
    #[test]
    fn a_guarding_barrier_inside_a_block_is_credited() {
        let mut errors = Vec::new();
        check_loop_back_edge(
            &[barrier(), exit_guard(), Node::Block(vec![barrier()])],
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "a Block executes unconditionally, so a barrier inside one orders the back edge: \
             {errors:?}"
        );
    }

    /// Locks out crediting a CONDITIONAL barrier as ordering. A barrier inside an
    /// `If` after the exit orders nothing for an invocation that skips the
    /// branch, so the race survives. Accepting this would be strictly worse than
    /// a shallow check, because it looks thorough while being wrong.
    #[test]
    fn a_guarding_barrier_inside_an_if_is_not_credited() {
        let mut errors = Vec::new();
        check_loop_back_edge(
            &[
                barrier(),
                exit_guard(),
                Node::If {
                    cond: Expr::bool(true),
                    then: vec![barrier()],
                    otherwise: Vec::new(),
                },
            ],
            &mut errors,
        );
        assert!(
            errors.iter().any(|error| error.message().contains("V055")),
            "a barrier only reached on one branch does not order the back edge"
        );
    }

    /// A barrier inside a nested `Loop` triggers collectiveness but must NOT be
    /// credited as a guard: a nested loop with a zero trip count never executes
    /// its body, so the barrier may not run at all.
    #[test]
    fn a_barrier_only_inside_a_nested_loop_triggers_but_never_guards() {
        let nested = Node::Loop {
            var: "inner".into(),
            from: Expr::u32(0),
            to: Expr::u32(0),
            body: vec![barrier()],
        };
        let mut errors = Vec::new();
        check_loop_back_edge(&[exit_guard(), nested], &mut errors);
        assert!(
            errors.iter().any(|error| error.message().contains("V055")),
            "a nested-loop barrier makes the loop collective but cannot guard the back edge"
        );
    }
}
