//! Barrier placement validation.
//!
//! Workgroup barriers in GPU shaders must only appear in uniform control
//! flow: every thread in the workgroup must reach the barrier or none
//! must reach it. This module checks that barrier nodes are not placed
//! inside divergent branches, catching a class of bugs that would
//! otherwise deadlock or produce undefined behavior on the GPU.

use crate::validate::{ValidationLocation, ValidationPhase};
mod exit_uniformity;

use rustc_hash::FxHashMap;

use self::exit_uniformity::exits_after_last_barrier_are_uniform;
use crate::ir_inner::model::expr::Ident;
use crate::ir_inner::model::node::Node;
use crate::memory_model::MemoryOrdering;
use crate::validate::binding::Binding;
use crate::validate::{err, ValidationError};
use crate::visit::any_descendant;

/// Ensure a barrier is not placed inside divergent control flow.
///
/// A barrier inside an `If` or `Loop` whose condition is not uniform
/// across the workgroup is illegal in vyre. This function appends a
/// validation error when `divergent` is `true`.
///
/// # Examples
///
/// `check_barrier` is `pub(crate)`; it's exercised indirectly through
/// [`crate::validate::rule_pipeline::validate`] when a program contains a
/// `Node::Barrier { ordering: vyre_foundation::ir::MemoryOrdering::SeqCst }` inside a divergent `Node::If`. See the unit tests on
/// [`crate::validate::rule_pipeline::validate`] for a runnable example.
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
            "V010",
            ValidationPhase::Memory,
            ValidationLocation::Program,
            "barrier may be reached by only part of a workgroup".to_string(),
            "move the barrier to uniform control flow.".to_string(),
        ));
    }
    if !ordering.is_valid_for_barrier() {
        errors.push(err(
            "V043",
            ValidationPhase::Memory,
            ValidationLocation::Program,
            format!(
                "barrier uses memory ordering `{ordering:?}`, but barriers must synchronize memory"
            ),
            "use Acquire, Release, AcqRel, or SeqCst; use no barrier at all for Relaxed."
                .to_string(),
        ));
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
/// behind `scallop_join`, and the DCE fixpoint in `vyre-pass-engine`. The
/// first was root-caused from an intermittent wrong answer downstream, not from
/// a hang, which is what this rule's error message means by costing answers
/// rather than liveness (reported by `ExactnessRegression`, whose measurement of
/// the original was 4 wrong results in 30 runs with the barrier removed against
/// 0 in 60 with it restored).
///
/// The repair cost depends on the exit proof. `persistent_fixpoint` reused a
/// consecutive barrier. The lineage programs retain a genuine trailing barrier
/// because their exit paths are not proven collective. The DCE fixpoint no
/// longer pays that barrier: its exit reads one scalar address immediately
/// after an acquiring barrier, with no intervening write.
///
/// V055 derives this carve-out conservatively. It accepts an unconditional
/// return or a return guarded only by uniform expressions and barrier-settled
/// loads at uniform indices. A store, atomic, asynchronous write, collective,
/// opaque node, divergent index, lane-dependent guard, or release-only barrier
/// invalidates the proof. The ordinary uniformity analyzer still rejects loads;
/// only this back-edge analysis credits the explicit synchronization.
///
/// This distinction is load-bearing. A collective return means every lane exits
/// or every lane takes the back edge, so no sibling can be stranded. Any
/// uncertainty remains V055 and requires a trailing unconditional barrier.
///
/// # Errors
///
/// Appends a `ValidationError` with code `V055` when `body` contains a barrier
/// and a potentially lane-dependent invocation can return after the last one.
pub(crate) fn check_loop_back_edge(
    body: &[Node],
    scope: &FxHashMap<Ident, Binding>,
    errors: &mut Vec<ValidationError>,
) {
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
    // A barrier-settled exit does not need another barrier when its complete
    // control path is proven workgroup-uniform. Every lane then returns together
    // or every lane takes the back edge. The proof is deliberately local to the
    // last unconditional barrier: any intervening write invalidates uniform
    // loads from that buffer, and any lane-dependent guard keeps V055 active.
    if exits_after_last_barrier_are_uniform(&steps[..=last_exit], scope) {
        return;
    }
    errors.push(err(
        "V055",
        ValidationPhase::Memory,
        ValidationLocation::Program,
        "an invocation can return from a synchronizing loop body after its last barrier, \
         so the exit and the next iteration's writes are unordered across the back edge. One \
         invocation can take the back edge and write while a sibling has not yet reached the \
         exit; the sibling then leaves the kernel while the rest keep iterating, freezing the \
         data it owns partway through. Nothing hangs, because a barrier does not count \
         invocations that already returned, so this costs answers and not liveness, and one \
         workgroup is enough to hit it"
            .to_string(),
        "either put an unconditional barrier after the \
         early exit as the last loop-body node, or make every return guard workgroup-uniform; \
         a guard that loads writable memory needs an acquiring barrier immediately before it, \
         one uniform index, and no intervening write."
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
            // Only transparent wrappers (`Block`, `Region`) splice: an unknown variant stays a single statement, which is the conservative reading for barrier ordering.
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
    any_descendant(node, &mut |n| matches!(n, Node::Barrier { .. }))
}

/// True when executing `node` can leave the kernel.
///
/// `Node::Return` exits the whole invocation rather than the enclosing loop, so
/// a nested one still ends participation in the outer loop's barriers and is
/// counted here.
fn can_return(node: &Node) -> bool {
    any_descendant(node, &mut |n| matches!(n, Node::Return))
}

#[cfg(test)]
mod back_edge_depth_tests;
#[cfg(test)]
mod tests;
