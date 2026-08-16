//! Async copy tag discipline.
//!
//! `AsyncLoad` and `AsyncStore` start a copy under a tag and `AsyncWait` ends
//! it. Both reference executors hold the same two rules at run time: a tag that
//! is already in flight cannot be started again, and a wait names a tag that a
//! transfer started. Until this pass existed nothing said so at compile time,
//! so a program that a backend accepts and silently miscopies reached the
//! backend, and the diagnosis arrived only from a reference run nobody had to
//! do.
//!
//! ## Why it decides multi-stage pipelining
//!
//! A depth-D pipeline issues the copy for tile `i + D` while the compute for
//! tile `i` runs, so D copies are in flight at once. The tag is the only thing
//! that distinguishes them, and it is a name in the node rather than a value,
//! so depth D needs D distinct tags and a wait on each before its slot comes
//! round again. A depth-2 pipeline written with one tag is the exact program
//! this pass rejects, and rejecting it is what makes the rotated shape safe to
//! generate: the rotation is correct when, and only when, every tag is waited
//! before the next start of that tag.
//!
//! ## What is reported and what is not
//!
//! Reported only when the defect happens on EVERY path, so a branch that starts
//! a copy and a join that waits for it is not a finding:
//!
//! * a start of a tag that is in flight on every path reaching it,
//! * a wait on a tag that is in flight on no path reaching it.
//!
//! A copy started and never waited is NOT reported. Both executors drop a
//! pending transfer at the end of an invocation rather than failing, a runtime
//! may drain a store after the dispatch, and a program that starts one copy and
//! returns is accepted today.
//!
//! A `Loop` body is analysed twice when its literal bounds prove at least two
//! iterations, so a tag started in the body and left in flight across the back
//! edge is caught. Identical findings from the two analyses collapse, so the
//! second one reports the back edge and nothing else.

use rustc_hash::FxHashSet;

use crate::ir::{Expr, Ident, Node, Program};
use crate::validate::{err, ValidationError, ValidationLocation, ValidationPhase};

/// Tags in flight, held twice: on every path and on some path.
///
/// `must` decides a duplicate start, because reporting one needs the copy to be
/// in flight however the program got here. `may` decides an unmatched wait,
/// because a wait is wrong only when no path could have started the tag.
#[derive(Clone, Default)]
struct InFlight {
    must: FxHashSet<Ident>,
    may: FxHashSet<Ident>,
}

impl InFlight {
    fn start(&mut self, tag: &Ident) {
        self.must.insert(tag.duplicate_handle());
        self.may.insert(tag.duplicate_handle());
    }

    fn finish(&mut self, tag: &Ident) {
        self.must.remove(tag);
        self.may.remove(tag);
    }

    /// State after control flow that could have taken either side.
    fn merge(&mut self, other: &Self) {
        self.must.retain(|tag| other.must.contains(tag));
        self.may.extend(other.may.iter().map(Ident::duplicate_handle));
    }
}

/// Every async tag discipline violation in `program`.
///
/// Findings are deduplicated, which is what lets a loop body be analysed twice
/// without reporting its straight-line defects twice.
#[must_use]
pub(crate) fn check_async_pipeline(program: &Program) -> Vec<ValidationError> {
    if !program.stats().has_any_node_kind(
        crate::ir::stats::NODE_KIND_ASYNC_LOAD
            | crate::ir::stats::NODE_KIND_ASYNC_STORE
            | crate::ir::stats::NODE_KIND_ASYNC_WAIT,
    ) {
        return Vec::new();
    }

    let mut found = Vec::new();
    let mut state = InFlight::default();
    walk_sequence(program.entry(), &mut state, &mut found);

    let mut seen = FxHashSet::default();
    found
        .into_iter()
        .filter(|finding| seen.insert(finding.clone()))
        .map(Finding::into_error)
        .collect()
}

/// One violation, kept as data so two analyses of the same body collapse.
#[derive(Clone, PartialEq, Eq, Hash)]
enum Finding {
    /// A tag started while a copy under the same tag is still in flight.
    DuplicateStart(Ident),
    /// A wait on a tag no transfer started.
    UnmatchedWait(Ident),
}

impl Finding {
    fn into_error(self) -> ValidationError {
        match self {
            Self::DuplicateStart(tag) => err(
                "V131",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                format!(
                    "async transfer tag `{tag}` starts a copy while a copy under the same tag is \
                     still in flight on every path reaching it, so the second copy lands in a \
                     destination the first has not finished writing"
                ),
                "wait the tag before starting it again, or give the second copy its own tag: a \
                 depth-D pipeline needs D tags and one wait per tag before its slot comes round.",
            ),
            Self::UnmatchedWait(tag) => err(
                "V132",
                ValidationPhase::Memory,
                ValidationLocation::Program,
                format!(
                    "async wait on tag `{tag}` has no transfer to wait for on any path reaching it"
                ),
                "start the copy with AsyncLoad or AsyncStore under that tag before waiting, or \
                 drop the wait.",
            ),
        }
    }
}

/// Whether execution can reach the node after the one just walked.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SequenceOutcome {
    Continues,
    Diverged,
}

/// Walk `nodes` in program order, stopping where the sequence leaves the
/// invocation.
fn walk_sequence(
    nodes: &[Node],
    state: &mut InFlight,
    found: &mut Vec<Finding>,
) -> SequenceOutcome {
    for node in nodes {
        if walk_node(node, state, found) == SequenceOutcome::Diverged {
            return SequenceOutcome::Diverged;
        }
    }
    SequenceOutcome::Continues
}

fn walk_node(node: &Node, state: &mut InFlight, found: &mut Vec<Finding>) -> SequenceOutcome {
    match node {
        Node::AsyncLoad { tag, .. } | Node::AsyncStore { tag, .. } => {
            if state.must.contains(tag) {
                found.push(Finding::DuplicateStart(tag.duplicate_handle()));
            }
            state.start(tag);
            SequenceOutcome::Continues
        }
        Node::AsyncWait { tag } => {
            if !state.may.contains(tag) {
                found.push(Finding::UnmatchedWait(tag.duplicate_handle()));
            }
            state.finish(tag);
            SequenceOutcome::Continues
        }
        Node::Return => SequenceOutcome::Diverged,
        Node::If {
            then, otherwise, ..
        } => {
            let mut taken = state.clone();
            let taken_outcome = walk_sequence(then, &mut taken, found);
            let mut skipped = state.clone();
            let skipped_outcome = walk_sequence(otherwise, &mut skipped, found);
            match (taken_outcome, skipped_outcome) {
                (SequenceOutcome::Diverged, SequenceOutcome::Diverged) => {
                    SequenceOutcome::Diverged
                }
                // A branch that leaves the invocation reaches no successor, so
                // it contributes nothing to what is in flight after the `If`.
                (SequenceOutcome::Diverged, SequenceOutcome::Continues) => {
                    *state = skipped;
                    SequenceOutcome::Continues
                }
                (SequenceOutcome::Continues, SequenceOutcome::Diverged) => {
                    *state = taken;
                    SequenceOutcome::Continues
                }
                (SequenceOutcome::Continues, SequenceOutcome::Continues) => {
                    taken.merge(&skipped);
                    *state = taken;
                    SequenceOutcome::Continues
                }
            }
        }
        Node::Loop { from, to, body, .. } => {
            let entry = state.clone();
            if walk_sequence(body, state, found) == SequenceOutcome::Diverged {
                // Every path through the body leaves the invocation, so the
                // only way past the loop is the body never running.
                *state = entry;
                return SequenceOutcome::Continues;
            }
            if runs_at_least_twice(from, to) {
                walk_sequence(body, state, found);
            }
            // The body may not run at all, so what is in flight after the loop
            // is the merge of the body's exit state and the state at entry.
            state.merge(&entry);
            SequenceOutcome::Continues
        }
        Node::Block(body) => walk_sequence(body, state, found),
        Node::Region { body, .. } => walk_sequence(body, state, found),
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::Barrier { .. }
        | Node::IndirectDispatch { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::Opaque(_) => SequenceOutcome::Continues,
    }
}

/// True when literal bounds prove the body runs at least twice.
///
/// A non-literal bound proves nothing, so the back edge is not analysed and no
/// finding is reported for it.
fn runs_at_least_twice(from: &Expr, to: &Expr) -> bool {
    match (from, to) {
        (Expr::LitU32(lo), Expr::LitU32(hi)) => hi.saturating_sub(*lo) >= 2,
        (Expr::LitI32(lo), Expr::LitI32(hi)) => hi.saturating_sub(*lo) >= 2,
        _ => false,
    }
}
