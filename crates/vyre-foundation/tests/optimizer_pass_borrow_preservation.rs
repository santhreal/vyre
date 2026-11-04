//! A pass that reports no change must not have rebuilt the program.
//!
//! WHY: `Program::map_entry` takes the entry by value and hands back a new one,
//! so it has no way to say "nothing here changed". Every structural pass used
//! it, which means each one allocated a fresh entry tree on every invocation
//! and dropped the cached facts hanging off the old one, including on the
//! invocations that changed nothing. Under the optimizer fixpoint that is the
//! common case: a pass runs to completion once, then runs again to prove it has
//! converged, and the second run was as expensive as the first.
//!
//! The observable contract is that a `PassResult` with `changed: false` carries
//! the caller's program, down to the `Arc` behind a `Region` body. This suite
//! asserts it for every pass the live registry returns, so a new pass has to
//! either preserve the borrow or be recorded in `REBUILDS_WHEN_UNCHANGED`
//! below. It cannot be added silently.
//!
//! The mirror obligation is that `changed: true` means the program differs.
//! Without it an empty [`REBUILDS_WHEN_UNCHANGED`] proves less the more passes
//! overreport: a pass that claims a change it did not make is exempt from the
//! borrow assertion, and it also costs the fixpoint another whole iteration to
//! discover the same thing. `Program` compares structurally, so the claim is
//! checked against the value rather than against a list of names.
//!
//! What this does NOT catch: a pass that reports `changed: true` and did change
//! something is free to rebuild, and this suite says nothing about whether the
//! change was worth making.

use std::sync::Arc;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::optimizer::registered_passes;
use vyre_foundation::visit::any_descendant;

/// Passes that still allocate a new entry tree while reporting no change.
///
/// Empty, and it stays empty. A pass enters this list only when somebody
/// records why its descent cannot report "no change"; it leaves by gating the
/// rewrite on the same analysis the scheduler already consults, so the
/// converged fixpoint iteration hands the caller's own program straight back.
const REBUILDS_WHEN_UNCHANGED: &[&str] = &[];

/// Passes whose change report on the inert fixture is a real change.
///
/// The fixture is inert to the *statement tree*, not to buffer and geometry
/// facts, and these two read those: `region_inline` removes the wrapper
/// `Program::wrapped` created, and `vectorization` promotes layout hints from
/// buffer shape facts.
///
/// `autotune` was a third until it was deleted: it rewrote dispatch dimensions
/// and workgroup bounds, which is schedule selection, and selection has one
/// owner.
///
/// `dead_buffer_elim` was a fourth until the observable-output assertion below
/// caught what it was changing: it rooted liveness in `is_output()` rather than
/// the cross-backend `is_backend_allocated_output()`, so a plain `WriteOnly`
/// storage buffer was not live and the only store in the program was deleted.
///
/// Membership is not what proves anything: the suite compares each pass's
/// report against the value it returned, and holds every one of them to the
/// fixture's observable output. The list records which passes are expected to
/// have work here, so one quietly going inert is visible too.
const CHANGES_AN_INERT_PROGRAM: &[&str] = &["region_inline", "vectorization"];

/// A program no structural pass has anything to do with: one region, one store
/// of a literal to a distinct buffer, no control flow, no dead binding, no
/// redundant load. A pass that reports a change here is reporting a change it
/// did not make.
fn inert_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::WriteOnly, DataType::U32).with_count(4),
            BufferDecl::storage("src", 1, BufferAccess::ReadOnly, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::load("src", Expr::u32(0)),
        )],
    )
}

/// Whether `program` still stores to the buffer the fixture exists to write.
///
/// Every pass here is ABI-preserving, so whatever it does to the wrapper, the
/// buffer set, or the geometry, the store to `out` has to survive. Without this
/// a pass could satisfy both assertions below by deleting the program.
///
/// The descent is `visit::any_descendant` rather than a match here: a pass may
/// legitimately move the store under a guard or a loop, and a hand-written walk
/// that knew only about `Region` would read that as a deletion, which is what
/// it did.
fn writes_the_output(program: &Program) -> bool {
    let out = Ident::from("out");
    let mut is_store = |node: &Node| matches!(node, Node::Store { buffer, .. } if *buffer == out);
    program
        .entry()
        .iter()
        .any(|node| any_descendant(node, &mut is_store))
}

/// The `Arc` behind the first `Region` body under `program`'s entry.
///
/// `Program::wrapped` puts the entry inside a region wrapper, and that `Arc` is
/// the cheapest observable witness of a rebuild: a walk that reconstructs the
/// tree cannot preserve it, and one that leaves the body alone cannot lose it.
fn region_body_arc(program: &Program) -> Arc<Vec<Node>> {
    fn find(nodes: &[Node]) -> Option<Arc<Vec<Node>>> {
        for node in nodes {
            if let Node::Region { body, .. } = node {
                return Some(Arc::clone(body));
            }
        }
        None
    }
    find(program.entry()).expect("Fix: the fixture must keep its region wrapper")
}

#[test]
fn a_pass_that_reports_no_change_hands_back_the_callers_program() {
    let passes = registered_passes().expect("Fix: the live pass registry must be valid");
    assert!(
        passes.len() >= 19,
        "the registry returned {} passes, which is too few to be the live set",
        passes.len()
    );

    let mut rebuilt: Vec<&'static str> = Vec::new();
    let mut claimed_change: Vec<&'static str> = Vec::new();
    let mut claimed_without_changing: Vec<&'static str> = Vec::new();
    for pass in &passes {
        let program = inert_program();
        let before = region_body_arc(&program);
        let result = pass.transform(program);
        assert!(
            writes_the_output(&result.program),
            "`{}` dropped the fixture's observable output. The store to `out` is \
             the whole program; a pass that removes it has optimized away the \
             thing the program exists to do.",
            pass.pass_id()
        );
        if result.changed {
            claimed_change.push(pass.pass_id());
            if result.program == inert_program() {
                claimed_without_changing.push(pass.pass_id());
            }
            continue;
        }
        let after = region_body_arc(&result.program);
        if !Arc::ptr_eq(&before, &after) {
            rebuilt.push(pass.pass_id());
        }
    }

    claimed_without_changing.sort_unstable();
    assert_eq!(
        claimed_without_changing,
        Vec::<&str>::new(),
        "a pass reported `changed: true` and returned a structurally identical \
         program. That costs the fixpoint another whole iteration to discover \
         the same thing, and exempts the pass from the borrow assertion below. \
         Fix: report the change its analysis actually found."
    );

    let mut claimed_expected: Vec<&str> = CHANGES_AN_INERT_PROGRAM.to_vec();
    claimed_expected.sort_unstable();
    claimed_change.sort_unstable();
    assert_eq!(
        claimed_change, claimed_expected,
        "the set of passes with something to do on the inert fixture moved. \
         Fix: if a pass stopped acting on it, say so in CHANGES_AN_INERT_PROGRAM; \
         if one started, check that the change is one the fixture warrants."
    );
    assert!(
        claimed_change.len() < passes.len(),
        "no pass left the fixture alone, so the fixture is not inert and this suite proves nothing"
    );
    let mut expected: Vec<&str> = REBUILDS_WHEN_UNCHANGED.to_vec();
    expected.sort_unstable();
    rebuilt.sort_unstable();
    assert_eq!(
        rebuilt, expected,
        "a pass reporting `changed: false` rebuilt the entry tree. Fix: route its \
         descent through a rewrite that can report no change, or record it in \
         REBUILDS_WHEN_UNCHANGED with the reason it cannot."
    );
}

#[test]
fn the_witness_would_see_a_rebuild() {
    // Guards the assertion above: if `region_body_arc` could not tell a rebuilt
    // body from a preserved one, the suite would pass for every pass in the
    // registry, including one that rebuilds on every call.
    let program = inert_program();
    let before = region_body_arc(&program);
    let same = region_body_arc(&program);
    assert!(Arc::ptr_eq(&before, &same));

    let rebuilt = program.map_entry(|entry| {
        entry
            .into_iter()
            .map(|node| match node {
                Node::Region {
                    generator,
                    source_region,
                    body,
                } => Node::Region {
                    generator,
                    source_region,
                    body: Arc::new(body.as_ref().clone()),
                },
                other => other,
            })
            .collect()
    });
    assert!(
        !Arc::ptr_eq(&before, &region_body_arc(&rebuilt)),
        "an identical-but-reallocated body must not compare equal by pointer"
    );
    assert_eq!(
        Ident::from("out"),
        match &region_body_arc(&rebuilt)[0] {
            Node::Store { buffer, .. } => buffer.clone(),
            other => panic!("Fix: the fixture body must stay a store, got {other:?}"),
        },
        "the rebuilt program must still be the same program"
    );
}
