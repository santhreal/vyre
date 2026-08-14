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
//! What this does NOT catch: a pass that reports `changed: true` is free to
//! rebuild, and this suite says nothing about whether it needed to.

use std::sync::Arc;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Ident, Node, Program};
use vyre_foundation::optimizer::registered_passes;

/// Passes that still allocate a new entry tree while reporting no change.
///
/// Every name here is outstanding work, not an exemption on principle. A pass
/// leaves this list by routing its descent through a borrowing rewrite; it
/// enters the list only when somebody records why it cannot.
const REBUILDS_WHEN_UNCHANGED: &[&str] = &[
    // Each of these still descends with an owned walk that cannot report "no
    // change", so it allocates a fresh entry tree even on the runs where its
    // rule never fires. Three carry context down the tree, which the shared
    // borrowing descent does not take yet: `loop_var_range_fold` needs the
    // enclosing induction range, `loop_licm` needs the loop it is hoisting out
    // of, and `loop_software_pipeline` needs the stage assignment. The rest are
    // whole-body rewriters whose rule is a fold over a sibling sequence rather
    // than a per-node replacement.
    "barrier_coalesce",
    "canonicalize",
    "cse",
    "dce",
    "fusion",
    "loop_licm",
    "loop_redundant_bound_check_elide",
    "loop_software_pipeline",
    "loop_var_range_fold",
    "normalize_atomics",
    "region_fusion_hint",
    "rematerialize_cheap_let",
];

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
    let mut unchanged = 0usize;
    for pass in &passes {
        let program = inert_program();
        let before = region_body_arc(&program);
        let result = pass.transform(program);
        if result.changed {
            continue;
        }
        unchanged += 1;
        let after = region_body_arc(&result.program);
        if !Arc::ptr_eq(&before, &after) {
            rebuilt.push(pass.pass_id());
        }
    }

    assert!(
        unchanged > 0,
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
