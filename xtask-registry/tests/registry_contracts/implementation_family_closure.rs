//! Every row of the implementation-family taxonomy names an operation the live
//! registry publishes.
//!
//! WHY: the taxonomy in `xtask::gates::implementation_family` is a table rather
//! than a match so this closure can be proved, and until this file existed
//! nothing proved it. A row whose operation was renamed, withdrawn or demoted to
//! a composition classifies nothing: `implementation_family_id` never matches
//! it, `same_implementation_family` never groups it, and the dedup audit keeps
//! reporting zero findings while a reader takes the row as evidence the
//! operation was reviewed. That is how `vyre-libs::math::sinkhorn_scale` stayed
//! in the table after it stopped being a registration, and its family then held
//! exactly one live claimant whose body the named builder does not emit.
//!
//! The registered set comes from the registry at run time, so withdrawing a
//! registration turns this red in the same commit rather than at the next
//! reading of the table.

use std::collections::BTreeSet;

use xtask::gates::implementation_family::{
    DISTINCT_FAMILY_PAIRS, IMPLEMENTATION_FAMILY_ROWS, REVIEWED_DISTINCT_OPERATIONS,
};

fn registered_operation_ids() -> BTreeSet<&'static str> {
    vyre_registry_link::operation::live_operation_registry()
        .iter()
        .map(|entry| entry.id)
        .collect()
}

/// WHY: an empty table would satisfy every closure assertion below, so the
/// registry walk itself is asserted to be a walk over a real registry first.
#[test]
fn the_registry_and_the_taxonomy_are_both_populated() {
    let registered = registered_operation_ids();
    assert!(
        registered.len() > 200,
        "Fix: the live registry published {} operations; a walk this small means a registration source was dropped at link time and every closure check below would pass for the wrong reason",
        registered.len()
    );
    assert!(
        IMPLEMENTATION_FAMILY_ROWS.len() > 50,
        "Fix: the family taxonomy holds {} rows; a table this small classifies nothing and the audit's suppressions are unrecorded",
        IMPLEMENTATION_FAMILY_ROWS.len()
    );
    assert!(
        !REVIEWED_DISTINCT_OPERATIONS.is_empty(),
        "Fix: no reviewed-distinct pair is recorded, so no shape verdict carries a reason"
    );
    assert!(
        !DISTINCT_FAMILY_PAIRS.is_empty(),
        "Fix: no distinct-family pair is recorded, so no family separation is recorded"
    );
}

#[test]
fn every_family_row_names_a_registered_operation() {
    let registered = registered_operation_ids();
    for (op_id, family) in IMPLEMENTATION_FAMILY_ROWS {
        assert!(
            registered.contains(op_id),
            "Fix: the family row `{op_id}` -> `{family}` names no registered operation; delete the row if the operation was withdrawn, or name the id the registry publishes if it was renamed"
        );
    }
}

#[test]
fn every_reviewed_pair_names_two_registered_operations() {
    let registered = registered_operation_ids();
    for (one, other, _) in REVIEWED_DISTINCT_OPERATIONS {
        for op_id in [one, other] {
            assert!(
                registered.contains(op_id),
                "Fix: the reviewed-distinct pair `{one}` / `{other}` names `{op_id}`, which no registration publishes; a pair the audit can never form records a judgment nothing consults"
            );
        }
    }
}

/// WHY: a family exists to group live operations. A family whose only claimants
/// were withdrawn still reads as a shared builder, and the in-crate test that
/// counts claimants counts rows, not registrations, so it cannot see this.
#[test]
fn every_family_groups_at_least_two_registered_operations_or_is_paired() {
    let registered = registered_operation_ids();
    let families: BTreeSet<&str> = IMPLEMENTATION_FAMILY_ROWS
        .iter()
        .map(|(_, family)| *family)
        .collect();
    for family in families {
        let live_claimants = IMPLEMENTATION_FAMILY_ROWS
            .iter()
            .filter(|(op_id, claimed)| *claimed == family && registered.contains(op_id))
            .count();
        let paired = DISTINCT_FAMILY_PAIRS
            .iter()
            .any(|(one, other)| *one == family || *other == family);
        assert!(
            live_claimants > 1 || paired,
            "Fix: `{family}` is claimed by {live_claimants} registered operation(s) and is in no distinct-family pair; either the family groups nothing and its rows go, or the second claimant is missing"
        );
    }
}
