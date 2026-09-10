//! A tier reader returns exactly its own tier, so reading one tier through
//! another reader returns nothing rather than a partial answer.
//!
//! Closes: the class where a rule reads library compositions through an
//! intrinsic-tier projection. `vyre_primitives::operation_catalog` selects
//! `OperationTier::Intrinsic` and `vyre_libs::operation_catalog` selects
//! `OperationTier::Library`, so a caller that filters the intrinsic reader for a
//! `vyre-libs::` id iterates an empty set and every assertion under that filter
//! passes without checking anything. `vyre-libs-bitset` shipped exactly that:
//! its bitset coverage loop read the intrinsic catalog, so the loop body never
//! ran and the only reachable assertion failed on a registration that was
//! present.
//!
//! The tier space is [`OperationTier::ALL`], which `vyre-foundation` closes at
//! compile time: a new variant stops that roster compiling until it is listed.
//! Every tier read here therefore comes from source at run time, and a tier
//! with no recorded reader decision panics instead of passing.
//!
//! Does not catch: a caller that names the wrong reader for its own purpose.
//! Nothing in the registry records which tier a given assertion meant to read,
//! so this file pins what each reader returns and leaves the choice of reader to
//! the call site. It also says nothing about `vyre_primitives::hardware`, which
//! selects on the registration category rather than the tier and is asserted
//! against the tier readers in `conform/vyre-conform/tests/op_matrix_truth`.

use std::collections::BTreeSet;

use vyre_foundation::operation::{OperationRegistry, OperationTier, SemanticOperation};

/// What reads a tier, decided one tier at a time.
enum TierReader {
    /// The named reader projects exactly this tier.
    Reader(&'static str, fn() -> Vec<SemanticOperation>),
    /// No catalog reader projects this tier, so no reader may return it.
    Unread,
}

/// The reader recorded for `tier`.
///
/// `OperationTier` is `#[non_exhaustive]`, so a downstream match needs a
/// catch-all and cannot refuse a new variant at compile time. The catch-all
/// panics instead: a tier added upstream and left undecided here turns this
/// suite red rather than passing under a default.
fn reader_for(tier: OperationTier) -> TierReader {
    match tier {
        OperationTier::Intrinsic => TierReader::Reader(
            "vyre_primitives::operation_catalog::intrinsic_entries",
            || {
                vyre_primitives::operation_catalog::intrinsic_entries()
                    .collect::<Vec<SemanticOperation>>()
            },
        ),
        OperationTier::Library => TierReader::Reader(
            "vyre_libs::operation_catalog::library_entries",
            || vyre_libs::operation_catalog::library_entries().collect::<Vec<SemanticOperation>>(),
        ),
        // Foundation IR operations, external extensions and unnamespaced ids are
        // submitted by consumers and test fixtures, not by a workspace catalog.
        OperationTier::Foundation | OperationTier::External | OperationTier::Unknown => {
            TierReader::Unread
        }
        undecided => panic!(
            "Fix: `OperationTier::{undecided:?}` has no recorded reader decision. Record the reader that projects it, or record it as unread and state which crate submits it."
        ),
    }
}

/// Registered ids of one tier, read from the registry at run time.
fn registered_ids(tier: OperationTier) -> BTreeSet<&'static str> {
    OperationRegistry::global()
        .iter()
        .filter(|entry| entry.tier == tier)
        .map(|entry| entry.id)
        .collect()
}

fn reader_ids(read: fn() -> Vec<SemanticOperation>) -> BTreeSet<&'static str> {
    read().into_iter().map(|entry| entry.id).collect()
}

/// The registry must carry both read tiers, or the assertions below range over
/// nothing and prove nothing about either reader.
fn assert_both_read_tiers_are_populated() {
    for tier in [OperationTier::Intrinsic, OperationTier::Library] {
        assert!(
            !registered_ids(tier).is_empty(),
            "Fix: this binary linked no `{tier:?}` registration, so the tier-reader assertions range over an empty registry. Name an item from the crate that submits that tier so its inventory section is retained."
        );
    }
}

#[test]
fn every_tier_reader_returns_exactly_its_own_tier() {
    assert_both_read_tiers_are_populated();

    for tier in OperationTier::ALL.iter().copied() {
        let registered = registered_ids(tier);
        match reader_for(tier) {
            TierReader::Reader(name, read) => {
                let returned = reader_ids(read);
                assert_eq!(
                    returned, registered,
                    "Fix: `{name}` must return exactly the {tier:?} tier. It returned {} ids while the registry holds {} at that tier, so a rule reading through it judges a different population than the one that registered.",
                    returned.len(),
                    registered.len()
                );
            }
            TierReader::Unread => {
                for other in OperationTier::ALL.iter().copied() {
                    let TierReader::Reader(name, read) = reader_for(other) else {
                        continue;
                    };
                    let leaked: Vec<&str> = reader_ids(read)
                        .intersection(&registered)
                        .copied()
                        .collect();
                    assert!(
                        leaked.is_empty(),
                        "Fix: `{name}` returned {tier:?}-tier ids {leaked:?}. A reader projects one tier, and {tier:?} has no reader."
                    );
                }
            }
        }
    }
}

/// WHY: this is the defect itself, stated as an invariant. A library id reached
/// through the intrinsic reader is what let a coverage loop filter for
/// `vyre-libs::bitset::` against an intrinsic-only catalog and iterate nothing.
///
/// The two id sets come from the registry at run time, so a newly registered
/// operation of either tier is covered with nothing else edited.
#[test]
fn neither_read_tier_is_reachable_through_the_other_reader() {
    assert_both_read_tiers_are_populated();

    let intrinsic_registered = registered_ids(OperationTier::Intrinsic);
    let library_registered = registered_ids(OperationTier::Library);

    let TierReader::Reader(intrinsic_name, read_intrinsic) = reader_for(OperationTier::Intrinsic)
    else {
        panic!("Fix: the intrinsic tier must have a reader for this invariant to mean anything.")
    };
    let TierReader::Reader(library_name, read_library) = reader_for(OperationTier::Library) else {
        panic!("Fix: the library tier must have a reader for this invariant to mean anything.")
    };

    let through_intrinsic = reader_ids(read_intrinsic);
    let through_library = reader_ids(read_library);

    let library_through_intrinsic: Vec<&str> = through_intrinsic
        .intersection(&library_registered)
        .copied()
        .collect();
    assert!(
        library_through_intrinsic.is_empty(),
        "Fix: `{intrinsic_name}` returned library ids {library_through_intrinsic:?}. Read them through `{library_name}`."
    );

    let intrinsic_through_library: Vec<&str> = through_library
        .intersection(&intrinsic_registered)
        .copied()
        .collect();
    assert!(
        intrinsic_through_library.is_empty(),
        "Fix: `{library_name}` returned intrinsic ids {intrinsic_through_library:?}. Read them through `{intrinsic_name}`."
    );

    let missing_from_library: Vec<&str> = library_registered
        .difference(&through_library)
        .copied()
        .collect();
    assert!(
        missing_from_library.is_empty(),
        "Fix: `{library_name}` dropped registered library ids {missing_from_library:?}, so a rule reading it judges a partial roster."
    );

    let missing_from_intrinsic: Vec<&str> = intrinsic_registered
        .difference(&through_intrinsic)
        .copied()
        .collect();
    assert!(
        missing_from_intrinsic.is_empty(),
        "Fix: `{intrinsic_name}` dropped registered intrinsic ids {missing_from_intrinsic:?}, so a rule reading it judges a partial roster."
    );
}

/// WHY: `vyre_libs::link_anchor` returns a count, and a count of the wrong tier
/// is the same silent pass. Its callers read it as the number of library
/// operations the active feature set registers, so it is pinned against the
/// library tier read from the registry.
#[test]
fn the_libs_link_anchor_counts_the_library_tier() {
    let anchored = vyre_libs::link_anchor();
    let registered = registered_ids(OperationTier::Library).len();
    assert_eq!(
        anchored, registered,
        "Fix: `vyre_libs::link_anchor` returned {anchored} while the registry holds {registered} library operations. The anchor reports the library tier it just linked, not another tier's count."
    );
}
