use super::{
    operation_id_namespace, validate_identity, IdNamespace, OperationRegistration,
    OperationRegistryError, OperationTier,
};
use std::collections::BTreeSet;

/// The tier roster carries every variant of the tier enum.
#[test]
fn the_roster_carries_every_tier() {
    let mut seen = BTreeSet::new();
    for tier in OperationTier::ALL {
        seen.insert(match tier {
            OperationTier::Foundation => 0,
            OperationTier::Intrinsic => 1,
            OperationTier::Library => 2,
            OperationTier::External => 3,
            OperationTier::Unknown => 4,
        });
    }
    assert_eq!(
        seen.len(),
        5,
        "OperationTier::ALL must list every variant the match above names"
    );
}

/// A namespace is a minting fact, never a placement one.
#[test]
fn the_namespace_never_answers_with_a_tier() {
    assert_eq!(
        operation_id_namespace("vyre-primitives::graph::toposort"),
        IdNamespace::Workspace("vyre-primitives")
    );
}

/// A workspace id cannot carry a tier only a consumer identity has.
#[test]
fn a_workspace_id_declaring_an_external_tier_is_rejected() {
    let entry = OperationRegistration::new(
        "vyre-libs::scan::literal_set",
        OperationTier::External,
        None,
        None,
        None,
    );
    assert_eq!(
        validate_identity(&entry),
        Err(OperationRegistryError::InvalidTier {
            id: "vyre-libs::scan::literal_set",
            declared: OperationTier::External,
            origin: "workspace",
        })
    );
}

/// A consumer id carries the external tier and no other.
#[test]
fn an_external_id_declaring_a_workspace_tier_is_rejected() {
    let entry = OperationRegistration::new(
        "community_pack::scan::signature",
        OperationTier::Library,
        None,
        None,
        None,
    );
    assert_eq!(
        validate_identity(&entry),
        Err(OperationRegistryError::InvalidTier {
            id: "community_pack::scan::signature",
            declared: OperationTier::Library,
            origin: "external",
        })
    );
    assert_eq!(
        validate_identity(&OperationRegistration::new(
            "community_pack::scan::signature",
            OperationTier::External,
            None,
            None,
            None,
        )),
        Ok(())
    );
}

/// Every tier a workspace crate can mint is accepted, and the two that name
/// no minting crate are not.
#[test]
fn a_workspace_id_carries_every_workspace_tier() {
    for tier in [
        OperationTier::Foundation,
        OperationTier::Intrinsic,
        OperationTier::Library,
    ] {
        assert_eq!(
            validate_identity(&OperationRegistration::new(
                "vyre-primitives::hardware::popcount_u32",
                tier,
                None,
                None,
                None,
            )),
            Ok(()),
            "{tier:?} is a tier a workspace crate mints"
        );
    }
    assert_eq!(
        validate_identity(&OperationRegistration::new(
            "vyre-primitives::hardware::popcount_u32",
            OperationTier::Unknown,
            None,
            None,
            None,
        )),
        Err(OperationRegistryError::InvalidTier {
            id: "vyre-primitives::hardware::popcount_u32",
            declared: OperationTier::Unknown,
            origin: "workspace",
        })
    );
}

/// An id that names no crate is refused before any tier question.
#[test]
fn an_id_naming_no_crate_is_refused_whatever_it_declares() {
    for id in ["not_a_namespace", "core.indirect_dispatch", "vyre-libs::"] {
        assert_eq!(
            validate_identity(&OperationRegistration::new(
                id,
                OperationTier::Library,
                None,
                None,
                None,
            )),
            Err(OperationRegistryError::UnknownNamespace { id })
        );
    }
}
