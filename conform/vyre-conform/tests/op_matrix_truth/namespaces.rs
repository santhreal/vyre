//! One kernel, one registered identity, in the namespace of the crate that
//! owns it.

use std::collections::BTreeMap;

use vyre_foundation::operation::OperationTier as OpTier;

use super::registry::registered_ops;

#[test]
fn registry_namespaces_do_not_pollute_other_tiers() {
    for entry in vyre_primitives::hardware::all_entries() {
        assert!(
            entry.id.starts_with("vyre-primitives::hardware::"),
            "Fix: hardware-category entry `{}` must use the vyre-primitives::hardware namespace.",
            entry.id
        );
    }

    for entry in vyre_primitives::operation_catalog::all_entries() {
        assert!(
            entry.id.starts_with("vyre-primitives::"),
            "Fix: intrinsic-tier entry `{}` must use the vyre-primitives namespace.",
            entry.id
        );
    }

    for entry in vyre_libs::operation_catalog::all_entries() {
        assert!(
            matches!(entry.tier, OpTier::Library | OpTier::External),
            "Fix: shared harness entry `{}` must be a composition or an external consumer op, not {:?}.",
            entry.id,
            entry.tier
        );
    }
}

/// One kernel gets one registered identity. `structure-gate` screens for this by
/// reading source text, which it must do to run on a workspace that does not
/// compile, and that parser cannot see ids built from constants or macros. This
/// is the authority: it enumerates the linked registry, so a shadow registration
/// of an existing kernel under a second crate's namespace fails here regardless
/// of how the id is spelled in source.
///
/// Two ops inside ONE crate may share a terminal name (`vyre-primitives::bitset::any`
/// and `vyre-primitives::reduce::any` are different kernels over different layouts).
/// The same terminal name claimed by two crates means the higher layer re-registered
/// the lower layer's kernel instead of calling it.
#[test]
fn no_kernel_is_registered_under_two_crate_namespaces() {
    let mut claims: BTreeMap<&str, BTreeMap<&str, Vec<&str>>> = BTreeMap::new();
    let registered = registered_ops();
    for op in &registered {
        let (crate_name, rest) = op
            .id
            .split_once("::")
            .unwrap_or_else(|| panic!("Fix: registered id `{}` must be `<crate>::<path>`.", op.id));
        let leaf = rest.rsplit("::").next().unwrap_or(rest);
        claims
            .entry(leaf)
            .or_default()
            .entry(crate_name)
            .or_default()
            .push(op.id.as_str());
    }

    assert!(
        registered.len() > 300,
        "Fix: registry enumeration collapsed to {} ops; the inventory link is broken and this \
         test would pass vacuously.",
        registered.len()
    );

    let collisions = claims
        .iter()
        .filter(|(_, by_crate)| by_crate.len() > 1)
        .map(|(leaf, by_crate)| {
            let ids = by_crate
                .values()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
                .join(", ");
            format!("`{leaf}` is registered by {} crates: {ids}", by_crate.len())
        })
        .collect::<Vec<_>>();

    assert!(
        collisions.is_empty(),
        "Fix: delete the shadow registration and call the surviving op instead. One kernel, \
         one id.\n{}",
        collisions.join("\n")
    );
}
