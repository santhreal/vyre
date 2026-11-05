//! Discovered universe of IR extension variants and resolvers.

use std::collections::BTreeSet;
use vyre_foundation::extension::{OpaqueExprResolver, OpaqueNodeResolver};

/// Return all registered `Expr` extension kind strings.
#[must_use]
pub fn registered_expr_extension_kinds() -> BTreeSet<String> {
    inventory::iter::<OpaqueExprResolver>
        .into_iter()
        .map(|r| r.kind.to_string())
        .collect()
}

/// Return all registered `Node` extension kind strings.
#[must_use]
pub fn registered_node_extension_kinds() -> BTreeSet<String> {
    inventory::iter::<OpaqueNodeResolver>
        .into_iter()
        .map(|r| r.kind.to_string())
        .collect()
}

/// Assert that all registered extensions have valid kind names and resolvers.
///
/// # Panics
/// Panics if any extension resolver has an empty or malformed kind name.
pub fn assert_extension_registry_complete() {
    for resolver in inventory::iter::<OpaqueExprResolver> {
        assert!(
            !resolver.kind.is_empty(),
            "OpaqueExprResolver has empty kind"
        );
    }
    for resolver in inventory::iter::<OpaqueNodeResolver> {
        assert!(
            !resolver.kind.is_empty(),
            "OpaqueNodeResolver has empty kind"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extension_registrations_are_well_formed() {
        assert_extension_registry_complete();
    }
}
