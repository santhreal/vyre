//! Authoritative universe of linked semantic operation registrations.

use std::collections::BTreeSet;
use vyre_foundation::operation::OperationRegistry;

/// Minimum operation registrations expected in a foundation build.
pub const OPERATION_REGISTRATION_FLOOR: usize = 40;

/// Return the full set of operation IDs registered in the global registry.
#[must_use]
pub fn registered_operation_ids() -> BTreeSet<String> {
    let registry = OperationRegistry::global();
    registry
        .iter()
        .map(|entry| entry.id.to_string())
        .collect()
}

/// Assert that the global operation registry is well-formed and complete.
///
/// # Panics
/// Panics if the registry has fewer entries than `OPERATION_REGISTRATION_FLOOR` or if
/// any entry violates invariant contracts.
pub fn assert_operation_registry_complete() {
    let registry = OperationRegistry::global();
    let count = registry.iter().count();
    assert!(
        count >= OPERATION_REGISTRATION_FLOOR,
        "operation registry contains only {count} operations, below floor {}",
        OPERATION_REGISTRATION_FLOOR
    );

    for entry in registry.iter() {
        assert!(
            !entry.id.is_empty(),
            "registered operation has empty ID"
        );
        assert!(
            entry.semantic_version > 0,
            "operation `{}` has semantic_version == 0",
            entry.id
        );
        assert!(
            entry.build.is_some() || entry.signature.is_some(),
            "operation `{}` has neither builder nor signature",
            entry.id
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn global_operation_registry_is_complete() {
        assert_operation_registry_complete();
    }
}
