//! Canonical semantic-operation support sets.
//!
//! Backends that consume semantic regions derive their advertised operation IDs
//! from the foundation-owned registry. Language-level node IDs remain a separate
//! frozen IR vocabulary and are unioned only by
//! [`dialect_and_language_supported_ops`].

use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use vyre_foundation::ir::OpId;
use vyre_foundation::operation::OperationRegistry;

/// Canonical semantic operation ids without language-level IR node ids.
#[must_use]
pub fn dialect_only_supported_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(|| {
        OperationRegistry::global()
            .iter()
            .map(|registration| Arc::<str>::from(registration.id))
            .collect()
    });
    &OPS
}

// Inline: `vyre_runtime::backend` is `pub(crate)`, so no integration test can reach what this suite
// exercises.
#[cfg(test)]
mod tests {
    use super::*;

    /// The semantic half is the registry, not a driver-local list.
    ///
    /// The driver used to register five host capabilities as operations so this
    /// set would contain them. It now advertises exactly what the two
    /// operation-owning crates registered, so a set built from a hardcoded id
    /// list would be the defect.
    #[test]
    fn semantic_set_mirrors_the_operation_registry() {
        let ops = dialect_only_supported_ops();
        let registry = OperationRegistry::global();

        assert_eq!(ops.len(), registry.iter().len());
        for operation in registry.iter() {
            assert!(
                ops.iter().any(|id| id.as_ref() == operation.id),
                "semantic set is missing registered operation `{}`",
                operation.id
            );
        }
    }
}
