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

/// The union of every canonical semantic operation id and frozen language-level
/// IR operation id. The set is computed once and reused by all consumers.
#[must_use]
pub fn dialect_and_language_supported_ops() -> &'static HashSet<OpId> {
    static OPS: LazyLock<HashSet<OpId>> = LazyLock::new(|| {
        let language_ops = super::validation::default_supported_ops();
        let semantic_ops = dialect_only_supported_ops();
        let mut set = HashSet::with_capacity(language_ops.len().saturating_add(semantic_ops.len()));
        set.extend(language_ops.iter().cloned());
        set.extend(semantic_ops.iter().cloned());
        set
    });
    &OPS
}

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

// Inline: `vyre_driver::backend` is `pub(crate)`, so no integration test can reach what this suite
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

    /// The union is exactly its two sources, with no third contributor.
    #[test]
    fn union_is_exactly_language_plus_semantic() {
        let union = dialect_and_language_supported_ops();
        let language = super::super::validation::default_supported_ops();
        let semantic = dialect_only_supported_ops();

        assert!(union.contains(&OpId::from("vyre.node.store")));
        for id in language.iter().chain(semantic.iter()) {
            assert!(union.contains(id), "union dropped `{id}`");
        }
        let expected: HashSet<&OpId> = language.iter().chain(semantic.iter()).collect();
        assert_eq!(union.len(), expected.len());
    }
}
