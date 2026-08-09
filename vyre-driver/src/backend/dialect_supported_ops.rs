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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_set_contains_io_ops() {
        let ops = dialect_only_supported_ops();
        for op in [
            "io.dma_from_nvme",
            "io.write_back_to_nvme",
            "mem.zerocopy_map",
            "mem.unmap",
        ] {
            assert!(
                ops.iter().any(|o| o.as_ref() == op),
                "dialect set missing {op}; saw {:?}",
                ops.iter().map(|o| o.as_ref()).collect::<Vec<_>>().len()
            );
        }
    }

    #[test]
    fn union_includes_both_sources() {
        let union = dialect_and_language_supported_ops();
        assert!(union.iter().any(|o| o.as_ref() == "vyre.node.store"));
        assert!(union.iter().any(|o| o.as_ref() == "io.dma_from_nvme"));
    }

    #[test]
    fn union_size_exceeds_language_alone() {
        let lang = super::super::validation::default_supported_ops().len();
        let union = dialect_and_language_supported_ops().len();
        assert!(union > lang);
    }
}
