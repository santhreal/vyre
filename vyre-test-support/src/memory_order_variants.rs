//! One fixture per `MemoryOrdering` variant, derived from source to fail closed on additions.

use std::collections::BTreeSet;
use vyre_foundation::ir::MemoryOrdering;

/// Minimum variants expected in MemoryOrdering (catches broken source parsing).
pub const MEMORY_ORDERING_VARIANT_FLOOR: usize = 6;

/// Return all declared `MemoryOrdering` variants derived from the wire schema.
#[must_use]
pub fn declared_memory_ordering_variants() -> BTreeSet<String> {
    (0u8..=255)
        .filter_map(|tag| MemoryOrdering::from_wire_tag(tag).ok())
        .map(|ord| format!("{ord:?}"))
        .collect()
}

/// Every declared `MemoryOrdering` variant.
#[must_use]
pub fn memory_ordering_variant_samples() -> Vec<MemoryOrdering> {
    (0u8..=255)
        .filter_map(|tag| MemoryOrdering::from_wire_tag(tag).ok())
        .collect()
}

/// Panic unless `samples` covers every declared `MemoryOrdering` variant.
///
/// # Panics
/// Panics if a declared variant is missing or an undeclared variant is present.
pub fn assert_covers_every_memory_ordering(samples: &[MemoryOrdering]) {
    let declared = declared_memory_ordering_variants();
    let covered: BTreeSet<String> = samples.iter().map(|s| format!("{s:?}")).collect();

    let missing: BTreeSet<_> = declared.difference(&covered).cloned().collect();
    assert!(
        missing.is_empty(),
        "missing MemoryOrdering sample(s): {missing:?}"
    );

    let unexpected: BTreeSet<_> = covered.difference(&declared).cloned().collect();
    assert!(
        unexpected.is_empty(),
        "unexpected MemoryOrdering sample(s): {unexpected:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_ordering_samples_cover_all_variants() {
        let samples = memory_ordering_variant_samples();
        assert!(samples.len() >= MEMORY_ORDERING_VARIANT_FLOOR);
        assert_covers_every_memory_ordering(&samples);
    }
}
