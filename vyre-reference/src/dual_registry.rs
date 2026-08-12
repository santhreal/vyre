//! Reference-owned registry for independent oracle pairs.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::dual::ReferenceFn;

/// Two independently written reference implementations for one operation.
pub struct DualReferenceFacet {
    /// Reference-owned operation identity.
    pub operation_id: &'static str,
    /// First implementation.
    pub reference_a: ReferenceFn,
    /// Second implementation.
    pub reference_b: ReferenceFn,
}

impl DualReferenceFacet {
    /// Construct one dual-reference facet.
    #[must_use]
    pub const fn new(
        operation_id: &'static str,
        reference_a: ReferenceFn,
        reference_b: ReferenceFn,
    ) -> Self {
        Self {
            operation_id,
            reference_a,
            reference_b,
        }
    }
}

inventory::collect!(DualReferenceFacet);

static FACETS: LazyLock<BTreeMap<&'static str, &'static DualReferenceFacet>> =
    LazyLock::new(|| {
        let mut facets = BTreeMap::new();
        for facet in inventory::iter::<DualReferenceFacet> {
            assert!(
                facets.insert(facet.operation_id, facet).is_none(),
                "duplicate dual-reference facet `{}`; keep one reference owner per operation",
                facet.operation_id
            );
        }
        facets
    });

/// Resolve both independent references for one operation.
#[must_use]
pub fn resolve_dual(operation_id: &str) -> Option<(ReferenceFn, ReferenceFn)> {
    FACETS
        .get(operation_id)
        .map(|facet| (facet.reference_a, facet.reference_b))
}

/// Return every dual-reference operation identity in stable order.
#[must_use]
pub fn dual_op_ids() -> &'static [&'static str] {
    static IDS: LazyLock<Box<[&'static str]>> = LazyLock::new(|| {
        FACETS
            .keys()
            .copied()
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    &IDS
}
