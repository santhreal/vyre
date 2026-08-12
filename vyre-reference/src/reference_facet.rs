//! Reference implementations keyed by canonical semantic operation identity.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use vyre_spec::CpuFn;

/// One portable flat-byte reference implementation.
pub struct ReferenceFacet {
    /// Canonical semantic operation identifier.
    pub operation_id: &'static str,
    /// Portable reference implementation.
    pub execute: CpuFn,
}

impl ReferenceFacet {
    /// Construct a reference facet for one canonical semantic operation.
    #[must_use]
    pub const fn new(operation_id: &'static str, execute: CpuFn) -> Self {
        Self {
            operation_id,
            execute,
        }
    }
}

inventory::collect!(ReferenceFacet);

static FACETS: LazyLock<BTreeMap<&'static str, &'static ReferenceFacet>> = LazyLock::new(|| {
    let mut facets = BTreeMap::new();
    for facet in inventory::iter::<ReferenceFacet> {
        assert!(
            vyre_foundation::operation::OperationRegistry::global()
                .get(facet.operation_id)
                .is_some(),
            "reference facet `{}` has no canonical semantic operation",
            facet.operation_id
        );
        assert!(
            facets.insert(facet.operation_id, facet).is_none(),
            "duplicate reference facet `{}`; keep one reference owner per operation",
            facet.operation_id
        );
    }
    facets
});

/// Resolve the portable reference implementation for an operation.
#[must_use]
pub fn reference_fn(operation_id: &str) -> Option<CpuFn> {
    FACETS.get(operation_id).map(|facet| facet.execute)
}

/// Iterate reference facets in stable operation-id order.
pub fn reference_facets() -> impl ExactSizeIterator<Item = &'static ReferenceFacet> {
    FACETS.values().copied()
}
