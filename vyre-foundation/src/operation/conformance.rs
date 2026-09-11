//! Conformance registry holding all identity-joined conformance-case providers.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use super::records::ConformanceProvider;
use super::registration::OperationRegistration;

/// Conformance registry holding all operation fixture and oracle providers,
/// isolated from the production catalog bundle.
#[derive(Clone, Debug)]
pub struct ConformanceRegistry {
    providers: BTreeMap<&'static str, ConformanceProvider>,
}

impl ConformanceRegistry {
    /// Return the process-wide conformance registry, built once from the
    /// global inventory.
    ///
    /// The walk grows with the registration count, so it happens here and
    /// once. Every caller probes this one rather than reading the inventory
    /// again.
    #[must_use]
    pub fn global() -> &'static Self {
        static REGISTRY: LazyLock<ConformanceRegistry> =
            LazyLock::new(ConformanceRegistry::from_registry);
        &REGISTRY
    }

    /// Compute the conformance registry from global inventory submissions.
    ///
    /// Reachable only through `global`, which runs it once.
    #[must_use]
    pub(crate) fn from_registry() -> Self {
        let mut providers = BTreeMap::new();

        for prov in inventory::iter::<ConformanceProvider> {
            providers.insert(prov.id, *prov);
        }

        for reg in inventory::iter::<OperationRegistration> {
            providers
                .entry(reg.id)
                .or_insert_with(|| reg.conformance_provider());
        }

        Self { providers }
    }

    /// Look up a conformance provider by stable operation identifier.
    #[must_use]
    pub fn provider(&self, id: &str) -> Option<&ConformanceProvider> {
        self.providers.get(id)
    }

    /// Check whether a conformance provider exists for this operation identifier.
    #[must_use]
    pub fn contains(&self, id: &str) -> bool {
        self.providers.contains_key(id)
    }

    /// Return the number of conformance providers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Check whether the conformance registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }

    /// Iterate all conformance providers in stable operation-id order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &ConformanceProvider> + '_ {
        self.providers.values()
    }
}
