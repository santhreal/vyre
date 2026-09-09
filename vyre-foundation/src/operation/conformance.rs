//! Conformance registry holding all identity-joined conformance-case providers.

use std::collections::BTreeMap;

use super::records::ConformanceProvider;
use super::registration::OperationRegistration;

/// Conformance registry holding all operation fixture and oracle providers,
/// isolated from the production catalog bundle.
#[derive(Clone, Debug)]
pub struct ConformanceRegistry {
    providers: BTreeMap<&'static str, ConformanceProvider>,
}

impl ConformanceRegistry {
    /// Compute the conformance registry from global inventory submissions.
    #[must_use]
    pub fn from_registry() -> Self {
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
