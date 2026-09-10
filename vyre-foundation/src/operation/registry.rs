//! Process-wide validated operation registry view.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::operation::call_graph::CallGraphClosure;
use crate::operation::catalog_bundle::OperationCatalogBundle;
use crate::operation::registration::OperationRegistration;
use crate::operation::registry_error::{validate_identity, OperationRegistryError};
use crate::operation::semantic_op::SemanticOperation;
use crate::operation::semantics::OperationEffects;
use crate::program_caps::RequiredCapabilities;

/// Immutable validated view over every linked semantic operation registration.
pub struct OperationRegistry {
    ordered: Vec<&'static OperationRegistration>,
    by_id: BTreeMap<&'static str, &'static OperationRegistration>,
    call_graph: CallGraphClosure,
    catalog_bundle: OperationCatalogBundle,
}

impl OperationRegistry {
    fn build() -> Result<Self, OperationRegistryError> {
        let mut ordered = inventory::iter::<OperationRegistration>
            .into_iter()
            .collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|entry| entry.id);
        let mut by_id = BTreeMap::new();
        let known_laws: std::collections::BTreeSet<&str> =
            vyre_spec::law_catalog().iter().copied().collect();

        for entry in &ordered {
            if entry.semantic_version == 0 {
                return Err(OperationRegistryError::InvalidVersion { id: entry.id });
            }
            if entry.build.is_none() && entry.signature.is_none() {
                return Err(OperationRegistryError::MissingSemantics { id: entry.id });
            }
            validate_identity(entry)?;
            if !entry.has_transform_decision() {
                return Err(OperationRegistryError::MissingTransformDecision { id: entry.id });
            }
            for &law in entry.laws {
                if !known_laws.contains(law) {
                    return Err(OperationRegistryError::UnknownAlgebraicLaw { id: entry.id, law });
                }
            }
            let record = entry.contract_record();
            if let Err(err) = record.validate() {
                return Err(OperationRegistryError::ContractValidationFailed {
                    id: entry.id,
                    message: err.to_string(),
                });
            }
            if by_id.insert(entry.id, *entry).is_some() {
                return Err(OperationRegistryError::DuplicateId { id: entry.id });
            }
        }
        let call_graph = CallGraphClosure::solve_from_registrations(ordered.iter().copied());
        let catalog_bundle = OperationCatalogBundle::from_registry();
        Ok(Self {
            ordered,
            by_id,
            call_graph,
            catalog_bundle,
        })
    }

    /// Return the process-wide validated semantic operation registry.
    ///
    /// # Panics
    ///
    /// Panics if the static operation inventory fails validation or contains duplicate IDs.
    #[must_use]
    pub fn global() -> &'static Self {
        static REGISTRY: LazyLock<OperationRegistry> = LazyLock::new(|| {
            OperationRegistry::build()
                .unwrap_or_else(|error| panic!("invalid semantic operation registry: {error}"))
        });
        &REGISTRY
    }

    /// Return the immutable precomputed call graph closure over all registered operations.
    #[must_use]
    pub fn call_graph_closure(&self) -> &CallGraphClosure {
        &self.call_graph
    }

    /// Return transitive effects for an operation.
    #[must_use]
    pub fn transitive_effects(&self, id: &str) -> Option<OperationEffects> {
        self.call_graph.transitive_effects(id)
    }

    /// Return transitive required capabilities for an operation.
    #[must_use]
    pub fn transitive_capabilities(&self, id: &str) -> Option<RequiredCapabilities> {
        self.call_graph.transitive_capabilities(id)
    }

    /// Return direct callees for an operation.
    #[must_use]
    pub fn callees(&self, id: &str) -> Option<&[&'static str]> {
        self.call_graph.callees(id)
    }

    /// Return deterministic call-graph closure identity.
    #[must_use]
    pub fn call_graph_closure_identity(&self) -> u64 {
        self.call_graph.closure_identity()
    }

    /// Return effective composite version for an operation.
    #[must_use]
    pub fn composite_version(&self, id: &str) -> Option<u64> {
        let entry = self.by_id.get(id)?;
        self.call_graph
            .composite_version(id, entry.semantic_version)
    }

    /// Resolve one stable operation identity.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<SemanticOperation> {
        self.by_id.get(id).copied().map(SemanticOperation::from)
    }

    /// Iterate registrations in stable operation-id order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = SemanticOperation> + '_ {
        self.ordered.iter().copied().map(SemanticOperation::from)
    }

    /// Return the immutable catalog bundle.
    #[must_use]
    pub fn catalog_bundle(&self) -> &OperationCatalogBundle {
        &self.catalog_bundle
    }
}
