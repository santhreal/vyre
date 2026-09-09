//! Canonical semantic operation registration, three identity-joined records,
//! and derived catalog views.

mod call_graph;
mod catalog_bundle;
mod conformance;
mod records;
mod registration;
mod registry;
mod registry_error;
mod semantic_op;
mod semantics;
mod target_facet;

pub use self::call_graph::CallGraphClosure;
pub use self::catalog_bundle::{CatalogBundle, ExtensionProvenance};
pub use self::conformance::ConformanceRegistry;
pub use self::records::{
    ConformanceProvider, LoweringProvider, OperationFixtures, SemanticDescriptor,
};
pub use self::registration::OperationRegistration;
pub use self::registry::OperationRegistry;
pub use self::registry_error::OperationRegistryError;
pub use self::semantic_op::SemanticOperation;
pub use self::semantics::{operation_id_namespace, IdNamespace, OperationEffects, OperationTier};
pub use self::target_facet::{TargetId, TargetOperationFacet};

/// Declarative operation macro generating three identity-joined submissions:
/// `SemanticDescriptor`, `LoweringProvider`, and `ConformanceProvider`.
#[macro_export]
macro_rules! declare_operation {
    (
        id: $id:expr,
        tier: $tier:expr,
        $(semantic_version: $sem_ver:expr,)?
        $(signature: $sig:expr,)?
        $(category: $cat:expr,)?
        $(laws: $laws:expr,)?
        $(numeric: $num:expr,)?
        $(geometry_requirements: $geom:expr,)?
        $(explicit_effects: $eff:expr,)?
        $(explicit_capabilities: $caps:expr,)?
        $(build: $build:expr,)?
        $(test_inputs: $inputs:expr,)?
        $(expected_output: $expected:expr)?
    ) => {
        const _: () = {
            $crate::inventory::submit! {
                $crate::operation::SemanticDescriptor {
                    id: $id,
                    semantic_version: 1 $( - 1 + $sem_ver )?,
                    signature: None $(.or(Some($sig)))?,
                    tier: $tier,
                    category: None $(.or(Some($cat)))?,
                    laws: &[] $(.or($laws))?,
                    numeric: $crate::numeric::NumericContract::EXACT $(.or($num))?,
                    geometry_requirements: $crate::geometry::GeometryRequirements::agnostic() $(.or($geom))?,
                    explicit_effects: None $(.or(Some($eff)))?,
                    explicit_capabilities: None $(.or(Some($caps)))?,
                }
            }
            $crate::inventory::submit! {
                $crate::operation::LoweringProvider {
                    id: $id,
                    build: None $(.or(Some($build)))?,
                }
            }
            $crate::inventory::submit! {
                $crate::operation::ConformanceProvider {
                    id: $id,
                    test_inputs: None $(.or(Some($inputs)))?,
                    expected_output: None $(.or(Some($expected)))?,
                }
            }
        };
    };
}

#[cfg(test)]
mod tests;
