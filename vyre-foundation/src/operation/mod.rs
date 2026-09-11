//! Canonical semantic operation registration, three identity-joined records,
//! and derived catalog views.

/// Declares one operation record: the identity fields every operation record
/// carries, then the fields that record adds.
///
/// `SemanticDescriptor`, `SemanticOperation`, and `OperationRegistration`
/// answer the same identity questions about an operation and differ only in
/// how they hold the signature and what they add, so the identity field list
/// is declared once and a field added to it reaches all three.
macro_rules! declare_operation_record {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            signature: $signature:ty,
            $($(#[$field_meta:meta])* $field:ident: $field_ty:ty,)*
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            /// Stable operation identifier.
            pub id: &'static str,
            /// Semantic schema version.
            pub semantic_version: u32,
            /// Explicit callable signature when the operation is used through `Expr::Call`.
            pub signature: $signature,
            /// Semantic tier.
            pub tier: OperationTier,
            /// Derived dialect or category namespace.
            pub category: Option<&'static str>,
            /// Algebraic or semantic law identifiers.
            pub laws: &'static [&'static str],
            /// What the result is allowed to be.
            pub numeric: NumericContract,
            /// Recorded target-neutral schedule constraints.
            pub geometry_requirements: GeometryRequirements,
            /// Optional explicit closed effects.
            pub explicit_effects: Option<OperationEffects>,
            /// Optional explicit closed capabilities.
            pub explicit_capabilities: Option<RequiredCapabilities>,
            /// Recorded decision when the operation declares no unconditional law.
            pub absence: Option<AbsenceDecision>,
            $($(#[$field_meta])* pub $field: $field_ty,)*
        }
    };
}

/// Reads the facts a semantic contract record is derived from off one operation
/// record.
///
/// `OperationRegistration` owns its signature and `SemanticOperation` borrows a
/// `'static` one, so the signature is an argument. Every other fact is read
/// through an accessor both records answer, which keeps one list of facts.
macro_rules! contract_facts_of {
    ($record:expr, $signature:expr) => {
        ContractFacts {
            id: $record.id,
            signature: $signature,
            effects: $record.direct_effects(),
            capabilities: $record.direct_required_capabilities(),
            numeric: $record.numeric,
            laws: $record.laws,
            absence: $record.absence,
            program: $record.program(),
        }
    };
}

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
pub use self::catalog_bundle::{ExtensionProvenance, OperationCatalogBundle};
pub use self::conformance::ConformanceRegistry;
pub use self::records::{
    AbsenceDecision, ConformanceProvider, ContractProvider, LoweringProvider,
    OperationContractBuilder, OperationFixtures, SemanticDescriptor,
};
pub use self::registration::OperationRegistration;
pub use self::registry::OperationRegistry;
pub use self::registry_error::OperationRegistryError;
pub use self::semantic_op::SemanticOperation;
pub use self::semantics::{operation_id_namespace, IdNamespace, OperationEffects, OperationTier};
pub use self::target_facet::{TargetId, TargetOperationFacet};

/// Declarative operation macro generating four identity-joined submissions:
/// `SemanticDescriptor`, `LoweringProvider`, `ConformanceProvider`, and `ContractProvider`.
#[macro_export]
macro_rules! declare_operation {
    (
        id: $id:expr,
        tier: $tier:expr,
        $(semantic_version: $sem_ver:expr,)?
        $(signature: $sig:expr,)?
        $(category: $cat:expr,)?
        $(laws: $laws:expr,)?
        $(absence: $absence:expr,)?
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
                    laws: {
                        let mut val: &'static [&'static str] = &[];
                        $(val = $laws;)?
                        val
                    },
                    numeric: {
                        let mut val = $crate::numeric::NumericContract::EXACT;
                        $(val = $num;)?
                        val
                    },
                    geometry_requirements: {
                        let mut val = $crate::geometry::GeometryRequirements::agnostic();
                        $(val = $geom;)?
                        val
                    },
                    explicit_effects: None $(.or(Some($eff)))?,
                    explicit_capabilities: None $(.or(Some($caps)))?,
                    absence: None $(.or(Some($absence)))?,
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
            $crate::inventory::submit! {
                $crate::operation::ContractProvider {
                    id: $id,
                    contract: Some(|| {
                        let desc = $crate::operation::SemanticDescriptor {
                            id: $id,
                            semantic_version: 1 $( - 1 + $sem_ver )?,
                            signature: None $(.or(Some($sig)))?,
                            tier: $tier,
                            category: None $(.or(Some($cat)))?,
                            laws: {
                                let mut val: &'static [&'static str] = &[];
                                $(val = $laws;)?
                                val
                            },
                            numeric: {
                                let mut val = $crate::numeric::NumericContract::EXACT;
                                $(val = $num;)?
                                val
                            },
                            geometry_requirements: {
                                let mut val = $crate::geometry::GeometryRequirements::agnostic();
                                $(val = $geom;)?
                                val
                            },
                            explicit_effects: None $(.or(Some($eff)))?,
                            explicit_capabilities: None $(.or(Some($caps)))?,
                            absence: None $(.or(Some($absence)))?,
                        };
                        let op = $crate::operation::SemanticOperation {
                            id: desc.id,
                            semantic_version: desc.semantic_version,
                            signature: desc.signature,
                            tier: desc.tier,
                            category: desc.category,
                            build: None $(.or(Some($build)))?,
                            test_inputs: None $(.or(Some($inputs)))?,
                            expected_output: None $(.or(Some($expected)))?,
                            laws: desc.laws,
                            numeric: desc.numeric,
                            geometry_requirements: desc.geometry_requirements,
                            source_file: file!(),
                            explicit_effects: desc.explicit_effects,
                            explicit_capabilities: desc.explicit_capabilities,
                            absence: desc.absence,
                        };
                        op.contract_record()
                    }),
                }
            }
        };
    };
}

#[cfg(test)]
mod tests;
