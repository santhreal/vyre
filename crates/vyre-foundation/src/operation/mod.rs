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
mod records;
mod registration;
mod registry;
mod registry_error;
mod semantic_op;
mod semantics;
mod target_facet;

pub use self::call_graph::CallGraphClosure;
pub use self::catalog_bundle::{ExtensionProvenance, OperationCatalogBundle};
pub use self::records::{
    AbsenceDecision, ConformanceProvider, LoweringProvider, OperationFixtures, SemanticDescriptor,
};
pub use self::registration::OperationRegistration;
pub use self::registry::OperationRegistry;
pub use self::registry_error::OperationRegistryError;
pub use self::semantic_op::SemanticOperation;
pub use self::semantics::{operation_id_namespace, IdNamespace, OperationEffects, OperationTier};
pub use self::target_facet::{TargetId, TargetOperationFacet};

#[cfg(test)]
mod tests;
