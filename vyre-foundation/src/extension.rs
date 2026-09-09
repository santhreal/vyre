//! Open-IR extension surface  -  traits and inventory registration for
//! third-party Expr / Node / DataType / BinOp / UnOp / AtomicOp /
//! RuleCondition variants.
//!
//! vyre-spec defines the per-kind extension ids and trait contracts
//! (`ExtensionDataType`, `ExtensionBinOp`, `ExtensionUnOp`,
//! `ExtensionAtomicOp`). This module provides the link-time registration
//! types that downstream crates submit via `inventory::submit!`, plus
//! frozen-after-init resolvers that materialize `&'static dyn Trait`
//! pointers.
//!
//! # Runtime cost
//!
//! Every resolver is a `LazyLock<FxHashMap<ExtensionXxxId, &'static dyn
//! ExtensionXxx>>`. First call walks the inventory once. Every subsequent
//! call is one hash + one table probe  -  sub-ns, no allocation, no lock.
//! The prior implementation called `inventory::iter` per lookup which
//! scaled linearly with the registration count, which is the hot-path
//! invariant a resolver lookup must not break.

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::LazyLock;

use rustc_hash::FxHashMap;
use vyre_spec::{
    ExtensionAtomicOp, ExtensionAtomicOpId, ExtensionBinOp, ExtensionBinOpId, ExtensionDataType,
    ExtensionDataTypeId, ExtensionIdentity, ExtensionRuleConditionId, ExtensionSchema,
    ExtensionSchemaDigest, ExtensionUnOp, ExtensionUnOpId,
};

pub use vyre_spec::{
    ExtensionField as DeclExtensionField, ExtensionFieldType as DeclExtensionFieldType,
    ExtensionIdentity as DeclExtensionIdentity, ExtensionNamespace as DeclExtensionNamespace,
    ExtensionNumericalContract as DeclExtensionNumericalContract,
    ExtensionOperand as DeclExtensionOperand, ExtensionOperandKind as DeclExtensionOperandKind,
    ExtensionProofFieldKind as DeclExtensionProofFieldKind,
    ExtensionProofFields as DeclExtensionProofFields,
    ExtensionResourceBounds as DeclExtensionResourceBounds, ExtensionSchema as DeclExtensionSchema,
    ExtensionSchemaDigest as DeclExtensionSchemaDigest, ExtensionSemVer as DeclExtensionSemVer,
    ExtensionShapeRule as DeclExtensionShapeRule,
};

/// Error produced when validating or registering in a [`CatalogBundle`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExtensionCatalogError {
    /// An extension with identical identity is already registered in the bundle.
    #[error("Duplicate extension identity in bundle: {0}")]
    DuplicateIdentity(ExtensionIdentity),
    /// An extension with same namespace and semantic version already exists with a different digest.
    #[error("Conflicting schema definition for {namespace}@{version}: existing {first_id}, incoming {second_id}")]
    DuplicateNamespaceVersion {
        /// Extension namespace.
        namespace: vyre_spec::ExtensionNamespace,
        /// Extension semantic version.
        version: vyre_spec::ExtensionSemVer,
        /// Existing identity in bundle.
        first_id: ExtensionIdentity,
        /// Incoming colliding identity.
        second_id: ExtensionIdentity,
    },
    /// Computed schema digest does not match the claimed identity digest.
    #[error("Schema digest mismatch for {identity}: expected {expected:?}, computed {actual:?}")]
    DigestMismatch {
        /// Extension identity.
        identity: ExtensionIdentity,
        /// Expected digest from identity.
        expected: ExtensionSchemaDigest,
        /// Actual digest recomputed from fields.
        actual: ExtensionSchemaDigest,
    },
    /// Required proof field missing or empty.
    #[error("Required extension proof field missing: {0}")]
    MissingProofField(String),
    /// Extension schema failed validation.
    #[error("Invalid extension schema for {0}: {1}")]
    InvalidSchema(String, String),
}

/// Closed declarative extension catalog bundle.
///
/// Replaces process-global opaque callbacks with an explicit, versioned,
/// serializable bundle of declarative extension schemas.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CatalogBundle {
    /// Bundle identifier.
    pub bundle_id: String,
    /// Registered extension schemas keyed by collision-resistant identity.
    pub schemas: FxHashMap<ExtensionIdentity, ExtensionSchema>,
}

impl CatalogBundle {
    /// Create a new empty catalog bundle.
    #[must_use]
    pub fn new(bundle_id: impl Into<String>) -> Self {
        Self {
            bundle_id: bundle_id.into(),
            schemas: FxHashMap::default(),
        }
    }

    /// Register a declarative extension schema into this bundle.
    ///
    /// Fails closed if the identity is already registered or if the schema
    /// digest is inconsistent.
    pub fn register(&mut self, schema: ExtensionSchema) -> Result<(), ExtensionCatalogError> {
        if schema.proof_fields.target_capability.trim().is_empty() {
            return Err(ExtensionCatalogError::MissingProofField(
                "target_capability must not be empty".to_string(),
            ));
        }
        let expected_digest = ExtensionSchema::compute_digest(
            schema.identity.namespace.as_str(),
            &schema.identity.version,
            &schema.fields,
            &schema.operands,
            &schema.result_types,
            &schema.proof_fields,
        );
        if schema.identity.schema_digest != expected_digest {
            return Err(ExtensionCatalogError::DigestMismatch {
                identity: schema.identity.clone(),
                expected: schema.identity.schema_digest,
                actual: expected_digest,
            });
        }
        if self.schemas.contains_key(&schema.identity) {
            return Err(ExtensionCatalogError::DuplicateIdentity(schema.identity));
        }
        for existing in self.schemas.values() {
            if existing.identity.namespace == schema.identity.namespace
                && existing.identity.version == schema.identity.version
            {
                return Err(ExtensionCatalogError::DuplicateNamespaceVersion {
                    namespace: schema.identity.namespace.clone(),
                    version: schema.identity.version,
                    first_id: existing.identity.clone(),
                    second_id: schema.identity.clone(),
                });
            }
        }
        self.schemas.insert(schema.identity.clone(), schema);
        Ok(())
    }

    /// Lookup an extension schema by exact identity.
    #[must_use]
    pub fn get(&self, identity: &ExtensionIdentity) -> Option<&ExtensionSchema> {
        self.schemas.get(identity)
    }

    /// Lookup all extension schemas matching a given namespace.
    #[must_use]
    pub fn get_by_namespace(&self, namespace: &str) -> Vec<&ExtensionSchema> {
        self.schemas
            .values()
            .filter(|s| s.identity.namespace.as_str() == namespace)
            .collect()
    }

    /// Lookup the latest semantic version of an extension in a namespace.
    #[must_use]
    pub fn get_latest(&self, namespace: &str) -> Option<&ExtensionSchema> {
        self.schemas
            .values()
            .filter(|s| s.identity.namespace.as_str() == namespace)
            .max_by_key(|s| (s.identity.version.major, s.identity.version.minor, s.identity.version.patch))
    }

    /// Validate all registered schemas within the bundle.
    pub fn validate(&self) -> Result<(), ExtensionCatalogError> {
        for (identity, schema) in &self.schemas {
            if &schema.identity != identity {
                return Err(ExtensionCatalogError::InvalidSchema(
                    identity.to_string(),
                    "Schema identity mismatch with key".to_string(),
                ));
            }
            let expected = ExtensionSchema::compute_digest(
                identity.namespace.as_str(),
                &identity.version,
                &schema.fields,
                &schema.operands,
                &schema.result_types,
                &schema.proof_fields,
            );
            if identity.schema_digest != expected {
                return Err(ExtensionCatalogError::DigestMismatch {
                    identity: identity.clone(),
                    expected: identity.schema_digest,
                    actual: expected,
                });
            }
        }
        Ok(())
    }

    /// Merge another catalog bundle into this one.
    pub fn merge(&mut self, other: CatalogBundle) -> Result<(), ExtensionCatalogError> {
        for (_, schema) in other.schemas {
            self.register(schema)?;
        }
        Ok(())
    }

    /// Number of registered extensions in this bundle.
    #[must_use]
    pub fn len(&self) -> usize {
        self.schemas.len()
    }

    /// Whether this bundle is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.schemas.is_empty()
    }

    /// Compute a canonical fingerprint over the entire catalog bundle.
    #[must_use]
    pub fn canonical_fingerprint(&self) -> [u8; 32] {
        let mut identities: Vec<_> = self.schemas.keys().map(|id| id.to_canonical_string()).collect();
        identities.sort();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(self.bundle_id.as_bytes());
        for id_str in identities {
            bytes.extend_from_slice(id_str.as_bytes());
        }
        *blake3::hash(&bytes).as_bytes()
    }
}

/// Opaque rule condition extension  -  lets third-party rule-engine crates
/// compose bespoke predicates without editing the facade or foundation model.
pub trait RuleConditionExt: Debug + Send + Sync + 'static {
    /// Stable extension id.
    fn extension_id(&self) -> ExtensionRuleConditionId;
    /// Canonical fingerprint for cache invalidation.
    fn stable_fingerprint(&self) -> [u8; 32];
    /// Buffer declarations the rule builder must add when this condition
    /// appears in a program.
    fn required_buffers(&self) -> Vec<crate::ir::BufferDecl> {
        Vec::new()
    }
    /// Serialize the extension payload into stable bytes for wire round-trip.
    fn wire_payload(&self) -> Vec<u8> {
        Vec::new()
    }
    /// Evaluate against an opaque rule context (crate-specific payload).
    fn evaluate_opaque(&self, _ctx: &dyn std::any::Any) -> bool {
        false
    }
}

// ---------------------------------------------------------------------
// Registration types (one per extendable IR kind).
// ---------------------------------------------------------------------

/// Link-time registration for an extension-declared `DataType`.
///
/// The `vtable` pointer is what `resolve_data_type` returns  -  it bypasses
/// any further registry lookup on subsequent accesses.
pub struct ExtensionDataTypeRegistration {
    /// Stable id this registration serves.
    pub id: ExtensionDataTypeId,
    /// Implementation pointer. Must outlive the process (`'static`).
    pub vtable: &'static dyn ExtensionDataType,
}

/// Link-time registration for an extension-declared binary operator.
pub struct ExtensionBinOpRegistration {
    /// Stable id this registration serves.
    pub id: ExtensionBinOpId,
    /// Implementation pointer.
    pub vtable: &'static dyn ExtensionBinOp,
}

/// Link-time registration for an extension-declared unary operator.
pub struct ExtensionUnOpRegistration {
    /// Stable id this registration serves.
    pub id: ExtensionUnOpId,
    /// Implementation pointer.
    pub vtable: &'static dyn ExtensionUnOp,
}

/// Link-time registration for an extension-declared atomic operator.
pub struct ExtensionAtomicOpRegistration {
    /// Stable id this registration serves.
    pub id: ExtensionAtomicOpId,
    /// Implementation pointer.
    pub vtable: &'static dyn ExtensionAtomicOp,
}

inventory::collect!(ExtensionDataTypeRegistration);
inventory::collect!(ExtensionBinOpRegistration);
inventory::collect!(ExtensionUnOpRegistration);
inventory::collect!(ExtensionAtomicOpRegistration);

/// Deserializer function matched to the bytes produced by
/// [`crate::ir::ExprNode::wire_payload`] for `Expr::Opaque` round-trip.
pub type ExprExtensionDeserializer =
    fn(&[u8]) -> Result<std::sync::Arc<dyn crate::ir::ExprNode>, String>;

/// Deserializer function matched to the bytes produced by
/// [`crate::ir::NodeExtension::wire_payload`] for `Node::Opaque` round-trip.
pub type NodeExtensionDeserializer =
    fn(&[u8]) -> Result<std::sync::Arc<dyn crate::ir::NodeExtension>, String>;

/// Inventory record pairing an `ExprNode` extension kind to its wire-format
/// deserializer. Wire tag `0x80` on an `Expr` discriminant triggers a
/// kind-keyed lookup against these records.
pub struct OpaqueExprResolver {
    /// Stable extension kind  -  must match [`crate::ir::ExprNode::extension_kind`].
    pub kind: &'static str,
    /// Deserializer for the extension's `wire_payload` bytes.
    pub deserialize: ExprExtensionDeserializer,
}

/// Inventory record pairing a `NodeExtension` extension kind to its decoder.
pub struct OpaqueNodeResolver {
    /// Stable extension kind  -  must match [`crate::ir::NodeExtension::extension_kind`].
    pub kind: &'static str,
    /// Deserializer for the extension's `wire_payload` bytes.
    pub deserialize: NodeExtensionDeserializer,
}

inventory::collect!(OpaqueExprResolver);
inventory::collect!(OpaqueNodeResolver);

fn collect_unique_by<K, V, I>(
    registrations: I,
    registry_name: &str,
) -> Result<FxHashMap<K, V>, String>
where
    K: Eq + Hash + Copy + std::fmt::Debug,
    I: IntoIterator<Item = (K, V, &'static str)>,
{
    let mut map = FxHashMap::default();
    let mut owners: FxHashMap<K, &'static str> = FxHashMap::default();
    for (key, value, owner) in registrations {
        if let Some(previous_owner) = owners.insert(key, owner) {
            return Err(format!(
                "{registry_name} duplicate registration for {key:?}: first registrant `{previous_owner}`, second registrant `{owner}`. Fix: pick one stable tag/kind owner."
            ));
        }
        map.insert(key, value);
    }
    Ok(map)
}

fn frozen_opaque_expr_registry(
) -> Result<&'static FxHashMap<&'static str, ExprExtensionDeserializer>, String> {
    static FROZEN: LazyLock<Result<FxHashMap<&'static str, ExprExtensionDeserializer>, String>> =
        LazyLock::new(|| {
            collect_unique_by(
                inventory::iter::<OpaqueExprResolver>
                    .into_iter()
                    .map(|reg| (reg.kind, reg.deserialize, reg.kind)),
                "OpaqueExprResolver",
            )
        });
    FROZEN.as_ref().map_err(Clone::clone)
}

fn frozen_opaque_node_registry(
) -> Result<&'static FxHashMap<&'static str, NodeExtensionDeserializer>, String> {
    static FROZEN: LazyLock<Result<FxHashMap<&'static str, NodeExtensionDeserializer>, String>> =
        LazyLock::new(|| {
            collect_unique_by(
                inventory::iter::<OpaqueNodeResolver>
                    .into_iter()
                    .map(|reg| (reg.kind, reg.deserialize, reg.kind)),
                "OpaqueNodeResolver",
            )
        });
    FROZEN.as_ref().map_err(Clone::clone)
}

/// Decode an opaque expression extension payload into an `Expr::Opaque` value.
pub fn decode_opaque_expr(kind: &str, payload: &[u8]) -> Result<crate::ir::Expr, String> {
    let registry = frozen_opaque_expr_registry()?;
    if let Some(deserialize) = registry.get(kind) {
        let node = deserialize(payload)?;
        let re_encoded = node.wire_payload();
        if re_encoded.as_slice() != payload {
            return Err(format!(
                "Canonical decode/re-encode mismatch for opaque expr `{kind}`: payload length {}, re-encoded length {}. Fix: ensure deserialized extension round-trips byte-for-byte to its canonical wire payload.",
                payload.len(),
                re_encoded.len()
            ));
        }
        Ok(crate::ir::Expr::Opaque(node))
    } else {
        Err(format!(
            "Fix: no OpaqueExprResolver registered for extension kind `{kind}`. Link the crate that owns this extension and ensure it submits `inventory::submit! {{ OpaqueExprResolver {{ kind, deserialize }} }}`."
        ))
    }
}

/// Decode an opaque statement extension payload into a `Node::Opaque` value.
pub fn decode_opaque_node(kind: &str, payload: &[u8]) -> Result<crate::ir::Node, String> {
    let registry = frozen_opaque_node_registry()?;
    if let Some(deserialize) = registry.get(kind) {
        let extension = deserialize(payload)?;
        let re_encoded = extension.wire_payload();
        if re_encoded.as_slice() != payload {
            return Err(format!(
                "Canonical decode/re-encode mismatch for opaque node `{kind}`: payload length {}, re-encoded length {}. Fix: ensure deserialized extension round-trips byte-for-byte to its canonical wire payload.",
                payload.len(),
                re_encoded.len()
            ));
        }
        Ok(crate::ir::Node::Opaque(extension))
    } else {
        Err(format!(
            "Fix: no OpaqueNodeResolver registered for extension kind `{kind}`. Link the crate that owns this extension and ensure it submits `inventory::submit! {{ OpaqueNodeResolver {{ kind, deserialize }} }}`."
        ))
    }
}

// ---------------------------------------------------------------------
// Frozen resolvers. First call walks the inventory; every subsequent
// call is hash + probe. No locks on the hot path.
// ---------------------------------------------------------------------

fn frozen_data_type_registry(
) -> Result<&'static FxHashMap<ExtensionDataTypeId, &'static dyn ExtensionDataType>, String> {
    static FROZEN: LazyLock<
        Result<FxHashMap<ExtensionDataTypeId, &'static dyn ExtensionDataType>, String>,
    > = LazyLock::new(|| {
        collect_unique_by(
            inventory::iter::<ExtensionDataTypeRegistration>
                .into_iter()
                .map(|reg| (reg.id, reg.vtable, reg.vtable.display_name())),
            "ExtensionDataTypeRegistration",
        )
    });
    FROZEN.as_ref().map_err(Clone::clone)
}

fn frozen_bin_op_registry(
) -> Result<&'static FxHashMap<ExtensionBinOpId, &'static dyn ExtensionBinOp>, String> {
    static FROZEN: LazyLock<
        Result<FxHashMap<ExtensionBinOpId, &'static dyn ExtensionBinOp>, String>,
    > = LazyLock::new(|| {
        collect_unique_by(
            inventory::iter::<ExtensionBinOpRegistration>
                .into_iter()
                .map(|reg| (reg.id, reg.vtable, reg.vtable.display_name())),
            "ExtensionBinOpRegistration",
        )
    });
    FROZEN.as_ref().map_err(Clone::clone)
}

fn frozen_un_op_registry(
) -> Result<&'static FxHashMap<ExtensionUnOpId, &'static dyn ExtensionUnOp>, String> {
    static FROZEN: LazyLock<
        Result<FxHashMap<ExtensionUnOpId, &'static dyn ExtensionUnOp>, String>,
    > = LazyLock::new(|| {
        collect_unique_by(
            inventory::iter::<ExtensionUnOpRegistration>
                .into_iter()
                .map(|reg| (reg.id, reg.vtable, reg.vtable.display_name())),
            "ExtensionUnOpRegistration",
        )
    });
    FROZEN.as_ref().map_err(Clone::clone)
}

fn frozen_atomic_op_registry(
) -> Result<&'static FxHashMap<ExtensionAtomicOpId, &'static dyn ExtensionAtomicOp>, String> {
    static FROZEN: LazyLock<
        Result<FxHashMap<ExtensionAtomicOpId, &'static dyn ExtensionAtomicOp>, String>,
    > = LazyLock::new(|| {
        collect_unique_by(
            inventory::iter::<ExtensionAtomicOpRegistration>
                .into_iter()
                .map(|reg| (reg.id, reg.vtable, reg.vtable.display_name())),
            "ExtensionAtomicOpRegistration",
        )
    });
    FROZEN.as_ref().map_err(Clone::clone)
}

// ---------------------------------------------------------------------
// Public lookup API. Every function is hot-path safe (one hash + one
// table probe; no allocation; no iteration).
// ---------------------------------------------------------------------

/// Resolve a `DataType::Opaque(id)` to its extension implementation.
///
/// Returns `None` for ids that no linked crate has registered; callers
/// surface a typed error, never a panic.
#[must_use]
pub fn resolve_data_type(id: ExtensionDataTypeId) -> Option<&'static dyn ExtensionDataType> {
    try_resolve_data_type(id).ok().flatten()
}

/// Resolve a `DataType::Opaque(id)` and surface registry construction errors.
pub fn try_resolve_data_type(
    id: ExtensionDataTypeId,
) -> Result<Option<&'static dyn ExtensionDataType>, String> {
    Ok(frozen_data_type_registry()?.get(&id).copied())
}

/// Resolve a `BinOp::Opaque(id)` to its extension implementation.
#[must_use]
pub fn resolve_bin_op(id: ExtensionBinOpId) -> Option<&'static dyn ExtensionBinOp> {
    try_resolve_bin_op(id).ok().flatten()
}

/// Resolve a `BinOp::Opaque(id)` and surface registry construction errors.
pub fn try_resolve_bin_op(
    id: ExtensionBinOpId,
) -> Result<Option<&'static dyn ExtensionBinOp>, String> {
    Ok(frozen_bin_op_registry()?.get(&id).copied())
}

/// Resolve a `UnOp::Opaque(id)` to its extension implementation.
#[must_use]
pub fn resolve_un_op(id: ExtensionUnOpId) -> Option<&'static dyn ExtensionUnOp> {
    try_resolve_un_op(id).ok().flatten()
}

/// Resolve a `UnOp::Opaque(id)` and surface registry construction errors.
pub fn try_resolve_un_op(
    id: ExtensionUnOpId,
) -> Result<Option<&'static dyn ExtensionUnOp>, String> {
    Ok(frozen_un_op_registry()?.get(&id).copied())
}

/// Resolve an `AtomicOp::Opaque(id)` to its extension implementation.
#[must_use]
pub fn resolve_atomic_op(id: ExtensionAtomicOpId) -> Option<&'static dyn ExtensionAtomicOp> {
    try_resolve_atomic_op(id).ok().flatten()
}

/// Resolve an `AtomicOp::Opaque(id)` and surface registry construction errors.
pub fn try_resolve_atomic_op(
    id: ExtensionAtomicOpId,
) -> Result<Option<&'static dyn ExtensionAtomicOp>, String> {
    Ok(frozen_atomic_op_registry()?.get(&id).copied())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyre_spec::*;

    #[test]
    fn per_kind_resolvers_are_empty_by_default() {
        // Foundation links no extension crates in its own test binary.
        // Every resolver must return None for any id.
        let data_type_id = ExtensionDataTypeId::from_name("tensor.gather");
        assert!(resolve_data_type(data_type_id).is_none());
        let bin_op_id = ExtensionBinOpId::from_name("bit.parity");
        assert!(resolve_bin_op(bin_op_id).is_none());
        let un_op_id = ExtensionUnOpId::from_name("bit.reverse_nibbles");
        assert!(resolve_un_op(un_op_id).is_none());
        let atomic_id = ExtensionAtomicOpId::from_name("atomic.clamp");
        assert!(resolve_atomic_op(atomic_id).is_none());
    }

    #[test]
    fn duplicate_typed_extension_ids_name_both_registrants() {
        let err = collect_unique_by(
            [
                (
                    ExtensionDataTypeId::from_name("dialect.duplicate"),
                    10usize,
                    "dialect.alpha",
                ),
                (
                    ExtensionDataTypeId::from_name("dialect.duplicate"),
                    20usize,
                    "dialect.beta",
                ),
            ],
            "ExtensionDataTypeRegistration",
        )
        .expect_err("Fix: duplicate registrations must return an error");

        assert!(err.contains("dialect.alpha"));
        assert!(err.contains("dialect.beta"));
    }

    #[test]
    fn catalog_bundle_distinguishes_distinct_extension_identities() {
        let mut bundle = CatalogBundle::new("test_bundle");
        let name_a = "d5pj";
        let name_b = "x.ta";

        let id_val_a = ExtensionDataTypeId::from_name(name_a);
        let id_val_b = ExtensionDataTypeId::from_name(name_b);
        assert_ne!(id_val_a, id_val_b, "Extension IDs must not collide");

        let ns_a = ExtensionNamespace::new(name_a).unwrap();
        let ns_b = ExtensionNamespace::new(name_b).unwrap();
        let ver_a = ExtensionSemVer::new(1, 0, 0);
        let ver_b = ExtensionSemVer::new(1, 0, 0);

        let proof_a = ExtensionProofFields {
            host_shareable: true,
            is_pure: true,
            cse_eligible: true,
            is_divergent: false,
            may_alias: false,
            terminates: true,
            target_capability: "generic".into(),
        };
        let proof_b = ExtensionProofFields {
            host_shareable: true,
            is_pure: true,
            cse_eligible: true,
            is_divergent: false,
            may_alias: false,
            terminates: true,
            target_capability: "generic".into(),
        };

        let digest_a = ExtensionSchema::compute_digest(name_a, &ver_a, &[], &[], &[], &proof_a);
        let digest_b = ExtensionSchema::compute_digest(name_b, &ver_b, &[], &[], &[], &proof_b);

        let id_a = ExtensionIdentity::new(ns_a, ver_a, digest_a);
        let id_b = ExtensionIdentity::new(ns_b, ver_b, digest_b);

        let schema_a = ExtensionSchema {
            identity: id_a.clone(),
            display_name: "Extension A".into(),
            description: "First extension".into(),
            fields: Vec::new(),
            operands: Vec::new(),
            result_types: Vec::new(),
            side_effects: vyre_spec::SideEffectClass::Pure,
            shape_rules: Vec::new(),
            numerical_contract: vyre_spec::ExtensionNumericalContract::default(),
            laws: Vec::new(),
            resource_bounds: vyre_spec::ExtensionResourceBounds::default(),
            proof_fields: proof_a,
        };

        let schema_b = ExtensionSchema {
            identity: id_b.clone(),
            display_name: "Extension B".into(),
            description: "Second extension".into(),
            fields: Vec::new(),
            operands: Vec::new(),
            result_types: Vec::new(),
            side_effects: vyre_spec::SideEffectClass::Pure,
            shape_rules: Vec::new(),
            numerical_contract: vyre_spec::ExtensionNumericalContract::default(),
            laws: Vec::new(),
            resource_bounds: vyre_spec::ExtensionResourceBounds::default(),
            proof_fields: proof_b,
        };

        bundle.register(schema_a).expect("schema A registers cleanly");
        bundle.register(schema_b).expect("schema B registers cleanly");

        assert_eq!(bundle.len(), 2);
        assert_eq!(bundle.get(&id_a).unwrap().display_name, "Extension A");
        assert_eq!(bundle.get(&id_b).unwrap().display_name, "Extension B");
        bundle.validate().expect("bundle validates cleanly");
    }
}
