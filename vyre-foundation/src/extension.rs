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
use vyre_spec::extension::{
    ExtensionAtomicOp, ExtensionAtomicOpId, ExtensionBinOp, ExtensionBinOpId, ExtensionDataType,
    ExtensionDataTypeId, ExtensionRuleConditionId, ExtensionUnOp, ExtensionUnOpId,
};

/// Opaque rule condition extension  -  lets third-party rule-engine crates
/// compose bespoke predicates without editing the facade or foundation model.
pub trait RuleConditionExt: Debug + Send + Sync + 'static {
    /// Stable extension id.
    fn extension_id(&self) -> ExtensionRuleConditionId;
    /// Evaluate against an opaque rule context (crate-specific payload).
    fn evaluate_opaque(&self, ctx: &dyn std::any::Any) -> bool;
    /// Canonical fingerprint for cache invalidation.
    fn stable_fingerprint(&self) -> [u8; 32];
    /// Buffer declarations the rule builder must add when this condition
    /// appears in a program. Extensions that need private scratch
    /// buffers for their evaluator return them here; frozen conditions
    /// return an empty `Vec`. The rule builder merges these into the
    /// canonical six-buffer set before construction.
    fn required_buffers(&self) -> Vec<crate::ir::BufferDecl> {
        Vec::new()
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
}
