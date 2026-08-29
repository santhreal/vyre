//! Declarative dialect descriptors and global dialect metadata registry.

use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::dialect_lookup::Signature;
use crate::operation::OperationTier;

/// Immutable descriptor for a registered dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DialectDescriptor {
    /// Stable dialect namespace identifier, e.g. `vyre-libs::logical`.
    pub id: &'static str,
    /// Human-readable dialect name.
    pub name: &'static str,
    /// Dialect schema version.
    pub version: u32,
    /// Minimum dialect schema version supported without migration.
    pub min_supported_version: u32,
    /// Semantic tier of this dialect.
    pub tier: OperationTier,
    /// Coarse taxonomy category.
    pub category: &'static str,
    /// Operations declared in this dialect.
    pub operations: &'static [DialectOpDescriptor],
    /// Short summary of the dialect purpose.
    pub summary: &'static str,
}

impl DialectDescriptor {
    /// Look up an operation descriptor by its operation identifier.
    #[must_use]
    pub fn find_op(&self, op_id: &str) -> Option<&'static DialectOpDescriptor> {
        self.operations.iter().find(|op| op.id == op_id)
    }

    /// Check whether this dialect contains the given operation identifier.
    #[must_use]
    pub fn contains_op(&self, op_id: &str) -> bool {
        self.find_op(op_id).is_some()
    }
}

/// Metadata descriptor for a single operation in a declarative dialect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DialectOpDescriptor {
    /// Fully qualified operation identifier (e.g. `vyre-libs::logical::nand`).
    pub id: &'static str,
    /// Owning dialect identifier.
    pub dialect: &'static str,
    /// Short operation name within dialect (e.g. `nand`).
    pub name: &'static str,
    /// Dialect schema version at which this operation was introduced.
    pub version: u32,
    /// Stable callable signature.
    pub signature: &'static Signature,
    /// Whether this operation is a composition over existing IR (true) or intrinsic (false).
    pub is_composable: bool,
    /// Documentation summary.
    pub summary: &'static str,
}

/// Link-time registration record for a dialect descriptor.
pub struct DialectDescriptorRegistration {
    /// Registered dialect descriptor.
    pub descriptor: &'static DialectDescriptor,
}

inventory::collect!(DialectDescriptorRegistration);

/// Global registry of declarative dialects.
pub struct DialectRegistry;

static REGISTRY: LazyLock<BTreeMap<&'static str, &'static DialectDescriptor>> =
    LazyLock::new(|| {
        let mut map = BTreeMap::new();
        for registration in inventory::iter::<DialectDescriptorRegistration> {
            let desc = registration.descriptor;
            assert!(
                map.insert(desc.id, desc).is_none(),
                "Fix: duplicate dialect registration for `{}`; each dialect identifier must be unique",
                desc.id
            );
        }
        map
    });

impl DialectRegistry {
    /// Get the global dialect descriptor map.
    #[must_use]
    pub fn global() -> &'static BTreeMap<&'static str, &'static DialectDescriptor> {
        &REGISTRY
    }

    /// Look up a dialect descriptor by its dialect identifier.
    #[must_use]
    pub fn get(dialect_id: &str) -> Option<&'static DialectDescriptor> {
        REGISTRY.get(dialect_id).copied()
    }

    /// Look up the dialect descriptor that owns a given operation identifier.
    #[must_use]
    pub fn find_by_op_id(op_id: &str) -> Option<&'static DialectDescriptor> {
        REGISTRY
            .values()
            .find(|dialect| dialect.contains_op(op_id))
            .copied()
    }
}
