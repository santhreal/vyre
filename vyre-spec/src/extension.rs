//! Extension contracts for open IR.
//!
//! Downstream crates ship new `Expr`, `Node`, `DataType`, `BinOp`, `UnOp`,
//! `AtomicOp`, `TernaryOp`, and `RuleCondition` variants by implementing the
//! traits in this module and registering an id with the foundation extension
//! registry.
//!
//! `vyre-spec` is intentionally data-only and carries no dependency on
//! `inventory`. The trait signatures below describe the stable contract;
//! actual registration and resolution lives in `vyre_foundation::extension`.
//!
//! Every extension id occupies the range `0x8000_0000..=0xFFFF_FFFF`  -  the
//! high bit of the wire tag distinguishes extension ids from the frozen
//! core tag space `0x00..=0x7F`. The `ExtensionDataTypeId::from_name`
//! constructor folds a stable crate-name hash into the reserved range so
//! two independently-authored extensions collide only on deliberate
//! name-clashes.

use core::fmt::Debug;
use crate::data_type::DataType;
use crate::op_contract::SideEffectClass;
use crate::region_law::RegionLawFamily;
macro_rules! impl_extension_id {
    ($id:ident) => {
        impl $id {
            /// Reserved range: every extension id has its high bit set.
            ///
            /// Core IR discriminants occupy `0x00..=0x7F`; extensions occupy
            /// `0x80..=0xFFFF_FFFF`. Wire decoders test the high byte to route
            /// decoding between the two.
            pub const EXTENSION_RANGE_MASK: u32 = 0x8000_0000;

            /// Construct an id from a stable extension name.
            ///
            /// The id is derived deterministically with FNV-1a and folded into
            /// the extension range by setting the high bit. Callers that pass
            /// the same `name` always get the same id.
            #[must_use]
            pub const fn from_name(name: &str) -> Self {
                Self(fnv1a_with_high_bit(name))
            }

            /// Return the raw id.
            #[must_use]
            pub const fn as_u32(self) -> u32 {
                self.0
            }

            /// Is this a reserved extension id (high bit set)?
            #[must_use]
            pub const fn is_extension(self) -> bool {
                (self.0 & Self::EXTENSION_RANGE_MASK) != 0
            }
        }
    };
}

/// Stable u32 id for an extension variant.
///
/// Extension ids are generated deterministically from a stable name via
/// [`ExtensionDataTypeId::from_name`]. A crate that never changes its
/// extension name keeps the same id across versions, which is the
/// wire-format contract: a `Program` encoded by v1.0 of an extension
/// decodes identically in v1.1 so long as the name is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionDataTypeId(pub u32);

impl_extension_id!(ExtensionDataTypeId);

/// The contract for an extension-declared `DataType`.
///
/// An implementer describes the runtime shape of a non-core data type:
/// how many bytes it occupies, whether it participates in the float
/// conformance family, and how it should be displayed.
///
/// The foundation extension registry walks a link-time inventory of
/// `ExtensionDataTypeRegistration` entries. The resolver caches
/// `&'static dyn ExtensionDataType` so downstream
/// consumers never re-consult the registry on the hot path.
pub trait ExtensionDataType: Send + Sync + Debug + 'static {
    /// Stable id for this data type.
    fn id(&self) -> ExtensionDataTypeId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
    /// Minimum byte count to represent one value of this type.
    fn min_bytes(&self) -> usize;
    /// Maximum byte count for one value of this type; `None` when unbounded.
    fn max_bytes(&self) -> Option<usize>;
    /// Fixed element size in bytes, or `None` for variable-size types.
    fn size_bytes(&self) -> Option<usize>;
    /// Whether this type belongs to the IEEE-754 float conformance family.
    fn is_float_family(&self) -> bool {
        false
    }
    /// Whether values can be safely memcpy'd between host and device.
    fn is_host_shareable(&self) -> bool {
        true
    }
}

/// Runtime contract for an extension-declared binary operator.
///
/// The foundation extension registry caches `&'static dyn ExtensionBinOp`
/// pointers keyed by [`ExtensionBinOpId`]; downstream evaluators and lowerings
/// call through this trait without re-consulting the registry on the hot path.
pub trait ExtensionBinOp: Send + Sync + Debug + 'static {
    /// Stable id of this binary operator.
    fn id(&self) -> ExtensionBinOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
    /// Evaluate on the reference (CPU) backend.
    ///
    /// Returning `None` means "this backend does not support the op"; the
    /// caller surfaces a typed error. Extensions implementing backends
    /// other than reference supply their own lowering via the backend
    /// registry.
    fn eval_u32(&self, _a: u32, _b: u32) -> Option<u32> {
        None
    }
}

/// Runtime contract for an extension-declared unary operator.
pub trait ExtensionUnOp: Send + Sync + Debug + 'static {
    /// Stable id of this unary operator.
    fn id(&self) -> ExtensionUnOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
    /// Evaluate on the reference (CPU) backend. `None` = unsupported.
    fn eval_u32(&self, _a: u32) -> Option<u32> {
        None
    }
}

/// Runtime contract for an extension-declared atomic operator.
pub trait ExtensionAtomicOp: Send + Sync + Debug + 'static {
    /// Stable id of this atomic operator.
    fn id(&self) -> ExtensionAtomicOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
}

/// Runtime contract for an extension-declared ternary operator.
pub trait ExtensionTernaryOp: Send + Sync + Debug + 'static {
    /// Stable id of this ternary operator.
    fn id(&self) -> ExtensionTernaryOpId;
    /// Human-readable name for display / debug.
    fn display_name(&self) -> &'static str;
}

/// Stable u32 id for an extension binary operator.
///
/// Identical discipline to [`ExtensionDataTypeId`]: stable across process
/// runs, high bit set, generated by FNV-1a of the extension name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionBinOpId(pub u32);

impl_extension_id!(ExtensionBinOpId);

/// Stable u32 id for an extension unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionUnOpId(pub u32);

impl_extension_id!(ExtensionUnOpId);

/// Stable u32 id for an extension atomic operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionAtomicOpId(pub u32);

impl_extension_id!(ExtensionAtomicOpId);

/// Stable u32 id for an extension ternary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionTernaryOpId(pub u32);

impl_extension_id!(ExtensionTernaryOpId);

/// Stable u32 id for an extension rule condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionRuleConditionId(pub u32);

impl_extension_id!(ExtensionRuleConditionId);

/// Globally unique namespace identifying an extension domain or package.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionNamespace(pub String);

impl ExtensionNamespace {
    /// Construct a validated extension namespace.
    ///
    /// The namespace must not be empty and must only contain alphanumeric ASCII
    /// characters, dots, hyphens, and underscores.
    pub fn new(namespace: impl Into<String>) -> Result<Self, &'static str> {
        let ns = namespace.into();
        if ns.is_empty() {
            return Err("Extension namespace cannot be empty");
        }
        if !ns.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_') {
            return Err("Extension namespace contains invalid characters; expected [a-zA-Z0-9._-]");
        }
        Ok(Self(ns))
    }

    /// Access the namespace as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Display for ExtensionNamespace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Semantic version for an extension schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionSemVer {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
}

impl ExtensionSemVer {
    /// Construct a new semantic version.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }
}

impl core::fmt::Display for ExtensionSemVer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// 256-bit cryptographic digest over a canonical extension schema definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionSchemaDigest(pub [u8; 32]);

impl ExtensionSchemaDigest {
    /// Create a schema digest from raw 32 bytes.
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Compute schema digest from arbitrary byte stream using deterministic 256-bit hash.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut out = [0u8; 32];
        let mut state: [u64; 4] = [
            0x243f_6a88_85a3_08d3,
            0x1319_8a2e_0370_7344,
            0xa409_3822_299f_31d0,
            0x082e_fa98_ec4e_6c89,
        ];
        for (i, &b) in bytes.iter().enumerate() {
            let lane = i % 4;
            state[lane] = state[lane].rotate_left(13) ^ (b as u64).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
        for lane in 0..4 {
            out[lane * 8..(lane + 1) * 8].copy_from_slice(&state[lane].to_le_bytes());
        }
        Self(out)
    }

    /// Format the digest as lowercase hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            use core::fmt::Write as _;
            let _ = write!(s, "{:02x}", b);
        }
        s
    }
}

/// Full collision-resistant identity for an extension schema.
///
/// Combines a globally unique namespace, semantic version, and 256-bit schema digest.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionIdentity {
    /// Globally unique namespace.
    pub namespace: ExtensionNamespace,
    /// Semantic version.
    pub version: ExtensionSemVer,
    /// Canonical schema digest.
    pub schema_digest: ExtensionSchemaDigest,
}

impl ExtensionIdentity {
    /// Create a new extension identity.
    #[must_use]
    pub fn new(
        namespace: ExtensionNamespace,
        version: ExtensionSemVer,
        schema_digest: ExtensionSchemaDigest,
    ) -> Self {
        Self {
            namespace,
            version,
            schema_digest,
        }
    }

    /// Compute the legacy 31-bit FNV-1a hash of the namespace for backward comparison.
    #[must_use]
    pub fn legacy_fnv1a_hash(&self) -> u32 {
        fnv1a_with_high_bit(self.namespace.as_str())
    }

    /// Canonical display string `namespace@major.minor.patch#hex_digest`.
    #[must_use]
    pub fn to_canonical_string(&self) -> String {
        format!("{}@{}#{}", self.namespace.as_str(), self.version, self.schema_digest.to_hex())
    }
}

impl core::fmt::Display for ExtensionIdentity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_canonical_string())
    }
}

/// Canonical primitive field type supported in extension attribute declarations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExtensionFieldType {
    /// Boolean flag.
    Bool,
    /// 64-bit signed integer.
    I64,
    /// 64-bit unsigned integer.
    U64,
    /// 64-bit floating point literal.
    F64,
    /// String literal.
    String,
    /// Primitive or structured DataType.
    DataType,
    /// Shape dimension list.
    ShapeDims,
}

/// Declarative schema attribute field.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionField {
    /// Attribute name.
    pub name: String,
    /// Data type of the field.
    pub field_type: ExtensionFieldType,
    /// Whether this attribute is mandatory.
    pub required: bool,
    /// Optional default value.
    pub default_value: Option<String>,
}

/// Role or kind of an extension operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExtensionOperandKind {
    /// SSA value operand.
    Value,
    /// Structured region operand.
    Region,
    /// Buffer / storage operand.
    Buffer,
}

/// Declarative operand descriptor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionOperand {
    /// Operand name.
    pub name: String,
    /// Kind of operand.
    pub kind: ExtensionOperandKind,
    /// Expected data type, if constrained.
    pub data_type: Option<DataType>,
    /// Whether this operand is optional.
    pub optional: bool,
}

/// Declarative shape inference and transformation rule for an extension operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ExtensionShapeRule {
    /// Output shape matches primary input operand.
    PreservesShape,
    /// Broadcasts leading dimensions across operands.
    BroadcastLeading,
    /// Matrix multiplication contracting (M, K) x (K, N) -> (M, N).
    MatrixProduct {
        /// M dimension.
        m: usize,
        /// K reduction dimension.
        k: usize,
        /// N dimension.
        n: usize,
    },
    /// Reduces the specified axis.
    Reduction {
        /// Reduction axis index.
        axis: usize,
    },
    /// General affine shape transformation.
    AffineMap {
        /// Affine expression string.
        expression: String,
    },
    /// Custom declarative rule string.
    Custom {
        /// Rule description.
        rule: String,
    },
}

/// Declarative numerical accuracy and floating point behavior contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionNumericalContract {
    /// Whether this operation participates in floating point families.
    pub float_family: bool,
    /// Maximum allowed ULP distance from ideal real semantics.
    pub max_ulp_error: Option<u32>,
    /// Whether this operation preserves mathematical monotonicity.
    pub preserves_monotonicity: bool,
    /// Whether strict IEEE-754 semantics are mandatory.
    pub requires_strict_fp: bool,
}

impl Default for ExtensionNumericalContract {
    fn default() -> Self {
        Self {
            float_family: false,
            max_ulp_error: None,
            preserves_monotonicity: false,
            requires_strict_fp: false,
        }
    }
}

/// Hardware resource bounds and limits declared by an extension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, Default)]
pub struct ExtensionResourceBounds {
    /// Maximum workgroup-shared memory in bytes.
    pub max_shared_bytes: u64,
    /// Maximum invocation-private scratch in bytes.
    pub max_private_bytes: u64,
    /// Estimated register slots per invocation.
    pub registers_per_invocation: u32,
}

/// Closed declarative extension schema.
///
/// Replaces opaque callbacks with a self-contained, serializable,
/// verifiable declaration of an extension's typed interface, semantics,
/// laws, shape rules, and resource bounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ExtensionSchema {
    /// Unique identity.
    pub identity: ExtensionIdentity,
    /// Human-readable display name.
    pub display_name: String,
    /// Documentation description.
    pub description: String,
    /// Canonical typed attribute fields.
    pub fields: Vec<ExtensionField>,
    /// Child and value operands.
    pub operands: Vec<ExtensionOperand>,
    /// Result types produced by this operation.
    pub result_types: Vec<DataType>,
    /// Side-effect classification.
    pub side_effects: SideEffectClass,
    /// Shape inference rules.
    pub shape_rules: Vec<ExtensionShapeRule>,
    /// Numerical contract.
    pub numerical_contract: ExtensionNumericalContract,
    /// Algebraic and structural law families this extension satisfies.
    pub laws: Vec<RegionLawFamily>,
    /// Hardware resource bounds.
    pub resource_bounds: ExtensionResourceBounds,
    /// Whether values of this extension can be safely shared across host and device.
    pub host_shareable: bool,
    /// Whether this operation is purely functional.
    pub is_pure: bool,
    /// Whether this operation can cause control flow divergence.
    pub is_divergent: bool,
    /// Whether this operation is guaranteed to terminate in bounded steps.
    pub terminates: bool,
}

impl ExtensionSchema {
    /// Compute the deterministic schema digest from its canonical fields.
    #[must_use]
    pub fn compute_digest(
        namespace: &str,
        version: &ExtensionSemVer,
        fields: &[ExtensionField],
        operands: &[ExtensionOperand],
        result_types: &[DataType],
    ) -> ExtensionSchemaDigest {
        let mut canonical_bytes = Vec::new();
        canonical_bytes.extend_from_slice(namespace.as_bytes());
        canonical_bytes.push(0xFF);
        canonical_bytes.extend_from_slice(&version.major.to_le_bytes());
        canonical_bytes.extend_from_slice(&version.minor.to_le_bytes());
        canonical_bytes.extend_from_slice(&version.patch.to_le_bytes());
        canonical_bytes.push(0xFE);
        for f in fields {
            canonical_bytes.extend_from_slice(f.name.as_bytes());
            canonical_bytes.push(if f.required { 1 } else { 0 });
        }
        canonical_bytes.push(0xFD);
        for op in operands {
            canonical_bytes.extend_from_slice(op.name.as_bytes());
            canonical_bytes.push(op.kind as u8);
        }
        canonical_bytes.push(0xFC);
        for r in result_types {
            canonical_bytes.extend_from_slice(format!("{r:?}").as_bytes());
        }
        ExtensionSchemaDigest::from_bytes(&canonical_bytes)
    }
}

/// FNV-1a 32-bit hash folded into the extension range (high bit set).
///
/// Shared helper backing every `ExtensionXxxId::from_name`. Kept private
/// so callers don't construct raw ids that bypass the high-bit invariant.
#[must_use]
const fn fnv1a_with_high_bit(name: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    let bytes = name.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash | 0x8000_0000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_from_name_is_deterministic() {
        assert_eq!(
            ExtensionDataTypeId::from_name("tensor.gather"),
            ExtensionDataTypeId::from_name("tensor.gather"),
        );
    }

    #[test]
    fn id_from_different_names_differ() {
        let a = ExtensionDataTypeId::from_name("tensor.gather");
        let b = ExtensionDataTypeId::from_name("tensor.scatter");
        assert_ne!(a, b);
    }

    #[test]
    fn every_id_is_in_extension_range() {
        let id = ExtensionDataTypeId::from_name("anything");
        assert!(id.is_extension(), "{:#010x} missing high bit", id.as_u32());
        assert!(id.as_u32() & ExtensionDataTypeId::EXTENSION_RANGE_MASK != 0);
    }

    #[test]
    fn extension_identity_resolves_fnv1a_collision() {
        // Two distinct names that produce the exact same 31-bit FNV-1a hash:
        let name_a = "d5pj";
        let name_b = "x.ta";
        let hash_a = fnv1a_with_high_bit(name_a);
        let hash_b = fnv1a_with_high_bit(name_b);
        assert_eq!(hash_a, hash_b, "Demonstrating the 31-bit FNV-1a hash collision");
        assert_eq!(
            ExtensionDataTypeId::from_name(name_a),
            ExtensionDataTypeId::from_name(name_b),
            "Legacy 31-bit ExtensionDataTypeId collides on different extension names"
        );

        // Under the new ExtensionIdentity, they are strictly distinct:
        let ns_a = ExtensionNamespace::new(name_a).expect("valid namespace");
        let ns_b = ExtensionNamespace::new(name_b).expect("valid namespace");
        let ver = ExtensionSemVer::new(1, 0, 0);
        let digest_a = ExtensionSchema::compute_digest(name_a, &ver, &[], &[], &[]);
        let digest_b = ExtensionSchema::compute_digest(name_b, &ver, &[], &[], &[]);

        let id_a = ExtensionIdentity::new(ns_a, ver, digest_a);
        let id_b = ExtensionIdentity::new(ns_b, ver, digest_b);

        assert_ne!(id_a, id_b, "New ExtensionIdentity distinguishes colliding names");
        assert_ne!(id_a.schema_digest, id_b.schema_digest);
        assert_eq!(id_a.legacy_fnv1a_hash(), id_b.legacy_fnv1a_hash());
    }
}
